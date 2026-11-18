"""
ONE prompt template used for every model (Phase 1.3).

Every model in the project (base 0.5B/1.5B, the 7B comparison, fine-tuned
versions, GRPO) sees exactly the same input, built here. If the prompt changes,
all results must be re-run, so treat this file as frozen once baselines exist.

A prompt has two messages (the "chat format" the model was trained on):
  system: who the model is and the house rules
  user:   database schema + sample values, the BIRD hint (if any), the question,
          and the instruction to answer with SQL only inside a ```sql block

Main functions:
  get_schema(db_path)                         schema text for one database (cached)
  build_messages(db_path, question, evidence) list of chat messages (used for training, 2.1)
  build_prompt(db_path, question, evidence, tokenizer)
                                              final text the model reads (used for generation)
  measure_prompt_lengths(records, paths, tokenizer)
                                              token-length statistics for a set of questions

Colab use:
    from transformers import AutoTokenizer
    from src.prompts import build_prompt
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    text = build_prompt(db_file, question, evidence, tok)
"""

import sqlite3
import statistics
from collections import defaultdict
from functools import lru_cache

# ---------------------------------------------------------------- settings
SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct SQLite queries. "
    "Use only the tables and columns that exist in the given database schema."
)
INSTRUCTION = (
    "Write one SQLite query that answers the question. "
    "Return only the SQL, inside a ```sql code block, with no explanation."
)
N_SAMPLES = 3            # sample values shown per column
MAX_VALUE_CHARS = 40     # long text values are cut to this length
SAMPLE_SCAN_ROWS = 1000  # only look at the first N rows of a table when picking samples
                         # (keeps huge BIRD tables fast: no full-table DISTINCT scan)


# ---------------------------------------------------------------- helpers
def _quote_ident(name):
    """Wrap a table/column name in double quotes so names with spaces or
    symbols (common in BIRD, e.g. "Free Meal Count (K-12)") are safe in SQL."""
    return '"' + name.replace('"', '""') + '"'


def _format_value(value):
    """Turn one database value into short, SQL-looking text.
    Text gets single quotes (like in a WHERE clause), numbers stay bare.
    Returns None for values we don't want to show (binary blobs)."""
    if isinstance(value, bytes):
        return None
    if isinstance(value, str):
        value = " ".join(value.split())  # squash newlines/tabs into single spaces
        if len(value) > MAX_VALUE_CHARS:
            value = value[:MAX_VALUE_CHARS] + "..."
        return "'" + value.replace("'", "''") + "'"
    return repr(value)  # int / float


def _sample_values(con, table, column, n=N_SAMPLES):
    """Up to n distinct non-empty values of one column, from the first rows."""
    t, c = _quote_ident(table), _quote_ident(column)
    sql = (f"SELECT DISTINCT {c} FROM "
           f"(SELECT {c} FROM {t} WHERE {c} IS NOT NULL AND {c} != '' LIMIT {SAMPLE_SCAN_ROWS}) "
           f"LIMIT {n}")
    try:
        rows = con.execute(sql).fetchall()
    except sqlite3.Error:
        return []  # odd column (e.g. broken type): just show no samples
    shown = [_format_value(r[0]) for r in rows]
    return [v for v in shown if v is not None]


def _clean_ddl(ddl):
    """Drop blank lines and trailing spaces from a CREATE TABLE statement (saves tokens)."""
    return "\n".join(line.rstrip() for line in ddl.splitlines() if line.strip())


# ---------------------------------------------------------------- schema
@lru_cache(maxsize=None)
def get_schema(db_path):
    """Schema text for one database: each CREATE TABLE followed by sample values.

    Cached: BIRD dev has 1,534 questions over only 11 databases, so each schema
    is built once and reused (pass db_path as a str so the cache key is stable).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only, like the exam
    # Some BIRD databases contain text that isn't valid UTF-8; replace bad bytes
    # instead of crashing.
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()  # ORDER BY rowid = the order the tables were created in

        blocks = []
        for table, ddl in tables:
            lines = [_clean_ddl(ddl)]
            samples = []
            for col in con.execute(f"PRAGMA table_info({_quote_ident(table)})"):
                col_name, col_type = col[1], (col[2] or "").upper()
                if "BLOB" in col_type:
                    continue  # images/binary data: not useful to the model
                values = _sample_values(con, table, col_name)
                if values:
                    samples.append(f"  {col_name}: {', '.join(values)}")
            if samples:
                lines.append("/* Sample values:\n" + "\n".join(samples) + "\n*/")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
    finally:
        con.close()


# ---------------------------------------------------------------- prompt
def build_messages(db_path, question, evidence=""):
    """The prompt as chat messages (system + user).

    Phase 2.1 adds the gold SQL as a third, 'assistant' message for training.
    """
    parts = ["Database schema:", get_schema(str(db_path)), ""]
    if evidence:  # BIRD's hint, e.g. "eligible free rate = Free Meal Count / Enrollment"
        parts += [f"Hint: {evidence}", ""]
    parts += [f"Question: {question}", "", INSTRUCTION]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_prompt(db_path, question, evidence, tokenizer):
    """The exact text the model reads, in the model's own chat format.

    add_generation_prompt=True ends the text with the 'assistant' header, so
    the model's next tokens are its answer.
    """
    return tokenizer.apply_chat_template(
        build_messages(db_path, question, evidence),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------- lengths
def _percentile(sorted_vals, p):
    """Value below which p% of the (sorted) numbers fall."""
    i = min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def measure_prompt_lengths(records, paths, tokenizer, name="", limits=(1024, 2048, 4096)):
    """Count prompt tokens for every record; print and return summary stats.

    Why: training memory on a T4 grows with sequence length, so these numbers
    decide max_seq_length in Phase 3 (and how many tokens vLLM must handle).
    """
    from src.data_prep import db_path as record_db_path

    lengths, by_db = [], defaultdict(list)
    for r in records:
        text = build_prompt(record_db_path(r, paths), r["question"], r["evidence"], tokenizer)
        n = len(tokenizer(text)["input_ids"])
        lengths.append(n)
        by_db[r["db_id"]].append(n)

    s = sorted(lengths)
    stats = {
        "set": name,
        "questions": len(s),
        "mean": round(statistics.mean(s)),
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "max": s[-1],
        **{f"over_{lim}": sum(n > lim for n in s) for lim in limits},
        "longest_dbs": sorted(((max(v), db) for db, v in by_db.items()), reverse=True)[:3],
    }

    print(f"\n{name}: {stats['questions']} prompts | mean {stats['mean']} | "
          f"p50 {stats['p50']} | p95 {stats['p95']} | max {stats['max']} tokens")
    for lim in limits:
        k = stats[f"over_{lim}"]
        print(f"  over {lim:>5} tokens: {k:>5} ({100 * k / len(s):.1f}%)")
    print("  longest databases:", ", ".join(f"{db} ({n})" for n, db in stats["longest_dbs"]))
    return stats