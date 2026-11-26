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

Few-shot (Phase 1.5b): pass examples=... to build_messages / build_prompt.
  pick_fewshot_examples(train_records, paths) chooses 3 solved PRACTICE questions by a
  fixed rule; save_fewshot_examples / load_fewshot_examples store them (with their
  sheets already built) so the GPU session doesn't need the practice databases.
  Without examples, the prompt is exactly the zero-shot prompt above.

Colab use:
    from transformers import AutoTokenizer
    from src.prompts import build_prompt
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    text = build_prompt(db_file, question, evidence, tok)
"""

import json
import random
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
def user_message(db_path, question, evidence=""):
    """The 'user' text for one question: schema, optional hint, question, instruction."""
    parts = ["Database schema:", get_schema(str(db_path)), ""]
    if evidence:  # BIRD's hint, e.g. "eligible free rate = Free Meal Count / Enrollment"
        parts += [f"Hint: {evidence}", ""]
    parts += [f"Question: {question}", "", INSTRUCTION]
    return "\n".join(parts)


def format_answer(sql):
    """How a correct answer is written: the SQL inside a ```sql block.
    Used for few-shot examples now and for training answers in Phase 2.1."""
    return f"```sql\n{sql.strip()}\n```"


def build_messages(db_path, question, evidence="", examples=None):
    """The prompt as chat messages.

    Zero-shot (examples=None): system + user.
    Few-shot: system, then for each example a user message + its correct
    assistant answer, then the real user message last.
    Phase 2.1 adds the gold SQL as a final 'assistant' message for training.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples or []:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user_message(db_path, question, evidence)})
    return messages


def build_prompt(db_path, question, evidence, tokenizer, examples=None):
    """The exact text the model reads, in the model's own chat format.

    add_generation_prompt=True ends the text with the 'assistant' header, so
    the model's next tokens are its answer.
    """
    return tokenizer.apply_chat_template(
        build_messages(db_path, question, evidence, examples),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------- few-shot
# Fixed rule for choosing the 3 examples: one slot per SQL pattern, each from a
# different PRACTICE database (never an exam database), always the database with
# the smallest schema so the extra sheets stay short. Chosen by rule, not by hand,
# so the examples aren't tuned to the exam's mistakes.
FEWSHOT_SLOTS = [
    ("BIRD question with a hint and a JOIN",
     lambda r: r["source"] == "bird_train" and r["evidence"] and " JOIN " in r["gold_sql"].upper()),
    ("Spider question with GROUP BY",
     lambda r: r["source"] == "spider" and "GROUP BY" in r["gold_sql"].upper()),
    ("Spider question with a JOIN",
     lambda r: r["source"] == "spider" and " JOIN " in r["gold_sql"].upper()
     and "GROUP BY" not in r["gold_sql"].upper()),
]


def pick_fewshot_examples(train_records, paths, seed=42):
    """Choose one practice question per slot in FEWSHOT_SLOTS and build its sheet."""
    from src.data_prep import db_path as record_db_path

    rng = random.Random(seed)
    schema_size = {}  # characters of schema text per database
    used, chosen = set(), []
    for why, matches in FEWSHOT_SLOTS:
        cands = [r for r in train_records if matches(r) and (r["source"], r["db_id"]) not in used]
        if not cands:
            raise ValueError(f"No practice question fits the slot: {why}")
        for r in cands:
            key = (r["source"], r["db_id"])
            if key not in schema_size:
                schema_size[key] = len(get_schema(str(record_db_path(r, paths))))
        # the smallest database for this slot (db_id breaks ties, so the choice is fixed)
        best = min({(r["source"], r["db_id"]) for r in cands},
                   key=lambda k: (schema_size[k], k[1]))
        pool = sorted((r for r in cands if (r["source"], r["db_id"]) == best), key=lambda r: r["id"])
        r = rng.choice(pool)
        used.add(best)
        chosen.append({
            "why": why, "id": r["id"], "source": r["source"], "db_id": r["db_id"],
            "question": r["question"], "evidence": r["evidence"], "gold_sql": r["gold_sql"],
            "user": user_message(record_db_path(r, paths), r["question"], r["evidence"]),
            "assistant": format_answer(r["gold_sql"]),
        })
    return chosen


def save_fewshot_examples(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
    print(f"Saved {len(examples)} examples to {path}")


def load_fewshot_examples(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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