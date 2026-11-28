"""
Data preparation for the project.

Phase 1.2 — Split by database.
  make_splits(paths) reads the raw BIRD/Spider question files and writes:
    train.jsonl        questions the model may learn from
    val.jsonl          questions for choosing between experiments (held-out training DBs)
    test_bird.jsonl    BIRD dev   = main exam (databases never used in training)
    test_spider.jsonl  Spider dev = second exam
  Every database appears in exactly ONE split, so the model is always tested on
  databases it has never seen.

Phase 2.1/2.2 — Training files.
  make_chat_files(paths) writes sft_train.jsonl / sft_val.jsonl: chat-format
  conversations whose gold SQL has been run and works (check_gold), with tidy spacing
  (tidy_sql, verified by verify_tidy). 2.4 (leakage check) will add functions here too.

Colab use:
    from src.get_data import unpack_all
    from src.data_prep import make_splits
    paths = unpack_all(include_bird_train=True)
    summary = make_splits(paths)
"""

import json
import random
import re
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

DRIVE_PROCESSED = Path("/content/drive/MyDrive/text2sql/data/processed")
SEED = 42            # fixed seed -> the same validation databases every time
VAL_FRACTION = 0.10  # ~10% of training questions go to validation


# ---------------------------------------------------------------- loading
def load_records(paths):
    """Read every raw question file into one common format.

    BIRD and Spider name their fields differently (BIRD: "SQL", Spider: "query"),
    so we convert both into the same keys here, once.
    """
    records = []
    for source, info in paths.items():          # source: bird_dev, bird_train, spider
        for split, qpath in info["questions"].items():
            raw = json.loads(Path(qpath).read_text(encoding="utf-8"))
            for i, q in enumerate(raw):
                records.append({
                    "id": f"{source}-{split}-{i}",
                    "source": source,            # which database folder to use
                    "orig_split": split,         # the dataset's own split name
                    "db_id": q["db_id"],
                    "question": q["question"].strip(),
                    "evidence": (q.get("evidence") or "").strip(),  # BIRD hint; "" for Spider
                    "gold_sql": (q.get("SQL") or q.get("query")).strip(),
                    "difficulty": q.get("difficulty", ""),          # BIRD dev only
                })
    return records


def db_path(record, paths):
    """Full path to a record's SQLite file (paths come from unpack_all)."""
    db_dir = Path(paths[record["source"]]["db_dir"])
    return db_dir / record["db_id"] / f"{record['db_id']}.sqlite"


# ---------------------------------------------------------------- schema leaks
SCHEMA_SIMILARITY_LIMIT = 0.5   # remove training DBs at or above this similarity
SCHEMA_CONTAINMENT_LIMIT = 0.8  # remove if one schema is mostly inside the other
SCHEMA_REVIEW_LEVEL = 0.3       # print pairs above this so you can eyeball them


def schema_signature(db_file):
    """The set of 'table.column' names in a database (lowercase).
    Two databases built from the same source share most of these, even if the
    database itself has a different name."""
    try:
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        tables = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        cols = set()
        for t in tables:
            safe = t.replace('"', '""')
            for row in con.execute(f'PRAGMA table_info("{safe}")'):
                cols.add(f"{t.lower()}.{row[1].lower()}")
        con.close()
        return cols
    except sqlite3.Error as e:
        print(f"  ⚠ could not read schema of {db_file}: {e}")
        return set()


def jaccard(a, b):
    """Overlap score from 0 (nothing shared) to 1 (identical)."""
    return len(a & b) / len(a | b) if (a | b) else 0.0


def containment(a, b):
    """Share of the SMALLER schema found in the bigger one (0 to 1).
    Catches cut-down copies: a database with 6 of another's 7 tables scores
    high here even if the 7th table is huge and drags Jaccard down."""
    return len(a & b) / min(len(a), len(b)) if (a and b) else 0.0


def find_train_leaks(train_pool, test_records, paths):
    """Find training databases that match a test database by name or schema.
    Returns {(source, db_id): reason} for databases to remove from training."""
    def sig(key):
        return schema_signature(db_path({"source": key[0], "db_id": key[1]}, paths))

    test_keys = sorted({(r["source"], r["db_id"]) for r in test_records})
    train_keys = sorted({(r["source"], r["db_id"]) for r in train_pool})
    test_sigs = {k: sig(k) for k in test_keys}

    print(f"Comparing {len(train_keys)} training DBs with {len(test_keys)} test DBs...")
    leaks = {}
    for k in train_keys:
        s = sig(k)
        for tk, ts in test_sigs.items():
            score = jaccard(s, ts)
            inside = containment(s, ts)
            same_name = k[1].lower() == tk[1].lower()
            scores = f"similarity {score:.2f}, containment {inside:.2f}"
            if same_name or score >= SCHEMA_SIMILARITY_LIMIT or inside >= SCHEMA_CONTAINMENT_LIMIT:
                leaks[k] = f"matches test DB {tk[0]}/{tk[1]} (name match: {same_name}, {scores})"
                break
            if score >= SCHEMA_REVIEW_LEVEL:
                print(f"  review: {k[0]}/{k[1]} vs {tk[0]}/{tk[1]} {scores} (kept)")
    for k, reason in leaks.items():
        print(f"  removed from training: {k[0]}/{k[1]} — {reason}")
    return leaks


# ---------------------------------------------------------------- splitting
def pick_val_dbs(records, fraction=VAL_FRACTION, seed=SEED):
    """Choose whole databases for validation until ~fraction of questions are covered."""
    counts = Counter(r["db_id"] for r in records)
    db_ids = sorted(counts)                 # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(db_ids)

    target = fraction * len(records)
    chosen, total = [], 0
    for db in db_ids:
        if total >= target:
            break
        chosen.append(db)
        total += counts[db]
    return chosen


def check_no_db_overlap(splits):
    """Fail loudly if any database name appears in two splits."""
    db_sets = {name: {r["db_id"] for r in recs} for name, recs in splits.items()}
    for a, b in combinations(db_sets, 2):
        shared = db_sets[a] & db_sets[b]
        if shared:
            raise AssertionError(f"Database overlap between {a} and {b}: {sorted(shared)}")
    print("✓ No database appears in more than one split")


def check_db_files(splits, paths):
    """Fail loudly if any question points to a missing database file."""
    missing = {
        (r["source"], r["db_id"])
        for recs in splits.values() for r in recs
        if not db_path(r, paths).exists()
    }
    if missing:
        raise FileNotFoundError(f"{len(missing)} database files missing, e.g. {sorted(missing)[:5]}")
    print("✓ Every question's database file exists")


def write_jsonl(records, path):
    """One JSON object per line: easy to stream and to read with pandas or datasets."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_splits(paths, out_dir=DRIVE_PROCESSED):
    records = load_records(paths)

    # 1. Test sets: the official dev sets (their databases are not in any train set)
    test_bird = [r for r in records if r["source"] == "bird_dev"]
    test_spider = [r for r in records if r["source"] == "spider" and r["orig_split"] == "dev"]

    # 2. Training pool: the official train sets
    train_pool = [r for r in records if r["orig_split"] == "train"]
    if not any(r["source"] == "bird_train" for r in train_pool):
        print("⚠ BIRD train not found: splitting Spider train only (re-run once BIRD train is unpacked)")

    # 2b. Remove training databases that duplicate a test database (by name or schema).
    #     The test sets are fixed benchmarks, so training is what gives way.
    leaks = find_train_leaks(train_pool, test_bird + test_spider, paths)
    train_pool = [r for r in train_pool if (r["source"], r["db_id"]) not in leaks]

    # 3. Validation: hold out whole databases, ~10% from each source separately,
    #    so validation reflects both BIRD-style and Spider-style questions
    val_keys = set()
    for source in sorted({r["source"] for r in train_pool}):
        group = [r for r in train_pool if r["source"] == source]
        val_keys |= {(source, db) for db in pick_val_dbs(group)}

    val = [r for r in train_pool if (r["source"], r["db_id"]) in val_keys]
    train = [r for r in train_pool if (r["source"], r["db_id"]) not in val_keys]

    splits = {"train": train, "val": val, "test_bird": test_bird, "test_spider": test_spider}

    # 4. Safety checks before saving anything
    check_no_db_overlap(splits)
    check_db_files(splits, paths)

    # 5. Save the splits and a small summary (the summary goes into GitHub)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "val_fraction": VAL_FRACTION, "splits": {}}
    for name, recs in splits.items():
        write_jsonl(recs, out_dir / f"{name}.jsonl")
        summary["splits"][name] = {
            "questions": len(recs),
            "databases": len({r["db_id"] for r in recs}),
            "by_source": dict(Counter(r["source"] for r in recs)),
        }
    summary["val_databases"] = sorted(f"{s}/{db}" for s, db in val_keys)
    summary["removed_from_train"] = {f"{s}/{db}": why for (s, db), why in sorted(leaks.items())}
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved to {out_dir}")
    for name, s in summary["splits"].items():
        print(f"  {name:12s} {s['questions']:6d} questions  {s['databases']:4d} databases  {s['by_source']}")
    return summary

# ---------------------------------------------------------------- 2.1 chat format
# Databases never used for training. works_cycles' schema alone is ~22,000 tokens
# (decided in 1.3): it can't fit any training length we could afford on a T4.
TRAIN_EXCLUDE_DBS = {"works_cycles", "retail_world"}  # retail_world: its gold SQL uses tables/columns (Order Details, Territories, CompanyName...) its DB file doesn't have; 330 of 373 fail


def read_jsonl(path):
    """Read a JSONL file back into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_chat_example(record, paths):
    """One question -> one training conversation.

    The system + user messages come from build_messages(), the SAME function the
    exam uses, so the model trains on exactly the prompt it will see at test time.
    The gold SQL becomes the assistant message, written the way we ask the model
    to answer (inside a ```sql block). Zero-shot format: no few-shot examples.
    """
    from src.prompts import build_messages, format_answer

    messages = build_messages(db_path(record, paths), record["question"], record["evidence"])
    messages.append({"role": "assistant", "content": format_answer(record["gold_sql"])})
    # id/source/db_id ride along so 2.2 (filtering) and 2.4 (leak check) can use them.
    # The trainer only reads "messages".
    return {"id": record["id"], "source": record["source"], "db_id": record["db_id"],
            "messages": messages}


def make_chat_files(paths, splits=("train", "val"), in_dir=DRIVE_PROCESSED, out_dir=DRIVE_PROCESSED,
                    check_gold_sql=True, tidy=True):
    """Write sft_<split>.jsonl (chat format) for each split and return counts.

    Steps per split:
      1. drop databases in TRAIN_EXCLUDE_DBS                    (2.1)
      2. run every gold query; drop the ones that fail        (2.2, check_gold)
      3. tidy the spacing of the gold SQL that's kept           (2.2, tidy_sql)
      4. turn each question into a chat conversation          (2.1, to_chat_example)
    Reads the split files from 1.2 (train.jsonl, val.jsonl) and never overwrites them.
    Needs BIRD train unpacked: unpack_all(include_bird_train=True).
    Also saves sft_summary.json (counts + every dropped question) next to the files.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in splits:
        records = read_jsonl(in_dir / f"{name}.jsonl")
        kept = [r for r in records if r["db_id"] not in TRAIN_EXCLUDE_DBS]
        excluded = len(records) - len(kept)

        dropped, empty = [], 0
        if check_gold_sql:
            checks = check_gold(kept, paths, out_dir / f"gold_check_{name}.jsonl")
            dropped = [{"id": r["id"], "db_id": r["db_id"], **checks[r["id"]]}
                       for r in kept if not checks[r["id"]]["ok"]]
            empty = sum(checks[r["id"]]["ok"] and checks[r["id"]]["rows"] == 0 for r in kept)
            kept = [r for r in kept if checks[r["id"]]["ok"]]

        n_tidied = 0
        if tidy:
            verify_tidy(kept, paths)   # stops everything if tidying ever changes a result
            tidied = []
            for r in kept:
                t = tidy_sql(r["gold_sql"])
                n_tidied += t != r["gold_sql"]
                tidied.append({**r, "gold_sql": t})
            kept = tidied

        chats = []
        for i, r in enumerate(kept, 1):
            chats.append(to_chat_example(r, paths))
            if i % 2000 == 0:
                print(f"  {name}: formatted {i}/{len(kept)}")

        out_file = out_dir / f"sft_{name}.jsonl"
        write_jsonl(chats, out_file)
        size_mb = out_file.stat().st_size / 1e6
        summary[name] = {"questions_in": len(records), "excluded_db": excluded,
                         "gold_failed": len(dropped),
                         "gold_failed_by_outcome": dict(Counter(d["outcome"] for d in dropped)),
                         "gold_empty_result_kept": empty, "gold_tidied": n_tidied,
                         "examples_out": len(chats),
                         "by_source": dict(Counter(c["source"] for c in chats)),
                         "file_mb": round(size_mb, 1),
                         "dropped": dropped}
        print(f"✓ {name}: {len(chats)} examples | excluded {excluded} ({sorted(TRAIN_EXCLUDE_DBS)}) | "
              f"gold failed {len(dropped)} {summary[name]['gold_failed_by_outcome']} | "
              f"empty result (kept) {empty} | tidied {n_tidied} -> {out_file} ({size_mb:.0f} MB)")

    (out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"Saved {out_dir / 'sft_summary.json'}")
    return summary


def check_chat_file(path, tokenizer, n=5, seed=SEED):
    """Spot-check a chat file with the model's own chat template.

    The key test: the full training text must START WITH the exact text the model
    reads at answer time (build_prompt's output). If it does, training and testing
    see identical prompts, and the only new part is the SQL answer.
    """
    rows = read_jsonl(path)
    for row in random.Random(seed).sample(rows, min(n, len(rows))):
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"], row["id"]
        assert msgs[-1]["content"].startswith("```sql\n") and msgs[-1]["content"].endswith("\n```"), row["id"]
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        assert full.startswith(prompt), f"{row['id']}: training text doesn't start with the test prompt"
    print(f"✓ {path}: {len(rows)} rows; {min(n, len(rows))} spot-checked, training text = test prompt + answer")
    print("\nEnd of one example (what the model learns to produce comes after 'assistant'):")
    print("..." + full[-400:])


# ---------------------------------------------------------------- 2.2 keep only working examples
def check_gold(records, paths, cache_file, timeout_s=None):
    """Run every record's gold SQL once with the exam's own run_sql (read-only, same
    30 s limit). Returns {id: {"ok", "outcome", "error", "rows", "seconds"}}.

    Results are appended to cache_file (on Drive) one line at a time, so if Colab
    disconnects, re-running skips everything already checked and carries on.
    """
    import time
    from src.evaluate import run_sql, TIMEOUT_S
    timeout_s = timeout_s or TIMEOUT_S

    cache_file = Path(cache_file)
    done = {}
    if cache_file.exists():
        done = {row["id"]: row for row in read_jsonl(cache_file)}
    todo = [r for r in records if r["id"] not in done]
    print(f"Gold check: {len(done)} already in {cache_file.name}, {len(todo)} to run")

    start = time.monotonic()
    with open(cache_file, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            t0 = time.monotonic()
            rows, label, msg = run_sql(db_path(r, paths), r["gold_sql"], timeout_s)
            row = {"id": r["id"], "ok": label is None, "outcome": label or "ok", "error": msg,
                   "rows": len(rows) if rows is not None else None,
                   "seconds": round(time.monotonic() - t0, 2)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()                 # written to Drive now, not at the end
            done[r["id"]] = row
            if i % 500 == 0:
                print(f"  checked {i}/{len(todo)} ({time.monotonic() - start:.0f} s)")
    return {r["id"]: done[r["id"]] for r in records}


# Quoted parts of SQL that must never be touched: 'text', "names", `names`, [names].
# '' and "" inside are SQL's way of writing a quote character, so they stay part of the match.
_QUOTED = re.compile(r"""('(?:[^']|'')*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])""")


def tidy_sql(sql):
    """Squash runs of spaces/newlines/tabs into one space, OUTSIDE quoted parts,
    and drop a trailing ';'. Example: 'a  =  "Solveig"' -> 'a = "Solveig"'.

    Safe because SQL ignores how much whitespace sits between words; only text
    inside quotes is meaningful, and that is left exactly as it was. Queries with
    comments (-- or /* */) are returned unchanged: joining lines could swallow code
    into a '--' comment.
    """
    parts = _QUOTED.split(sql)         # even positions: plain SQL, odd: quoted
    plain = parts[0::2]
    if any("--" in p or "/*" in p for p in plain):
        return sql
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\s+", " ", parts[i])
    return "".join(parts).strip().rstrip(";").strip()


def verify_tidy(records, paths, n=200, seed=SEED, timeout_s=None):
    """Proof on a random sample: tidied gold SQL must return exactly the same rows
    as the original. Pass records that already passed check_gold."""
    from src.evaluate import run_sql, same_result, TIMEOUT_S
    timeout_s = timeout_s or TIMEOUT_S

    changed = [r for r in records if tidy_sql(r["gold_sql"]) != r["gold_sql"]]
    sample = random.Random(seed).sample(changed, min(n, len(changed)))
    bad = []
    for r in sample:
        a, la, _ = run_sql(db_path(r, paths), r["gold_sql"], timeout_s)
        b, lb, _ = run_sql(db_path(r, paths), tidy_sql(r["gold_sql"]), timeout_s)
        if la or lb or not same_result(a, b):
            bad.append(r["id"])
    print(f"Tidy check: {len(changed)} of {len(records)} queries change; "
          f"{len(sample) - len(bad)}/{len(sample)} sampled give identical results")
    if bad:
        raise AssertionError(f"tidy_sql changed the result of: {bad[:10]}")