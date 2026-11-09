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

Later phases (2.1 chat formatting, 2.2 filtering, 2.4 leakage check) will add
functions to this same file.

Colab use:
    from src.get_data import unpack_all
    from src.data_prep import make_splits
    paths = unpack_all(include_bird_train=True)
    summary = make_splits(paths)
"""

import json
import random
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
            same_name = k[1].lower() == tk[1].lower()
            if same_name or score >= SCHEMA_SIMILARITY_LIMIT:
                leaks[k] = f"matches test DB {tk[0]}/{tk[1]} (name match: {same_name}, schema similarity {score:.2f})"
            elif score >= SCHEMA_REVIEW_LEVEL:
                print(f"  review: {k[0]}/{k[1]} vs {tk[0]}/{tk[1]} similarity {score:.2f} (kept)")
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