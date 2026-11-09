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
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved to {out_dir}")
    for name, s in summary["splits"].items():
        print(f"  {name:12s} {s['questions']:6d} questions  {s['databases']:4d} databases  {s['by_source']}")
    return summary