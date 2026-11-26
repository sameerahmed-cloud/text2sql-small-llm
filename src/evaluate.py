"""
Every score in the project comes from this file.

For each question it:
  1. pulls the SQL out of the model's answer text   (extract_sql)
  2. runs the model's SQL and the correct SQL on the database,
     read-only and with a time limit                 (run_sql)
  3. compares the two results, ignoring row order    (same_result)
  4. labels the outcome: correct, or what went wrong (grade_one)
Then it adds up the score, with a 95% confidence interval, and saves a
results file with one entry per question            (evaluate)

Colab use:
    from src.evaluate import evaluate
    # predictions: {question id: the model's full answer text}
    report = evaluate(records, predictions, paths, run_id="baseline-0")
"""

import json
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

TIMEOUT_S = 30  # a query running longer than this counts as "timeout"

# Outcome labels (one per question)
CORRECT = "correct"
WRONG_RESULT = "runs but wrong result"
SYNTAX = "syntax error"
UNKNOWN_NAME = "no such column/table"
TIMEOUT = "timeout"
OTHER_ERROR = "other error"
NO_SQL = "no SQL found"
GOLD_ERROR = "gold SQL failed"  # the dataset's own SQL failed: a problem with the data, not the model


# ---------------------------------------------------------------- 1. extract
def extract_sql(text):
    """Take the SQL out of the model's answer.
    Uses the first ```sql ... ``` block; else any ``` ... ``` block; else the whole text."""
    m = re.search(r"```sql\s*(.*?)```", text, re.S | re.I) or re.search(r"```\s*(.*?)```", text, re.S)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip()


# ---------------------------------------------------------------- 2. run
def run_sql(db_file, sql, timeout_s=TIMEOUT_S):
    """Run one query read-only with a time limit.
    Returns (rows, None) on success or (None, outcome_label, message) on failure."""
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)  # ro = cannot change the database
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    # SQLite calls this every ~10,000 steps of work; returning True stops the query
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        return con.execute(sql).fetchall(), None, ""
    except sqlite3.Error as e:
        msg = str(e)
        if msg == "interrupted":
            label = TIMEOUT
        elif "syntax error" in msg or "incomplete input" in msg or "unrecognized token" in msg:
            label = SYNTAX
        elif "no such column" in msg or "no such table" in msg:
            label = UNKNOWN_NAME
        else:
            label = OTHER_ERROR
        return None, label, msg
    finally:
        con.close()


# ---------------------------------------------------------------- 3. compare
def same_result(rows_a, rows_b):
    """True if both results hold the same rows, in any order.
    Counter counts each row, so duplicates must match too (a 'multiset')."""
    return Counter(rows_a) == Counter(rows_b)


# ---------------------------------------------------------------- 4. grade
def grade_one(db_file, pred_text, gold_sql, timeout_s=TIMEOUT_S):
    """Grade one answer. Returns a dict describing what happened."""
    pred_sql = extract_sql(pred_text or "")
    out = {"pred_sql": pred_sql, "correct": False, "outcome": "", "error": ""}

    gold_rows, gold_label, gold_msg = run_sql(db_file, gold_sql, timeout_s)
    if gold_label:
        out.update(outcome=GOLD_ERROR, error=f"{gold_label}: {gold_msg}")
        return out
    if not pred_sql:
        out["outcome"] = NO_SQL
        return out

    pred_rows, label, msg = run_sql(db_file, pred_sql, timeout_s)
    if label:
        out.update(outcome=label, error=msg)
    elif same_result(pred_rows, gold_rows):
        out.update(outcome=CORRECT, correct=True)
    else:
        out["outcome"] = WRONG_RESULT
    return out


# ---------------------------------------------------------------- score
def bootstrap_ci(correct_flags, n_resamples=1000, seed=42):
    """95% confidence interval for accuracy (in %).
    Re-draw the same number of questions at random (with repeats) 1,000 times,
    score each draw, and take the middle 95% of those scores."""
    rng = random.Random(seed)
    n = len(correct_flags)
    scores = sorted(
        100 * sum(rng.choice(correct_flags) for _ in range(n)) / n for _ in range(n_resamples)
    )
    return round(scores[int(0.025 * n_resamples)], 2), round(scores[int(0.975 * n_resamples) - 1], 2)


def evaluate(records, predictions, paths, run_id, out_dir="results", timeout_s=TIMEOUT_S):
    """Grade every record and save results/<run_id>.json.

    records:     list of question dicts (from the split JSONL files)
    predictions: {record id: model answer text}; missing ids count as "no SQL found"
    paths:       from get_data.unpack_all(), used to find each database file
    """
    from src.data_prep import db_path

    per_question = []
    for i, r in enumerate(records, 1):
        g = grade_one(db_path(r, paths), predictions.get(r["id"], ""), r["gold_sql"], timeout_s)
        per_question.append({"id": r["id"], "db_id": r["db_id"], "question": r["question"],
                             "gold_sql": r["gold_sql"], "difficulty": r.get("difficulty", ""), **g})
        if i % 250 == 0:
            print(f"  graded {i}/{len(records)}")

    flags = [q["correct"] for q in per_question]
    counts = Counter(q["outcome"] for q in per_question)
    report = {
        "run_id": run_id,
        "questions": len(flags),
        "accuracy": round(100 * sum(flags) / len(flags), 2),
        "ci95": bootstrap_ci(flags),
        "outcomes": dict(counts.most_common()),
        "per_question": per_question,
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / f"{run_id}.json"
    out_file.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{run_id}: {report['accuracy']}% correct of {len(flags)} "
          f"(95% CI {report['ci95'][0]}–{report['ci95'][1]}%)")
    for outcome, k in counts.most_common():
        print(f"  {outcome:24} {k:>5}")
    print(f"Saved {out_file}")
    return report