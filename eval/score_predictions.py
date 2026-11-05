#!/usr/bin/env python3
"""Re-score saved eval predictions with exact and category-level metrics.

Reads any `*_predictions.jsonl` written by run_baseline.py and recomputes micro
P/R/F1 under two matching rules:

- **exact**: full ICD code must match (what run_baseline.py already reports).
- **category**: codes are truncated to their 3-character prefix (E11.9 -> E11)
  before matching, giving partial credit for "right category, wrong subcode".

The baseline eval found cancer staging/laterality codes were frequently
close-but-wrong (right category, wrong digit), which exact match zeroes out —
this quantifies how much of the gap is subcode precision vs. genuinely missing
the diagnosis. Works retroactively on every run already on disk, which is why
it lives here rather than inside run_baseline.py.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

DEGENERATE_PRED_THRESHOLD = 15  # keep in sync with run_baseline.py


def category(code: str) -> str:
    return code.split(".")[0][:3]


def micro(records: list[dict], key) -> dict:
    tp = fp = fn = 0
    for r in records:
        pred = {key(c) for c in r["pred"]}
        gold = {key(c) for c in r["gold"]}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return {"precision": p, "recall": rc, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def describe(path: Path) -> dict:
    records = [json.loads(line) for line in path.open()]
    clean = [r for r in records if len(r["pred"]) <= DEGENERATE_PRED_THRESHOLD]
    n_unique = len({tuple(sorted(r["pred"])) for r in clean})
    n_exact_admissions = sum(1 for r in clean if set(r["pred"]) == set(r["gold"]))
    n_zero_overlap = sum(1 for r in clean if not (set(r["pred"]) & set(r["gold"])))
    return {
        "run": path.name.replace("_predictions.jsonl", ""),
        "n_samples": len(records),
        "n_degenerate_outliers": len(records) - len(clean),
        "exact": micro(clean, lambda c: c),
        "category": micro(clean, category),
        "n_unique_predictions": n_unique,
        "unique_prediction_rate": n_unique / len(clean) if clean else 0.0,
        "n_exact_set_matches": n_exact_admissions,
        "zero_overlap_rate": n_zero_overlap / len(clean) if clean else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        nargs="*",
        help="Prediction jsonl paths (default: every *_predictions.jsonl in eval/results).",
    )
    parser.add_argument("--out", default=None, help="Write the comparison JSON here.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.predictions] or sorted(RESULTS_DIR.glob("*_predictions.jsonl"))
    if not paths:
        parser.error(f"no prediction files found in {RESULTS_DIR}")

    reports = [describe(p) for p in paths]

    hdr = f"{'run':<52} {'exact F1':>9} {'cat F1':>9} {'cat/exact':>10} {'uniq':>7} {'zero-ov':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        ratio = r["category"]["f1"] / r["exact"]["f1"] if r["exact"]["f1"] else float("inf")
        print(
            f"{r['run']:<52} {r['exact']['f1']:>9.4f} {r['category']['f1']:>9.4f} "
            f"{ratio:>10.2f} {r['unique_prediction_rate']:>7.1%} {r['zero_overlap_rate']:>8.1%}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
