#!/usr/bin/env python3
"""Stratify micro-F1 by ICD code training frequency (the head/tail breakdown).

Each code is bucketed by how often it appears in the v2 training set; a TP/FP/FN is credited to the
bucket of the code involved (gold code's bucket for TP/FN, predicted code's bucket for FP). This shows
where each method's accuracy comes from (frequent "head" codes) and where it fails (rare "tail"), and
in particular whether retrieval/agentic reach tail codes the supervised head cannot.

No new compute: reads existing *_predictions.jsonl.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BUCKETS = [("very_freq (>=1000)", 1000, 10**9), ("freq (100-999)", 100, 999),
           ("medium (25-99)", 25, 99), ("rare (2-24)", 2, 24), ("singleton (1)", 1, 1)]


def bucket_of(count):
    for name, lo, hi in BUCKETS:
        if lo <= count <= hi:
            return name
    return "unseen (0)"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", required=True, help="predictions jsonl files")
    ap.add_argument("--labels", nargs="+", default=None, help="display labels (parallel to --preds)")
    args = ap.parse_args()

    tr = pd.read_parquet(ROOT / "data/processed/train.parquet")
    tr = tr[tr["ver"] == "10"]
    cnt = Counter(c.upper() for cs in tr["icd_codes"] for c in cs)
    labels = args.labels or [Path(p).name.replace("_predictions.jsonl", "") for p in args.preds]

    for pred_file, label in zip(args.preds, labels):
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
        gold_tot = defaultdict(int)
        for line in open(pred_file):
            r = json.loads(line)
            gold = {c.upper() for c in r["gold"]}; pred = {c.upper() for c in r["pred"]}
            for c in gold:
                gold_tot[bucket_of(cnt.get(c, 0))] += 1
            for c in gold & pred:
                tp[bucket_of(cnt.get(c, 0))] += 1
            for c in pred - gold:
                fp[bucket_of(cnt.get(c, 0))] += 1
            for c in gold - pred:
                fn[bucket_of(cnt.get(c, 0))] += 1
        print(f"\n=== {label} ===")
        print(f"{'bucket':<20}{'gold':>7}{'P':>7}{'R':>7}{'F1':>7}")
        for name, _, _ in BUCKETS:
            p, r, f = prf(tp[name], fp[name], fn[name])
            print(f"{name:<20}{gold_tot[name]:>7}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
        # overall
        P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        print(f"{'OVERALL':<20}{sum(gold_tot.values()):>7}{P:>7.3f}{R:>7.3f}{F:>7.3f}")


if __name__ == "__main__":
    main()
