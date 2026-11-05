#!/usr/bin/env python3
"""Selective-coding operating curve from per-code confidence scores.

Treats each (admission, candidate-label) pair as an auto-code decision with a confidence = the
supervised probe's sigmoid probability. Sweeping the confidence threshold traces precision vs how much
of the true coding workload is auto-captured (recall / coverage). The headline question: at a
coder-grade precision bar (>=0.90), what fraction of the gold codes can be auto-coded, deferring the
rest to a human?

Also reports the selective-risk / precision-coverage summary and precision at fixed autonomy levels.
Reads the probs/labels/gold dumped by train_supervised_baseline.py (--dump).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(ROOT / "eval/results_v2/supervised_selcurve"),
                    help="prefix of *_probs.npy / _labels.npy / _gold.json")
    ap.add_argument("--precision-bars", type=float, nargs="+", default=[0.95, 0.90, 0.85, 0.80])
    args = ap.parse_args()

    P = np.load(f"{args.prefix}_probs.npy")            # (n_adm, n_labels)
    labels = np.load(f"{args.prefix}_labels.npy")
    gold = json.load(open(f"{args.prefix}_gold.json"))  # list of lists
    lab2i = {c: i for i, c in enumerate(labels)}
    n_adm, n_lab = P.shape
    total_gold = sum(len(g) for g in gold)             # ALL gold codes (incl. out-of-label-space)
    reachable = sum(1 for g in gold for c in g if c in lab2i)
    print(f"n_adm={n_adm} n_labels={n_lab} total_gold={total_gold} "
          f"reachable_in_label_space={reachable} ({reachable/total_gold:.3f})")

    # Build a gold mask aligned to P: gmask[a, j] = 1 if label j is a gold code for admission a.
    gmask = np.zeros_like(P, dtype=bool)
    for a, g in enumerate(gold):
        for c in g:
            j = lab2i.get(c)
            if j is not None:
                gmask[a, j] = True

    # Flatten every (admission, label) cell -> (prob, is_correct). Sort by prob desc.
    prob = P.reshape(-1)
    correct = gmask.reshape(-1)
    order = np.argsort(-prob)
    prob_s = prob[order]; correct_s = correct[order]
    cum_tp = np.cumsum(correct_s)
    cum_n = np.arange(1, len(prob_s) + 1)
    precision = cum_tp / cum_n
    recall = cum_tp / total_gold                        # coverage of the TRUE workload

    # Precision at each coder-grade bar: the max coverage achievable while precision stays >= bar.
    print("\n=== Coder-grade autonomy frontier (supervised-probe confidence) ===")
    print(f"{'precision bar':>14}{'threshold':>11}{'auto-codes':>11}{'coverage(recall)':>18}{'precision':>11}")
    for bar in args.precision_bars:
        ok = np.where(precision >= bar)[0]
        if len(ok) == 0:
            print(f"{bar:>14.2f}{'—':>11}{'—':>11}{'never reached':>18}")
            continue
        i = ok[-1]                                       # deepest point still >= bar
        print(f"{bar:>14.2f}{prob_s[i]:>11.3f}{int(cum_n[i]):>11}{recall[i]:>18.3f}{precision[i]:>11.3f}")

    # Precision at fixed autonomy (volume auto-coded as fraction of total gold workload).
    print("\n=== Precision at fixed autonomy (auto-code volume / total gold) ===")
    print(f"{'autonomy':>10}{'threshold':>11}{'precision':>11}{'recall':>9}")
    for frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        i = min(int(frac * total_gold), len(prob_s) - 1)
        print(f"{frac:>10.2f}{prob_s[i]:>11.3f}{precision[i]:>11.3f}{recall[i]:>9.3f}")

    # Area under the precision-recall curve (selective-risk summary), trapezoid over recall.
    # Dedup to monotone recall grid for a stable AUC.
    aupr = float(np.trapz(precision, recall))
    print(f"\nArea under precision-coverage curve (AUC-PR proxy): {aupr:.4f}")


if __name__ == "__main__":
    main()
