#!/usr/bin/env python3
"""Per-code-threshold selective coding: how much can we auto-code at a coder-grade precision bar?

A single global confidence threshold is wasteful: some codes (frequent, distinctive) are auto-codable
at high precision, others never are. Following the "per-code thresholds / variable error cost"
recommendation (Eight Recommendations for clinical-coding AI, 2024), we learn a per-code threshold on a
CALIBRATION split and apply it to a disjoint EVAL split:

  for each code j: the lowest threshold at which code j's precision on calibration is >= target
  (with >= min_support auto-accepts); codes that can never reach the target are ALWAYS deferred.

Then: auto-accept (admission, code) on eval iff prob >= that code's threshold. Report the overall
precision on the auto-coded slice and the coverage (recall) of the true workload -- the honest
"safely-automatable fraction". No test leakage (thresholds fit on calibration only).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def eval_at(P, gmask, gold_counts, targets, min_support, calib_idx, eval_idx):
    n_lab = P.shape[1]
    results = {}
    total_gold_eval = sum(gold_counts[a] for a in eval_idx)
    for target in targets:
        thr = np.full(n_lab, np.inf)  # inf = always defer
        for j in range(n_lab):
            pj = P[calib_idx, j]; cj = gmask[calib_idx, j]
            order = np.argsort(-pj)
            ps = pj[order]; cs = cj[order].astype(int)
            cum_tp = np.cumsum(cs); cum_n = np.arange(1, len(cs) + 1)
            prec = cum_tp / cum_n
            ok = np.where((prec >= target) & (cum_n >= min_support))[0]
            if len(ok):
                thr[j] = ps[ok[-1]]  # deepest point still >= target
        # apply to eval
        tp = fp = 0
        for a in eval_idx:
            accept = np.where(P[a] >= thr)[0]
            for j in accept:
                if gmask[a, j]: tp += 1
                else: fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        cov = tp / total_gold_eval if total_gold_eval else 0.0
        n_auto_codes = int((thr < np.inf).sum())
        results[target] = (prec, cov, tp + fp, n_auto_codes)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(ROOT / "eval/results_v2/supervised_selcurve"))
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.90, 0.85, 0.80])
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = np.load(f"{args.prefix}_probs.npy")
    labels = np.load(f"{args.prefix}_labels.npy")
    gold = json.load(open(f"{args.prefix}_gold.json"))
    lab2i = {c: i for i, c in enumerate(labels)}
    n_adm, n_lab = P.shape
    gmask = np.zeros_like(P, dtype=bool)
    gold_counts = np.zeros(n_adm, dtype=int)
    for a, g in enumerate(gold):
        gold_counts[a] = len(g)
        for c in g:
            j = lab2i.get(c)
            if j is not None: gmask[a, j] = True

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_adm)
    calib_idx = perm[: n_adm // 2]; eval_idx = perm[n_adm // 2:]
    print(f"n_adm={n_adm} (calib {len(calib_idx)} / eval {len(eval_idx)}), n_labels={n_lab}, "
          f"min_support={args.min_support}")

    res = eval_at(P, gmask, gold_counts, args.targets, args.min_support, calib_idx, eval_idx)
    print(f"\n=== Per-code-threshold autonomy frontier (eval split) ===")
    print(f"{'target prec':>12}{'eval precision':>16}{'coverage(recall)':>18}{'auto-codes':>12}{'#codes usable':>14}")
    for t in args.targets:
        prec, cov, n_auto, n_codes = res[t]
        print(f"{t:>12.2f}{prec:>16.3f}{cov:>18.3f}{n_auto:>12}{n_codes:>14}")


if __name__ == "__main__":
    main()
