#!/usr/bin/env python3
"""Does the router's precision guarantee survive having to pick the threshold in advance?

`eval/selective_curve.py` takes the threshold at the deepest point where TEST precision still clears
the bar. That is the honest way to draw an operating curve in a paper, but it means the advertised
precision is in-sample: the bar was chosen on the very rows it is then reported on. A hospital cannot
do that -- it has to fix a cut-off today and live with whatever precision next month's admissions give.

So: split the held-out admissions in half, fit the global cut-off on the `cal` half at each bar, and
measure what it actually delivers on the disjoint `eval` half. The oracle columns re-pick the threshold
on the eval half itself, so the two are like-for-like on the same rows and the difference is purely the
cost of having to choose in advance.

Companion to `eval/selective_percode.py`, which asks the same question of per-code thresholds -- those
are the ones that do not survive (they undershoot every bar at this calibration size).

    python eval/threshold_transfer.py
    python eval/threshold_transfer.py --prefix eval/results_v2/plmicd_cuda_full_nodx_ep1
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = [
    ("qwen3 probe", "eval/results_v2/supervised_selcurve_qwen3"),
    ("PLM-ICD head (seed 42)", "eval/results_v2/plmicd_cuda_full_nodx_ep1"),
]


def gold_mask(gold, labels, shape):
    lab2i = {c: i for i, c in enumerate(labels)}
    gm = np.zeros(shape, dtype=bool)
    for a, g in enumerate(gold):
        for c in g:
            j = lab2i.get(c)
            if j is not None:
                gm[a, j] = True
    return gm


def deepest_point(P_sub, gm_sub, bar):
    """Threshold at the deepest point where precision over the flattened cells still clears `bar`."""
    p = P_sub.reshape(-1)
    o = np.argsort(-p)
    prec = np.cumsum(gm_sub.reshape(-1)[o]) / np.arange(1, p.size + 1)
    ok = np.where(prec >= bar)[0]
    if len(ok) == 0:
        return None, None, None
    i = ok[-1]
    return float(p[o][i]), float(prec[i]), int(i + 1)


def report(name, prefix, bars, seed):
    P = np.load(f"{prefix}_probs.npy")
    labels = np.load(f"{prefix}_labels.npy", allow_pickle=True)
    gold = json.load(open(f"{prefix}_gold.json"))
    gm = gold_mask(gold, labels, P.shape)
    counts = [len(g) for g in gold]

    perm = np.random.default_rng(seed).permutation(P.shape[0])
    cal, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
    gold_ev = sum(counts[a] for a in ev)
    print(f"\n### {name}   cal {len(cal)} / eval {len(ev)} admissions   gold codes in eval {gold_ev}")
    print(f"{'bar':>5}{'thr(cal)':>10}{'cal P':>8}{'eval P':>8}{'eval cov':>10}"
          f"{'oracle P':>10}{'oracle cov':>12}")
    for bar in bars:
        thr, calP, _ = deepest_point(P[cal], gm[cal], bar)
        if thr is None:
            print(f"{bar:>5.2f}   never reached on the calibration half")
            continue
        acc = P[ev] >= thr                               # the prospective decision
        n = int(acc.sum()); tp = int((acc & gm[ev]).sum())
        evP = tp / n if n else float("nan")
        # Oracle: re-pick on the eval half, i.e. what selective_curve.py would print for these rows.
        oThr, oP, oN = deepest_point(P[ev], gm[ev], bar)
        oTp = int(((P[ev] >= oThr) & gm[ev]).sum()) if oThr is not None else 0
        print(f"{bar:>5.2f}{thr:>10.3f}{calP:>8.3f}{evP:>8.3f}{tp / gold_ev:>10.3f}"
              f"{(oP if oP else float('nan')):>10.3f}{oTp / gold_ev:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", action="append", default=None,
                    help="prefix of *_probs.npy / _labels.npy / _gold.json; repeatable")
    ap.add_argument("--bars", type=float, nargs="+", default=[0.95, 0.90, 0.85, 0.80])
    ap.add_argument("--seed", type=int, default=42, help="matches eval/selective_percode.py's split")
    args = ap.parse_args()
    targets = [(p, p) for p in args.prefix] if args.prefix else \
              [(n, str(ROOT / p)) for n, p in DEFAULTS]
    for name, prefix in targets:
        report(name, prefix, args.bars, args.seed)
    print("\ncoverage = recall of ALL gold codes (out-of-label-space ones included), as in the paper.")


if __name__ == "__main__":
    main()
