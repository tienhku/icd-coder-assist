#!/usr/bin/env python3
"""What kind of wrong are the auto-accepted errors? Wrong diagnosis, or the coder's own diagnosis at
the wrong digit?

The selective frontier reports exact micro-precision on the auto-accepted slice, which counts
`J96.00` against a coder's `J96.90` as a false positive AND a false negative -- two errors for one
code at the wrong fourth character. That is the right thing for a reproducible metric and the wrong
thing for deciding whether a slice is safe to auto-accept: a specificity disagreement sends a coder to
one field of one code, a wrong category sends them to re-read the chart.

So split the false positives in the auto-accepted slice by whether they share a 3-character category
with a code the coder actually assigned to that admission. The two numbers together bracket the error
rate a coder would experience: exact precision is the pessimistic end, "not contradicting the coder"
(treating a same-category digit miss as a near-hit rather than an error) the optimistic one.

This generalizes what the review workbench showed on one case (`tools/workbench`, 2026-10-23).
Bootstrap is over admissions, as everywhere else in this program.

    python eval/error_anatomy.py
    python eval/error_anatomy.py --prefix eval/results_v2/plmicd_cuda_full_nodx_ep1 --bars 0.90
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=str(ROOT / "eval/results_v2/supervised_selcurve_qwen3"))
    ap.add_argument("--bars", type=float, nargs="+", default=[0.95, 0.90, 0.85, 0.80])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = np.load(f"{args.prefix}_probs.npy")
    labels = np.load(f"{args.prefix}_labels.npy", allow_pickle=True)
    gold = json.load(open(f"{args.prefix}_gold.json"))
    lab2i = {c: i for i, c in enumerate(labels)}
    gm = np.zeros(P.shape, dtype=bool)
    for a, g in enumerate(gold):
        for c in g:
            j = lab2i.get(c)
            if j is not None:
                gm[a, j] = True
    total_gold = sum(len(g) for g in gold)
    gold_cats = [{c[:3] for c in g} for g in gold]
    rng = np.random.default_rng(args.seed)

    print(f"n_adm={P.shape[0]} n_labels={P.shape[1]} total_gold={total_gold}")
    for bar in args.bars:
        # Threshold the way the paper does it: deepest point on the whole test set still >= bar.
        p = P.reshape(-1)
        o = np.argsort(-p)
        prec = np.cumsum(gm.reshape(-1)[o]) / np.arange(1, p.size + 1)
        ok = np.where(prec >= bar)[0]
        if len(ok) == 0:
            print(f"\n=== bar {bar:.2f}: never reached ===")
            continue
        thr = float(p[o][ok[-1]])
        acc = P >= thr

        # Per-admission tallies, so the bootstrap can resample admissions.
        tp = np.zeros(P.shape[0], dtype=np.int32)
        near = np.zeros(P.shape[0], dtype=np.int32)   # FP sharing a category with a gold code
        far = np.zeros(P.shape[0], dtype=np.int32)    # FP in a category the coder never used
        for a in range(P.shape[0]):
            g = set(gold[a])
            for j in np.where(acc[a])[0]:
                c = labels[j]
                if c in g:
                    tp[a] += 1
                elif c[:3] in gold_cats[a]:
                    near[a] += 1
                else:
                    far[a] += 1
        n = int(tp.sum() + near.sum() + far.sum())
        fp = int(near.sum() + far.sum())

        def ratio(idx, num, den):
            nu = num[idx].sum(); de = den[idx].sum()
            return nu / de if de else float("nan")

        all_i = np.arange(P.shape[0])
        strict = ratio(all_i, tp, tp + near + far)
        lenient = ratio(all_i, tp + near, tp + near + far)
        boot = np.array([ratio(rng.integers(0, P.shape[0], P.shape[0]), tp + near, tp + near + far)
                         for _ in range(args.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        print(f"\n=== bar {bar:.2f}  threshold {thr:.3f}  auto-accepted {n}  "
              f"coverage {tp.sum() / total_gold:.4f} ===")
        print(f"  true positives                     {int(tp.sum()):6d}")
        print(f"  FP, same 3-char category as gold   {int(near.sum()):6d}  "
              f"{near.sum() / fp:5.1%} of errors")
        print(f"  FP, category the coder never used  {int(far.sum()):6d}  "
              f"{far.sum() / fp:5.1%} of errors")
        print(f"  exact precision                    {strict:.4f}")
        print(f"  not-contradicting-the-coder prec.  {lenient:.4f} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
