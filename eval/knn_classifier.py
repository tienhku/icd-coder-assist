#!/usr/bin/env python3
"""Retrieval-only discriminative classifier + probe⊕retrieval fusion, as a stronger base predictor.

The agentic 'propose' stage (weighted k-NN vote over the M most similar past admissions) was only ever
used to feed an LLM verifier; it was never evaluated as a stand-alone classifier. Its oracle recall@40
is 0.667 — the retrieval signal is strong. Here we (1) score k-NN vote as a classifier, (2) fuse it with
the frozen-probe sigmoid probability, and (3) report exact + category micro-F1 and the selective
autonomy frontier for the best signal. Thresholds are tuned on a calibration split, evaluated on a
disjoint split (no leakage).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def micro(pred_sets, gold_sets, key=lambda c: c):
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        P = {key(c) for c in p}; G = {key(c) for c in g}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    return (2*pr*rc/(pr+rc) if pr+rc else 0), pr, rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-emb", default=str(ROOT/"data/embeddings/test_n5000_seed42"))
    ap.add_argument("--pool-emb", default=str(ROOT/"data/embeddings/train_pool_full"))
    ap.add_argument("--probe-prefix", default=str(ROOT/"eval/results_v2/supervised_selcurve"))
    ap.add_argument("--m-neighbors", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default=str(ROOT/"eval/results_v2/knn_fusion"))
    args = ap.parse_args()

    te = np.load(f"{args.test_emb}.npy"); te_acc = np.load(f"{args.test_emb}.acc_no.npy")
    pool = np.load(f"{args.pool_emb}.npy"); pool_acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(ROOT/"data/processed/train.parquet").set_index("acc_no")
    pool_codes = [set(c.upper() for c in cs) for cs in train.loc[pool_acc,"icd_codes"]]
    probe_acc = np.load(f"{args.probe_prefix}_test_acc.npy")
    assert (te_acc == probe_acc).all(), "probe/test admissions misaligned"
    P = np.load(f"{args.probe_prefix}_probs.npy")
    labels = list(np.load(f"{args.probe_prefix}_labels.npy"))
    lab2i = {str(c): i for i, c in enumerate(labels)}
    gold = [set(map(str, g)) for g in json.load(open(f"{args.probe_prefix}_gold.json"))]
    n = len(te_acc); K = len(labels)
    print(f"n_test={n}, probe labels K={K}, pool={len(pool_acc)}, M={args.m_neighbors}")

    # k-NN weighted vote: full-code (for standalone) + aligned to probe label space (for fusion).
    knn_full = [defaultdict(float) for _ in range(n)]           # code -> vote (all codes)
    knn_mat = np.zeros((n, K), dtype=np.float32)               # aligned to probe labels
    for i in range(n):
        sims = pool @ te[i]
        nn = np.argpartition(-sims, args.m_neighbors+1)[:args.m_neighbors+1]
        nn = nn[np.argsort(-sims[nn])]
        seen = 0
        for j in nn:
            if pool_acc[j] == te_acc[i]:
                continue
            w = float(sims[j])
            for c in pool_codes[j]:
                knn_full[i][c] += w
                k = lab2i.get(c)
                if k is not None: knn_mat[i, k] += w
            seen += 1
            if seen >= args.m_neighbors: break
        if (i+1) % 1000 == 0: print(f"  knn {i+1}/{n}")

    rng = np.random.default_rng(args.seed); perm = rng.permutation(n)
    cal, ev = perm[:n//2], perm[n//2:]
    gold_ev = [gold[a] for a in ev]

    def best_threshold(score_rows, cand_lists, idx, grid):
        best_t, best_f = grid[0], -1
        for t in grid:
            preds = [{cand_lists[a][j] for j in np.where(score_rows[a] >= t)[0]} for a in idx]
            f = micro(preds, [gold[a] for a in idx])[0]
            if f > best_f: best_f, best_t = f, t
        return best_t

    # --- standalone k-NN vote (full code space) ---
    all_codes = [sorted(knn_full[i]) for i in range(n)]
    knn_scores = [np.array([knn_full[i][c] for c in all_codes[i]]) for i in range(n)]
    grid = np.quantile(np.concatenate([knn_scores[a] for a in cal]), np.linspace(0.5, 0.99, 25))
    t_knn = best_threshold(knn_scores, all_codes, cal, grid)
    knn_pred = [{all_codes[a][j] for j in np.where(knn_scores[a] >= t_knn)[0]} for a in ev]
    f_knn = micro(knn_pred, gold_ev); fc_knn = micro(knn_pred, gold_ev, key=lambda c: c.replace('.','')[:3])

    # --- probe standalone (probe label space) ---
    cand = [labels]*n
    grid_p = np.linspace(0.1, 0.6, 26)
    t_p = best_threshold([P[a] for a in range(n)], cand, cal, grid_p)
    probe_pred = [{labels[j] for j in np.where(P[a] >= t_p)[0]} for a in ev]
    f_probe = micro(probe_pred, gold_ev); fc_probe = micro(probe_pred, gold_ev, key=lambda c: c.replace('.','')[:3])

    # --- fusion: normalized probe prob + normalized knn vote (probe label space) ---
    knn_norm = knn_mat / (knn_mat.max(axis=1, keepdims=True) + 1e-9)
    for alpha in [0.3, 0.5, 0.7]:
        fused = alpha*P + (1-alpha)*knn_norm
        t_f = best_threshold([fused[a] for a in range(n)], cand, cal, np.linspace(0.1,0.7,31))
        fpred = [{labels[j] for j in np.where(fused[a] >= t_f)[0]} for a in ev]
        ff = micro(fpred, gold_ev); ffc = micro(fpred, gold_ev, key=lambda c: c.replace('.','')[:3])
        print(f"  fusion α={alpha}: exact-F1 {ff[0]:.4f} (P{ff[1]:.3f} R{ff[2]:.3f})  cat-F1 {ffc[0]:.4f}")

    print(f"\n=== base predictors (eval split, n={len(ev)}) ===")
    print(f"probe:      exact-F1 {f_probe[0]:.4f} (P{f_probe[1]:.3f} R{f_probe[2]:.3f})  cat-F1 {fc_probe[0]:.4f}")
    print(f"k-NN vote:  exact-F1 {f_knn[0]:.4f} (P{f_knn[1]:.3f} R{f_knn[2]:.3f})  cat-F1 {fc_knn[0]:.4f}")

    # save fused scores (best alpha=0.5) for the selective curve
    np.save(f"{args.out_prefix}_knnmat.npy", knn_mat)
    print(f"Wrote knn matrix -> {args.out_prefix}_knnmat.npy")


if __name__ == "__main__":
    main()
