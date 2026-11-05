#!/usr/bin/env python3
"""PLM-ICD (head) + k-NN retrieval (tail) ensemble — does combining beat PLM-ICD alone (0.460)?

PLM-ICD owns the frequent head but is 0.000 on the tail (its label space is train-count>=25). A weighted
k-NN vote over the 258k pool reaches tail codes PLM-ICD cannot. We test: (a) union of the two at the
code level; (b) a head/tail ROUTER (PLM-ICD for its label space, k-NN only for codes outside it, i.e.
the true tail). Reports exact + category micro-F1 and tail recall vs PLM-ICD alone. n=5000 test (same
admissions/order as the PLM-ICD eval).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def micro(preds, golds, key=lambda c: c):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        P = {key(c) for c in p}; G = {key(c) for c in g}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    return (2*pr*rc/(pr+rc) if pr+rc else 0), pr, rc


def freq_recall(preds, golds, cnt, lo, hi):
    tp = fn = 0
    for p, g in zip(preds, golds):
        for c in g:
            if lo <= cnt.get(c, 0) <= hi:
                if c in p: tp += 1
                else: fn += 1
    return tp/(tp+fn) if tp+fn else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plmicd", default=str(ROOT/"eval/results_v2/plmicd_cuda_full_nodx_ep1"))
    ap.add_argument("--test-emb", default=str(ROOT/"data/embeddings/test_n5000_seed42"))
    ap.add_argument("--pool-emb", default=str(ROOT/"data/embeddings/train_pool_full"))
    ap.add_argument("--m", type=int, default=100); ap.add_argument("--knn-th", type=float, default=0.55)
    ap.add_argument("--plm-th", type=float, default=0.7)
    args = ap.parse_args()

    P = np.load(f"{args.plmicd}_probs.npy"); labels = list(map(str, np.load(f"{args.plmicd}_labels.npy")))
    gold = [set(map(str, g)) for g in json.load(open(f"{args.plmicd}_gold.json"))]
    plm_labelset = set(labels); n = P.shape[0]
    plm_pred = [{labels[j] for j in np.where(P[a] >= args.plm_th)[0]} for a in range(n)]

    # k-NN weighted vote over the pool (full code space) for the SAME test admissions/order
    te = np.load(f"{args.test_emb}.npy")[:n]; te_acc = np.load(f"{args.test_emb}.acc_no.npy")[:n]
    pool = np.load(f"{args.pool_emb}.npy"); pool_acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(ROOT/"data/processed/train.parquet").set_index("acc_no")
    pool_codes = [set(c.upper() for c in cs) for cs in train.loc[pool_acc, "icd_codes"]]
    cnt = {}
    for cs in train[train["ver"]=="10"]["icd_codes"]:
        for c in cs: cnt[c.upper()] = cnt.get(c.upper(), 0) + 1
    knn_vote = [defaultdict(float) for _ in range(n)]
    for i in range(n):
        sims = pool @ te[i]; nn = np.argpartition(-sims, args.m+1)[:args.m+1]
        for j in nn:
            if pool_acc[j] == te_acc[i]: continue
            w = float(sims[j])
            for c in pool_codes[j]: knn_vote[i][c] += w
        if (i+1) % 1000 == 0: print(f"  knn {i+1}/{n}")
    # normalize per-admission and threshold
    knn_pred = []
    for i in range(n):
        if not knn_vote[i]: knn_pred.append(set()); continue
        mx = max(knn_vote[i].values())
        knn_pred.append({c for c, v in knn_vote[i].items() if v/mx >= args.knn_th})

    # (a) plain union ; (b) head/tail router: PLM-ICD in its label space + k-NN ONLY for tail codes
    union = [plm_pred[a] | knn_pred[a] for a in range(n)]
    router = [plm_pred[a] | {c for c in knn_pred[a] if c not in plm_labelset} for a in range(n)]

    cat = lambda c: c.replace('.', '')[:3]
    print(f"\n{'method':<26}{'exact-F1':>9}{'cat-F1':>8}{'P':>7}{'R':>7}{'rare-recall':>12}")
    for name, pred in [("PLM-ICD alone", plm_pred), ("k-NN alone", knn_pred),
                       ("union (PLM ∪ kNN)", union), ("router (PLM + kNN-tail)", router)]:
        f, p, r = micro(pred, gold); fc, _, _ = micro(pred, gold, key=cat)
        rr = freq_recall(pred, gold, cnt, 2, 24)
        print(f"{name:<26}{f:>9.4f}{fc:>8.4f}{p:>7.3f}{r:>7.3f}{rr:>12.3f}")


if __name__ == "__main__":
    main()
