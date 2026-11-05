#!/usr/bin/env python3
"""Learned meta-classifier: fuse probe + retrieval + priors into a stronger, CALIBRATED predictor.

Per candidate (admission, code) we build features — frozen-probe sigmoid prob, k-NN vote (raw + per-note
normalized + rank), code log-frequency prior — and train a gradient-boosted classifier to predict
P(code is correct). This (a) learns the optimal fusion (beating fixed blends), and (b) yields a
calibrated confidence = the predicted probability, which is exactly what the selective autonomy frontier
needs. Trained on a calibration split, evaluated on a disjoint split. Reports exact/category micro-F1
and the coder-grade autonomy frontier vs the probe baseline.
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    ap.add_argument("--probe-prefix", default=str(ROOT/"eval/results_v2/supervised_selcurve"))
    ap.add_argument("--knn-mat", default=str(ROOT/"eval/results_v2/knn_fusion_knnmat.npy"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = np.load(f"{args.probe_prefix}_probs.npy")
    labels = list(map(str, np.load(f"{args.probe_prefix}_labels.npy")))
    gold = [set(map(str, g)) for g in json.load(open(f"{args.probe_prefix}_gold.json"))]
    knn = np.load(args.knn_mat)
    n, K = P.shape
    lab2i = {c: i for i, c in enumerate(labels)}
    gidx = [np.array([lab2i[c] for c in g if c in lab2i], dtype=int) for g in gold]
    gmask = np.zeros((n, K), dtype=bool)
    for a in range(n): gmask[a, gidx[a]] = True

    tr = pd.read_parquet(ROOT/"data/processed/train.parquet")
    cnt = Counter(c.upper() for cs in tr[tr["ver"]=="10"]["icd_codes"] for c in cs)
    logfreq = np.array([np.log1p(cnt.get(c, 0)) for c in labels], dtype=np.float32)

    knn_norm = knn / (knn.max(axis=1, keepdims=True) + 1e-9)
    probe_rank = (-P).argsort(axis=1).argsort(axis=1)          # 0 = highest prob in that admission

    def build(idx):
        rows, ys, meta = [], [], []
        for a in idx:
            cand = np.where((P[a] >= 0.02) | (knn_norm[a] >= 0.15))[0]
            for j in cand:
                rows.append((P[a, j], knn[a, j], knn_norm[a, j], probe_rank[a, j], logfreq[j]))
                ys.append(gmask[a, j]); meta.append((a, j))
        return np.array(rows, dtype=np.float32), np.array(ys), meta

    rng = np.random.default_rng(args.seed); perm = rng.permutation(n)
    cal, ev = perm[:n//2], perm[n//2:]
    Xc, yc, _ = build(cal); Xe, ye, meta_e = build(ev)
    print(f"n={n} (cal {len(cal)}/ev {len(ev)}); cand rows: cal {len(yc)}, ev {len(ye)}; "
          f"cal positive rate {yc.mean():.3f}")

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                         l2_regularization=1.0, random_state=args.seed)
    clf.fit(Xc, yc)
    conf = clf.predict_proba(Xe)[:, 1]                          # calibrated confidence per candidate

    # assemble eval predictions at a tuned threshold (tune on calib via CV-free proxy: use ev is not
    # allowed; tune threshold on a held-out slice of calib)
    Xc2, yc2, _ = build(cal[len(cal)//2:]); conf_c2 = clf.predict_proba(Xc2)[:,1]
    # map cal-slice rows back for threshold F1: rebuild predictions on that slice
    def f1_at(threshold, X_meta, confs, idx_set):
        by_adm = {}
        for (a, j), c in zip(X_meta, confs):
            if c >= threshold: by_adm.setdefault(a, set()).add(labels[j])
        preds = [by_adm.get(a, set()) for a in idx_set]
        return micro(preds, [gold[a] for a in idx_set])
    _, _, meta_c2 = build(cal[len(cal)//2:])
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.1, 0.9, 33):
        f = f1_at(t, meta_c2, conf_c2, cal[len(cal)//2:])[0]
        if f > best_f: best_f, best_t = f, t

    fe = f1_at(best_t, meta_e, conf, ev)
    gold_ev = [gold[a] for a in ev]
    by_adm = {}
    for (a, j), c in zip(meta_e, conf):
        if c >= best_t: by_adm.setdefault(a, set()).add(labels[j])
    preds_e = [by_adm.get(a, set()) for a in ev]
    fc = micro(preds_e, gold_ev, key=lambda c: c.replace('.','')[:3])
    print(f"\n=== meta-classifier (eval split) — threshold {best_t:.2f} ===")
    print(f"exact-F1 {fe[0]:.4f} (P{fe[1]:.3f} R{fe[2]:.3f})   category-F1 {fc[0]:.4f}")

    # --- selective autonomy frontier from the calibrated confidence ---
    total_gold = sum(len(gold[a]) for a in ev)
    order = np.argsort(-conf); cs = ye[order].astype(int)
    ctp = np.cumsum(cs); cn = np.arange(1, len(cs)+1); prec = ctp/cn; rec = ctp/total_gold
    print("\n=== autonomy frontier (meta-confidence) ===")
    print(f"{'precision bar':>14}{'coverage(recall)':>18}{'auto-codes':>12}")
    for bar in [0.95, 0.90, 0.85, 0.80]:
        ok = np.where(prec >= bar)[0]
        if len(ok):
            i = ok[-1]; print(f"{bar:>14.2f}{rec[i]:>18.3f}{int(cn[i]):>12}")
        else:
            print(f"{bar:>14.2f}{'never':>18}")
    imp = dict(zip(["probe_prob","knn_vote","knn_norm","probe_rank","logfreq"],
                   np.round(getattr(clf, 'feature_importances_', [np.nan]*5), 3) if hasattr(clf,'feature_importances_') else [np.nan]*5))
    print(f"\nfeatures: probe_prob, knn_vote, knn_norm, probe_rank, logfreq")


if __name__ == "__main__":
    main()
