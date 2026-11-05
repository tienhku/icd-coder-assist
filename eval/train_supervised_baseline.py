#!/usr/bin/env python3
"""Supervised baseline: a multi-label linear classifier on frozen bge-m3 note embeddings.

Motivation (docs/taiwan_related_work.md): every Taiwan ICD-10 study is *supervised* and scores
0.70-0.77 (diagnosis-bearing, per-hospital labels). Reviewers will ask why our RAG's ~0.20 is so
much lower. This shows the *supervised ceiling on OUR data and setting* — a linear head trained on
the labels over frozen multilingual embeddings, same test set as RAG — so RAG can be framed as the
no-training alternative, and the head-vs-tail split (supervised strong on frequent codes, retrieval
needed for the long tail) is made explicit.

Frozen-encoder linear probe: no fine-tuning of bge-m3, just a 1024->K linear layer (BCE) over the
K codes with support >= --min-count. The long tail (codes below threshold) is unreachable by
construction — that is the point.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent.parent


def micro_f1(preds, golds, key=lambda c: c):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        P = {key(c) for c in p}
        G = {key(c) for c in g}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_full"))
    ap.add_argument("--test-emb", default=str(ROOT / "data/embeddings/test_n1000_seed42"))
    ap.add_argument("--min-count", type=int, default=25, help="Min train support for a code to be a label.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.03, help="Train slice held out for threshold tuning.")
    ap.add_argument("--tag", default="supervised_linear_bgem3")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_n1000"))
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    Xtr = np.load(f"{args.pool_emb}.npy")
    acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(ROOT / "data/processed/train.parquet").set_index("acc_no")
    codes_tr = [[c.upper() for c in cs] for cs in train.loc[acc, "icd_codes"]]

    cnt = Counter(c for cs in codes_tr for c in cs)
    labels = sorted([c for c, v in cnt.items() if v >= args.min_count])
    lab2i = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    cover = sum(v for c, v in cnt.items() if v >= args.min_count) / sum(cnt.values())
    print(f"Train {len(Xtr):,} notes, K={K} labels (support>={args.min_count}, covers {cover*100:.1f}% of code occurrences)")

    # per-sample label index lists (build multi-hot per batch to save memory)
    Y_idx = [np.array([lab2i[c] for c in cs if c in lab2i], dtype=np.int32) for cs in codes_tr]

    n = len(Xtr)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_ix, tr_ix = perm[:n_val], perm[n_val:]

    model = nn.Linear(Xtr.shape[1], K)
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, X, Y):
        return mx.mean(nn.losses.binary_cross_entropy(model(X), Y, with_logits=True))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def make_Y(idx_batch):
        Y = np.zeros((len(idx_batch), K), dtype=np.float32)
        for r, s in enumerate(idx_batch):
            if len(Y_idx[s]):
                Y[r, Y_idx[s]] = 1.0
        return Y

    print("Training linear head...")
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(tr_ix)
        tot = 0.0; nb = 0
        for i in range(0, len(tr_ix), args.batch_size):
            b = tr_ix[i:i + args.batch_size]
            X = mx.array(Xtr[b]); Y = mx.array(make_Y(b))
            loss, grads = loss_and_grad(model, X, Y)
            opt.update(model, grads); mx.eval(model.parameters(), opt.state)
            tot += float(loss); nb += 1
        if ep % 4 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:2d} loss {tot/nb:.4f}  ({time.time()-t0:.0f}s)")

    # threshold tuning on the held-out train slice
    def predict_probs(X):
        out = []
        for i in range(0, len(X), 4096):
            out.append(np.array(mx.sigmoid(model(mx.array(X[i:i+4096])))))
        return np.concatenate(out)

    Pval = predict_probs(Xtr[val_ix])
    val_gold = [set(codes_tr[s]) for s in val_ix]
    best_th, best_f1 = 0.5, -1
    for th in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        preds = [{labels[j] for j in np.where(Pval[r] >= th)[0]} for r in range(len(val_ix))]
        f1, _, _ = micro_f1(preds, val_gold)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"Best threshold {best_th} (val micro-F1 {best_f1:.4f})")

    # evaluate on the standard test embedding
    Xte = np.load(f"{args.test_emb}.npy")
    te_acc = np.load(f"{args.test_emb}.acc_no.npy")
    test = pd.read_parquet(ROOT / "data/processed/test.parquet").set_index("acc_no")
    te_gold = [ {c.upper() for c in test.loc[a, "icd_codes"]} for a in te_acc ]
    Pte = predict_probs(Xte)
    te_pred = [{labels[j] for j in np.where(Pte[r] >= best_th)[0]} for r in range(len(te_acc))]

    cat = lambda c: c.split(".")[0][:3]
    f1e, pe, re_ = micro_f1(te_pred, te_gold)
    f1c, _, _ = micro_f1(te_pred, te_gold, key=cat)
    # tail coverage: fraction of gold codes that are even in the label space
    reachable = sum(1 for g in te_gold for c in g if c in lab2i)
    total_gold = sum(len(g) for g in te_gold)
    summary = {
        # Derive the encoder label from the embeddings actually loaded -- hardcoding "bge-m3" mislabelled
        # every ablation run's summary json.
        "mode": "supervised_linear", "encoder": f"{Path(args.pool_emb).name} (frozen)", "n_labels": K,
        "min_count": args.min_count, "label_coverage_of_train": cover, "threshold": best_th,
        "n_test": len(te_acc), "exact_micro_f1": f1e, "precision": pe, "recall": re_,
        "category_micro_f1": f1c,
        "gold_in_label_space": reachable / total_gold,
    }
    print("\n--- Supervised baseline ---")
    print(json.dumps(summary, indent=2))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    recs = [{"acc_no": int(a), "gold": sorted(g), "pred": sorted(p),
             "tp": len(p & g), "fp": len(p - g), "fn": len(g - p)}
            for a, g, p in zip(te_acc, te_gold, te_pred)]
    with open(out_dir / f"{args.tag}_note_text_no_diagnosis_predictions.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    with open(out_dir / f"{args.tag}_note_text_no_diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Dump the full test probability matrix + labels + gold for the selective-coding curve (the
    # per-code sigmoid probability is the continuous confidence signal).
    np.save(out_dir / f"{args.tag}_probs.npy", Pte.astype(np.float32))
    np.save(out_dir / f"{args.tag}_labels.npy", np.array(labels))
    np.save(out_dir / f"{args.tag}_test_acc.npy", te_acc)
    with open(out_dir / f"{args.tag}_gold.json", "w") as f:
        json.dump([sorted(g) for g in te_gold], f)
    print(f"Wrote -> {out_dir}/{args.tag}_* (incl. probs/labels/gold for selective curve)")


if __name__ == "__main__":
    main()
