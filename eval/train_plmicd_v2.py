#!/usr/bin/env python3
"""PLM-ICD (LAAT) for the v2 full-record corpus, with long-document CHUNKING.

The v1 train_plmicd.py assumed short (~40-token) diagnosis-stripped notes and used a single 256-token
window. v2 notes are ~2,250 tokens, so we use PLM-ICD's long-doc approach: split the note into S fixed
segments of L tokens, encode each segment independently through the biomedical encoder, concatenate the
token representations into one (S*L) sequence, and apply a single label-wise attention (LAAT) pool over
the whole concatenated sequence. Encoder fine-tuned end-to-end. This is the trained-network supervised
ceiling above the frozen linear probe; same label space (train support >= --min-count), same test
admissions, same exact + category micro-F1 as every other method. Trains on the Mac (MPS).
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class LAATChunked(nn.Module):
    """Segment-encode + concatenate tokens + label-wise attention over the full sequence."""
    def __init__(self, encoder_name, n_labels, proj=512):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder_name)
        h = self.enc.config.hidden_size
        self.first = nn.Linear(h, proj)
        self.second = nn.Linear(proj, n_labels)   # per-label attention scores over tokens
        self.out = nn.Linear(h, n_labels)          # per-label output weight + bias

    def forward(self, input_ids, attention_mask):
        # input_ids / attention_mask: (B, S, L)
        B, S, L = input_ids.shape
        ids = input_ids.reshape(B * S, L)
        am = attention_mask.reshape(B * S, L)
        H = self.enc(input_ids=ids, attention_mask=am).last_hidden_state   # (B*S, L, h)
        H = H.reshape(B, S * L, -1)                                          # (B, S*L, h)
        M = attention_mask.reshape(B, S * L)                                 # (B, S*L)
        Z = torch.tanh(self.first(H))
        scores = self.second(Z).masked_fill(M.unsqueeze(-1) == 0, -1e4)      # (B, S*L, n_labels)
        alpha = torch.softmax(scores, dim=1)                                 # over all tokens
        V = torch.bmm(alpha.transpose(1, 2), H)                              # (B, n_labels, h)
        logits = (V * self.out.weight).sum(-1) + self.out.bias               # (B, n_labels)
        return logits


def micro_f1(preds, golds, key=lambda c: c):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        P = {key(c) for c in p}; G = {key(c) for c in g}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--seg-len", type=int, default=512, help="Tokens per segment.")
    ap.add_argument("--max-segments", type=int, default=4, help="Segments per note (seg-len*this = token cap).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-train", type=int, default=0, help="Cap train size (0=all).")
    ap.add_argument("--pos-weight-cap", type=float, default=8.0,
                    help="Cap on per-label BCE pos_weight. 8 is the project standard: every valid run "
                         "uses it. The old default of 30 over-predicts badly (P~0.05, exact-F1~0.088) "
                         "and has silently invalidated two runs.")
    ap.add_argument("--test-emb-accno", default=str(ROOT / "data/embeddings/test_n5000_seed42.acc_no.npy"))
    ap.add_argument("--tag", default="plmicd_v2_biomedbert")
    ap.add_argument("--seed", type=int, default=0, help="Training seed (torch+numpy) for multi-seed robustness.")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Micro-batches per optimizer step (effective batch = batch-size * this). "
                         "Large encoders (e.g. BioLinkBERT-large) under-converge at effective batch 4.")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="Fraction of total optimizer steps spent in linear LR warmup (then linear decay). "
                         "0 = constant LR (previous behaviour).")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_v2"))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"device={DEV} encoder={args.encoder} seg={args.seg_len}x{args.max_segments} col={args.text_column} seed={args.seed}")

    tr = pd.read_parquet(ROOT / "data/processed/train.parquet")
    mask = (tr["ver"] == "10")
    if "english_predominant" in tr.columns:
        mask &= tr["english_predominant"]
    tr = tr[mask].reset_index(drop=True)
    codes_tr = [[c.upper() for c in cs] for cs in tr["icd_codes"]]
    cnt = Counter(c for cs in codes_tr for c in cs)
    labels = sorted([c for c, v in cnt.items() if v >= args.min_count])
    lab2i = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    notes_tr = tr[args.text_column].astype(str).tolist()
    if args.max_train and args.max_train < len(notes_tr):
        # Random (seeded) subsample for representativeness -- not the first N rows, which the parquet
        # ordering could bias. Labels/vocab are still fit on the FULL train set above.
        sub = np.random.default_rng(42).choice(len(notes_tr), size=args.max_train, replace=False)
        notes_tr = [notes_tr[i] for i in sub]; codes_tr = [codes_tr[i] for i in sub]
    print(f"train {len(notes_tr):,} notes, K={K} labels")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = LAATChunked(args.encoder, K).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Per-label positive weighting to counter extreme multi-label imbalance (~0.16% positive rate):
    # plain BCE collapses to an all-negative solution (loss ~0 but recall ~0). pos_weight[j] =
    # min(neg/pos, cap) upweights rare-label positives; cap prevents singleton labels from exploding.
    sub_cnt = Counter(c for cs in codes_tr for c in cs if c in lab2i)
    pos = np.array([max(sub_cnt.get(c, 1), 1) for c in labels], dtype=np.float64)
    Ntr = len(codes_tr)
    pos_weight = np.minimum((Ntr - pos) / pos, args.pos_weight_cap)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(DEV))
    print(f"pos_weight: median {np.median(pos_weight):.1f}, cap {args.pos_weight_cap}")
    total_len = args.seg_len * args.max_segments

    def encode(batch_notes):
        e = tok(batch_notes, padding="max_length", truncation=True, max_length=total_len, return_tensors="pt")
        B = e["input_ids"].shape[0]
        ids = e["input_ids"].reshape(B, args.max_segments, args.seg_len)
        am = e["attention_mask"].reshape(B, args.max_segments, args.seg_len)
        return ids.to(DEV), am.to(DEV)

    def make_Y(idxs):
        Y = torch.zeros(len(idxs), K)
        for r, s in enumerate(idxs):
            for c in codes_tr[s]:
                if c in lab2i: Y[r, lab2i[c]] = 1.0
        return Y.to(DEV)

    n = len(notes_tr)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cat = lambda c: c.split(".")[0][:3]

    # Load test once; eval + dump AFTER EACH EPOCH so a killed run is never a total loss and we get the
    # per-epoch F1 trajectory (overfitting shows up as a peak-then-decline).
    test = pd.read_parquet(ROOT / "data/processed/test.parquet").set_index("acc_no")
    te_acc = np.load(args.test_emb_accno)
    te_notes = test.loc[te_acc, args.text_column].astype(str).tolist()
    te_gold = [{c.upper() for c in test.loc[a, "icd_codes"]} for a in te_acc]

    def predict_probs(note_list):
        out_ = []
        with torch.no_grad():
            for i in range(0, len(note_list), args.batch_size):
                ids, am = encode(note_list[i:i + args.batch_size])
                out_.append(torch.sigmoid(model(ids, am)).cpu().numpy())
        return np.concatenate(out_)

    def evaluate(suffix):
        model.eval()
        P = predict_probs(te_notes)
        val_idx = np.random.default_rng(0).choice(n, size=min(2000, n), replace=False)
        Pv = predict_probs([notes_tr[s] for s in val_idx]); vg = [set(codes_tr[s]) for s in val_idx]
        best_th, best = 0.3, -1
        for th in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]:
            pr = [{labels[j] for j in np.where(Pv[r] >= th)[0]} for r in range(len(val_idx))]
            f1, _, _ = micro_f1(pr, vg)
            if f1 > best: best, best_th = f1, th
        te_pred = [{labels[j] for j in np.where(P[r] >= best_th)[0]} for r in range(len(te_acc))]
        f1e, pe, re_ = micro_f1(te_pred, te_gold)
        f1c, _, _ = micro_f1(te_pred, te_gold, key=cat)
        summary = {"mode": "plmicd_laat_chunked", "encoder": args.encoder, "text_column": args.text_column,
                   "seg_len": args.seg_len, "max_segments": args.max_segments, "n_labels": K,
                   "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "effective_batch": args.batch_size * args.grad_accum, "warmup_frac": args.warmup_frac,
                   "epochs": args.epochs, "seed": args.seed,
                   "threshold": best_th, "val_f1": best, "n_test": len(te_acc), "exact_micro_f1": f1e,
                   "precision": pe, "recall": re_, "category_micro_f1": f1c}
        print(f"\n--- PLM-ICD v2 eval{suffix} --- {json.dumps(summary)}", flush=True)
        with open(out / f"{args.tag}{suffix}_predictions.jsonl", "w") as f:
            for a, g, p in zip(te_acc, te_gold, te_pred):
                f.write(json.dumps({"acc_no": int(a), "gold": sorted(g), "pred": sorted(p),
                                    "tp": len(p & g), "fp": len(p - g), "fn": len(g - p)}) + "\n")
        # dump probs for the selective curve (per-code confidence signal)
        np.save(out / f"{args.tag}{suffix}_probs.npy", P.astype(np.float32))
        np.save(out / f"{args.tag}{suffix}_labels.npy", np.array(labels))
        with open(out / f"{args.tag}{suffix}_gold.json", "w") as f:
            json.dump([sorted(g) for g in te_gold], f)
        with open(out / f"{args.tag}{suffix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        model.train()

    # LR schedule: linear warmup then linear decay over the whole run. Warmup is what keeps a large
    # encoder from wrecking its pretrained weights in the first few hundred high-LR steps.
    steps_per_epoch = math.ceil(math.ceil(n / args.batch_size) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_frac)
    if args.warmup_frac > 0:
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    else:
        sched = None
    print(f"grad_accum={args.grad_accum} (effective batch {args.batch_size * args.grad_accum}) "
          f"opt_steps={total_steps} warmup={warmup_steps}", flush=True)

    print("training...")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); order = np.random.permutation(n); tot = 0.0; nb = 0
        opt.zero_grad()
        for i in range(0, n, args.batch_size):
            b = order[i:i + args.batch_size]
            ids, am = encode([notes_tr[s] for s in b])
            loss = bce(model(ids, am), make_Y(b))
            tot += loss.item(); nb += 1
            (loss / args.grad_accum).backward()   # scale so the accumulated grad is the mean, not the sum
            if nb % args.grad_accum == 0:
                opt.step()
                if sched is not None: sched.step()
                opt.zero_grad()
            if nb % 200 == 0:
                lr_now = opt.param_groups[0]["lr"]
                print(f"  ep{ep} step {nb} loss {tot/nb:.4f} lr {lr_now:.2e} "
                      f"({time.time()-t0:.0f}s, {i/n*100:.0f}% epoch)", flush=True)
        if nb % args.grad_accum:      # flush a trailing partial accumulation group
            opt.step()
            if sched is not None: sched.step()
            opt.zero_grad()
        print(f"  epoch {ep} done, loss {tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
        evaluate(f"_ep{ep}")   # per-epoch eval + dump (survives a later crash)
    # final canonical outputs = last epoch's
    evaluate("")
    torch.save(model.state_dict(), out / f"{args.tag}_model.pt")  # save weights for cheap re-eval
    print(f"Wrote -> {out}/{args.tag}_* (incl. _model.pt)")


if __name__ == "__main__":
    main()
