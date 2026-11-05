#!/usr/bin/env python3
"""Measure the RAG copy rate for a predictions file.

Copy rate = fraction of a model's predicted codes that were AVAILABLE TO COPY from the k retrieved
few-shot examples (i.e. present in at least one retrieved neighbour's gold codes). This operationalises
the copy-suppression mechanism: a base model leans on the retrieved examples (high copy rate); a
fine-tuned model leans on its parametric memory instead (lower copy rate), which on this task hurts.

Recomputes the retrieved neighbours exactly as eval/run_rag.py does (same embeddings, same k, same
same-episode + exact-duplicate guards) so the copyable set matches what the model actually saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="RAG predictions jsonl (has acc_no, pred).")
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_full"))
    ap.add_argument("--test-emb", default=str(ROOT / "data/embeddings/test_n1000_seed42"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--dedup-exact-note", action="store_true", default=True)
    args = ap.parse_args()

    preds = {}
    for line in open(args.pred):
        r = json.loads(line)
        preds[int(r["acc_no"])] = {c.upper() for c in r["pred"]}

    test_acc = np.load(f"{args.test_emb}.acc_no.npy")
    test_emb = np.load(f"{args.test_emb}.npy")
    pool_emb = np.load(f"{args.pool_emb}.npy")
    pool_acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(TRAIN_PARQUET).set_index("acc_no")
    pool_codes = [set(c.upper() for c in cs) for cs in train.loc[pool_acc, "icd_codes"]]
    pool_notes = train.loc[pool_acc, args.text_column].tolist()
    test = pd.read_parquet(TEST_PARQUET).set_index("acc_no")

    acc2row = {int(a): i for i, a in enumerate(test_acc)}
    total_pred = total_copied = 0
    per_adm_rates = []
    n_used = 0
    for acc, pred in preds.items():
        if acc not in acc2row or not pred:
            continue
        i = acc2row[acc]
        query_note = test.loc[acc, args.text_column]
        sims = pool_emb @ test_emb[i]
        order = np.argsort(-sims)
        copyable = set()
        seen = 0
        for j in order:
            if pool_acc[j] == acc:
                continue
            if args.dedup_exact_note and pool_notes[j] == query_note:
                continue
            copyable |= pool_codes[j]
            seen += 1
            if seen >= args.k:
                break
        copied = len(pred & copyable)
        total_pred += len(pred)
        total_copied += copied
        per_adm_rates.append(copied / len(pred))
        n_used += 1

    micro = total_copied / total_pred if total_pred else 0.0
    macro = float(np.mean(per_adm_rates)) if per_adm_rates else 0.0
    print(json.dumps({
        "pred_file": args.pred, "k": args.k, "n_admissions": n_used,
        "copy_rate_micro": round(micro, 4), "copy_rate_macro": round(macro, 4),
        "total_predicted_codes": total_pred, "total_copyable": total_copied,
    }, indent=2))


if __name__ == "__main__":
    main()
