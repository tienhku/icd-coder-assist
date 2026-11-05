#!/usr/bin/env python3
"""Experiment C: retrieval-classifier shortlist -> LLM audit (the "encoder shortlist + LLM verify"
pipeline the 2026 systematic review calls the best pattern).

Per test admission:
  1. Weighted k-NN over the full train note pool: aggregate the codes of the top-M nearest notes,
     each voting with its cosine similarity; take the top-K codes as a candidate shortlist. This is
     a discriminative aggregator over many neighbors — unlike note-RAG, which shows the LLM only 5
     whole cases. Its oracle recall (top-K codes contain the gold) is much higher: ~0.55 at M=100,
     K=30 vs note-RAG's 0.37.
  2. LLM audits: given the note + the candidate (code: description) shortlist, select the codes that
     apply (reusing the description-RAG assignment prompt).
Same test sample / seed / scoring as the other runners.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import build_desc_rag_prompt, parse_codes  # noqa: E402

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
GEN_URL = "http://localhost:11434/api/generate"


def ollama_gen(model, prompt, url, timeout=180):
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 400}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--gen-url", default=GEN_URL)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_full"))
    ap.add_argument("--test-emb", default=str(ROOT / "data/embeddings/test_n1000_seed42"))
    ap.add_argument("--m-neighbors", type=int, default=100)
    ap.add_argument("--k-codes", type=int, default=30, help="Shortlist size handed to the LLM.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_n1000"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(TEST_PARQUET)
    test = test[test["ver"] == "10"].sample(n=max(args.n, 1000), random_state=args.seed)
    test = test.reset_index(drop=True).iloc[: args.n]
    test_emb = np.load(f"{args.test_emb}.npy")[: args.n]
    test_acc = np.load(f"{args.test_emb}.acc_no.npy")[: args.n]
    assert (test["acc_no"].to_numpy() == test_acc).all()

    pool_emb = np.load(f"{args.pool_emb}.npy")
    pool_acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(TRAIN_PARQUET).set_index("acc_no")
    pool_codes = [set(c.upper() for c in cs) for cs in train.loc[pool_acc, "icd_codes"]]
    desc = pd.read_csv(ROOT / "data/icd_ref/icd10cm_descriptions.tsv", sep="\t").set_index("code")
    descU = {k.upper(): v for k, v in desc["description"].items()}
    print(f"shortlist-audit: {args.n} queries, M={args.m_neighbors}, K={args.k_codes}, model={args.model}")

    tag = args.tag or f"shortlist_{args.model.replace(':', '_')}_{args.text_column}"
    records = []
    total_tp = total_fp = total_fn = n_errors = 0
    t0 = time.time()
    for i in range(args.n):
        sims = pool_emb @ test_emb[i]
        nn = np.argsort(-sims)[: args.m_neighbors]
        vote = defaultdict(float)
        for j in nn:
            if pool_acc[j] == test_acc[i]:
                continue
            for c in pool_codes[j]:
                vote[c] += float(sims[j])
        shortlist = sorted(vote, key=lambda c: -vote[c])[: args.k_codes]
        candidates = [(c, descU.get(c, c)) for c in shortlist]
        try:
            raw = ollama_gen(args.model, build_desc_rag_prompt(test.iloc[i][args.text_column], candidates), args.gen_url)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{args.n}] ERROR: {e}")
            if n_errors >= 10 and not records:
                sys.exit("ABORT: backend unreachable.")
            continue
        pred = parse_codes(raw)
        gold = {c.upper() for c in test.iloc[i]["icd_codes"]}
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        total_tp += tp; total_fp += fp; total_fn += fn
        records.append({"acc_no": int(test.iloc[i]["acc_no"]), "gold": sorted(gold), "pred": sorted(pred),
                        "shortlist": shortlist, "raw_response": raw, "tp": tp, "fp": fp, "fn": fn})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n}] elapsed={time.time()-t0:.0f}s tp={total_tp} fp={total_fp} fn={total_fn}")

    if not records:
        sys.exit("ABORT: no successful predictions.")
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    summary = {"mode": "shortlist_audit", "model": args.model, "m_neighbors": args.m_neighbors,
               "k_codes": args.k_codes, "text_column": args.text_column, "n_samples": len(records),
               "n_errors": n_errors, "micro_precision": p, "micro_recall": r, "micro_f1": f1,
               "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn, "elapsed_sec": time.time() - t0}
    print("\n--- Summary ---"); print(json.dumps(summary, indent=2))
    with open(out_dir / f"{tag}_predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote -> {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
