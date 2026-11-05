#!/usr/bin/env python3
"""Multi-role agentic ICD coding: Propose -> Physician -> Coder -> Reviewer.

The 5-agent / role-decomposition pattern from the recent agentic-coding preprints, adapted to our
propose->verify frame. The single-agent verify collapses extraction+assignment+audit into one LLM
call; here we separate them:

  1. Propose  (retrieval): weighted k-NN vote over the 100 most similar past admissions -> top-K
     candidate codes with evidence (description + support count). [same as run_agentic.py]
  2. Physician (LLM): distill the note into the admission's active problem list (clinical phrases).
  3. Coder     (LLM): map the problem list onto the candidate codes.
  4. Reviewer  (LLM): audit the coder's codes against the note, drop unsupported.

Same test sample / seed / scoring as every other method for a like-for-like comparison. Three LLM
calls per admission (vs one for single-agent verify) -- the cost of role separation.
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
from prompting import (  # noqa: E402
    build_agent_physician_prompt,
    build_agent_coder_prompt,
    build_agent_reviewer_prompt,
    parse_codes,
    parse_entities,
)

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
GEN_URL = "http://localhost:11434/api/generate"


def ollama_gen(model, prompt, url, timeout=240, num_predict=400):
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": num_predict}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="medgemma:27b")
    ap.add_argument("--gen-url", default=GEN_URL)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_full"))
    ap.add_argument("--test-emb", default=str(ROOT / "data/embeddings/test_n1000_seed42"))
    ap.add_argument("--m-neighbors", type=int, default=100)
    ap.add_argument("--k-candidates", type=int, default=40)
    ap.add_argument("--note-chars", type=int, default=8000)
    ap.add_argument("--no-reviewer", action="store_true", help="Ablate the reviewer stage (physician->coder only).")
    ap.add_argument("--tag", default="v2_agentic_multirole")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_v2"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"multirole: {args.n} queries, propose M={args.m_neighbors} K={args.k_candidates}, "
          f"physician->coder->{'(no reviewer)' if args.no_reviewer else 'reviewer'} by {args.model}")

    records = []
    total_tp = total_fp = total_fn = n_errors = 0
    t0 = time.time()
    for i in range(args.n):
        # Stage 1 -- Propose: weighted k-NN vote over the M nearest past admissions.
        sims = pool_emb @ test_emb[i]
        nn = np.argsort(-sims)[: args.m_neighbors]
        vote = defaultdict(float); support = defaultdict(int)
        for j in nn:
            if pool_acc[j] == test_acc[i]:
                continue
            for c in pool_codes[j]:
                vote[c] += float(sims[j]); support[c] += 1
        ranked = sorted(vote, key=lambda c: -vote[c])[: args.k_candidates]
        candidates = [(c, descU.get(c, c), support[c]) for c in ranked]

        note = str(test.iloc[i][args.text_column])[: args.note_chars]
        try:
            # Stage 2 -- Physician: distill the problem list.
            problems_raw = ollama_gen(args.model, build_agent_physician_prompt(note), args.gen_url)
            problems = "\n".join(parse_entities(problems_raw)[:40]) or problems_raw[:1500]
            # Stage 3 -- Coder: map problem list onto candidates.
            coder_raw = ollama_gen(args.model, build_agent_coder_prompt(problems, candidates), args.gen_url)
            pred = parse_codes(coder_raw)
            # Stage 4 -- Reviewer: audit against the note.
            if not args.no_reviewer:
                rev_raw = ollama_gen(args.model, build_agent_reviewer_prompt(note, candidates, pred), args.gen_url)
                pred = parse_codes(rev_raw)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{args.n}] ERROR: {e}")
            if n_errors >= 10 and not records:
                sys.exit("ABORT: backend unreachable.")
            continue
        # Agent must pick from candidates; intersect defensively.
        cand_set = {c for c, _, _ in candidates}
        pred = pred & cand_set
        gold = {c.upper() for c in test.iloc[i]["icd_codes"]}
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        total_tp += tp; total_fp += fp; total_fn += fn
        records.append({"acc_no": int(test.iloc[i]["acc_no"]), "gold": sorted(gold), "pred": sorted(pred),
                        "n_candidates": len(candidates), "tp": tp, "fp": fp, "fn": fn})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n}] elapsed={time.time()-t0:.0f}s tp={total_tp} fp={total_fp} fn={total_fn}")

    if not records:
        sys.exit("ABORT: no successful predictions.")
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    summary = {"mode": "agentic_multirole", "model": args.model, "m_neighbors": args.m_neighbors,
               "k_candidates": args.k_candidates, "reviewer": not args.no_reviewer,
               "text_column": args.text_column, "n_samples": len(records), "n_errors": n_errors,
               "micro_precision": p, "micro_recall": r, "micro_f1": f1,
               "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn, "elapsed_sec": time.time() - t0}
    print("\n--- Agentic (multi-role physician->coder->reviewer) ---"); print(json.dumps(summary, indent=2))
    with open(out_dir / f"{args.tag}_predictions.jsonl", "w") as f:
        for rec in records: f.write(json.dumps(rec) + "\n")
    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote -> {out_dir}/{args.tag}_*")


if __name__ == "__main__":
    main()
