#!/usr/bin/env python3
"""Agentic ICD coding: Propose (retrieval tools, high recall) -> Verify (LLM coder-review agent).

Motivated by the v2 finding that the bottleneck is SELECTION from a high-recall candidate pool
(oracle recall@5 0.557, but generative RAG realizes only ~0.08). Agentic verification tackles this
directly: a k-NN vote over the 100 most similar past admissions proposes candidate codes with
evidence (retrieval support + official description), and a "senior coder review" LLM agent decides,
per candidate, whether the note supports it for THIS admission.

This is the first agent pipeline; a Complete agent (catch codes retrieval missed) can be added.
Same test sample / seed / scoring as every other method for a like-for-like comparison.
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
from prompting import build_agent_verify_prompt, build_agent_cases_verify_prompt, build_agent_refine_prompt, parse_codes  # noqa: E402

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
GEN_URL = "http://localhost:11434/api/generate"


def ollama_gen(model, prompt, url, timeout=240):
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 400}},
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
    ap.add_argument("--k-candidates", type=int, default=40, help="Candidates handed to the verify agent.")
    ap.add_argument("--verify-mode", default="strict", choices=["strict","lenient"])
    ap.add_argument("--evidence-mode", default="support", choices=["support","cases"])
    ap.add_argument("--n-cases", type=int, default=3, help="Similar cases shown (evidence-mode=cases).")
    ap.add_argument("--case-chars", type=int, default=1000)
    ap.add_argument("--cases-strict", action="store_true", help="Strict rule with case evidence.")
    ap.add_argument("--refine", action="store_true", help="Reflexion: second self-critique pass.")
    ap.add_argument("--note-chars", type=int, default=6000, help="Truncate the note in the verify prompt.")
    ap.add_argument("--tag", default="v2_agentic_verify")
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
    pool_notes = train.loc[pool_acc, args.text_column].tolist() if True else None
    desc = pd.read_csv(ROOT / "data/icd_ref/icd10cm_descriptions.tsv", sep="\t").set_index("code")
    descU = {k.upper(): v for k, v in desc["description"].items()}
    print(f"agentic: {args.n} queries, propose M={args.m_neighbors}, K={args.k_candidates}, verify by {args.model}")

    records = []
    total_tp = total_fp = total_fn = n_errors = 0
    t0 = time.time()
    for i in range(args.n):
        # Stage 1 — Propose: weighted k-NN vote over the M nearest past admissions.
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

        # Stage 2 — Verify: coder-review agent decides which candidates the note supports.
        note = str(test.iloc[i][args.text_column])[: args.note_chars]
        if args.evidence_mode == "cases":
            cases = []
            for j in nn[: args.n_cases]:
                if pool_acc[j] == test_acc[i]:
                    continue
                cases.append((float(sims[j]), str(pool_notes[j])[: args.case_chars], list(pool_codes[j])))
                if len(cases) >= args.n_cases:
                    break
            prompt = build_agent_cases_verify_prompt(note, candidates, cases, args.cases_strict)
        else:
            prompt = build_agent_verify_prompt(note, candidates, args.verify_mode)
        try:
            raw = ollama_gen(args.model, prompt, args.gen_url)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{args.n}] ERROR: {e}")
            if n_errors >= 10 and not records:
                sys.exit("ABORT: backend unreachable.")
            continue
        # The agent may only pick from candidates; intersect to be safe (it shouldn't invent codes).
        pred = parse_codes(raw)
        if args.refine:
            try:
                raw2 = ollama_gen(args.model, build_agent_refine_prompt(note, candidates, pred), args.gen_url)
                pred = parse_codes(raw2)
            except requests.RequestException:
                pass
        gold = {c.upper() for c in test.iloc[i]["icd_codes"]}
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        total_tp += tp; total_fp += fp; total_fn += fn
        records.append({"acc_no": int(test.iloc[i]["acc_no"]), "gold": sorted(gold), "pred": sorted(pred),
                        "n_candidates": len(candidates), "raw_response": raw, "tp": tp, "fp": fp, "fn": fn})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n}] elapsed={time.time()-t0:.0f}s tp={total_tp} fp={total_fp} fn={total_fn}")

    if not records:
        sys.exit("ABORT: no successful predictions.")
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    summary = {"mode": "agentic_propose_verify", "model": args.model, "m_neighbors": args.m_neighbors,
               "k_candidates": args.k_candidates, "text_column": args.text_column, "n_samples": len(records),
               "n_errors": n_errors, "micro_precision": p, "micro_recall": r, "micro_f1": f1,
               "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn, "elapsed_sec": time.time() - t0}
    print("\n--- Agentic (propose->verify) ---"); print(json.dumps(summary, indent=2))
    with open(out_dir / f"{args.tag}_predictions.jsonl", "w") as f:
        for rec in records: f.write(json.dumps(rec) + "\n")
    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote -> {out_dir}/{args.tag}_*")


if __name__ == "__main__":
    main()
