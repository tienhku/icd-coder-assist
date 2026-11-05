#!/usr/bin/env python3
"""Experiment A: code-description-retrieval RAG (GAVS-style), for the note-vs-description head-to-head.

Pipeline per test admission:
  1. LLM extracts diagnostic entities from the note (short phrases).
  2. Each entity is embedded (bge-m3) and matched against the ICD-10-CM code-description vector DB;
     top-per-entity candidate codes are pooled (deduped, capped).
  3. LLM assigns the final codes from the candidate (code: description) list + the note.
Same test sample / seed / scoring as run_rag.py and run_baseline.py.

Direct note->description retrieval was near-useless (oracle recall 0.037 @top-50 vs note-retrieval's
0.37 @k5) because a whole note doesn't embed near short code descriptions — hence the entity step.
Unlike note-retrieval, this can reach long-tail codes with no similar note, since every ICD-10-CM
code is in the pool.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import (  # noqa: E402
    ENTITY_EXTRACTION_PROMPT,
    build_desc_rag_prompt,
    parse_codes,
    parse_entities,
)

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
GEN_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embed"


def ollama_gen(model, prompt, url, num_predict, timeout=180):
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0, "num_predict": num_predict}}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def embed(texts, model, url, timeout=60):
    r = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    return np.asarray(r.json()["embeddings"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--gen-url", default=GEN_URL)
    ap.add_argument("--embed-url", default=EMBED_URL)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--desc-emb", default=str(ROOT / "data/embeddings/icd10cm_desc"))
    ap.add_argument("--per-entity", type=int, default=5, help="Candidate codes retrieved per entity.")
    ap.add_argument("--max-candidates", type=int, default=40, help="Cap on total candidate codes.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_n1000"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(TEST_PARQUET)
    test = test[test["ver"] == "10"].sample(n=max(args.n, 1000), random_state=args.seed)
    test = test.reset_index(drop=True).iloc[: args.n]

    desc_emb = np.load(f"{args.desc_emb}.npy")
    desc_codes = np.load(f"{args.desc_emb}.codes.npy")
    desc_df = pd.read_csv(ROOT / "data/icd_ref/icd10cm_descriptions.tsv", sep="\t").set_index("code")
    print(f"desc-RAG: {args.n} queries, desc pool {len(desc_codes)}, model={args.model}")

    tag = args.tag or f"ragdesc_{args.model.replace(':', '_')}_{args.text_column}"
    records = []
    total_tp = total_fp = total_fn = n_errors = 0
    t0 = time.time()
    for i in range(args.n):
        note = test.iloc[i][args.text_column]
        try:
            ent_raw = ollama_gen(args.model, ENTITY_EXTRACTION_PROMPT.format(note_text=note), args.gen_url, 200)
            entities = parse_entities(ent_raw)[:12] or [note[:200]]
            ent_vecs = embed(entities, args.embed_model, args.embed_url)
            ent_vecs /= np.linalg.norm(ent_vecs, axis=1, keepdims=True) + 1e-8
            # top per-entity candidate codes, pooled by best similarity
            best = {}
            for v in ent_vecs:
                sims = desc_emb @ v
                for j in np.argsort(-sims)[: args.per_entity]:
                    c = str(desc_codes[j])
                    best[c] = max(best.get(c, -1), float(sims[j]))
            cand = sorted(best, key=lambda c: -best[c])[: args.max_candidates]
            candidates = [(c, str(desc_df.loc[c, "description"])) for c in cand if c in desc_df.index]
            raw = ollama_gen(args.model, build_desc_rag_prompt(note, candidates), args.gen_url, 400)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{args.n}] ERROR: {e}")
            if n_errors >= 10 and not records:
                sys.exit(f"ABORT: {n_errors} failures, 0 successes.")
            continue
        pred = parse_codes(raw)
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
    summary = {"mode": "rag_desc", "model": args.model, "text_column": args.text_column,
               "n_samples": len(records), "n_errors": n_errors, "micro_precision": p,
               "micro_recall": r, "micro_f1": f1, "total_tp": total_tp, "total_fp": total_fp,
               "total_fn": total_fn, "elapsed_sec": time.time() - t0}
    print("\n--- Summary ---"); print(json.dumps(summary, indent=2))
    with open(out_dir / f"{tag}_predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote -> {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
