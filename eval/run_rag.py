#!/usr/bin/env python3
"""RAG eval: retrieve similar train notes as few-shot context, then code via a local model.

For each test admission, embed the note (bge-m3, precomputed), retrieve the top-k most similar
TRAIN admissions by cosine, and prompt medgemma with those (note -> codes) pairs as examples
before the test note. Same test sample / seed / scoring as run_baseline.py, so RAG vs zero-shot
is a like-for-like comparison — only the retrieved context differs.

Leakage guard: the retrieval pool is the TRAIN split, disjoint from the TEST split by construction
(build_corpus.py splits on a hash of acc_no). We additionally drop any pool row whose acc_no equals
a query's, belt-and-suspenders.
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
sys.path.insert(0, str(ROOT / "eval"))
from prompting import build_rag_prompt, parse_codes  # noqa: E402
from run_baseline import make_mlx_responder  # noqa: E402  (reuse the exact MLX generation path)

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
GEN_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embed"


def call_ollama(model, prompt, url, timeout=180):
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 400},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def score(pred, gold):
    return len(pred & gold), len(pred - gold), len(gold - pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5, help="Retrieved examples per query.")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mlx"])
    ap.add_argument("--model", default="medgemma", help="Ollama model name (--backend ollama).")
    ap.add_argument(
        "--mlx-model", default="google/medgemma-4b-it", help="HF repo/path (--backend mlx)."
    )
    ap.add_argument(
        "--adapter-path", default=None, help="LoRA adapter for MLX (RAG + fine-tuning combined)."
    )
    ap.add_argument("--gen-url", default=GEN_URL)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--test-emb", default=str(ROOT / "data/embeddings/test_n1000_seed42"))
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_50k"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results_n1000"))
    ap.add_argument("--max-example-chars", type=int, default=600, help="Truncate long retrieved notes.")
    ap.add_argument(
        "--dedup-exact-note",
        action="store_true",
        help="Drop any retrieved pool row whose note text is byte-identical to the query's. "
        "Removes templated/re-admission exact duplicates for a conservative headline; different "
        "acc_no with merely *similar* text is a legitimate past-case retrieval and is kept.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test queries: must match run_baseline.py's sampling exactly (same n/seed/ver/column).
    test = pd.read_parquet(TEST_PARQUET)
    test = test[test["ver"] == "10"].sample(n=max(args.n, 1000), random_state=args.seed)
    test = test.reset_index(drop=True).iloc[: args.n]
    test_emb = np.load(f"{args.test_emb}.npy")[: args.n]
    test_acc = np.load(f"{args.test_emb}.acc_no.npy")[: args.n]
    assert (test["acc_no"].to_numpy() == test_acc).all(), "test embedding order != eval sampling order"

    # Retrieval pool (train).
    pool_emb = np.load(f"{args.pool_emb}.npy")
    pool_acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(TRAIN_PARQUET).set_index("acc_no")
    pool_notes = train.loc[pool_acc, args.text_column].tolist()
    pool_codes = train.loc[pool_acc, "icd_codes"].tolist()
    # Generation backend: Ollama (5090, remote HTTP) or MLX (local, unquantized, optional adapter).
    if args.backend == "mlx":
        mlx_respond = make_mlx_responder(args.mlx_model, args.adapter_path)
        respond = mlx_respond
        model_label = args.mlx_model + (f"+{Path(args.adapter_path).name}" if args.adapter_path else "")
    else:
        respond = lambda prompt: call_ollama(args.model, prompt, args.gen_url)  # noqa: E731
        model_label = args.model
    print(f"RAG: {args.n} queries, pool {len(pool_acc)}, k={args.k}, backend={args.backend}, model={model_label}")

    tag = args.tag or f"rag_k{args.k}_{args.backend}_{model_label.replace(':', '_').replace('/', '_')}_{args.text_column}"
    records = []
    total_tp = total_fp = total_fn = n_errors = 0
    t0 = time.time()
    for i in range(args.n):
        sims = pool_emb @ test_emb[i]  # cosine (both L2-normalized)
        order = np.argsort(-sims)
        query_note = test.iloc[i][args.text_column]
        examples, seen = [], 0
        for j in order:
            if pool_acc[j] == test_acc[i]:  # same-episode leakage guard
                continue
            if args.dedup_exact_note and pool_notes[j] == query_note:  # exact-duplicate guard
                continue
            note = pool_notes[j][: args.max_example_chars]
            examples.append((note, list(pool_codes[j])))
            seen += 1
            if seen >= args.k:
                break

        prompt = build_rag_prompt(test.iloc[i][args.text_column], examples)
        try:
            raw = respond(prompt)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{args.n}] ERROR: {e}")
            if n_errors >= 10 and not records:
                sys.exit(f"ABORT: {n_errors} failures, 0 successes — backend unreachable.")
            continue
        pred = parse_codes(raw)
        gold = {c.upper() for c in test.iloc[i]["icd_codes"]}
        tp, fp, fn = score(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        records.append(
            {
                "acc_no": int(test.iloc[i]["acc_no"]),
                "gold": sorted(gold),
                "pred": sorted(pred),
                "retrieved_acc_no": [int(pool_acc[j]) for j, _ in zip(order, range(args.k))],
                "raw_response": raw,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n}] elapsed={time.time()-t0:.0f}s tp={total_tp} fp={total_fp} fn={total_fn}")

    if not records:
        sys.exit("ABORT: no successful predictions.")
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    summary = {
        "mode": "rag",
        "k": args.k,
        "backend": args.backend,
        "model": model_label,
        "adapter_path": args.adapter_path,
        "pool_size": len(pool_acc),
        "text_column": args.text_column,
        "n_samples": len(records),
        "n_errors": n_errors,
        "micro_precision": p,
        "micro_recall": r,
        "micro_f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "elapsed_sec": time.time() - t0,
    }
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    with open(out_dir / f"{tag}_predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote -> {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
