#!/usr/bin/env python3
"""Build RAG-format LoRA training data: teach the model to USE retrieved context.

Naively bolting RAG onto a zero-shot-trained adapter hurts (docs/rag_results.md): the adapter's
"guess frequent codes" prior fights RAG's copy-from-context mechanism. The fix is to fine-tune on
the same prompt shape the model sees at RAG eval time — retrieved (note -> codes) examples, then the
target note -> its codes — so it learns to select from and adjust the retrieved codes rather than
override them.

Each training row: retrieve the target admission's top-k neighbors from the embedded pool, build the
exact RAG prompt (prompting.build_rag_prompt), and use the target's own gold codes as the assistant
turn. Chat format, matching prepare_finetune_data.py / the mlx_lm.lora ChatDataset.

Leakage guard mirroring eval: exclude the target's own acc_no from its retrieval (at eval time the
query note is never in the pool, so training must not get a verbatim self-match it won't see later).
Near-duplicates (same patient's other admissions, different acc_no) are KEPT — they legitimately
occur at eval time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import build_rag_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-emb", default=str(ROOT / "data/embeddings/train_pool_50k"))
    ap.add_argument("--column", default="note_text_no_diagnosis")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-valid", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-example-chars", type=int, default=600)
    ap.add_argument("--out-dir", default=str(ROOT / "data/finetune/rag_v1"))
    args = ap.parse_args()

    emb = np.load(f"{args.pool_emb}.npy")
    acc = np.load(f"{args.pool_emb}.acc_no.npy")
    train = pd.read_parquet(ROOT / "data/processed/train.parquet").set_index("acc_no")
    notes = train.loc[acc, args.column].tolist()
    codes = train.loc[acc, "icd_codes"].tolist()

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(acc), size=args.n_train + args.n_valid, replace=False)

    rows = []
    for idx in pick:
        sims = emb @ emb[idx]
        order = np.argsort(-sims)
        examples, seen = [], 0
        for j in order:
            if acc[j] == acc[idx]:  # exclude self (verbatim answer leak)
                continue
            examples.append((notes[j][: args.max_example_chars], list(codes[j])))
            seen += 1
            if seen >= args.k:
                break
        prompt = build_rag_prompt(notes[idx][: args.max_example_chars], examples)
        target = json.dumps(sorted(c.upper() for c in codes[idx]))
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": target}]})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for r in rows[: args.n_train]:
            f.write(json.dumps(r) + "\n")
    with open(out / "valid.jsonl", "w") as f:
        for r in rows[args.n_train :]:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {args.n_train} train / {args.n_valid} valid RAG-format examples -> {out}")


if __name__ == "__main__":
    main()
