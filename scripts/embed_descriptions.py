#!/usr/bin/env python3
"""Embed ICD-10-CM code descriptions via the 5090's bge-m3, for code-description-retrieval RAG.

Companion to embed_notes.py, but the retrieval pool here is the ICD-10-CM code space itself
(code -> official description), not past admissions. This lets retrieval reach rare/long-tail codes
that never appear in the note pool — the ceiling that note-retrieval's oracle recall (0.37) hits.
Output mirrors embed_notes.py: <prefix>.npy (L2-normalized) + <prefix>.codes.npy (aligned codes).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

EMBED_URL = "http://localhost:11434/api/embed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, help="code<TAB>description TSV with header.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--url", default=EMBED_URL)
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    codes = df["code"].tolist()
    # Embed "code: description" so the code token anchors the vector too.
    texts = [f"{c}: {d}" for c, d in zip(codes, df["description"].tolist())]
    print(f"Embedding {len(texts)} ICD-10-CM descriptions via {args.model}")

    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        for attempt in range(3):
            try:
                r = requests.post(args.url, json={"model": args.model, "input": batch}, timeout=120)
                r.raise_for_status()
                vecs.extend(r.json()["embeddings"])
                break
            except requests.RequestException as e:
                if attempt == 2:
                    sys.exit(f"ABORT at row {i}: {e}")
                time.sleep(2)
        if (i // args.batch_size) % 20 == 0:
            done = min(i + args.batch_size, len(texts))
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(texts)}  {rate:.0f}/s  eta {(len(texts)-done)/rate:.0f}s")

    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    np.save(f"{args.out_prefix}.npy", arr)
    np.save(f"{args.out_prefix}.codes.npy", np.array(codes))
    print(f"Saved {arr.shape} -> {args.out_prefix}.npy ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
