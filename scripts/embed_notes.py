#!/usr/bin/env python3
"""Embed note text via the RTX 5090's Ollama embedding endpoint, save vectors + acc_no index.

bge-m3 is multilingual (verified: EN/ZH paraphrase cosine 0.70), which matters for the bilingual
notes. Embeddings of patient text run on the 5090 — allowed, it's a self-owned LAN machine
(see docs/rtx5090_node_calibration.md). Output: an L2-normalized float32 .npy (rows aligned to the
saved acc_no order), so retrieval is a plain dot product.

Long runs (the 258k train pool takes ~24h) CHECKPOINT every --checkpoint-every batches and resume
automatically, and a note that cannot be embedded at all is recorded as a zero vector rather than
killing the run — an earlier 10-hour qwen3 run was lost at 44% to a single bad note.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
EMBED_URL = "http://localhost:11434/api/embed"


def embed_batch(texts, model, url, timeout=120):
    resp = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _is_too_long(exc):
    """A 400 means the input exceeded the model's context — shrink. Anything else (connection reset,
    timeout, 500 from a momentarily overloaded ollama) is transient — back off and retry the same size."""
    resp = getattr(exc, "response", None)
    return resp is not None and resp.status_code == 400



def _stack(vecs, dim):
    """Stack vecs into an (n, dim) float32 array, zero-filling the rows that failed to embed.

    The obvious np.asarray([v if v is not None else [0.0]*dim for v in vecs]) builds one more giant
    Python list before the array exists, so at the peak the list-of-lists and the array are both
    resident. Pre-allocating and filling row by row avoids that peak.
    """
    out = np.zeros((len(vecs), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        if v is not None:
            out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--column", default="note_text_no_diagnosis")
    ap.add_argument("--out-prefix", required=True, help="Writes <prefix>.npy and <prefix>.acc_no.npy")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--url", default=EMBED_URL)
    ap.add_argument("--ver", default="10", help="Filter to this ICD version ('' = no filter).")
    ap.add_argument("--n", type=int, default=0, help="Sample this many rows (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=28000,
                    help="Truncate notes to this many chars before embedding (~8k tokens; bge-m3's "
                         "limit is 8192, and v2 full-record notes have a long tail up to 284k chars).")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Flush a resumable partial checkpoint every N batches (0 = never).")
    ap.add_argument("--restart", action="store_true", help="Ignore any existing checkpoint and start over.")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.ver:
        df = df[df["ver"] == args.ver]
    if args.n:
        df = df.sample(n=args.n, random_state=args.seed)
    df = df.reset_index(drop=True)
    texts = [str(t)[: args.max_chars] for t in df[args.column].tolist()]
    acc_nos = df["acc_no"].to_numpy()
    del df          # a 3.6GB DataFrame; texts and acc_nos are all the rest of the run needs
    print(f"Embedding {len(texts)} rows from {args.parquet} col='{args.column}' via {args.model}")

    # Resume is only safe if the row ORDER is identical, so key the checkpoint on the acc_no sequence.
    fingerprint = hashlib.sha256(acc_nos.tobytes()).hexdigest()[:16]
    ckpt = Path(f"{args.out_prefix}.partial.npy")
    ckpt_meta = Path(f"{args.out_prefix}.partial.meta.npy")

    vecs, start = [], 0
    if ckpt.exists() and ckpt_meta.exists() and not args.restart:
        meta = np.load(ckpt_meta, allow_pickle=True).item()
        if meta.get("fingerprint") == fingerprint and meta.get("model") == args.model:
            prev = np.load(ckpt)
            vecs = [row for row in prev]
            start = len(vecs)
            print(f"RESUMING from checkpoint: {start}/{len(texts)} rows already embedded "
                  f"({start/len(texts):.1%})")
        else:
            print("Checkpoint exists but does not match this run (different rows or model) — starting over.")

    failed = []

    def embed_one(text, idx):
        """One note, shrinking on 400s and backing off on transient errors. Never raises."""
        for cap in (args.max_chars, args.max_chars // 2, args.max_chars // 4, 2000, 1000, 500):
            for attempt in range(3):
                try:
                    return embed_batch([text[:cap]], args.model, args.url)[0]
                except requests.RequestException as e:
                    if _is_too_long(e):
                        break                      # too long at this cap — go smaller
                    time.sleep(2 ** attempt)       # transient — same cap, back off
        failed.append(idx)
        print(f"  WARN row {idx} could not be embedded at any length; writing a zero vector", flush=True)
        return None

    def embed_chunk(batch, base_idx):
        """Whole batch fast-path, then per-note fallback so one bad note cannot poison its neighbours."""
        for cap in (args.max_chars, args.max_chars // 2, args.max_chars // 4):
            for attempt in range(2):
                try:
                    return embed_batch([t[:cap] for t in batch], args.model, args.url)
                except requests.RequestException as e:
                    if _is_too_long(e):
                        break
                    time.sleep(2 ** attempt)
        return [embed_one(t, base_idx + j) for j, t in enumerate(batch)]

    dim = len(vecs[0]) if vecs else None
    t0 = time.time()
    nb = 0
    for i in range(start, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        got = embed_chunk(batch, i)
        for v in got:
            if v is None:
                vecs.append(None)          # placeholder, filled with zeros once dim is known
            else:
                dim = dim or len(v)
                # float32, not a Python list: a list of floats costs ~32 bytes per element
                # (a 24-byte float object plus an 8-byte pointer) against 4 bytes for float32.
                # One 4096-d vector is 160KB vs 16KB -- about 36GB over 250k notes.
                vecs.append(np.asarray(v, dtype=np.float32))
        nb += 1
        done = min(i + args.batch_size, len(texts))
        if nb % 20 == 0:
            rate = (done - start) / (time.time() - t0)
            print(f"  {done}/{len(texts)}  {rate:.0f} notes/s  eta {(len(texts)-done)/max(rate,1e-9):.0f}s",
                  flush=True)
        if args.checkpoint_every and nb % args.checkpoint_every == 0 and dim:
            partial = _stack(vecs, dim)
            np.save(ckpt, partial)
            np.save(ckpt_meta, np.array({"fingerprint": fingerprint, "model": args.model,
                                         "n_done": len(vecs), "failed": list(failed)}, dtype=object))
            print(f"  [checkpoint] {len(vecs)}/{len(texts)} rows saved", flush=True)

    if dim is None:
        sys.exit("ABORT: not a single note embedded successfully — check the ollama endpoint/model.")
    arr = _stack(vecs, dim)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8  # L2-normalize for dot-product cosine
    np.save(f"{args.out_prefix}.npy", arr)
    np.save(f"{args.out_prefix}.acc_no.npy", acc_nos)
    if failed:
        np.save(f"{args.out_prefix}.failed_idx.npy", np.array(failed))
        print(f"WARNING: {len(failed)} note(s) are zero vectors (never retrieved): "
              f"indices saved to {args.out_prefix}.failed_idx.npy")
    for p in (ckpt, ckpt_meta):
        p.unlink(missing_ok=True)
    print(f"Saved {arr.shape} -> {args.out_prefix}.npy  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
