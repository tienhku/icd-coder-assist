#!/usr/bin/env python3
"""Encoder-scale sweep for §5.7: does the frozen-embedding gain saturate, and is it just probe width?

The manuscript reports a frozen linear probe over bge-m3 (568M, 1024-d) at 0.334 and over
qwen3-embedding:8b (8B, 4096-d) at 0.456, and states as a limitation that we never swept the range
between. This assembles the sweep.

Two questions, one table:

  1. **Saturation.** Where on the 568M -> 8B curve does the gain arrive? If a 0.6B encoder already
     captures most of it, the replication claim in §5.7 gets much cheaper to act on.
  2. **Is it just probe width?** The 8B embedding is 4096-d, so its probe has 4x the parameters of the
     bge-m3 probe -- a confound we flagged. qwen3-embedding:0.6b is **1024-d, exactly matching
     bge-m3**, so comparing those two holds probe capacity fixed and varies only the representation.

Every probe is the same code (`train_supervised_baseline.py`), the same pool, the same test
admissions and the same label space, so the encoder is the only thing that moves.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval/results_v2"
EMB = ROOT / "data/embeddings"

# (display name, params, pool-embedding prefix, probe tag)
ENCODERS = [
    ("bge-m3",              "568M", "train_pool_full",          "supervised_selcurve"),
    ("qwen3-embedding:0.6b", "0.6B", "train_pool_full_qwen3_0_6b", "supervised_selcurve_qwen3_0_6b"),
    ("qwen3-embedding:4b",   "4B",   "train_pool_full_qwen3_4b",   "supervised_selcurve_qwen3_4b"),
    ("qwen3-embedding:8b",   "8B",   "train_pool_full_qwen3",      "supervised_selcurve_qwen3"),
]
TEST_FOR = {
    "train_pool_full": "test_n5000_seed42",
    "train_pool_full_qwen3_0_6b": "test_n5000_seed42_qwen3_0_6b",
    "train_pool_full_qwen3_4b": "test_n5000_seed42_qwen3_4b",
    "train_pool_full_qwen3": "test_n5000_seed42_qwen3",
}
BASELINE = "supervised_selcurve"   # bge-m3, the comparison anchor


def preds(tag):
    """{acc_no: (gold, pred)} from a probe's prediction dump."""
    p = RES / f"{tag}_note_text_no_diagnosis_predictions.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in open(p):
        d = json.loads(line)
        out[d["acc_no"]] = (set(d["gold"]), set(d["pred"]))
    return out


def counts(d, keys, key):
    c = np.zeros((len(keys), 3))
    for i, k in enumerate(keys):
        g, pr = d[k]
        G = {key(x) for x in g}; P = {key(x) for x in pr}
        c[i] = (len(P & G), len(P - G), len(G - P))
    return c


def f1_of(c):
    tp, fp, fn = c.sum(0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def paired_delta(a, b, level="exact", boot=2000, seed=42):
    """b minus a, paired over the admissions both scored."""
    keys = sorted(set(a) & set(b))
    key = (lambda c: c) if level == "exact" else (lambda c: c.split(".")[0][:3])
    ca, cb = counts(a, keys, key), counts(b, keys, key)
    rng = np.random.default_rng(seed)
    d = [f1_of(cb[ix]) - f1_of(ca[ix])
         for ix in (rng.integers(0, len(keys), len(keys)) for _ in range(boot))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f1_of(ca), f1_of(cb), f1_of(cb) - f1_of(ca), lo, hi


def train_missing(dry_run=False):
    """Run the probe for any encoder whose embeddings exist but whose probe has not been trained."""
    for name, _, pool, tag in ENCODERS:
        if (RES / f"{tag}_note_text_no_diagnosis_summary.json").exists():
            continue
        if not (EMB / f"{pool}.npy").exists():
            print(f"  skip {name}: embeddings not present ({pool}.npy)")
            continue
        cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "eval/train_supervised_baseline.py"),
               "--pool-emb", str(EMB / pool), "--test-emb", str(EMB / TEST_FOR[pool]),
               "--tag", tag, "--out", str(RES)]
        print(f"  training probe for {name}: {' '.join(cmd[-8:])}")
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="Train any probe that is missing first.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.train:
        print("Training missing probes...")
        train_missing(args.dry_run)

    base = preds(BASELINE)
    if base is None:
        sys.exit("bge-m3 baseline probe predictions missing -- nothing to compare against.")

    rows = []
    for name, params, pool, tag in ENCODERS:
        s = RES / f"{tag}_note_text_no_diagnosis_summary.json"
        if not s.exists():
            rows.append((name, params, None, None, None, None, None, None, "not run"))
            continue
        j = json.load(open(s))
        dim = int(np.load(EMB / f"{pool}.npy", mmap_mode="r").shape[1])
        d = preds(tag)
        if tag == BASELINE:
            rows.append((name, params, dim, j["exact_micro_f1"], j["category_micro_f1"],
                         0.0, 0.0, 0.0, "baseline"))
        else:
            _, _, delta, lo, hi = paired_delta(base, d, "exact")
            rows.append((name, params, dim, j["exact_micro_f1"], j["category_micro_f1"],
                         delta, lo, hi, ""))

    print("\n| encoder | params | dim | exact F1 | category F1 | Δ exact vs bge-m3 | 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for name, params, dim, ex, ca, delta, lo, hi, note in rows:
        if ex is None:
            print(f"| {name} | {params} | — | *{note}* | | | |")
            continue
        d_s = "— (baseline)" if note == "baseline" else f"{delta:+.4f}"
        ci = "" if note == "baseline" else f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"| {name} | {params} | {dim} | {ex:.4f} | {ca:.4f} | {d_s} | {ci} |")

    # The capacity control: same 1024-d probe, different representation.
    a = preds(BASELINE); b = preds("supervised_selcurve_qwen3_0_6b")
    if b is not None:
        fa, fb, delta, lo, hi = paired_delta(a, b, "exact")
        print(f"\nCapacity control (both 1024-d, probe size held fixed): "
              f"bge-m3 {fa:.4f} -> qwen3-0.6b {fb:.4f}, Δ {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
