#!/usr/bin/env python3
"""Probe LoRA checkpoints for mode collapse / divergence, cheaply.

Runs each checkpoint over a handful of *different* test admissions and reports how
many **distinct** answers it gives. That is the metric that matters: a collapsed model
still emits well-formed JSON — it just emits the *same* JSON for every note. A
single-prompt smoke test cannot tell a collapsed model from a working one, and during
the scaleup_v1 post-mortem it produced a confidently wrong "this checkpoint is fine"
call (see docs/scaleup_v1_failure.md).

Typical use — find where a run went bad, in ~30s per checkpoint instead of a 150-sample
eval per checkpoint:

    .venv/bin/python eval/probe_checkpoints.py models/adapters/scaleup_v2_no_diagnosis

Reading the output:
  distinct == n        healthy, input-dependent
  distinct ~ n/3       collapsing toward template answers
  distinct <= 2        collapsed; the adapter is answering from a prior, not the note
  empty > 0            diverged; no parseable codes at all
"""
import argparse
import collections
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import PROMPT_TEMPLATE, parse_codes  # noqa: E402

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter_dir", help="Dir holding adapter_config.json + NNNNNNN_adapters.safetensors")
    parser.add_argument("--mlx-model", default="google/medgemma-4b-it")
    parser.add_argument("--n", type=int, default=20, help="Admissions to probe per checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--every", type=int, default=1, help="Probe every Nth checkpoint.")
    parser.add_argument("--text-column", default="note_text_no_diagnosis")
    args = parser.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors

    adapter_dir = Path(args.adapter_dir)
    ckpts = sorted(adapter_dir.glob("0*_adapters.safetensors"))[:: args.every]
    if not ckpts:
        parser.error(f"no NNNNNNN_adapters.safetensors checkpoints in {adapter_dir}")
    config = adapter_dir / "adapter_config.json"
    if not config.exists():
        parser.error(f"missing {config}")

    df = pd.read_parquet(TEST_PARQUET)
    df = df[df["ver"] == "10"].sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    prompts = [PROMPT_TEMPLATE.format(note_text=r[args.text_column]) for _, r in df.iterrows()]
    golds = [{c.upper() for c in r["icd_codes"]} for _, r in df.iterrows()]

    logits_processors = make_logits_processors(repetition_penalty=1.3)
    print(f"Probing {len(ckpts)} checkpoints on {args.n} admissions (seed {args.seed})\n")
    print(f"{'ckpt':>8} {'distinct':>9} {'empty':>6} {'tp':>4}  {'most common answer':<40}")
    print("-" * 74)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "adapter"
        staged.mkdir()
        shutil.copy(config, staged / "adapter_config.json")
        for ckpt in ckpts:
            shutil.copy(ckpt, staged / "adapters.safetensors")
            model, tokenizer = load(args.mlx_model, adapter_path=str(staged))
            preds, tp = [], 0
            for prompt, gold in zip(prompts, golds):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
                )
                out = generate(
                    model,
                    tokenizer,
                    prompt=rendered,
                    max_tokens=args.max_tokens,
                    logits_processors=logits_processors,
                )
                codes = parse_codes(out)
                preds.append(tuple(sorted(codes)))
                tp += len(codes & gold)
            distinct = len(set(preds))
            empty = sum(1 for p in preds if not p)
            answer, count = collections.Counter(preds).most_common(1)[0]
            flag = "" if distinct > args.n / 2 else ("  <-- DIVERGED" if empty == len(preds) else "  <-- COLLAPSED")
            it = re.search(r"(\d+)_adapters", ckpt.name).group(1)
            print(
                f"{it:>8} {f'{distinct}/{args.n}':>9} {empty:>6} {tp:>4}  "
                f"{str(answer)[:32]:<33} x{count}{flag}"
            )
            del model, tokenizer


if __name__ == "__main__":
    main()
