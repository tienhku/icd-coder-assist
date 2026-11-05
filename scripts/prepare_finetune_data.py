#!/usr/bin/env python3
"""Build mlx_lm LoRA fine-tuning data from the corpus.

Samples VER=10 admissions from data/processed/{train,val}.parquet, renders
each into the same PROMPT_TEMPLATE used at eval time (scripts/prompting.py)
plus the gold ICD codes as a JSON-array assistant turn, and writes
{train,valid}.jsonl in mlx_lm's chat-format schema. See
docs/finetune_pilot_summary.md / the approved plan for design rationale.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from prompting import PROMPT_TEMPLATE

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def build_example(row, text_column: str) -> dict:
    user_content = PROMPT_TEMPLATE.format(note_text=row[text_column])
    assistant_content = json.dumps(list(row["icd_codes"]))
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def write_jsonl(df: pd.DataFrame, text_column: str, path: Path) -> None:
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(build_example(row, text_column)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-column", default="note_text_no_diagnosis")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "data" / "finetune" / "pilot"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    train_df = train_df[train_df["ver"] == "10"].sample(n=args.n_train, random_state=args.seed)
    val_df = pd.read_parquet(PROCESSED / "val.parquet")
    val_df = val_df[val_df["ver"] == "10"].sample(n=args.n_val, random_state=args.seed)

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    write_jsonl(train_df, args.text_column, train_path)
    write_jsonl(val_df, args.text_column, valid_path)
    print(f"Wrote {len(train_df):,} train examples -> {train_path}")
    print(f"Wrote {len(val_df):,} val examples -> {valid_path}")

    # Sanity-check rendered length against a token budget using the real
    # tokenizer (loaded once, cheap — this only loads the tokenizer files,
    # not the 8GB of model weights).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    lengths = []
    for _, row in pd.concat([train_df, val_df]).iterrows():
        ex = build_example(row, args.text_column)
        rendered = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        lengths.append(len(tokenizer.encode(rendered)))
    lengths = pd.Series(lengths)
    print("\nRendered chat-template token length distribution:")
    print(lengths.describe(percentiles=[0.5, 0.9, 0.99]).to_string())


if __name__ == "__main__":
    main()
