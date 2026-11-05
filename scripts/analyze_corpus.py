#!/usr/bin/env python3
"""Exploratory analysis of the built corpus: length distribution, label
long-tail, dis_diagnosis leakage check. Prints results and writes
docs/corpus_analysis.json for downstream chart-building.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def length_stats(df: pd.DataFrame) -> dict:
    chars = df["note_text"].str.len()
    words = df["note_text"].str.split().str.len()
    return {
        "char": chars.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "word": words.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
    }


def label_long_tail(vocab_path: Path) -> dict:
    vocab = pd.read_csv(vocab_path, sep="\t")
    total_codes = len(vocab)
    total_occurrences = vocab["count"].sum()
    singleton = (vocab["count"] == 1).sum()
    rare5 = (vocab["count"] <= 5).sum()
    vocab_sorted = vocab.sort_values("count", ascending=False).reset_index(drop=True)
    vocab_sorted["cum_pct"] = vocab_sorted["count"].cumsum() / total_occurrences
    coverage = {}
    for n in [50, 100, 500, 1000, 2000, 5000]:
        if n <= len(vocab_sorted):
            coverage[f"top_{n}"] = float(vocab_sorted.loc[n - 1, "cum_pct"])
    codes_for_80pct = int((vocab_sorted["cum_pct"] < 0.8).sum() + 1)
    codes_for_95pct = int((vocab_sorted["cum_pct"] < 0.95).sum() + 1)
    return {
        "total_unique_codes": int(total_codes),
        "total_occurrences": int(total_occurrences),
        "singleton_codes": int(singleton),
        "singleton_pct": float(singleton / total_codes),
        "codes_with_le5_occurrences": int(rare5),
        "codes_with_le5_pct": float(rare5 / total_codes),
        "coverage_by_top_n": coverage,
        "codes_needed_for_80pct_coverage": codes_for_80pct,
        "codes_needed_for_95pct_coverage": codes_for_95pct,
    }


def dis_diagnosis_leakage(df: pd.DataFrame, n_sample: int = 500) -> dict:
    """Heuristic: for each admission, check what fraction of gold ICD codes'
    3-char category prefix textually overlaps with tokens in dis_diagnosis.
    This is a crude proxy (no real ICD code->description dictionary available
    locally), so instead we measure whether dis_diagnosis is present/how long
    it is relative to note_text, plus a manual-read flag for the caller.
    """
    sample = df.sample(n=min(n_sample, len(df)), random_state=42)
    has_dis_dx = sample["dis_diagnosis"].str.len().fillna(0) > 0
    dis_dx_share_of_note = (
        sample["dis_diagnosis"].str.len().fillna(0) / sample["note_text"].str.len().clip(lower=1)
    )
    return {
        "pct_admissions_with_dis_diagnosis": float(has_dis_dx.mean()),
        "dis_diagnosis_share_of_note_text_chars": dis_dx_share_of_note.describe(
            percentiles=[0.5, 0.9]
        ).to_dict(),
    }


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    print(f"Loaded train.parquet: {len(df):,} rows\n")

    print("=== 1. note_text length distribution ===")
    lstats = length_stats(df)
    print(json.dumps(lstats, indent=2, default=float))

    print("\n=== 2. ICD label long-tail distribution ===")
    ltail = label_long_tail(PROCESSED / "icd_vocab.tsv")
    print(json.dumps(ltail, indent=2))

    print("\n=== 4. dis_diagnosis presence/size relative to note_text ===")
    leak = dis_diagnosis_leakage(df)
    print(json.dumps(leak, indent=2, default=float))

    out = {
        "n_train_rows": len(df),
        "length_stats": lstats,
        "label_long_tail": ltail,
        "dis_diagnosis": leak,
    }
    out_path = ROOT / "docs" / "corpus_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
