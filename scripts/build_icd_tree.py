#!/usr/bin/env python3
"""Build a navigable ICD-10-CM hierarchy tree from icd10cm_descriptions.tsv.

For the Code-Like-Humans-style navigation agent: a coder (or LLM) walks chapter -> block -> category ->
full code, choosing branches by description. The descriptions file already encodes every level:
  chapter   "1"          -> "Certain infectious ... (A00-B99)"
  block     "A00-A09"    -> "Intestinal infectious diseases (A00-A09)"
  category  "A00"        -> "Cholera"
  full code "A00.0"      -> "Cholera due to Vibrio cholerae 01, biovar cholerae"
We emit a JSON with, per node, its children (code -> description), so the agent can present the option
list at each level. Chapters link to blocks by the (A00-B99) range in their text; blocks to categories
by prefix range; categories to full codes by the 3-char prefix.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "data/icd_ref/icd10cm_descriptions.tsv"
OUT = ROOT / "data/icd_ref/icd10cm_tree.json"

RANGE_RE = re.compile(r"\(([A-Z]\d+)-([A-Z]\d+)\)")


def code_key(c):
    """Sortable key: letter, then number, so A00 < A09 < B00."""
    return (c[0], c[1:])


def in_block(cat3, lo, hi):
    return code_key(lo) <= code_key(cat3) <= code_key(hi)


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).dropna()
    rows = list(zip(df["code"], df["description"]))
    chapters, blocks, cats, fulls = {}, {}, {}, {}
    for c, d in rows:
        if re.fullmatch(r"\d+", c):
            chapters[c] = d
        elif re.fullmatch(r"[A-Z]\d+-[A-Z]\d+", c):
            blocks[c] = d
        elif re.fullmatch(r"[A-Z]\d{2}", c):
            cats[c] = d
        elif re.fullmatch(r"[A-Z]\d{2}\.\w+", c):
            fulls[c] = d

    # category -> full codes (by 3-char prefix)
    cat_children = {k: {} for k in cats}
    for fc, d in fulls.items():
        p = fc.split(".")[0]
        if p in cat_children:
            cat_children[p][fc] = d

    # block -> categories (by range parsed from the block code A00-A09)
    block_children = {}
    for b in blocks:
        lo, hi = b.split("-")
        block_children[b] = {c: cats[c] for c in cats if in_block(c, lo, hi)}

    # chapter -> blocks (by the (A00-B99) range in the chapter description)
    chap_children = {}
    for ch, d in chapters.items():
        m = RANGE_RE.search(d)
        if not m:
            chap_children[ch] = {}
            continue
        lo, hi = m.group(1), m.group(2)
        chap_children[ch] = {b: blocks[b] for b in blocks
                             if in_block(b.split("-")[0], lo, hi)}

    tree = {
        "chapters": chapters,
        "chapter_children": chap_children,   # chapter -> {block: desc}
        "block_children": block_children,    # block -> {category: desc}
        "category_children": cat_children,   # category -> {full_code: desc}
        "descriptions": {**chapters, **blocks, **cats, **fulls},
    }
    OUT.write_text(json.dumps(tree))
    # sanity report
    empty_ch = sum(1 for v in chap_children.values() if not v)
    empty_bl = sum(1 for v in block_children.values() if not v)
    orphan_cat = sum(1 for c in cats if not any(c in bc for bc in block_children.values()))
    print(f"chapters {len(chapters)}, blocks {len(blocks)}, categories {len(cats)}, full {len(fulls)}")
    print(f"empty chapters {empty_ch}, empty blocks {empty_bl}, categories not in any block {orphan_cat}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
