#!/usr/bin/env python3
"""Hierarchy-navigation agent (Code-Like-Humans-inspired, efficient hybrid) for the long tail.

Plain description-retrieval (desc-RAG) only offers the codes whose descriptions embed near the note's
entities, so rare codes with weak descriptions are never proposed (our desc-RAG tail recall = 0.046).
Here we add the hierarchy: after retrieving candidate codes per entity, we EXPAND each to its 3-char
category's sibling codes (from icd10cm_tree.json). A rare sibling that embedding-retrieval would miss
thus becomes selectable, and the LLM picks from a hierarchy-grouped candidate list. Two LLM calls per
note (entity extraction + selection) — cheap enough for local models, unlike a full per-level tree walk.

Ablate --expand to isolate the sibling-expansion (hierarchy) effect vs plain desc-retrieval.
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import ENTITY_EXTRACTION_PROMPT, parse_entities, parse_codes  # noqa: E402

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
GEN_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embed"

SELECT_HEADER = """You are an expert medical coder. Below are diagnostic entities extracted from a \
discharge note, then CANDIDATE ICD-10-CM codes grouped by category (with official descriptions). \
Select every code that the note's entities support; within a category choose the MOST SPECIFIC code \
that applies. Respond with ONLY a JSON array of codes. No prose.

Entities:
{entities}

Candidate codes by category:
"""

SELECT_HEADER_STRICT = """You are a meticulous senior medical coder. Below are diagnostic entities from \
a discharge note, then CANDIDATE ICD-10-CM codes grouped by category. Select a code ONLY when the entity \
list gives CLEAR, SPECIFIC evidence for it. Most candidates do NOT apply — they are near-misses shown for \
context. It is better to OMIT a code than to assign a wrong one. Pick at most one (the most specific) code \
per category, and only for entities that are actually present. Respond with ONLY a JSON array of codes.

Entities:
{entities}

Candidate codes by category (most are distractors — choose only well-supported ones):
"""


def ollama_gen(model, prompt, url, timeout=240, n=400):
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": n}}, timeout=timeout)
    r.raise_for_status(); return r.json().get("response", "")


def embed(texts, model, url, timeout=60):
    r = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status(); return np.asarray(r.json()["embeddings"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="medgemma:27b")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--gen-url", default=GEN_URL); ap.add_argument("--embed-url", default=EMBED_URL)
    ap.add_argument("--text-column", default="note_text_no_diagnosis")
    ap.add_argument("--desc-emb", default=str(ROOT / "data/embeddings/icd10cm_desc"))
    ap.add_argument("--tree", default=str(ROOT / "data/icd_ref/icd10cm_tree.json"))
    ap.add_argument("--per-entity", type=int, default=3, help="Codes retrieved per entity.")
    ap.add_argument("--expand", action="store_true", help="Add category-sibling codes (the hierarchy step).")
    ap.add_argument("--strict", action="store_true", help="Strict selector prompt (curb over-prediction).")
    ap.add_argument("--sib-cap", type=int, default=15, help="Max sibling codes added per category.")
    ap.add_argument("--max-cats", type=int, default=12, help="Max candidate categories shown.")
    ap.add_argument("--note-chars", type=int, default=6000)
    ap.add_argument("--tag", default="v2_hierarchy_agent"); ap.add_argument("--out", default=str(ROOT / "eval/results_v2"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(TEST_PARQUET)
    test = test[test["ver"] == "10"].sample(n=max(args.n, 1000), random_state=args.seed).reset_index(drop=True).iloc[:args.n]
    desc_emb = np.load(f"{args.desc_emb}.npy"); desc_codes = np.load(f"{args.desc_emb}.codes.npy").astype(str)
    tree = json.loads(Path(args.tree).read_text())
    cat_children = tree["category_children"]; descs = tree["descriptions"]
    cat_of = lambda c: c.split(".")[0]
    print(f"hierarchy-agent: {args.n} notes, expand={args.expand}, model={args.model}")

    records = []; tp = fp = fn = nerr = 0; t0 = time.time()
    for i in range(args.n):
        note = str(test.iloc[i][args.text_column])[: args.note_chars]
        gold = {c.upper() for c in test.iloc[i]["icd_codes"]}
        try:
            ents = parse_entities(ollama_gen(args.model, ENTITY_EXTRACTION_PROMPT.format(note_text=note), args.gen_url))[:12]
            if not ents:
                ents = [note[:200]]
            ev = embed(ents, args.embed_model, args.embed_url)
            # retrieve top codes per entity
            cand = set()
            for v in ev:
                sims = desc_emb @ v
                for j in np.argsort(-sims)[: args.per_entity]:
                    c = desc_codes[j].upper()
                    if "." in c or len(c) >= 3:  # keep code-like
                        cand.add(c)
            # group by category; optionally expand to siblings
            by_cat = defaultdict(dict)
            for c in cand:
                cc = cat_of(c)
                by_cat[cc][c] = descs.get(c, c)
                if args.expand and cc in cat_children:
                    sibs = list(cat_children[cc].items())[: args.sib_cap]
                    for sc, sd in sibs:
                        by_cat[cc][sc.upper()] = sd
            # cap categories by number of retrieved hits
            cats_sorted = sorted(by_cat, key=lambda k: -len([c for c in cand if cat_of(c) == k]))[: args.max_cats]
            lines = []
            for cc in cats_sorted:
                lines.append(f"[{cc} — {descs.get(cc, cc)}]")
                for code, d in list(by_cat[cc].items())[:20]:
                    lines.append(f"  {code}: {d}")
            hdr = SELECT_HEADER_STRICT if args.strict else SELECT_HEADER
            prompt = hdr.format(entities="\n".join(f"- {e}" for e in ents)) + "\n".join(lines) + \
                "\n\nJSON array of applicable ICD-10-CM codes:"
            pred = parse_codes(ollama_gen(args.model, prompt, args.gen_url))
        except requests.RequestException as e:
            nerr += 1; print(f"  [{i+1}] ERR {e}")
            if nerr >= 10 and not records: sys.exit("ABORT: backend unreachable")
            continue
        t_, f_, n_ = len(pred & gold), len(pred - gold), len(gold - pred)
        tp += t_; fp += f_; fn += n_
        records.append({"acc_no": int(test.iloc[i]["acc_no"]), "gold": sorted(gold), "pred": sorted(pred),
                        "tp": t_, "fp": f_, "fn": n_})
        if (i + 1) % 25 == 0:
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            print(f"  [{i+1}/{args.n}] {time.time()-t0:.0f}s P={p:.3f} R={r:.3f} F1={2*p*r/(p+r) if p+r else 0:.3f}")
    if not records: sys.exit("ABORT: no predictions")
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0; F = 2*P*R/(P+R) if P+R else 0
    summary = {"mode": "hierarchy_agent", "expand": args.expand, "model": args.model, "n_samples": len(records),
               "n_errors": nerr, "micro_precision": P, "micro_recall": R, "micro_f1": F,
               "total_tp": tp, "total_fp": fp, "total_fn": fn, "elapsed_sec": time.time()-t0}
    print("\n--- hierarchy agent ---"); print(json.dumps(summary, indent=2))
    with open(out / f"{args.tag}_predictions.jsonl", "w") as f:
        for r in records: f.write(json.dumps(r) + "\n")
    with open(out / f"{args.tag}_summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(f"Wrote -> {out}/{args.tag}_*")


if __name__ == "__main__":
    main()
