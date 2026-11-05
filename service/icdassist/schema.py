"""The input contract a deploying site has to meet, and nothing more.

Deliberately NOT our research schema. The research code reads a 23-column parquet keyed on hospital
accession numbers with an ICD-version column; none of that is portable. A site adopting this method
supplies JSONL records:

    {"id": "<opaque local id>", "text": "<note text>", "codes": ["I10", "E11.9"], "split": "train"}

  id     any string unique within the file. Never needs to be a patient or accession identifier --
         the toolkit only uses it to join predictions back to the source record on the site's side.
  text   the note. Whatever the site decides to feed the coder: we use a discharge summary with the
         diagnosis fields removed (see --strip-fields), because leaving them in leaks the label and
         inflates F1 about 8x (docs/withdiagnosis_results.md).
  codes  the gold codes for training/calibration rows; omit or leave empty for rows to be predicted.
         Treated as OPAQUE STRINGS. ICD-10-CM, ICD-10 WHO, ICD-9-CM and local extensions all work --
         the method never parses them except for the optional category rollup.
  split  "train" | "cal" | "eval" | "predict", or absent and assigned by --val-frac.

Three splits, not two, and this matters: the auto-accept threshold is FIT on `cal` and only then
MEASURED on `eval`. A threshold chosen on the same rows it is reported on overstates coverage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "cal", "eval", "predict")


@dataclass
class Record:
    id: str
    text: str
    codes: list[str] = field(default_factory=list)
    split: str | None = None


class SchemaError(ValueError):
    pass


def _norm_codes(raw, rid):
    if raw is None:
        return []
    if isinstance(raw, str):                      # "I10;E11.9" or "I10,E11.9" -- a very common export
        raw = [c for c in raw.replace(";", ",").split(",") if c.strip()]
    if not isinstance(raw, list):
        raise SchemaError(f"record {rid!r}: 'codes' must be a list or delimited string, got {type(raw).__name__}")
    out, seen = [], set()
    for c in raw:
        c = str(c).strip().upper()
        if c and c not in seen:                   # upper+dedup: exports differ on case and repeat codes
            seen.add(c); out.append(c)
    return out


def load_jsonl(path, *, require_codes=False, text_field="text"):
    """Read records, failing loudly on a malformed FILE but never on one odd row's codes."""
    recs, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{ln}: not valid JSON ({e.msg})") from None
            rid = str(d.get("id", ln))
            if rid in ids:
                raise SchemaError(f"{path}:{ln}: duplicate id {rid!r}")
            ids.add(rid)
            text = d.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise SchemaError(f"{path}:{ln}: record {rid!r} has no non-empty {text_field!r}")
            codes = _norm_codes(d.get("codes"), rid)
            if require_codes and not codes:
                raise SchemaError(f"{path}:{ln}: record {rid!r} has no codes, but this file needs them")
            split = d.get("split")
            if split is not None and split not in SPLITS:
                raise SchemaError(f"{path}:{ln}: split {split!r} not one of {SPLITS}")
            recs.append(Record(id=rid, text=text, codes=codes, split=split))
    if not recs:
        raise SchemaError(f"{path}: no records")
    return recs


def assign_splits(recs, *, cal_frac=0.10, eval_frac=0.10, seed=42):
    """Fill in missing splits. Rows that already declare one are left alone, so a site can pin its own."""
    import numpy as np
    free = [i for i, r in enumerate(recs) if r.split is None]
    rng = np.random.default_rng(seed)
    rng.shuffle(free)
    n_cal, n_eval = int(len(free) * cal_frac), int(len(free) * eval_frac)
    for i in free[:n_cal]:
        recs[i].split = "cal"
    for i in free[n_cal:n_cal + n_eval]:
        recs[i].split = "eval"
    for i in free[n_cal + n_eval:]:
        recs[i].split = "train"
    return recs


def split_counts(recs):
    from collections import Counter
    return dict(Counter(r.split or "unassigned" for r in recs))


# ---- category rollup -------------------------------------------------------------------------
# The research code hardcodes code.split('.')[0][:3], which is only right for ICD-10/9 style codes.
# A site on a different code system picks a rollup here, or 'none' to report exact only.

def rollup_icd_3char(code):
    return code.replace(".", "")[:3]


def rollup_none(code):
    return code


ROLLUPS = {"icd-3char": rollup_icd_3char, "none": rollup_none}


def get_rollup(name):
    if name not in ROLLUPS:
        raise SchemaError(f"unknown --rollup {name!r}; choose from {sorted(ROLLUPS)}")
    return ROLLUPS[name]
