"""Turn probabilities into an auto-accept / defer decision with a precision guarantee.

This is the deployable part of the method, and the one place the toolkit deliberately goes BEYOND the
research code. eval/router_headline.py reports the selective frontier on the test set, which is the
right thing to do in a paper. It is the wrong thing to do in a hospital: a threshold picked at the
deepest point where test precision still clears the bar is, by construction, optimistic on those very
rows. So here the threshold is FIT on a `cal` split and then MEASURED on a disjoint `eval` split, and
the two numbers are reported side by side. If eval precision lands under the bar, the bar was not met
-- the toolkit says so instead of quietly shipping it.

Two threshold policies:
  global     one cut-off for every code. Robust, and what the paper reports.
  per-code   a cut-off per code, which is what a deployment actually wants (I10 is far easier than
             a four-character neoplasm code). Only codes with enough calibration positives get their
             own threshold; the rest fall back to global. Without that support floor a code with two
             cal positives gets a threshold fitted to noise and auto-accepts garbage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Cells below this can never clear an auto-accept bar; dropping them keeps the sort and the
# bootstrap tractable on a (n_notes x n_codes) grid.
PROB_FLOOR = 0.02
DEFAULT_BARS = (0.95, 0.90, 0.85, 0.80)


def gold_mask(gold_lists, lab2i, shape):
    gm = np.zeros(shape, dtype=bool)
    for a, g in enumerate(gold_lists):
        for c in g:
            j = lab2i.get(c)
            if j is not None:
                gm[a, j] = True
    return gm


def frontier(prob, correct, total_gold):
    """Confidence-sorted cumulative precision and coverage. Coverage is recall of ALL gold codes,
    including those outside the label space -- a coverage figure that silently excludes unreachable
    codes is not a workload fraction, and workload is what a coding department budgets."""
    o = np.argsort(-prob)
    p, c = prob[o], correct[o]
    tp = np.cumsum(c)
    n = np.arange(1, len(c) + 1)
    return p, tp / n, tp / max(total_gold, 1)


def threshold_for_precision(prob, correct, total_gold, bar):
    """Deepest confidence cut-off whose cumulative precision still clears `bar`.

    Returns (threshold, coverage, precision, n_cells) or None when the bar is unreachable. The
    threshold is nudged to the midpoint between the last accepted cell and the next one so that a
    new note landing exactly on a training probability is not decided by a floating-point tie."""
    p, prec, cov = frontier(prob, correct, total_gold)
    ok = np.where(prec >= bar)[0]
    if len(ok) == 0:
        return None
    i = ok[-1]
    th = float(p[i]) if i + 1 >= len(p) else float((p[i] + p[i + 1]) / 2)
    return {"threshold": th, "coverage": float(cov[i]), "precision": float(prec[i]),
            "n_cells": int(i + 1)}


def ece(prob, correct, nbins=10, floor=None):
    """Expected calibration error. `floor` restricts it to the DECISION REGION.

    Reporting ECE over every (note, code) cell is close to meaningless here: 99.8% of cells are
    near-zero-probability true negatives, so the all-cells number is dominated by cells no one will
    ever act on and comes out ~0.006 for a model whose actual decision-region error is ~0.015-0.11.
    Both are returned by `calibration_report` so the comparison is explicit."""
    if floor is not None:
        m = prob >= floor
        prob, correct = prob[m], correct[m]
    if len(prob) == 0:
        return float("nan")
    e = 0.0
    for k in range(nbins):
        lo, hi = k / nbins, (k + 1) / nbins
        m = (prob >= lo) & (prob < hi if k < nbins - 1 else prob <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - prob[m].mean())
    return float(e)


def calibration_report(prob, correct, nbins=10, floor=PROB_FLOOR):
    return {"ece_all_cells": ece(prob, correct, nbins),
            "ece_decision_region": ece(prob, correct, nbins, floor=floor),
            "decision_region_floor": floor,
            "n_cells": int(len(prob)),
            "n_cells_decision_region": int((prob >= floor).sum())}


class Router:
    """Auto-accept / defer, with the calibration that produced the thresholds attached."""

    def __init__(self, labels, bars=DEFAULT_BARS):
        self.labels = list(labels)
        self.lab2i = {c: i for i, c in enumerate(self.labels)}
        self.bars = [float(b) for b in bars]
        self.global_threshold: dict[str, float] = {}
        self.per_code: dict[str, dict[str, float]] = {}
        self.report: dict = {}

    # -- fitting -------------------------------------------------------------------------------
    def fit(self, P_cal, gold_cal, *, per_code_min_support=20, margin=0.0, log=print):
        """`margin` fits each threshold to (bar + margin) instead of bar.

        Threshold transfer is slightly optimistic: a cut-off chosen at the deepest point where
        CALIBRATION precision still clears the bar tends to land just under the bar on fresh rows,
        because that deepest point is partly fitted to the calibration split's noise. A margin of
        0.01-0.02 absorbs that at the cost of a little coverage. The eval split is what tells you
        whether you needed it -- fit with margin 0, read the verdicts, raise it if they say MARGINAL."""
        n, K = P_cal.shape
        gm = gold_mask(gold_cal, self.lab2i, P_cal.shape)
        total_gold = sum(len(g) for g in gold_cal)
        prob, corr = P_cal.reshape(-1), gm.reshape(-1)

        fitted = {}
        for bar in self.bars:
            target = min(bar + margin, 0.999)
            r = threshold_for_precision(prob, corr, total_gold, target)
            if r is None:
                log(f"  bar {bar:.2f}: UNREACHABLE on the calibration split"
                    f"{f' at the margin-adjusted target {target:.3f}' if margin else ''}"
                    f" -- no threshold offered")
                continue
            self.global_threshold[f"{bar:.2f}"] = r["threshold"]
            r["fit_target"] = target
            fitted[f"{bar:.2f}"] = r
            log(f"  bar {bar:.2f}: threshold {r['threshold']:.3f}  cal coverage {r['coverage']:.3f}"
                + (f"  (fitted to {target:.3f})" if margin else ""))

        # per-code thresholds, only where there is support to fit one
        pc_counts = gm.sum(axis=0)
        eligible = np.where(pc_counts >= per_code_min_support)[0]
        for bar in self.bars:
            key = f"{bar:.2f}"
            if key not in self.global_threshold:
                continue
            tbl = {}
            for j in eligible:
                r = threshold_for_precision(P_cal[:, j], gm[:, j], int(pc_counts[j]),
                                            min(bar + margin, 0.999))
                if r is not None:
                    tbl[self.labels[j]] = r["threshold"]
            self.per_code[key] = tbl
            log(f"  bar {bar:.2f}: per-code thresholds for {len(tbl)}/{K} codes "
                f"({len(eligible)} had >= {per_code_min_support} cal positives)")

        self.report["calibration"] = {
            "n_cal_notes": int(n), "n_labels": int(K), "total_gold": int(total_gold),
            "per_code_min_support": per_code_min_support,
            "bar_margin": float(margin),
            "n_codes_with_cal_support": int(len(eligible)),
            "fitted": fitted,
            **calibration_report(prob, corr)}
        return self

    # -- routing -------------------------------------------------------------------------------
    def route(self, probs_row, bar, *, policy="global", top_defer=25):
        """One note -> ({code: prob} to auto-accept, [(code, prob)] to show a human, highest first)."""
        key = self._key(bar)
        gth = self.global_threshold[key]
        pc = self.per_code.get(key, {}) if policy == "per-code" else {}
        auto, defer = [], []
        order = np.argsort(-probs_row)
        for j in order:
            code, p = self.labels[j], float(probs_row[j])
            th = pc.get(code, gth)
            if p >= th:
                auto.append({"code": code, "prob": p, "threshold": th})
            elif p >= PROB_FLOOR and len(defer) < top_defer:
                defer.append({"code": code, "prob": p, "threshold": th})
        return auto, defer

    def route_batch(self, P, bar, *, policy="global"):
        key = self._key(bar)
        gth = self.global_threshold[key]
        th = np.full(len(self.labels), gth, dtype=np.float32)
        if policy == "per-code":
            for c, t in self.per_code.get(key, {}).items():
                th[self.lab2i[c]] = t
        return P >= th[None, :]

    def _key(self, bar):
        key = f"{float(bar):.2f}"
        if key not in self.global_threshold:
            raise KeyError(f"no threshold calibrated for precision bar {key}; "
                           f"available: {sorted(self.global_threshold)}")
        return key

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"bars": self.bars, "labels": self.labels,
                                    "global_threshold": self.global_threshold,
                                    "per_code": self.per_code, "report": self.report}, indent=2))
        return path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        r = cls(d["labels"], d["bars"])
        r.global_threshold = d["global_threshold"]
        r.per_code = d.get("per_code", {})
        r.report = d.get("report", {})
        return r
