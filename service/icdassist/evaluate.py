"""Measure what the calibrated router actually does on a held-out split, and say when it misses.

Everything here is paired-bootstrapped OVER NOTES, not over cells. Codes within one admission are
correlated -- a diabetic foot note carries E11.x and L97.x together -- so resampling cells would give
confidence intervals several times too narrow. Resampling admissions is the unit the project uses
everywhere and is the unit a hospital's caseload actually varies in.
"""
from __future__ import annotations

import numpy as np

from .calibrate import PROB_FLOOR, calibration_report, frontier, gold_mask

# ICD-10 chapter boundaries by first letter; used only for the per-chapter breakdown, and skipped
# entirely when a site uses a code system this does not describe.
CHAPTERS = [("A", "B", 1), ("C", "D", 2), ("E", "E", 4), ("F", "F", 5), ("G", "G", 6), ("H", "H", 7),
            ("I", "I", 9), ("J", "J", 10), ("K", "K", 11), ("L", "L", 12), ("M", "M", 13),
            ("N", "N", 14), ("O", "O", 15), ("P", "P", 16), ("Q", "Q", 17), ("R", "R", 18),
            ("S", "T", 19), ("V", "Y", 20), ("Z", "Z", 21)]


def chapter_of(code):
    for lo, hi, ch in CHAPTERS:
        if lo <= code[:1] <= hi:
            return ch
    return 0


def micro_f1(preds, golds, key=None):
    key = key or (lambda c: c)
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        P = {key(c) for c in p}
        G = {key(c) for c in g}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"micro_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def rollup(P, labels, fn):
    """Collapse the label axis by a rollup function, taking the max probability per group -- the
    confidence that SOME code in the group applies."""
    groups = [fn(c) for c in labels]
    uniq = sorted(set(groups))
    idx = {g: i for i, g in enumerate(uniq)}
    col = np.array([idx[g] for g in groups])
    out = np.zeros((P.shape[0], len(uniq)), dtype=np.float32)
    np.maximum.at(out.T, col, P.T)
    return out, uniq


def achieved(P_eval, gold_eval, router, bar, *, policy="global"):
    """What the cal-fitted threshold actually delivers on eval: precision, coverage, and a verdict."""
    acc = router.route_batch(P_eval, bar, policy=policy)
    gm = gold_mask(gold_eval, router.lab2i, P_eval.shape)
    tp = int((acc & gm).sum())
    n_auto = int(acc.sum())
    total_gold = sum(len(g) for g in gold_eval)
    prec = tp / n_auto if n_auto else float("nan")
    return {"bar": float(bar), "policy": policy,
            "threshold_source": "calibration split",
            "n_auto_accepted": n_auto, "n_correct": tp,
            "precision": prec, "coverage": tp / max(total_gold, 1),
            "total_gold": total_gold,
            "bar_met": bool(n_auto and prec >= bar)}


def verdict(bar, precision, precision_ci):
    """Three outcomes, because two would be dishonest in both directions.

    A point estimate of 0.945 against a 0.95 bar, on 400 notes, with a 95% CI of [0.932, 0.958], is
    not a failed bar -- it is a bar this much held-out data cannot resolve. Calling it a failure
    pushes an operator to discard a usable tier; calling it a pass hides that it might be under.
    So: MET when the point estimate clears the bar, MARGINAL when it does not but the interval still
    covers it, MISSED only when the whole interval sits below."""
    if precision != precision:                       # NaN: nothing was auto-accepted
        return "NO AUTO-ACCEPTS"
    if precision >= bar:
        return "MET"
    hi = precision_ci[1] if precision_ci else None
    if hi is not None and hi >= bar:
        return "MARGINAL"
    return "MISSED"


def bootstrap_achieved(P_eval, gold_eval, router, bars, *, policy="global", n_boot=1000, seed=42):
    """Paired bootstrap over notes on the FIXED cal-fitted thresholds. No re-fitting inside the
    resample: the threshold is a shipped constant, and its uncertainty belongs to the cal split."""
    rng = np.random.default_rng(seed)
    gm = gold_mask(gold_eval, router.lab2i, P_eval.shape)
    gpn = np.array([len(g) for g in gold_eval])
    out = {}
    for bar in bars:
        key = f"{float(bar):.2f}"
        if key not in router.global_threshold:
            continue
        acc = router.route_batch(P_eval, bar, policy=policy)
        hit = (acc & gm).sum(axis=1)       # per-note correct auto-accepts
        tot = acc.sum(axis=1)              # per-note auto-accepts
        precs, covs = [], []
        n = len(gpn)
        for _ in range(n_boot):
            s = rng.integers(0, n, n)
            t, h, g = tot[s].sum(), hit[s].sum(), gpn[s].sum()
            precs.append(h / t if t else np.nan)
            covs.append(h / g if g else np.nan)
        out[key] = {"precision_ci": [float(np.nanpercentile(precs, 2.5)),
                                     float(np.nanpercentile(precs, 97.5))],
                    "coverage_ci": [float(np.nanpercentile(covs, 2.5)),
                                    float(np.nanpercentile(covs, 97.5))],
                    "n_boot": n_boot}
    return out


def full_report(P_eval, gold_eval, router, *, policy="global", n_boot=1000, seed=42,
                rollup_fn=None, per_chapter=True):
    gm = gold_mask(gold_eval, router.lab2i, P_eval.shape)
    prob, corr = P_eval.reshape(-1), gm.reshape(-1)
    total_gold = sum(len(g) for g in gold_eval)
    reachable = sum(1 for g in gold_eval for c in g if c in router.lab2i)

    rep = {"n_eval_notes": int(len(gold_eval)),
           "n_labels": int(P_eval.shape[1]),
           "total_gold_codes": int(total_gold),
           "gold_in_label_space": reachable / max(total_gold, 1),
           "calibration": calibration_report(prob, corr),
           "bars": {}, "oracle_frontier": {}}

    boot = bootstrap_achieved(P_eval, gold_eval, router, router.bars,
                              policy=policy, n_boot=n_boot, seed=seed)
    for bar in router.bars:
        key = f"{bar:.2f}"
        if key not in router.global_threshold:
            rep["bars"][key] = {"status": "unreachable on the calibration split"}
            continue
        a = achieved(P_eval, gold_eval, router, bar, policy=policy)
        a.update(boot.get(key, {}))
        a["threshold"] = router.global_threshold[key]
        a["verdict"] = verdict(bar, a["precision"], a.get("precision_ci"))
        rep["bars"][key] = a

    # The oracle frontier -- thresholds chosen ON eval. Reported only as a ceiling, to show how much
    # of the gap between cal-fitted and oracle is threshold-transfer loss rather than model quality.
    p, prec, cov = frontier(prob, corr, total_gold)
    for bar in router.bars:
        ok = np.where(prec >= bar)[0]
        rep["oracle_frontier"][f"{bar:.2f}"] = (
            {"coverage": float(cov[ok[-1]]), "threshold": float(p[ok[-1]])} if len(ok)
            else {"coverage": 0.0, "threshold": None})

    if rollup_fn is not None:
        Pr, ulabels = rollup(P_eval, router.labels, rollup_fn)
        gr = [{rollup_fn(c) for c in g} for g in gold_eval]
        ridx = {c: i for i, c in enumerate(ulabels)}
        grm = np.zeros(Pr.shape, bool)
        for a, s in enumerate(gr):
            for c in s:
                if c in ridx:
                    grm[a, ridx[c]] = True
        pr, pc, rc = frontier(Pr.reshape(-1), grm.reshape(-1), sum(len(s) for s in gr))
        rep["rollup_oracle_frontier"] = {}
        for bar in router.bars:
            ok = np.where(pc >= bar)[0]
            rep["rollup_oracle_frontier"][f"{bar:.2f}"] = float(rc[ok[-1]]) if len(ok) else 0.0

    if per_chapter:
        lab_ch = np.array([chapter_of(c) for c in router.labels])
        if (lab_ch > 0).mean() > 0.5:          # only meaningful for ICD-10-shaped codes
            rows = []
            for ch in sorted(set(lab_ch[lab_ch > 0])):
                cols = np.where(lab_ch == ch)[0]
                tg = int(gm[:, cols].sum())
                if tg < 30:
                    continue
                _, pp, cc = frontier(P_eval[:, cols].reshape(-1), gm[:, cols].reshape(-1), tg)
                ok = np.where(pp >= 0.90)[0]
                rows.append({"chapter": int(ch), "gold": tg,
                             "coverage_at_0.90": float(cc[ok[-1]]) if len(ok) else 0.0})
            rep["per_chapter_at_0.90"] = sorted(rows, key=lambda r: -r["coverage_at_0.90"])
    return rep


def format_report(rep):
    L = []
    L.append(f"notes {rep['n_eval_notes']:,}   labels {rep['n_labels']:,}   "
             f"gold codes {rep['total_gold_codes']:,}   "
             f"in label space {rep['gold_in_label_space']:.3f}")
    c = rep["calibration"]
    line = (f"ECE  decision region (p>={c['decision_region_floor']}) {c['ece_decision_region']:.4f}"
            f"   all cells {c['ece_all_cells']:.4f}")
    # The caveat only earns its space when the two actually diverge. They diverge when the label
    # space is large enough that most cells are near-zero true negatives; with a few dozen codes
    # they coincide and printing a warning about a difference of 0.0000 just erodes trust.
    if abs(c["ece_all_cells"] - c["ece_decision_region"]) > 0.005:
        line += "  <- the all-cells number is the misleading one"
    L.append(line)
    L.append("")
    L.append(f"{'bar':>5}{'thresh':>9}{'coverage':>10}{'95% CI':>18}"
             f"{'precision':>11}{'95% CI':>18}{'oracle':>9}{'verdict':>11}")
    verdicts = {}
    for key in sorted(rep["bars"], reverse=True):
        b = rep["bars"][key]
        if "status" in b:
            L.append(f"{key:>5}{'—':>9}  {b['status']}")
            continue
        cl, ch = b.get("coverage_ci", [float("nan")] * 2)
        pl, ph = b.get("precision_ci", [float("nan")] * 2)
        orc = rep["oracle_frontier"].get(key, {}).get("coverage", float("nan"))
        v = b.get("verdict", "?")
        verdicts[key] = v
        L.append(f"{key:>5}{b['threshold']:>9.3f}{b['coverage']:>10.3f}"
                 f"   [{cl:.3f}, {ch:.3f}]{b['precision']:>11.3f}   [{pl:.3f}, {ph:.3f}]"
                 f"{orc:>9.3f}{v:>11}")
    L.append("")
    L.append("coverage = share of the site's TRUE coding workload auto-accepted; the rest defers to a coder.")
    L.append("oracle  = coverage if the threshold had been tuned on these same rows (a ceiling, not a claim).")
    L.append("verdict = MET: held-out precision clears the bar. MARGINAL: point estimate is under it")
    L.append("          but the 95% CI still covers it -- this much eval data cannot tell them apart.")
    L.append("          MISSED: the whole interval is below the bar.")
    if "MARGINAL" in verdicts.values():
        L.append("")
        L.append(f"MARGINAL at {', '.join(k for k, v in sorted(verdicts.items()) if v == 'MARGINAL')}. "
                 f"Threshold transfer is mildly")
        L.append("optimistic by construction. Either enlarge the calibration split, or refit with")
        L.append("--bar-margin 0.02 to buy precision back with a little coverage, then re-read this table.")
    if "MISSED" in verdicts.values():
        L.append("")
        L.append(f"WARNING: bar(s) {', '.join(k for k, v in sorted(verdicts.items()) if v == 'MISSED')} "
                 f"did not hold on held-out data.")
        L.append("Do not operate those tiers. Usual causes: too few calibration notes, or cal and eval")
        L.append("drawn from different periods, services or coding teams.")
    if "per_chapter_at_0.90" in rep:
        L.append("")
        L.append("Per-ICD-chapter auto-codable coverage @0.90 (oracle, top 8):")
        for r in rep["per_chapter_at_0.90"][:8]:
            L.append(f"  chapter {r['chapter']:>2}: gold {r['gold']:>6}  coverage {r['coverage_at_0.90']:.3f}")
    return "\n".join(L)
