"""Fast, offline tests. No embedding server, no clinical data, no network.

Run: cd service && python -m pytest tests -q      (or: python tests/test_icdassist.py)

The properties worth pinning here are the ones that would be silently wrong rather than loudly
broken: that the probe's gradient is actually the gradient, that a threshold fitted on one split is
not quietly reported on itself, and that the schema rejects the input shapes a site will inevitably
send on the first try.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icdassist.calibrate import Router, ece, threshold_for_precision          # noqa: E402
from icdassist.evaluate import micro_f1, rollup, verdict                      # noqa: E402
from icdassist.probe import (Adam, LinearProbe, bce_with_logits,              # noqa: E402
                             build_label_space, sigmoid)
from icdassist.schema import (SchemaError, assign_splits, get_rollup,         # noqa: E402
                             load_jsonl)


# ---- numerics --------------------------------------------------------------------------------
def test_sigmoid_is_stable_at_extremes():
    z = np.array([-900.0, -40.0, 0.0, 40.0, 900.0], dtype=np.float64)
    p = sigmoid(z)
    assert np.all(np.isfinite(p)), "sigmoid overflowed"
    assert p[0] == 0.0 and p[-1] == 1.0
    assert abs(p[2] - 0.5) < 1e-12


def test_bce_matches_naive_form_in_the_safe_range():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(40, 7))
    y = (rng.random((40, 7)) < 0.3).astype(np.float64)
    p = sigmoid(z)
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert abs(bce_with_logits(z, y) - naive) < 1e-10


def test_probe_gradient_matches_finite_differences():
    """The one piece of hand-written calculus in the package. If dZ is wrong the probe still
    trains, just to the wrong place, and nothing downstream would notice."""
    rng = np.random.default_rng(1)
    B, D, K = 6, 5, 4
    X = rng.normal(size=(B, D))
    Y = (rng.random((B, K)) < 0.4).astype(np.float64)
    W = rng.normal(scale=0.3, size=(D, K))
    b = rng.normal(scale=0.3, size=K)

    def loss(W, b):
        return bce_with_logits(X @ W + b, Y)

    Z = X @ W + b
    dZ = (sigmoid(Z) - Y) / (B * K)
    gW, gb = X.T @ dZ, dZ.sum(axis=0)

    eps = 1e-6
    for i in range(D):
        for j in range(K):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += eps; Wm[i, j] -= eps
            fd = (loss(Wp, b) - loss(Wm, b)) / (2 * eps)
            assert abs(fd - gW[i, j]) < 1e-7, f"dW[{i},{j}] {gW[i, j]} vs fd {fd}"
    for j in range(K):
        bp, bm = b.copy(), b.copy()
        bp[j] += eps; bm[j] -= eps
        fd = (loss(W, bp) - loss(W, bm)) / (2 * eps)
        assert abs(fd - gb[j]) < 1e-7


def test_adam_reduces_a_quadratic():
    x = [np.array([5.0, -3.0], dtype=np.float32)]
    opt = Adam([x[0].shape], lr=0.1)
    for _ in range(400):
        opt.step(x, [2 * x[0]])
    assert np.allclose(x[0], 0.0, atol=1e-2), x[0]


def test_probe_learns_a_separable_problem():
    rng = np.random.default_rng(2)
    n, D, K = 600, 12, 3
    W_true = rng.normal(size=(D, K))
    X = rng.normal(size=(n, D)).astype(np.float32)
    Y = (X @ W_true > 0.6)
    Y_idx = [np.where(r)[0].astype(np.int32) for r in Y]
    p = LinearProbe(D, ["A", "B", "C"], seed=3)
    hist = p.fit(X, Y_idx, epochs=120, batch_size=128, lr=0.05, log=lambda *a: None)
    assert hist[-1] < hist[0] / 3, f"loss barely moved: {hist[0]:.4f} -> {hist[-1]:.4f}"
    pred = p.probs(X) >= 0.5
    assert (pred == Y).mean() > 0.9, (pred == Y).mean()


def test_probe_roundtrips_through_disk():
    p = LinearProbe(8, ["X", "Y"], seed=4)
    X = np.random.default_rng(5).normal(size=(10, 8)).astype(np.float32)
    before = p.probs(X)
    with tempfile.TemporaryDirectory() as d:
        p.save(Path(d) / "probe.npz")
        q = LinearProbe.load(Path(d) / "probe.npz")
    assert q.labels == ["X", "Y"]
    assert np.allclose(before, q.probs(X))


# ---- label space -----------------------------------------------------------------------------
def test_label_space_applies_min_count_and_reports_coverage():
    codes = [["A"]] * 30 + [["A", "B"]] * 5 + [["C"]] * 2
    labels, info = build_label_space(codes, min_count=5)
    assert labels == ["A", "B"]
    assert info["n_codes_seen"] == 3
    # 40 occurrences of A/B out of 42 total
    assert abs(info["label_coverage_of_train"] - 40 / 42) < 1e-12


# ---- calibration -----------------------------------------------------------------------------
def test_threshold_for_precision_respects_the_bar():
    # 10 cells, descending confidence, the 4th is wrong
    prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    corr = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    r = threshold_for_precision(prob, corr, total_gold=4, bar=1.0)
    assert r["n_cells"] == 3 and r["precision"] == 1.0
    assert 0.6 < r["threshold"] < 0.7, r["threshold"]     # strictly between kept and dropped cell
    assert threshold_for_precision(prob, corr, 4, bar=0.999999) is not None
    assert threshold_for_precision(np.array([0.9]), np.array([False]), 1, bar=0.5) is None


def test_ece_decision_region_differs_from_all_cells_when_the_tail_is_huge():
    # 10 confident, well-calibrated cells plus 10,000 near-zero true negatives
    prob = np.concatenate([np.full(10, 0.6), np.full(10_000, 0.001)])
    corr = np.concatenate([np.ones(10, bool), np.zeros(10_000, bool)])
    assert ece(prob, corr) < 0.01                              # swamped by the tail
    assert ece(prob, corr, floor=0.02) > 0.3                   # the decision region is miscalibrated


def test_router_threshold_is_fitted_on_cal_and_persists():
    rng = np.random.default_rng(6)
    labels = [f"C{i}" for i in range(5)]
    P = rng.random((200, 5)).astype(np.float32)
    gold = [[labels[j] for j in range(5) if P[i, j] > 0.75] for i in range(200)]
    r = Router(labels, bars=(0.8,)).fit(P, gold, per_code_min_support=5, log=lambda *a: None)
    assert "0.80" in r.global_threshold
    with tempfile.TemporaryDirectory() as d:
        path = r.save(Path(d) / "router.json")
        q = Router.load(path)
    assert q.global_threshold == r.global_threshold
    assert q.labels == labels


def test_router_rejects_an_uncalibrated_bar():
    r = Router(["A"], bars=(0.9,))
    r.global_threshold = {"0.90": 0.5}
    try:
        r.route(np.array([0.9]), 0.75)
    except KeyError as e:
        assert "0.75" in e.args[0]
    else:
        raise AssertionError("routing at an uncalibrated bar must fail loudly")


def test_router_margin_raises_the_threshold():
    rng = np.random.default_rng(7)
    labels = [f"C{i}" for i in range(4)]
    P = rng.random((400, 4)).astype(np.float32)
    gold = [[labels[j] for j in range(4) if P[i, j] > 0.7] for i in range(400)]
    lo = Router(labels, (0.85,)).fit(P, gold, margin=0.0, log=lambda *a: None)
    hi = Router(labels, (0.85,)).fit(P, gold, margin=0.05, log=lambda *a: None)
    assert hi.global_threshold["0.85"] >= lo.global_threshold["0.85"]


# ---- evaluation ------------------------------------------------------------------------------
def test_verdict_is_three_way():
    assert verdict(0.90, 0.93, [0.91, 0.95]) == "MET"
    assert verdict(0.90, 0.895, [0.88, 0.91]) == "MARGINAL"
    assert verdict(0.90, 0.84, [0.82, 0.86]) == "MISSED"
    assert verdict(0.90, float("nan"), None) == "NO AUTO-ACCEPTS"


def test_micro_f1_and_category_rollup():
    m = micro_f1([{"I10", "E11.9"}], [{"I10", "J18.9"}])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    cat = get_rollup("icd-3char")
    m2 = micro_f1([{"E11.9"}], [{"E11.65"}], key=cat)
    assert m2["tp"] == 1, "E11.9 and E11.65 share the 3-character category E11"


def test_rollup_takes_the_max_probability_in_a_group():
    P = np.array([[0.2, 0.7, 0.4]], dtype=np.float32)
    out, labels = rollup(P, ["E11.9", "E11.65", "I10"], get_rollup("icd-3char"))
    assert labels == ["E11", "I10"]
    assert np.allclose(out, [[0.7, 0.4]])


# ---- schema ----------------------------------------------------------------------------------
def _write(tmp, rows):
    p = Path(tmp) / "n.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_schema_accepts_delimited_codes_and_normalises_case():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, [{"id": "1", "text": "note", "codes": "i10; e11.9 ,i10"}])
        recs = load_jsonl(p)
    assert recs[0].codes == ["I10", "E11.9"], recs[0].codes


def test_schema_rejects_the_usual_first_attempts():
    with tempfile.TemporaryDirectory() as d:
        for rows, why in [
            ([{"id": "1", "text": ""}], "empty text"),
            ([{"id": "1"}], "missing text"),
            ([{"id": "1", "text": "a"}, {"id": "1", "text": "b"}], "duplicate id"),
            ([{"id": "1", "text": "a", "split": "holdout"}], "unknown split"),
            ([{"id": "1", "text": "a", "codes": 7}], "codes not a list"),
        ]:
            try:
                load_jsonl(_write(d, rows))
            except SchemaError:
                pass
            else:
                raise AssertionError(f"should have rejected: {why}")


def test_schema_requires_codes_when_asked():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, [{"id": "1", "text": "a"}])
        load_jsonl(p)                                  # fine without codes
        try:
            load_jsonl(p, require_codes=True)
        except SchemaError:
            return
    raise AssertionError("require_codes must reject an uncoded row")


def test_assign_splits_honours_preassigned_rows_and_is_deterministic():
    from icdassist.schema import Record
    recs = [Record(id=str(i), text="t") for i in range(100)]
    recs[0].split = "eval"
    a = [r.split for r in assign_splits(list(recs), seed=1)]
    b = [r.split for r in assign_splits(list(recs), seed=1)]
    assert a == b, "same seed must give the same splits"
    assert a[0] == "eval", "a row that declares its split must keep it"
    assert set(a) <= {"train", "cal", "eval"}


def test_rollup_none_is_identity():
    assert get_rollup("none")("E11.9") == "E11.9"
    try:
        get_rollup("klingon")
    except SchemaError:
        return
    raise AssertionError("unknown rollup must be rejected")


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:                      # noqa: BLE001 -- this IS the test reporter
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
