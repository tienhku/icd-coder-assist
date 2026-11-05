"""fit / evaluate / predict, and the on-disk bundle they pass between each other.

A bundle is a directory, not a blob, so an operator can inspect every artifact before trusting it:

    bundle/
      settings.json        exactly which encoder, hyperparameters and splits produced this
      probe.npz           the dim x K weight matrix and bias -- the only learned parameters
      probe.meta.json
      router.json         the auto-accept thresholds and the calibration evidence behind them
      report.json         what the thresholds achieved on the held-out eval split
      report.txt          the same, readable
      labels.json         the site's own label space, with training support per code

The bundle contains no note text and no patient identifiers: the probe is a matrix over embedding
dimensions, and `labels.json` holds codes and counts. It is still a model trained on clinical data,
so it belongs inside the site's governance boundary like any other derived dataset.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import __version__
from .calibrate import Router
from .config import Settings
from .embed import Embedder
from .evaluate import format_report, full_report
from .probe import LinearProbe, build_label_space, multihot_indices
from .schema import assign_splits, get_rollup, load_jsonl, split_counts


def _embed_or_load(recs, prefix, st, log, restart=False):
    prefix = Path(prefix)
    ids_file = prefix.parent / f"{prefix.name}.ids.json"
    if prefix.with_suffix(".npy").exists() and ids_file.exists():
        meta = json.loads(ids_file.read_text())
        if meta["ids"] == [r.id for r in recs] and meta.get("model") == st.encoder:
            log(f"reusing embeddings at {prefix}.npy ({meta['dim']}d)")
            return np.load(f"{prefix}.npy", mmap_mode="r"), meta.get("failed_ids", [])
        log("existing embeddings do not match these records or this encoder -- re-embedding")
    emb = Embedder(st.encoder, st.embed_url, max_chars=st.max_chars, batch_size=st.batch_size)
    dim = emb.probe()
    log(f"endpoint OK: {st.encoder} returns {dim}d vectors")
    return emb.corpus([r.text for r in recs], [r.id for r in recs], prefix,
                      restart=restart, log=log)


def fit(notes_path, out_dir, st: Settings | None = None, *, log=print, restart_embed=False):
    st = st or Settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    recs = load_jsonl(notes_path)
    labelled = [r for r in recs if r.codes]
    if len(labelled) < 500:
        raise ValueError(f"only {len(labelled)} records carry codes. The probe needs a few thousand "
                         f"labelled notes to be worth anything; below ~500 it cannot even be "
                         f"calibrated honestly.")
    if len(labelled) != len(recs):
        log(f"NOTE: {len(recs) - len(labelled)} record(s) have no codes and are ignored by `fit` "
            f"(use `predict` for those)")
    assign_splits(labelled, cal_frac=st.cal_frac, eval_frac=st.eval_frac, seed=st.seed)
    counts = split_counts(labelled)
    log(f"{len(labelled):,} labelled notes  splits={counts}")
    for need in ("train", "cal", "eval"):
        if counts.get(need, 0) < 100:
            raise ValueError(f"split {need!r} has {counts.get(need, 0)} notes; at least 100 is "
                             f"needed for the threshold or its verification to mean anything")

    X, failed = _embed_or_load(labelled, out / "embeddings" / "notes", st, log, restart_embed)
    if failed:
        log(f"WARNING: {len(failed)} note(s) are zero vectors; they are kept but will predict noise")

    split_arr = np.array([r.split for r in labelled])
    idx = {s: np.where(split_arr == s)[0] for s in ("train", "cal", "eval")}
    codes = [r.codes for r in labelled]

    labels, info = build_label_space([codes[i] for i in idx["train"]], min_count=st.min_count)
    if not labels:
        raise ValueError(f"no code reaches --min-count {st.min_count} in the training split")
    log(f"label space: K={len(labels)} of {info['n_codes_seen']} codes seen "
        f"(covers {info['label_coverage_of_train']:.1%} of training code occurrences)")
    lab2i = {c: i for i, c in enumerate(labels)}

    class _Rows:
        def __init__(self, X, ix): self.X, self.ix = X, ix
        def __len__(self): return len(self.ix)
        def __getitem__(self, k): return self.X[self.ix[k]]

    tr = np.sort(idx["train"])
    probe = LinearProbe(X.shape[1], labels, seed=st.seed)
    log(f"fitting probe {X.shape[1]} -> {len(labels)} on {len(tr):,} notes "
        f"({st.epochs} epochs, batch {st.probe_batch_size}, lr {st.lr})")
    probe.fit(_Rows(X, tr), multihot_indices([codes[i] for i in tr], lab2i),
              epochs=st.epochs, batch_size=st.probe_batch_size, lr=st.lr, seed=st.seed, log=log)
    probe.meta["encoder"] = st.encoder
    probe.meta["n_train_notes"] = int(len(tr))
    probe.meta["label_space"] = info

    log("calibrating auto-accept thresholds on the cal split")
    cal = np.sort(idx["cal"])
    P_cal = probe.probs(np.asarray(X[cal]))
    router = Router(labels, st.bars).fit(P_cal, [codes[i] for i in cal],
                                         per_code_min_support=st.per_code_min_support,
                                         margin=st.bar_margin, log=log)

    log("verifying on the eval split (thresholds are NOT refitted here)")
    ev = np.sort(idx["eval"])
    P_ev = probe.probs(np.asarray(X[ev]))
    rep = full_report(P_ev, [codes[i] for i in ev], router, n_boot=st.n_boot, seed=st.seed,
                      rollup_fn=get_rollup(st.rollup) if st.rollup != "none" else None)
    rep["splits"] = counts
    rep["toolkit_version"] = __version__
    rep["encoder"] = st.encoder
    rep["fit_seconds"] = round(time.time() - t0, 1)

    probe.save(out / "probe.npz")
    router.save(out / "router.json")
    st.save(out / "settings.json")
    support = Counter(c for i in tr for c in codes[i])
    (out / "labels.json").write_text(json.dumps(
        {"labels": labels, "train_support": {c: support[c] for c in labels}}, indent=2))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    text = format_report(rep)
    (out / "report.txt").write_text(text + "\n")
    log("\n" + text)
    log(f"\nbundle written to {out}  ({rep['fit_seconds']}s total)")
    return out, rep


def load_bundle(bundle):
    b = Path(bundle)
    st = Settings.load(b / "settings.json")
    return LinearProbe.load(b / "probe.npz"), Router.load(b / "router.json"), st


def evaluate(bundle, notes_path, *, policy="global", log=print):
    probe, router, st = load_bundle(bundle)
    recs = load_jsonl(notes_path, require_codes=True)
    X, _ = _embed_or_load(recs, Path(bundle) / "embeddings" / f"eval_{Path(notes_path).stem}",
                          st, log)
    P = probe.probs(np.asarray(X))
    rep = full_report(P, [r.codes for r in recs], router, policy=policy, n_boot=st.n_boot,
                      seed=st.seed, rollup_fn=get_rollup(st.rollup) if st.rollup != "none" else None)
    rep["source"] = str(notes_path)
    log("\n" + format_report(rep))
    return rep


def predict(bundle, notes_path, out_path, *, bar=0.90, policy="global", log=print):
    probe, router, st = load_bundle(bundle)
    recs = load_jsonl(notes_path)
    X, failed = _embed_or_load(recs, Path(bundle) / "embeddings" / f"pred_{Path(notes_path).stem}",
                               st, log)
    P = probe.probs(np.asarray(X))
    failed = set(failed)
    n_auto = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            auto, defer = router.route(P[i], bar, policy=policy)
            n_auto += len(auto)
            f.write(json.dumps({"id": r.id, "bar": bar, "policy": policy,
                                "auto_accept": auto, "defer": defer,
                                "unembeddable": r.id in failed}, ensure_ascii=False) + "\n")
    log(f"{len(recs):,} notes -> {out_path}  ({n_auto:,} auto-accepted codes, "
        f"{n_auto / max(len(recs), 1):.1f} per note at bar {bar})")
    return out_path
