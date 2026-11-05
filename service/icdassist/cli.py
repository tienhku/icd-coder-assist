"""icdassist command line: check / embed / fit / evaluate / predict / serve."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .embed import EmbedError, Embedder
from .schema import SchemaError, load_jsonl, split_counts


def _settings(a, base=None):
    st = base or Settings.from_env()
    for f in ("encoder", "embed_url", "max_chars", "batch_size", "min_count", "epochs", "lr",
              "seed", "rollup", "n_boot", "cal_frac", "eval_frac", "per_code_min_support",
              "bar_margin"):
        v = getattr(a, f, None)
        if v is not None:
            setattr(st, f, v)
    if getattr(a, "bars", None):
        st.bars = tuple(a.bars)
    return st


def _add_common(p):
    p.add_argument("--encoder", help=f"embedding model name (default {Settings.encoder})")
    p.add_argument("--embed-url", help=f"embedding endpoint (default {Settings.embed_url})")
    p.add_argument("--max-chars", type=int)
    p.add_argument("--batch-size", type=int)


def cmd_check(a):
    st = _settings(a)
    print(f"icdassist {__version__}")
    ok = True
    try:
        dim = Embedder(st.encoder, st.embed_url, max_chars=st.max_chars).probe()
        print(f"  embedding endpoint  OK   {st.encoder} @ {st.embed_url} -> {dim}d")
    except EmbedError as e:
        print(f"  embedding endpoint  FAIL {e}")
        ok = False
    if a.notes:
        try:
            recs = load_jsonl(a.notes)
            n_lab = sum(1 for r in recs if r.codes)
            codes = {c for r in recs for c in r.codes}
            chars = sorted(len(r.text) for r in recs)
            print(f"  notes file          OK   {len(recs):,} records, {n_lab:,} with codes, "
                  f"{len(codes):,} distinct codes")
            print(f"                           note length chars: median {chars[len(chars) // 2]:,}  "
                  f"p95 {chars[int(len(chars) * 0.95)]:,}  max {chars[-1]:,}")
            if chars[int(len(chars) * 0.95)] > st.max_chars:
                print(f"                           NOTE >5% of notes exceed --max-chars "
                      f"{st.max_chars:,} and will be truncated")
            if r_splits := {k: v for k, v in split_counts(recs).items() if k != "unassigned"}:
                print(f"                           pre-assigned splits: {r_splits}")
        except SchemaError as e:
            print(f"  notes file          FAIL {e}")
            ok = False
    return 0 if ok else 1


def cmd_embed(a):
    st = _settings(a)
    recs = load_jsonl(a.notes)
    emb = Embedder(st.encoder, st.embed_url, max_chars=st.max_chars, batch_size=st.batch_size)
    print(f"endpoint returns {emb.probe()}d vectors")
    emb.corpus([r.text for r in recs], [r.id for r in recs], a.out, restart=a.restart)
    return 0


def cmd_fit(a):
    from .pipeline import fit
    st = _settings(a)
    fit(a.notes, a.out, st, restart_embed=a.restart_embed)
    return 0


def cmd_evaluate(a):
    from .pipeline import evaluate
    rep = evaluate(a.bundle, a.notes, policy=a.policy)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {a.json_out}")
    return 0


def cmd_predict(a):
    from .pipeline import predict
    predict(a.bundle, a.notes, a.out, bar=a.bar, policy=a.policy)
    return 0


def cmd_serve(a):
    from .api import serve
    serve(a.bundle, host=a.host, port=a.port, default_bar=a.bar, default_policy=a.policy)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="icdassist",
        description="Self-hosted confidence-tiered ICD coding assistance. Notes never leave the "
                    "network you point --embed-url at.")
    ap.add_argument("--version", action="version", version=f"icdassist {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="verify the embedding endpoint and a notes file before a long run")
    p.add_argument("--notes", help="notes.jsonl to validate")
    _add_common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("embed", help="embed a notes file and stop (resumable)")
    p.add_argument("--notes", required=True)
    p.add_argument("--out", required=True, help="output prefix: writes <out>.npy and <out>.ids.json")
    p.add_argument("--restart", action="store_true", help="ignore any checkpoint")
    _add_common(p)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("fit", help="embed, fit the probe, calibrate thresholds, verify on held-out eval")
    p.add_argument("--notes", required=True)
    p.add_argument("--out", required=True, help="bundle directory to create")
    p.add_argument("--min-count", type=int, help=f"min train support per code (default {Settings.min_count})")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--cal-frac", type=float)
    p.add_argument("--eval-frac", type=float)
    p.add_argument("--per-code-min-support", type=int)
    p.add_argument("--bar-margin", type=float,
                   help="fit thresholds to bar+margin to absorb threshold-transfer optimism "
                        "(try 0.02 if the eval verdicts come back MARGINAL)")
    p.add_argument("--bars", type=float, nargs="+", help="precision bars to calibrate")
    p.add_argument("--rollup", choices=["icd-3char", "none"])
    p.add_argument("--n-boot", type=int)
    p.add_argument("--restart-embed", action="store_true")
    _add_common(p)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("evaluate", help="measure a bundle on a fresh labelled file (drift check)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("--policy", choices=["global", "per-code"], default="global")
    p.add_argument("--json-out")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict", help="route a file of unlabelled notes to auto-accept / defer")
    p.add_argument("--bundle", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bar", type=float, default=0.90)
    p.add_argument("--policy", choices=["global", "per-code"], default="global")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("serve", help="run the local coder-facing API and web UI")
    p.add_argument("--bundle", required=True)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 by default: bind wider only behind "
                                                       "the site's own authentication")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bar", type=float, default=0.90)
    p.add_argument("--policy", choices=["global", "per-code"], default="global")
    p.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (SchemaError, EmbedError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
