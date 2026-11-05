#!/usr/bin/env python3
"""Zero-shot ICD-10-CM coding baseline via a local Ollama model.

Samples admissions from data/processed/test.parquet (VER='10' only), prompts
the model with note_text, parses predicted ICD-10-CM codes out of its
response, and scores against the pticd TYPE='I' ground truth with micro
precision/recall/F1. No RAG, no few-shot examples — this is a floor, not a
target.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prompting import PROMPT_TEMPLATE, parse_codes  # noqa: E402

TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(model: str, prompt: str, url: str = OLLAMA_URL, timeout: int = 120) -> str:
    # num_predict caps runaway generation (observed failure mode: small quantized
    # models sometimes get stuck enumerating every sub-code of a category, e.g.
    # O30.0, O30.1, ... O30.999, until hitting the token limit). repeat_penalty
    # discourages the same degenerate loop at the sampling level.
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 400},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def make_mlx_responder(model_id: str, adapter_path: str | None):
    """Load an MLX model (optionally with a LoRA adapter) once and return a
    prompt -> response function, mirroring call_ollama's interface. The chat
    template rendering (add_generation_prompt=True, assistant turn excluded)
    matches exactly what ChatDataset uses to build the prompt-side offset
    during mlx_lm.lora training, so eval-time formatting matches train-time.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors

    print(f"Loading MLX model '{model_id}'" + (f" with adapter '{adapter_path}'" if adapter_path else "") + "...")
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    logits_processors = make_logits_processors(repetition_penalty=1.3)

    def respond(prompt: str) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return generate(
            model,
            tokenizer,
            prompt=rendered,
            max_tokens=400,
            logits_processors=logits_processors,
        )

    return respond


def score(pred: set[str], gold: set[str]) -> tuple[int, int, int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma", help="Ollama model name (--backend ollama).")
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help="Ollama /api/generate endpoint. Point at a LAN node to use it as an inference "
        "backend, e.g. http://<your-lan-gpu>:11434/api/generate. Patient text "
        "may go to the two self-owned LAN machines only — never a cloud endpoint (see README).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Override the output filename tag. Use when the same model name means different "
        "artifacts (e.g. Ollama's Q4_K_M medgemma vs MLX's unquantized one).",
    )
    parser.add_argument(
        "--mlx-model", default="google/medgemma-4b-it", help="HF repo or local path (--backend mlx)."
    )
    parser.add_argument("--backend", default="ollama", choices=["ollama", "mlx"])
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter dir (--backend mlx only).")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-column",
        default="note_text",
        choices=["note_text", "note_text_no_diagnosis"],
        help="Which corpus column to prompt with (full vs diagnosis-stripped).",
    )
    parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    parser.add_argument(
        "--canary-n",
        type=int,
        default=20,
        help="Abort if the first N predictions are all empty or all identical (0 disables).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "ollama":
        respond = lambda prompt: call_ollama(args.model, prompt, url=args.ollama_url)  # noqa: E731
        tag = f"{args.model.replace(':', '_')}_{args.text_column}"
        print(f"Ollama backend: {args.ollama_url}")
    else:
        respond = make_mlx_responder(args.mlx_model, args.adapter_path)
        adapter_tag = Path(args.adapter_path).name if args.adapter_path else "zeroshot"
        tag = f"mlx_{adapter_tag}_{args.text_column}"
    if args.tag:
        tag = args.tag

    df = pd.read_parquet(TEST_PARQUET)
    df = df[df["ver"] == "10"].sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(df)} VER=10 admissions from {TEST_PARQUET}, using column '{args.text_column}'")

    records = []
    n_errors = 0
    total_tp = total_fp = total_fn = 0
    t0 = time.time()
    for i, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(note_text=row[args.text_column])
        try:
            raw = respond(prompt)
        except requests.RequestException as e:
            n_errors += 1
            print(f"  [{i+1}/{len(df)}] acc_no={row['acc_no']} ERROR: {e}")
            # Bail out on a wall of failures rather than skipping every sample and writing a
            # summary that reads micro_f1 0.0 — indistinguishable from a genuinely dead model
            # (scaleup_v1 scored a real 0.0). A run that scored nothing because it never reached
            # the backend must not be able to masquerade as a result. Seen for real: macOS Local
            # Network privacy silently blocks the venv python (Errno 65 "No route to host") while
            # curl and the system python work fine, so connectivity checks outside the venv pass.
            if n_errors >= 10 and not records:
                sys.exit(
                    f"ABORT: first {n_errors} requests all failed, zero successes — the backend "
                    f"is unreachable, not wrong. Nothing written. Last error: {e}"
                )
            continue
        pred = parse_codes(raw)
        gold = {c.upper() for c in row["icd_codes"]}
        tp, fp, fn = score(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        records.append(
            {
                "acc_no": int(row["acc_no"]),
                "gold": sorted(gold),
                "pred": sorted(pred),
                "raw_response": raw,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(df)}] elapsed={elapsed:.0f}s tp={total_tp} fp={total_fp} fn={total_fn}")

        # Canary. A diverged adapter emits no parseable code at all; a mode-collapsed one
        # emits the same answer for every note. Both are invisible in weight statistics and
        # only show up at inference, so check early rather than after the full run: the
        # scaleup_v1 adapter burned 39 minutes to produce 150 identical junk responses.
        # Both failures need varied inputs to detect — a single prompt can't tell a collapsed
        # model from a working one, since the fixed answer is still well-formed.
        if args.canary_n and len(records) == args.canary_n:
            uniq = {tuple(r["pred"]) for r in records}
            if uniq == {()}:
                sys.exit(
                    f"ABORT: first {args.canary_n} predictions are all empty — the model is "
                    f"emitting no parseable codes. Adapter likely diverged."
                )
            if len(uniq) == 1:
                sys.exit(
                    f"ABORT: first {args.canary_n} predictions are all identical "
                    f"({list(uniq)[0]}) — the model has collapsed to a fixed answer, "
                    f"ignoring its input."
                )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Flag likely decoding-degeneration outliers (runaway sub-code enumeration)
    # and report a second score excluding them, since a handful of these can
    # dominate the false-positive count and distort the aggregate metric.
    DEGENERATE_PRED_THRESHOLD = 15
    clean = [r for r in records if len(r["pred"]) <= DEGENERATE_PRED_THRESHOLD]
    n_outliers = len(records) - len(clean)
    clean_tp = sum(r["tp"] for r in clean)
    clean_fp = sum(r["fp"] for r in clean)
    clean_fn = sum(r["fn"] for r in clean)
    clean_p = clean_tp / (clean_tp + clean_fp) if (clean_tp + clean_fp) else 0.0
    clean_r = clean_tp / (clean_tp + clean_fn) if (clean_tp + clean_fn) else 0.0
    clean_f1 = 2 * clean_p * clean_r / (clean_p + clean_r) if (clean_p + clean_r) else 0.0

    if not records:
        sys.exit("ABORT: no successful predictions — nothing to score, nothing written.")
    if n_errors:
        print(
            f"\nWARNING: {n_errors}/{len(df)} requests failed and were dropped. Metrics below "
            f"cover only the {len(records)} that succeeded — the effective sample is smaller "
            f"than --n."
        )

    summary = {
        "backend": args.backend,
        "model": args.model if args.backend == "ollama" else args.mlx_model,
        "adapter_path": args.adapter_path,
        "text_column": args.text_column,
        "n_samples": len(records),
        "n_requested": len(df),
        "n_errors": n_errors,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "n_degenerate_outliers": n_outliers,
        "clean_micro_precision": clean_p,
        "clean_micro_recall": clean_r,
        "clean_micro_f1": clean_f1,
        "elapsed_sec": time.time() - t0,
    }
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))

    preds_path = out_dir / f"{tag}_predictions.jsonl"
    with open(preds_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary_path = out_dir / f"{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote predictions -> {preds_path}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
