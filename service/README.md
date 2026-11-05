# icdassist

Self-hosted, confidence-tiered ICD coding assistance. It suggests codes for a discharge note, splits
them into **auto-accept** and **defer to a coder**, and the split is calibrated so that the
auto-accepted codes clear a precision bar *you* choose and *you* verify on your own held-out data.

It is the deployable form of the method in our manuscript. Two things about it are unusual and both
are deliberate.

**Your notes never leave your network.** There is no cloud API, no account, and nothing to sign up
for. You run two containers: an embedding server and this service. The only place a note travels is
from this process to the embedding container on the same host.

**We ship no model trained on patients.** What we publish is a recipe and the code that runs it. You
fit the model on your own coded notes, against your own code set. That is not a privacy fig leaf —
it is the only thing that works. A probe fitted on our hospital's 3,150 codes, our coders' habits
and our note templates would transfer badly to yours, and a confidence threshold fitted on our case
mix would be actively unsafe on yours.

---

## What it actually does

```
note text ──▶ frozen embedding model ──▶ linear probe ──▶ per-code probability
                 (you run it, ~6 GB)      (dim × K, you fit it, ~2 min)
                                                              │
                                                calibrated threshold
                                                     ┌────────┴────────┐
                                              auto-accept           defer
                                      (clears your precision bar)  (a coder decides)
```

The encoder is frozen: it is never fine-tuned, so you never need a training GPU. The only thing
trained is a single linear layer, which on 250,000 notes takes about two minutes of CPU.

Our result, on 5,000 held-out admissions at our site, was that this frozen-encoder probe matched an
end-to-end fine-tuned PLM-ICD head (0.456 vs 0.460 exact micro-F1) with substantially better raw
calibration. Your numbers will differ. The point of the `fit` command is that it tells you *your*
numbers before you deploy anything.

## Requirements

| | |
|---|---|
| **Labelled notes** | a few thousand minimum; tens of thousands is where it gets good. Each needs its final code set. |
| **An embedding model** | `qwen3-embedding:8b` is what we measured best. Needs ~16 GB VRAM via Ollama, or a lot of patience on CPU. |
| **This** | Python 3.10+, numpy, requests. Or the container. |
| **No GPU for training** | the probe is a linear layer. |

Smaller encoders work with a real accuracy cost, measured at our site:

| encoder | params | exact micro-F1 | note |
|---|---|---|---|
| `qwen3-embedding:8b` | 8B | **0.456** | best; no plateau found below this |
| `qwen3-embedding:4b` | 4B | 0.418 | |
| `qwen3-embedding:0.6b` | 0.6B | 0.386 | runs on CPU |
| `bge-m3` | 568M | 0.334 | |

Scale genuinely matters here, and it had not flattened out by 8B. If you have the VRAM, use it.

## Quick start (containers)

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3-embedding:8b

# three fields per line: id, text, codes
cp /path/to/your/notes.jsonl ./data/notes.jsonl
docker compose run --rm icdassist check --notes /data/notes.jsonl

docker compose run --rm icdassist fit --notes /data/notes.jsonl --out /bundle
docker compose up -d icdassist          # http://127.0.0.1:8080
```

## Quick start (no containers)

```bash
pip install -r requirements.txt
export ICDASSIST_EMBED_URL=http://localhost:11434/api/embed

python -m icdassist check    --notes notes.jsonl
python -m icdassist fit      --notes notes.jsonl --out ./bundle
python -m icdassist serve    --bundle ./bundle
```

## Your input

One JSON object per line:

```json
{"id": "local-00001", "text": "CHIEF COMPLAINT: ...", "codes": ["I10", "E11.9"]}
```

- `id` — any string unique in the file. It never has to be a patient or admission identifier; it
  exists so you can join predictions back to your own records.
- `text` — the note. **Remove the discharge diagnosis statement.** Leaving it in leaks the answer
  and inflated our F1 roughly eightfold; a model fitted on the leaky field looks excellent and is
  worthless in production, because at coding time that field is what you are trying to produce.
- `codes` — your final code set, as strings. ICD-10-CM, ICD-10 WHO, ICD-9-CM and local extensions
  all work; the codes are opaque to everything except the optional 3-character rollup.
- `split` — optional, one of `train` / `cal` / `eval`. Supply it if you want to split by time rather
  than at random, which is the more honest test. Left out, rows are split randomly.

A full worked example, with fictional notes, is in [`demo/`](demo/).

## Reading the report

`fit` ends with a table. This is the entire product; everything else is plumbing.

```
  bar   thresh  coverage            95% CI  precision            95% CI   oracle    verdict
 0.95    0.395     0.466   [0.449, 0.483]      0.971   [0.960, 0.981]    0.513        MET
 0.90    0.330     0.549   [0.533, 0.568]      0.922   [0.905, 0.938]    0.566        MET
```

- **bar** — the precision you asked for.
- **coverage** — the share of your true coding workload that gets auto-accepted. This is the number
  that becomes staff time.
- **precision** — what the auto-accepted codes actually achieved, **on notes not used to pick the
  threshold**. CIs are paired-bootstrapped over admissions, not over cells: codes within one
  admission are correlated, and resampling cells would give you intervals several times too narrow.
- **oracle** — the coverage you would have reported if you had tuned the threshold on these same
  rows. A ceiling, printed so you can see how much is threshold-transfer loss.
- **verdict** — `MET`, `MARGINAL` (point estimate under the bar but the interval still covers it —
  this much eval data cannot tell them apart), or `MISSED`.

If you see `MARGINAL`, refit with `--bar-margin 0.02`. That fits the threshold to a slightly
stricter target and buys precision back with a little coverage. If you see `MISSED`, do not operate
that tier.

## What this is not

- **Not autonomous coding.** A deferred code is not a wrong code, it is an unanswered question, and
  the auto-accepted tier is wrong a few percent of the time by construction — that is what a
  precision bar of 0.95 means. A qualified coder stays accountable for the final code set.
- **Not a reimbursement tool.** Nothing here models payer rules, sequencing, principal-diagnosis
  selection or DRG grouping.
- **Not usable on the long tail.** Codes below `--min-count` (default 25 training occurrences) are
  excluded from the label space and can never be predicted. `fit` reports what fraction of your
  gold codes are reachable at all. Ours was 96%; the missing 4% is real and stays with the coder.
- **Not a medical device clearance.** Whether decision support of this kind needs regulatory
  approval is jurisdiction-specific and is yours to determine.

## Files

| | |
|---|---|
| `icdassist/schema.py` | the input contract, and why there are three splits |
| `icdassist/embed.py` | calls your embedding server; checkpoints, resumes, survives bad notes |
| `icdassist/probe.py` | the linear probe and a hand-written Adam, in numpy |
| `icdassist/calibrate.py` | thresholds, per-code thresholds, the safety margin |
| `icdassist/evaluate.py` | held-out measurement, paired bootstrap, the verdict logic |
| `icdassist/api.py` | the local HTTP service |
| `web/index.html` | the coder-facing page; no CDN, works offline |
| `demo/make_synthetic.py` | fictional notes, so you can try all of the above with no data |
| `tests/` | offline tests, including a finite-difference check on the probe's gradient |

See [DEPLOY.md](DEPLOY.md) for sizing, governance, drift monitoring and the things that go wrong.

## Licence and citation

See the repository root. If this is useful in published work, please cite the manuscript listed
there.
