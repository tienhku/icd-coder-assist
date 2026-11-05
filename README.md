# icd-coder-assist

Code for **confidence-tiered ICD-10 coding assistance**: embed discharge notes once with a frozen
encoder, fit a linear probe on your own coded notes, calibrate an auto-accept threshold to a
precision bar, and route each suggested code to either *auto-accept* or *a human coder*.

This repository is the method and the tooling. It contains **no patient data and no model trained on
patients**, and both omissions are deliberate — see *What is not here*.

Manuscript in preparation; a citation will be added here when it has an identifier.

---

## Where to start

| | |
|---|---|
| **[`service/`](service/)** | `icdassist` — the deployable toolkit. If you want to run the method at your own site, read [`service/README.md`](service/README.md) and nothing else. numpy + requests, no GPU needed for training, containers provided. |
| [`service/DEPLOY.md`](service/DEPLOY.md) | sizing, what each artifact contains, drift monitoring, and the failure modes we actually hit. |
| [`eval/`](eval/) | the research harness behind the paper: the PLM-ICD head, the frozen probes, the selective-coding router, the retrieval and agentic comparators, the scoring scripts. |
| [`scripts/`](scripts/) | embedding, ICD-hierarchy and prompt utilities shared by the above. |

Try the whole pipeline with no data at all:

```bash
cd service
python demo/make_synthetic.py --mode corpus --n 4000 --out data/synthetic.jsonl
python -m icdassist fit --notes data/synthetic.jsonl --out ./bundle_demo \
    --encoder qwen3-embedding:0.6b --embed-url http://localhost:11434/api/embed
python -m icdassist serve --bundle ./bundle_demo
```

Every note that command uses is generated from templates in `demo/make_synthetic.py`. The resulting
numbers mean nothing clinically — the synthetic task is far easier than real coding — but they prove
the pipeline runs end to end on your hardware.

## What is not here, and why

- **The corpus, and anything derived from it.** Full inpatient discharge records from one hospital.
  They cannot be shared, which is also why this is a study rather than a public benchmark.
- **A trained probe.** A probe fitted on our label space, our coders' habits and our note templates
  would transfer badly to another site, and a threshold calibrated on our case mix would be
  *unsafe* on another's. The toolkit's `fit` command exists so every site produces its own, which
  takes about two minutes of CPU once the notes are embedded.
- **Corpus construction.** Our discharge documentation is scattered across ~12 rows per admission in
  one hospital's EMR schema, and the script that pivots it into one note per admission is specific
  to that schema — it would not run anywhere else, and publishing it mostly publishes our database
  layout. What it does is described in the manuscript's Methods, and the input contract the toolkit
  actually needs is three fields, documented in
  [`service/icdassist/schema.py`](service/icdassist/schema.py). Available to editors and reviewers
  on request.
- **Result dumps, logs and figures.** They belong with the paper, not ahead of it.
- Some docstrings cite `docs/*.md`. Those are the manuscript's supporting documents in our private
  research repository, not files in this one.

## Reproducibility, honestly

You cannot reproduce the paper's numbers from this repository, because you cannot have the corpus.
What you can do is read every step that produced them, run the identical pipeline on your own notes,
and check our claims against your own result — which is the claim we actually care about, since the
paper's argument is that the method transfers, not that our number is large.

Two things to know before trusting a number this code prints:

- `eval/selective_curve.py` picks the threshold at the deepest point where **test** precision still
  clears the bar, which is the right way to draw an operating curve in a paper and the wrong way to
  ship one. `eval/threshold_transfer.py` measures what it costs to choose the threshold in advance
  instead, and the toolkit's `fit` always does it the prospective way: fit on a `cal` split, measure
  on a disjoint `eval` split, and print a verdict rather than a number.
- `eval/error_anatomy.py` splits the auto-accepted errors by whether they are a wrong diagnosis or
  the coder's own diagnosis at the wrong digit. The two numbers bracket what a coder experiences,
  and exact micro-precision alone is the pessimistic end of that bracket.

## Privacy

Notes go from the process you run to the embedding endpoint you configure, and nowhere else. There
is no cloud API, no telemetry and no account. The local service binds `127.0.0.1` by default, has no
authentication of its own (put your own reverse proxy and SSO in front of it), and never writes note
text to its access log. `service/DEPLOY.md` says which artifacts are derived clinical data and which
are aggregate statistics.

This is decision support. The auto-accepted tier is wrong a few percent of the time *by
construction* — that is what a precision bar means — and a qualified coder remains accountable for
the final code set. We make no regulatory claim.

## Licence

Apache-2.0, including the documentation. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contact

TODO before publication: a dedicated address, not a personal one, and the words "please do not send
us patient notes" next to it.
