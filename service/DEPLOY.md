# Deploying icdassist

Written for whoever has to answer for this running in a hospital: sizing, what the data does, how it
degrades, and the failure modes we actually hit.

## 1. Sizing

**Embedding is the only expensive step, and it is a one-off per note.**

At our site, `qwen3-embedding:8b` on a single RTX 5090 embedded roughly 250,000 discharge notes in
about a day. Scale from that: it is linear in notes and roughly linear in note length. The probe fit
that follows takes about two minutes on a laptop CPU, and re-fitting is cheap — the expensive
artifact is the embedding matrix, and you keep it.

Online, a single note costs one embedding call: on our hardware about 90–120 ms end to end including
the probe. That is per note, single-threaded, with no batching. A coding department reviewing a few
hundred discharges a day will not notice the service exists.

Disk: the embedding matrix is `n_notes × dim × 4` bytes. 250,000 notes at 4,096 dimensions is about
4 GB. The shipped bundle is far smaller — a `dim × K` matrix, about 50 MB at 4,096 × 3,150.

**If you have no GPU**, use `qwen3-embedding:0.6b`. It runs on CPU and cost us 0.070 exact micro-F1
against the 8B model (0.386 vs 0.456). That is a large, real loss, not a rounding error. The gain
from encoder scale had not plateaued at 8B in our measurements, so if you can borrow a GPU for the
one-off embedding pass, borrow it — you can run the service afterwards without one only if you also
embed new notes elsewhere, so in practice the encoder stays.

## 2. Split by time, not at random

`fit` splits randomly if you do not say otherwise. Random is the friendlier number and the less
honest one. Coding practice drifts: guidelines change, a new coder joins, a service reorganises.
Supply the `split` field yourself with the oldest notes as `train`, a middle period as `cal` and the
most recent as `eval`, and the report then answers the question you actually care about — *will the
threshold I fit today hold next month* — rather than *does it hold on notes interleaved with its own
training data*.

Expect the temporal number to be worse. That gap is information, not a defect.

## 3. What the artifacts contain

| artifact | contains | treat as |
|---|---|---|
| `embeddings/*.npy` | a float vector per note | derived clinical data. Embeddings are not plain text, but they are a dense representation of one, and no one should claim they are anonymous. Keep them where the notes are kept. |
| `probe.npz` | a `dim × K` matrix and a bias | a model trained on clinical data. No note text, no identifiers. Still inside your governance boundary. |
| `router.json` | thresholds and aggregate calibration counts | aggregate statistics. |
| `labels.json` | your code strings and their training frequencies | aggregate statistics; it does reveal your case mix. |
| `report.json` | aggregate metrics only | safe to share, and the right thing to show a reviewer. |

`embeddings/*.ids.json` holds whatever you put in `id`. If you put a medical record number there,
that file now holds medical record numbers. Use a local surrogate key.

## 4. Running the service

Default bind is `127.0.0.1`. The service has **no authentication of its own**. If you need it
reachable from coders' workstations:

1. put it behind your existing reverse proxy and SSO, and
2. set `ICDASSIST_TOKEN` as a second layer, and
3. do not publish the ollama container's port — only this service needs to reach it.

The access log deliberately records character counts and timings and never note text. If you add
request logging at a proxy, make sure you are not logging request bodies; that would put a corpus of
clinical notes into your log pipeline.

## 5. Monitoring drift

Re-run the check monthly on recently coded notes:

```bash
python -m icdassist evaluate --bundle ./bundle --notes last_month.jsonl
```

This re-measures the *existing* thresholds on new data without refitting anything. Watch for:

- **a bar slipping from `MET` to `MARGINAL` or `MISSED`** — the threshold no longer holds. Refit.
- **coverage falling while precision holds** — the model is getting more conservative, usually
  because case mix or note templates moved.
- **`gold_in_label_space` falling** — new codes are in use that the label space does not contain.
  Refit so they enter it, if they now clear `--min-count`.

A template change in the electronic record is the single most likely cause of a sudden drop. It
changes the text distribution without changing anything clinical.

## 6. Failure modes we hit, and what they look like

**Notes that will not embed.** One note once killed a ten-hour embedding run at 44% completion. Now
a note that cannot be embedded at any truncation length becomes a zero vector and is listed in
`failed_ids`. A zero vector produces meaningless probabilities, so check that list; it is usually a
handful of notes with a pathological attachment or encoding.

**Truncation.** Notes longer than `--max-chars` (default 28,000, about 8,000 tokens) are cut. We
have notes out to 284,000 characters. `check` tells you what fraction of yours exceed the limit
before you spend a day embedding them. Truncation drops the end of the note, which is where the
hospital course usually lives — if a large share of your notes are over the limit, that matters.

**Diagnosis leakage.** If your `text` still contains the discharge diagnosis, every number will look
superb and the deployed system will be far worse than the report promised. This is the most
expensive mistake available here and the easiest to make.

**A collapsed model.** If a model emits nearly the same code set for every note, F1 can look
plausible while the model has learned only the marginal label frequencies. We watched a 340M-parameter
fine-tuned encoder do exactly this — one distinct prediction set across 5,000 notes. The linear probe
here is far less prone to it, but the diagnostic is cheap and worth knowing: count distinct predicted
code sets across your eval split. If it is in the single digits, nothing is being learned from the
text.

**Thresholds that do not transfer.** The `MARGINAL` verdict exists because of this. A threshold cut
at the deepest point where *calibration* precision clears the bar is slightly fitted to that split's
noise, so held-out precision tends to land just under. `--bar-margin 0.02` is the fix.

## 7. Before you let it touch live coding

- [ ] `fit` reports `MET` for the tier you intend to operate, on a **temporal** eval split.
- [ ] You have read `gold_in_label_space` and accepted what the long tail costs you.
- [ ] A coder has reviewed a sample of auto-accepted codes and agrees with the precision estimate.
- [ ] You know the verdict holds on the *current* quarter's notes, not last year's.
- [ ] Coders know that auto-accepted codes are wrong a few percent of the time, and that they, not
      the tool, sign off the final code set.
- [ ] Someone owns the monthly drift check.
- [ ] Your local rules on clinical decision support have been checked. We make no regulatory claim.
