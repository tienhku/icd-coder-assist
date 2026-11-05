# Try the whole pipeline with no data

Every note this directory generates is **fictional**, from templates in `make_synthetic.py`. No
sentence, value or identifier comes from a real record. Read that file: the templates encode only
textbook presentations, so the corpus is safe to put in a container image, a CI run or a public demo.

```bash
# 4,000 fictional notes with gold codes
python make_synthetic.py --mode corpus --n 4000 --out ../data/synthetic.jsonl

cd ..
python -m icdassist check --notes data/synthetic.jsonl \
    --encoder qwen3-embedding:0.6b --embed-url http://localhost:11434/api/embed
python -m icdassist fit   --notes data/synthetic.jsonl --out ./bundle_demo \
    --encoder qwen3-embedding:0.6b --embed-url http://localhost:11434/api/embed
python -m icdassist serve --bundle ./bundle_demo
```

`--mode showcase` writes one longer note per presentation, which is what the public website's demo
uses.

**What the demo numbers mean: nothing, clinically.** Synthetic notes are generated from 14
presentations with 36 codes, so the task is far easier than real coding and the probe scores far
higher than it would on real notes. The demo proves the *plumbing* works end to end — embed, fit,
calibrate, verify, route — not that the method performs. For performance, see the manuscript.
