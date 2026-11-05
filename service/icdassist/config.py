"""Settings, from CLI flags or environment, with the defaults our runs actually used."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# The encoder we measured best (docs/qwen3_embedding_ablation.md: frozen probe 0.456 exact micro-F1,
# +0.123 over bge-m3). A site with less VRAM can drop to qwen3-embedding:4b (0.418) or :0.6b (0.386);
# we found no plateau below 8B, so bigger is still better if it fits.
DEFAULT_ENCODER = "qwen3-embedding:8b"
DEFAULT_EMBED_URL = "http://localhost:11434/api/embed"


@dataclass
class Settings:
    encoder: str = DEFAULT_ENCODER
    embed_url: str = DEFAULT_EMBED_URL
    max_chars: int = 28000        # ~8k tokens; v2 notes have a tail out to 284k chars
    batch_size: int = 32
    min_count: int = 25           # min training support for a code to enter the label space
    epochs: int = 20
    probe_batch_size: int = 1024
    lr: float = 1e-3
    seed: int = 42
    cal_frac: float = 0.10
    eval_frac: float = 0.10
    bars: tuple = (0.95, 0.90, 0.85, 0.80)
    per_code_min_support: int = 20
    bar_margin: float = 0.0       # fit thresholds to bar+margin; see Router.fit
    rollup: str = "icd-3char"
    n_boot: int = 1000

    @classmethod
    def from_env(cls):
        d = {}
        for f, t in [("encoder", str), ("embed_url", str), ("max_chars", int), ("batch_size", int),
                     ("min_count", int), ("epochs", int), ("lr", float), ("seed", int),
                     ("rollup", str), ("n_boot", int)]:
            v = os.environ.get(f"ICDASSIST_{f.upper()}")
            if v is not None:
                d[f] = t(v)
        return cls(**d)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d["bars"] = tuple(d.get("bars", cls.bars))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
