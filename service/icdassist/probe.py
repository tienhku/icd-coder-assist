"""The multi-label linear probe: frozen embedding dim -> K codes, trained with BCE.

This is the entire trainable part of the method. Our paper's finding is that this layer over a large
frozen encoder matches an end-to-end fine-tuned PLM-ICD head (0.456 vs 0.460 exact micro-F1) at a
fraction of the cost, so the deployable artifact is deliberately tiny: a dim x K matrix plus a bias.

NumPy, with a hand-written Adam, and no torch. Three reasons, in order of how much they mattered:
  1. a site must be able to refit on its own data, and `pip install numpy` always works where a
     matched CUDA/torch build does not;
  2. one backend means one set of numbers -- no CPU/MPS/CUDA reduction-order drift to explain to a
     reviewer or to a hospital's validation team;
  3. it is not the bottleneck. 258k x 4096 -> 3150 takes ~2 min for 20 epochs on a laptop CPU;
     embedding the same corpus takes hours. Optimising this would be optimising the wrong step.

Adam here omits bias correction, matching the MLX optimiser the research runs used (mlx.optimizers
.Adam updates w -= lr * m / (sqrt(v) + eps) with no 1/(1-beta^t) terms). Kept for comparability with
the published numbers; `bias_correction=True` is available and converges to the same place.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

BUNDLE_VERSION = 1


# ---- label space -----------------------------------------------------------------------------
def build_label_space(code_lists, min_count=25):
    """Codes with at least `min_count` training occurrences. Rarer codes are unreachable BY DESIGN:
    a linear probe cannot learn a code it has seen twice, and pretending otherwise is how you ship a
    model that silently never predicts half the book. The returned coverage is what to report."""
    cnt = Counter(c for cs in code_lists for c in cs)
    labels = sorted(c for c, v in cnt.items() if v >= min_count)
    total = sum(cnt.values())
    covered = sum(v for c, v in cnt.items() if v >= min_count)
    return labels, {"n_codes_seen": len(cnt), "n_labels": len(labels),
                    "min_count": min_count,
                    "label_coverage_of_train": covered / total if total else 0.0}


def multihot_indices(code_lists, lab2i):
    return [np.array([lab2i[c] for c in cs if c in lab2i], dtype=np.int32) for cs in code_lists]


# ---- numerics --------------------------------------------------------------------------------
def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])                      # exp of a negative number: no overflow
    out[~pos] = e / (1.0 + e)
    return out


def bce_with_logits(z, y):
    """mean over all cells, in the stable form max(z,0) - z*y + log1p(exp(-|z|))."""
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


class Adam:
    def __init__(self, shapes, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, bias_correction=False):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.bc = bias_correction
        self.m = [np.zeros(s, dtype=np.float32) for s in shapes]
        self.v = [np.zeros(s, dtype=np.float32) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m, v = self.m[i], self.v[i]
            if self.bc:
                m = m / (1 - self.b1 ** self.t)
                v = v / (1 - self.b2 ** self.t)
            p -= self.lr * m / (np.sqrt(v) + self.eps)


# ---- the probe -------------------------------------------------------------------------------
class LinearProbe:
    def __init__(self, dim, labels, *, seed=42):
        self.dim, self.labels = int(dim), list(labels)
        self.lab2i = {c: i for i, c in enumerate(self.labels)}
        K = len(self.labels)
        rng = np.random.default_rng(seed)
        # U(-1/sqrt(dim), 1/sqrt(dim)) on weight AND bias: what mlx.nn.Linear and torch.nn.Linear do.
        s = 1.0 / np.sqrt(self.dim)
        self.W = rng.uniform(-s, s, size=(self.dim, K)).astype(np.float32)
        self.b = rng.uniform(-s, s, size=(K,)).astype(np.float32)
        self.meta = {}

    # -- inference -----------------------------------------------------------------------------
    def logits(self, X):
        return X @ self.W + self.b

    def probs(self, X, chunk=4096):
        out = np.empty((len(X), len(self.labels)), dtype=np.float32)
        for i in range(0, len(X), chunk):
            out[i:i + chunk] = sigmoid(self.logits(np.asarray(X[i:i + chunk], dtype=np.float32)))
        return out

    # -- training ------------------------------------------------------------------------------
    def fit(self, X, Y_idx, *, epochs=20, batch_size=1024, lr=1e-3, seed=42, log=print,
            bias_correction=False):
        n, K = len(X), len(self.labels)
        rng = np.random.default_rng(seed)
        opt = Adam([self.W.shape, self.b.shape], lr=lr, bias_correction=bias_correction)
        order = np.arange(n)
        Yb = np.zeros((batch_size, K), dtype=np.float32)   # reused; multi-hot is far too big dense
        t0 = time.time()
        history = []
        for ep in range(epochs):
            rng.shuffle(order)
            tot, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx = np.sort(order[i:i + batch_size])   # sorted: mmap reads stay sequential
                B = len(idx)
                Xb = np.asarray(X[idx], dtype=np.float32)
                Y = Yb[:B]
                Y.fill(0.0)
                for r, row in enumerate(idx):
                    j = Y_idx[row]
                    if len(j):
                        Y[r, j] = 1.0
                Z = Xb @ self.W + self.b
                tot += bce_with_logits(Z, Y); nb += 1
                dZ = (sigmoid(Z) - Y) / (B * K)             # d(mean BCE)/dZ
                opt.step([self.W, self.b], [Xb.T @ dZ, dZ.sum(axis=0)])
            loss = tot / nb
            history.append(loss)
            if ep % 4 == 0 or ep == epochs - 1:
                log(f"  epoch {ep:2d}  loss {loss:.4f}  ({time.time() - t0:.0f}s)")
        self.meta.update({"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
                          "bias_correction": bias_correction, "final_loss": history[-1],
                          "fit_seconds": round(time.time() - t0, 1)})
        return history

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, W=self.W, b=self.b, labels=np.array(self.labels))
        path.with_suffix(".meta.json").write_text(
            json.dumps({"bundle_version": BUNDLE_VERSION, "dim": self.dim,
                        "n_labels": len(self.labels), **self.meta}, indent=2))
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        labels = [str(c) for c in z["labels"]]
        p = cls(z["W"].shape[0], labels)
        p.W, p.b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            p.meta = json.loads(meta_path.read_text())
            got = p.meta.get("bundle_version", 0)
            if got > BUNDLE_VERSION:
                raise ValueError(f"{path} was written by a newer toolkit (bundle_version {got} > "
                                 f"{BUNDLE_VERSION}); upgrade icdassist before loading it")
        return p
