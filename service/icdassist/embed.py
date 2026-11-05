"""Turn notes into frozen embeddings by calling a LOCAL embedding server.

Carried over from scripts/embed_notes.py, which earned its defences the hard way:

  * a single un-embeddable note once killed a 10-hour run at 44% -- so a note that fails at every
    length becomes a zero vector and an entry in `failed_ids`, never an exception;
  * long runs checkpoint and resume, keyed on a fingerprint of the id sequence so a resume cannot
    silently splice together two different row orders;
  * HTTP 400 ("too long") is distinguished from a connection reset or timeout. Shrinking the input
    on a transient error throws away good text; backing off on a genuine 400 loops forever.

Generalised for deployment: any Ollama or OpenAI-compatible /embeddings endpoint, so a site can use
qwen3-embedding, bge-m3, or an in-house encoder behind a compatible shim. No model is bundled.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import requests

OLLAMA = "ollama"
OPENAI = "openai"


class EmbedError(RuntimeError):
    pass


def detect_api(url):
    return OPENAI if "/v1/" in url or url.rstrip("/").endswith("/embeddings") else OLLAMA


def _post(url, payload, timeout):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _call(texts, model, url, api, timeout):
    if api == OPENAI:
        js = _post(url, {"model": model, "input": texts}, timeout)
        return [d["embedding"] for d in sorted(js["data"], key=lambda d: d["index"])]
    js = _post(url, {"model": model, "input": texts}, timeout)
    return js["embeddings"]


def _is_too_long(exc):
    """400 means the input exceeded the model's context -- shrink. Anything else (connection reset,
    timeout, 500 from a momentarily overloaded server) is transient -- back off at the same size."""
    resp = getattr(exc, "response", None)
    return resp is not None and resp.status_code in (400, 413, 422)


class Embedder:
    def __init__(self, model, url, *, api=None, max_chars=28000, batch_size=32, timeout=180):
        self.model, self.url = model, url
        self.api = api or detect_api(url)
        self.max_chars, self.batch_size, self.timeout = max_chars, batch_size, timeout

    def probe(self):
        """One tiny request, so a misconfigured endpoint fails in a second instead of an hour in."""
        try:
            v = _call(["probe"], self.model, self.url, self.api, 30)[0]
        except Exception as e:                                  # noqa: BLE001 -- surface anything as config error
            raise EmbedError(f"embedding endpoint {self.url} (model {self.model!r}) is not usable: {e}") from None
        return len(v)

    # -- single note ---------------------------------------------------------------------------
    def one(self, text):
        """Embed one note. Used by the API's online path; raises rather than zero-filling, because
        silently returning a zero vector to a live coder would produce confident nonsense."""
        last = None
        for cap in self._caps():
            for attempt in range(3):
                try:
                    v = _call([text[:cap]], self.model, self.url, self.api, self.timeout)[0]
                    return _l2(np.asarray(v, dtype=np.float32))
                except requests.RequestException as e:
                    last = e
                    if _is_too_long(e):
                        break
                    time.sleep(2 ** attempt)
        raise EmbedError(f"could not embed note at any length: {last}")

    def _caps(self):
        m = self.max_chars
        return (m, m // 2, m // 4, 2000, 1000, 500)

    def _one_or_none(self, text, rid, failed):
        for cap in self._caps():
            for attempt in range(3):
                try:
                    return _call([text[:cap]], self.model, self.url, self.api, self.timeout)[0]
                except requests.RequestException as e:
                    if _is_too_long(e):
                        break
                    time.sleep(2 ** attempt)
        failed.append(rid)
        print(f"  WARN {rid}: not embeddable at any length; writing a zero vector", flush=True)
        return None

    def _chunk(self, batch, ids, failed):
        """Whole-batch fast path, then per-note fallback so one bad note cannot poison its neighbours."""
        m = self.max_chars
        for cap in (m, m // 2, m // 4):
            for attempt in range(2):
                try:
                    return _call([t[:cap] for t in batch], self.model, self.url, self.api, self.timeout)
                except requests.RequestException as e:
                    if _is_too_long(e):
                        break
                    time.sleep(2 ** attempt)
        return [self._one_or_none(t, i, failed) for t, i in zip(batch, ids)]

    # -- corpus --------------------------------------------------------------------------------
    def corpus(self, texts, ids, out_prefix, *, checkpoint_every=50, restart=False, log=print):
        """Embed a whole corpus to <out_prefix>.npy + .ids.json, resumable. Returns (array, failed_ids)."""
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        fp = hashlib.sha256(("\x00".join(map(str, ids)) + f"|{self.model}").encode()).hexdigest()[:16]
        ck, ckm = Path(f"{out_prefix}.partial.npy"), Path(f"{out_prefix}.partial.meta.npy")

        vecs, start, failed = [], 0, []
        if ck.exists() and ckm.exists() and not restart:
            meta = np.load(ckm, allow_pickle=True).item()
            if meta.get("fingerprint") == fp:
                vecs = [r for r in np.load(ck)]
                start = len(vecs)
                failed = list(meta.get("failed", []))
                log(f"RESUMING at {start}/{len(texts)} ({start / len(texts):.1%})")
            else:
                log("checkpoint is for a different corpus or model -- starting over")

        dim = len(vecs[0]) if vecs else None
        t0, nb = time.time(), 0
        for i in range(start, len(texts), self.batch_size):
            batch, bids = texts[i:i + self.batch_size], ids[i:i + self.batch_size]
            for v in self._chunk(batch, bids, failed):
                if v is None:
                    vecs.append(None)
                else:
                    dim = dim or len(v)
                    vecs.append(v)
            nb += 1
            done = min(i + self.batch_size, len(texts))
            if nb % 20 == 0:
                rate = (done - start) / max(time.time() - t0, 1e-9)
                log(f"  {done}/{len(texts)}  {rate:.1f} notes/s  eta {(len(texts) - done) / max(rate, 1e-9):.0f}s")
            if checkpoint_every and nb % checkpoint_every == 0 and dim:
                np.save(ck, np.asarray([v if v is not None else [0.0] * dim for v in vecs], dtype=np.float32))
                np.save(ckm, np.array({"fingerprint": fp, "failed": failed}, dtype=object))
                log(f"  [checkpoint] {len(vecs)}/{len(texts)}")

        if dim is None:
            raise EmbedError("not a single note embedded -- check the endpoint and model name")
        arr = _l2(np.asarray([v if v is not None else [0.0] * dim for v in vecs], dtype=np.float32))
        np.save(f"{out_prefix}.npy", arr)
        (out_prefix.parent / f"{out_prefix.name}.ids.json").write_text(
            __import__("json").dumps({"ids": list(map(str, ids)), "failed_ids": list(map(str, failed)),
                                      "model": self.model, "dim": int(arr.shape[1])}, indent=2))
        for p in (ck, ckm):
            p.unlink(missing_ok=True)
        if failed:
            log(f"WARNING: {len(failed)} note(s) are zero vectors and will predict garbage -- "
                f"they are listed in {out_prefix.name}.ids.json under failed_ids")
        log(f"Saved {arr.shape} -> {out_prefix}.npy ({time.time() - t0:.0f}s)")
        return arr, failed


def _l2(a):
    if a.ndim == 1:
        return a / (np.linalg.norm(a) + 1e-8)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
