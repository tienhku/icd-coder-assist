"""The coder-facing local service: a JSON API plus the single-page UI that drives it.

Built on the standard library's ThreadingHTTPServer rather than FastAPI/uvicorn. The point is the
dependency list a hospital's IT review has to sign off: numpy and requests. The embedding endpoint is
the bottleneck by three orders of magnitude, so an async framework would buy nothing.

Operational rules baked in here, not left to the deployer to remember:
  * binds 127.0.0.1 unless told otherwise, and says so loudly when told otherwise;
  * NEVER logs note text -- the access log records character counts and timings only. A log file is
    the easiest place to accidentally spill a whole corpus of clinical notes;
  * an optional shared token (ICDASSIST_TOKEN) so a wider bind is not wide open. It is a speed bump,
    not an authentication system: put this behind the site's own reverse proxy and SSO.
"""
from __future__ import annotations

import json
import socketserver
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import __version__
from .embed import EmbedError, Embedder
from .pipeline import load_bundle

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_BODY = 4 * 1024 * 1024          # a 4MB note is already far past any encoder's context


class _State:
    def __init__(self, bundle, default_bar, default_policy, token):
        self.probe, self.router, self.settings = load_bundle(bundle)
        self.bundle = Path(bundle)
        self.default_bar = default_bar
        self.default_policy = default_policy
        self.token = token
        self.embedder = Embedder(self.settings.encoder, self.settings.embed_url,
                                 max_chars=self.settings.max_chars,
                                 batch_size=self.settings.batch_size)
        self.report = {}
        rp = self.bundle / "report.json"
        if rp.exists():
            self.report = json.loads(rp.read_text())
        self.started = time.time()
        self.n_requests = 0

    def code_one(self, text, bar, policy):
        v = self.embedder.one(text)
        if v.shape[0] != self.probe.dim:
            raise EmbedError(f"endpoint returned {v.shape[0]}d but the probe expects "
                             f"{self.probe.dim}d -- the bundle was fitted with a different encoder")
        p = self.probe.probs(v[None, :])[0]
        auto, defer = self.router.route(p, bar, policy=policy)
        return auto, defer

    def meta(self):
        bars = {}
        for key, th in self.router.global_threshold.items():
            b = self.report.get("bars", {}).get(key, {})
            bars[key] = {"threshold": th,
                         "eval_coverage": b.get("coverage"),
                         "eval_precision": b.get("precision"),
                         "coverage_ci": b.get("coverage_ci"),
                         "bar_met": b.get("bar_met"),
                         "n_per_code_thresholds": len(self.router.per_code.get(key, {}))}
        return {"toolkit_version": __version__,
                "encoder": self.settings.encoder,
                "embedding_dim": self.probe.dim,
                "n_labels": len(self.router.labels),
                "default_bar": self.default_bar,
                "default_policy": self.default_policy,
                "bars": bars,
                "fitted": self.probe.meta.get("n_train_notes"),
                "eval_notes": self.report.get("n_eval_notes"),
                "ece_decision_region": self.report.get("calibration", {}).get("ece_decision_region"),
                "uptime_s": round(time.time() - self.started, 1),
                "requests_served": self.n_requests}


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer minus the reverse DNS lookup in its bind path.

    http.server's HTTPServer.server_bind() calls socket.getfqdn() on the bind address, purely to
    populate self.server_name for CGI. The bind has already happened at that point but listen() has
    not, so on a host with no DNS egress -- an air-gapped clinical network, a locked-down container,
    a laptop on a captive portal -- the lookup blocks and the service prints that it is listening
    while accepting nothing. Exactly the environment this is meant to be deployed in, so we fill in
    server_name ourselves and never resolve anything."""

    daemon_threads = True          # a hung embedding call must not keep the process alive on Ctrl-C
    allow_reuse_address = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class Handler(BaseHTTPRequestHandler):
    server_version = f"icdassist/{__version__}"
    state: _State = None

    # -- plumbing ------------------------------------------------------------------------------
    def log_message(self, fmt, *args):
        """Deliberately overridden. The default logs the request line, which for a GET with a note
        in the query string would write clinical text to stderr. We log only what we choose to."""
        return

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")       # never cache a response about a note
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _authed(self):
        tok = self.state.token
        if not tok:
            return True
        got = self.headers.get("Authorization", "")
        if got.startswith("Bearer "):
            got = got[7:]
        return got == tok

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError(f"request body {n} bytes exceeds the {MAX_BODY} byte limit")
        if not n:
            raise ValueError("empty request body")
        return json.loads(self.rfile.read(n))

    # -- routes --------------------------------------------------------------------------------
    def do_GET(self):
        s = self.state
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            f = WEB / "index.html"
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8") if f.exists() \
                else self._send(404, {"error": "web/index.html not found"})
        if path == "/health":
            return self._send(200, {"status": "ok", "version": __version__})
        if not self._authed():
            return self._send(401, {"error": "missing or wrong bearer token"})
        if path == "/meta":
            return self._send(200, s.meta())
        if path == "/report":
            return self._send(200, s.report or {"error": "no report.json in this bundle"})
        if path == "/labels":
            return self._send(200, {"labels": s.router.labels})
        return self._send(404, {"error": f"no route {path}"})

    def do_POST(self):
        s = self.state
        path = self.path.split("?")[0]
        if not self._authed():
            return self._send(401, {"error": "missing or wrong bearer token"})
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        bar = float(body.get("bar", s.default_bar))
        policy = body.get("policy", s.default_policy)
        if policy not in ("global", "per-code"):
            return self._send(400, {"error": "policy must be 'global' or 'per-code'"})

        if path == "/code":
            text = body.get("text")
            if not isinstance(text, str) or not text.strip():
                return self._send(400, {"error": "field 'text' must be a non-empty string"})
            t0 = time.time()
            try:
                auto, defer = s.code_one(text, bar, policy)
            except KeyError as e:
                # str() on a KeyError re-quotes its argument; args[0] is the message we wrote.
                return self._send(400, {"error": e.args[0] if e.args else "unknown bar"})
            except EmbedError as e:
                return self._send(503, {"error": f"embedding backend: {e}"})
            s.n_requests += 1
            ms = round((time.time() - t0) * 1000, 1)
            print(f"[code] chars={len(text)} bar={bar} policy={policy} "
                  f"auto={len(auto)} defer={len(defer)} {ms}ms", flush=True)
            return self._send(200, {"bar": bar, "policy": policy, "auto_accept": auto,
                                    "defer": defer, "latency_ms": ms,
                                    "note_chars": len(text),
                                    "truncated": len(text) > s.settings.max_chars})

        if path == "/batch":
            notes = body.get("notes")
            if not isinstance(notes, list) or not notes:
                return self._send(400, {"error": "field 'notes' must be a non-empty list of "
                                                 "{id, text} objects"})
            out, t0 = [], time.time()
            for i, nd in enumerate(notes):
                text = (nd or {}).get("text")
                rid = str((nd or {}).get("id", i))
                if not isinstance(text, str) or not text.strip():
                    out.append({"id": rid, "error": "missing text"})
                    continue
                try:
                    auto, defer = s.code_one(text, bar, policy)
                    out.append({"id": rid, "auto_accept": auto, "defer": defer})
                except EmbedError as e:
                    out.append({"id": rid, "error": f"embedding backend: {e}"})
            s.n_requests += len(notes)
            ms = round((time.time() - t0) * 1000, 1)
            print(f"[batch] n={len(notes)} bar={bar} {ms}ms", flush=True)
            return self._send(200, {"bar": bar, "policy": policy, "results": out, "latency_ms": ms})

        return self._send(404, {"error": f"no route {path}"})


def serve(bundle, *, host="127.0.0.1", port=8080, default_bar=0.90, default_policy="global",
          token=None):
    import os
    token = token or os.environ.get("ICDASSIST_TOKEN")
    state = _State(bundle, default_bar, default_policy, token)
    Handler.state = state
    m = state.meta()
    print(f"icdassist {__version__}  bundle={bundle}")
    print(f"  encoder {m['encoder']} ({m['embedding_dim']}d)   K={m['n_labels']} codes")
    print(f"  bars calibrated: {', '.join(sorted(m['bars'], reverse=True))}   "
          f"default {default_bar} / {default_policy}")
    for key, b in sorted(m["bars"].items(), reverse=True):
        if b.get("bar_met") is False:
            print(f"  WARNING bar {key} did not hold on held-out data -- do not offer that tier")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING binding {host}: this service has no authentication of its own. "
              f"{'A shared token is set.' if token else 'NO token is set.'} "
              f"Put it behind your own reverse proxy and SSO.")
    if not state.probe.labels:
        raise ValueError("bundle has an empty label space")
    srv = _Server((host, port), Handler)                 # bind+listen before announcing it
    print(f"  listening on http://{host}:{port}  (UI at /, API at /code)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.server_close()
