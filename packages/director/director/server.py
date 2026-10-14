"""The director's interface, served locally (D1). Standard library only.

    python -m director.server --config ~/.config/director/director.json      # http://127.0.0.1:8740

    GET  /            the page (ui.html)
    GET  /state       capabilities (pages, statements, situations), the current plan, lane status
    GET  /events?since=N   events with seq ≥ N (the page polls this)
    POST /request     {"text": "..."} — the director plans and delegates; refused with 409 while busy

The config (private, JSON): {"memory_dir", "workdir", "log", "local_base_url", "local_model", "director_model",
"frontier_model", "frontier_tools", "plan_budget_usd", "task_budget_usd"}. It binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .core import Director

UI = Path(__file__).resolve().parent / "ui.html"


def make_handler(d: Director):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/state":
                return self._send(200, d.state())
            if u.path == "/events":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                return self._send(200, d.log.since(since))
            self._send(404, {"error": "no route"})

        def do_POST(self):
            if urlparse(self.path).path != "/request":
                return self._send(404, {"error": "no route"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            text = (body.get("text") or "").strip()
            if not text:
                return self._send(400, {"error": "empty request"})
            if d.busy:
                return self._send(409, {"error": "the director is still working on the previous request"})
            d.busy = True                       # claimed here, so two quick posts cannot both start
            threading.Thread(target=d.handle, args=(text,), daemon=True).start()
            self._send(202, {"ok": True})

        def log_message(self, *a):
            pass
    return H


def serve(cfg: dict, port: int = 8740) -> ThreadingHTTPServer:
    d = Director(cfg)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(d))
    srv.director = d
    return srv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8740)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).expanduser().read_text())
    srv = serve(cfg, a.port)
    print(f"[director] http://127.0.0.1:{a.port} · memory {cfg['memory_dir']} · log {cfg['log']}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
