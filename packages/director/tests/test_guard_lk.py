"""The D0 guard behind Accept (a failing guard leaves the memory as it was; a passing one is part of the agentvcs
gate) and the lora-kernel local lane (a specialist's local answer stays local; an abstention goes to the frontier)."""

import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_loop import AGENTVCS, PLAN, post, start, wait_for  # noqa: E402
import director.guard  # noqa: E402

GATE = Path(__file__).resolve().parents[1] / "director" / "learn_gate.py"


def through_reflection(tmp, monkeypatch, guard_result, **cfg):
    monkeypatch.setattr(director.guard, "run_guard", lambda cand, wd, g, progress=print: guard_result)
    srv, base = start(tmp, "http://127.0.0.1:9/v1", guard={"enabled": True, "k": 1, "min": 0.86}, **cfg)
    post(base, "/request", {"text": "update the plan doc"}); wait_for(base, "plan")
    post(base, "/run"); wait_for(base, "done")
    post(base, "/reflect", {"correction": "drafting goes local"})
    p = [e for e in wait_for(base, "proposal") if e["kind"] == "proposal"][0]
    code, out = post(base, "/proposal", {"id": p["task"], "accept": True})
    assert code == 200 and out["status"] == "guarding"
    return srv, base, p["task"]


def test_a_failing_guard_leaves_the_memory_unchanged(tmp_path, monkeypatch):
    srv, base, pid = through_reflection(tmp_path, monkeypatch, {"mean": 0.7, "min": 0.86, "n_cases": 14, "samples": 14,
                                                                "below_one": {}, "cost_usd": 0, "k": 1, "passed": False})
    evs = wait_for(base, "proposal_rejected")
    assert any(e["kind"] == "guard_done" for e in evs)
    assert not (tmp_path / "sandbox" / "memory" / "Lanes.md").exists()
    assert json.loads(urllib.request.urlopen(base + "/state").read())["proposals"][pid]["status"] == "guard_failed"
    srv.shutdown()


@pytest.mark.skipif(not AGENTVCS, reason="agentvcs CLI not available")
def test_a_passing_guard_is_part_of_the_agentvcs_gate(tmp_path, monkeypatch):
    srv, base, pid = through_reflection(tmp_path, monkeypatch, {"mean": 0.93, "min": 0.86, "n_cases": 14, "samples": 14,
                                                                "below_one": {}, "cost_usd": 0, "k": 1, "passed": True},
                                        agentvcs=AGENTVCS)
    ev = [e for e in wait_for(base, "memory_updated") if e["kind"] == "memory_updated"][0]
    assert ev["patch"]["gate"] == {"lint_problems": 0, "human_accepted": 1, "guard": 0.93}
    assert (tmp_path / "sandbox" / "memory" / "Lanes.md").exists()
    srv.shutdown()


def test_the_gate_command_reads_the_guard_and_refuses_an_incomplete_one(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"dimensions": {"memory.index": {"content": {"template": "# Index\n"}}}}))
    for n, want in ((14, 0.9), (13, 0.0)):
        g = tmp_path / "g.json"
        g.write_text(json.dumps({"mean": 0.9, "n_cases": n}))
        out = subprocess.run([sys.executable, str(GATE), str(g)], env={**os.environ, "AGENTVCS_MANIFEST_FILE": str(m)},
                             capture_output=True, text=True, check=True).stdout
        assert json.loads(out)["metrics"]["guard"] == want


class Gateway(BaseHTTPRequestHandler):
    """lora-kernel's gateway, faked: a tracker-ish request is served locally with a tool call; anything else is
    answered OUT OF SCOPE with route `frontier`, as tr-out-s0 does."""
    seen = []

    def do_GET(self):
        self._json({"data": [{"id": "auto"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Gateway.seen.append({"auth": self.headers.get("Authorization"), "session": self.headers.get("X-Session-Id"),
                             "model": body["model"]})
        text = body["messages"][-1]["content"]
        local = "draft" in text                       # PLAN's t2 prompt is "draft a two-line note"
        self._json({"choices": [{"message": {"content": "local: note drafted" if local else "OUT OF SCOPE"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4}, "x_route": "local" if local else "frontier",
                    "x_calls": [{"tool": "page_read", "args": {"ref": "notes"}, "result": "ok"}] if local else []})

    def _json(self, b):
        d = json.dumps(b).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)

    def log_message(self, *a):
        pass


def lk_stub(tmp: Path) -> Path:
    root = tmp / "lk"
    (root / "examples" / "common").mkdir(parents=True)
    (root / "examples" / "__init__.py").write_text("")
    (root / "examples" / "common" / "__init__.py").write_text("")
    (root / "examples" / "common" / "tokens.py").write_text("def issue(u, r, o):\n    return f'tok-{u}-{r}-{o}'\n")
    return root


def run_with_gateway(tmp, plan_lane_t1):
    gw = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    lk = {"path": str(lk_stub(tmp)), "user": "developer-harborworks", "role": "developer", "org": "harborworks",
          "domains": "the team tracker"}
    srv, base = start(tmp, f"http://127.0.0.1:{gw.server_address[1]}/v1", local_kind="lora-kernel", lora_kernel=lk)
    post(base, "/request", {"text": "update the plan doc"}); wait_for(base, "plan")
    if plan_lane_t1:
        post(base, "/move", {"task": "t1", "lane": "local"})
    post(base, "/run")
    evs = wait_for(base, "done")
    srv.shutdown(); gw.shutdown()
    return evs


def test_lora_kernel_lane_serves_its_domain_locally_with_a_fresh_session_and_a_role_token(tmp_path):
    Gateway.seen = []
    evs = run_with_gateway(tmp_path, False)
    assert any(e["lane"] == "local" and e["kind"] == "text" and e["text"] == "local: note drafted" for e in evs)
    assert any(e["lane"] == "local" and e["kind"] == "tool" and e["tool"] == "page_read" for e in evs)
    assert Gateway.seen[0]["auth"] == "Bearer tok-developer-harborworks-developer-harborworks"
    assert Gateway.seen[0]["model"] == "auto" and Gateway.seen[0]["session"].startswith("director-t2-")


def test_an_abstention_goes_to_the_frontier(tmp_path):
    Gateway.seen = []
    evs = run_with_gateway(tmp_path, True)                      # t1 ("run the mirror check") moved to local
    assert any(e["lane"] == "local" and e["kind"] == "abstained" and e["task"] == "t1" for e in evs)
    assert any(e["lane"] == "frontier" and e["kind"] == "text" and e["task"] == "t1" and "mirrors are level" in e["text"]
               for e in evs)
    assert not any(e["lane"] == "local" and e["kind"] == "text" and e.get("task") == "t1" for e in evs)
