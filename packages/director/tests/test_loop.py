"""D1's loop end to end, with a fake `claude` (stream-json on stdout, like the real one) and a fake local
OpenAI-compatible server: the director's walk and plan, a frontier task's transcript and a local task's reply
all land in the event log the interface reads; with no local server, the local task is logged offline and the
director never starts one."""

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director.core import capabilities, parse_plan  # noqa: E402
from director.server import serve  # noqa: E402

PLAN = {"summary": "check the mirror, then draft the note",
        "tasks": [{"id": "t1", "title": "check mirrors", "lane": "frontier", "why_lane": "needs the repo",
                   "prompt": "run the mirror check", "cites": ["Documentation#mirror_same_commit"]},
                  {"id": "t2", "title": "draft note", "lane": "local", "why_lane": "text only",
                   "prompt": "draft a two-line note", "cites": []}]}

FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, sys
PLAN = __PLAN__
prompt = sys.argv[sys.argv.index("-p") + 1]
def say(m): print(json.dumps(m), flush=True)
if "Walk your memory first" in prompt:
    say({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "/x/memory/Index.md"}}]}})
    text = "Walked Index and Documentation.\n```json\n" + json.dumps(PLAN) + "\n```"
else:
    say({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Grep", "input": {"pattern": "mirror"}}]}})
    text = "frontier: mirrors are level"
say({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
say({"type": "result", "result": text, "total_cost_usd": 0.01, "num_turns": 2, "is_error": False})
'''


def fake_claude(tmp: Path) -> str:
    p = tmp / "claude"
    p.write_text(FAKE_CLAUDE.replace("__PLAN__", repr(PLAN)))
    p.chmod(0o755)
    return str(p)


def memory(tmp: Path) -> Path:
    m = tmp / "sandbox" / "memory"
    m.mkdir(parents=True)
    (m / "Index.md").write_text("# Index\n- about to commit a document → [[Documentation#mirror_same_commit]]\n")
    (m / "Documentation.md").write_text("# Documentation\nDocs.\n\n## #mirror_same_commit\n**When:** x\n**Do:** y\n")
    return m


class Local(BaseHTTPRequestHandler):
    def do_GET(self):
        self._json({"data": [{"id": "local"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._json({"choices": [{"message": {"content": "local: note drafted"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4}})

    def _json(self, b):
        d = json.dumps(b).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)

    def log_message(self, *a):
        pass


def run_request(tmp, local_url):
    os.environ["DIRECTOR_CLAUDE"] = fake_claude(tmp)
    cfg = {"memory_dir": str(memory(tmp)), "workdir": str(tmp), "log": str(tmp / "events.jsonl"),
           "local_base_url": local_url}
    srv = serve(cfg, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    req = urllib.request.Request(base + "/request", data=json.dumps({"text": "update the plan doc"}).encode(),
                                 headers={"Content-Type": "application/json"})
    assert urllib.request.urlopen(req).status == 202
    for _ in range(100):
        evs = json.loads(urllib.request.urlopen(base + "/events?since=0").read())
        if any(e["kind"] in ("done", "error") and e["lane"] == "director" for e in evs):
            break
        time.sleep(0.1)
    state = json.loads(urllib.request.urlopen(base + "/state").read())
    page = urllib.request.urlopen(base + "/").read().decode()
    srv.shutdown()
    return evs, state, page


def test_plan_and_both_lanes_land_in_the_log(tmp_path):
    loc = ThreadingHTTPServer(("127.0.0.1", 0), Local)
    threading.Thread(target=loc.serve_forever, daemon=True).start()
    evs, state, page = run_request(tmp_path, f"http://127.0.0.1:{loc.server_address[1]}/v1")
    loc.shutdown()
    kinds = [(e["lane"], e["kind"]) for e in evs]
    assert ("director", "tool") in kinds and ("director", "plan") in kinds and ("director", "done") in kinds
    assert any(e["lane"] == "frontier" and e["kind"] == "text" and "mirrors are level" in e["text"] for e in evs)
    assert any(e["lane"] == "local" and e["kind"] == "text" and e["text"] == "local: note drafted" for e in evs)
    assert [e["seq"] for e in evs] == list(range(len(evs)))
    assert state["plan"]["tasks"][0]["cites"] == ["Documentation#mirror_same_commit"]
    assert state["capabilities"]["statements"] == 1 and len(state["capabilities"]["situations"]) == 1
    assert "Frontier — Claude Code" in page and "Local models" in page


def test_local_lane_offline_is_logged_and_never_started(tmp_path):
    evs, state, _ = run_request(tmp_path, "http://127.0.0.1:9/v1")      # nothing listens on port 9
    assert any(e["lane"] == "local" and e["kind"] == "offline" for e in evs)
    assert not state["local_online"]
    assert any(e["lane"] == "frontier" and e["kind"] == "task_end" for e in evs)


def test_parse_plan_rejects_a_reply_without_a_plan():
    assert parse_plan("no plan here") is None
    p = parse_plan("x\n```json\n" + json.dumps(PLAN) + "\n```")
    assert [t["lane"] for t in p["tasks"]] == ["frontier", "local"]


def test_capabilities_skip_index_and_readme(tmp_path):
    m = memory(tmp_path)
    (m / "README.md").write_text("# format\n## #not_a_statement\n")
    c = capabilities(m)
    assert list(c["pages"]) == ["Documentation"] and c["statements"] == 1
