"""D1's loop end to end, with a fake `claude` (stream-json on stdout, like the real one) and a fake local
OpenAI-compatible server: the director's walk and plan, a frontier task's transcript and a local task's reply
all land in the event log the interface reads; with no local server, the local task is logged offline and the
director never starts one."""

import json
import os
import sys
import threading
import time
import urllib.error
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
if "A task has just ended" in prompt:
    EDITS = {"edits": [{"op": "add", "page": "Lanes", "id": "draft_notes_locally", "about": "Where tasks run.",
                        "when": "a task only drafts text from what it is given", "do": "plan it on the local lane",
                        "because": "the user moved such a task to local", "links": ["Documentation#mirror_same_commit"],
                        "situations": ["choosing a lane for a task"]}]}
    text = "Reflected.\n```json\n" + json.dumps(EDITS) + "\n```"
elif "Walk your memory first" in prompt:
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


def post(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for(base, kind, n=1):
    for _ in range(150):
        evs = json.loads(urllib.request.urlopen(base + "/events?since=0").read())
        if sum(e["kind"] == kind and e["lane"] == "director" for e in evs) >= n:
            return evs
        time.sleep(0.1)
    raise AssertionError(f"no {kind} event")


def start(tmp, local_url, **extra):
    os.environ["DIRECTOR_CLAUDE"] = fake_claude(tmp)
    cfg = {"memory_dir": str(memory(tmp)), "workdir": str(tmp), "log": str(tmp / "live" / "events.jsonl"),
           "local_base_url": local_url, **extra}
    srv = serve(cfg, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def run_request(tmp, local_url):
    srv, base = start(tmp, local_url)
    assert post(base, "/request", {"text": "update the plan doc"})[0] == 202
    wait_for(base, "plan")
    assert post(base, "/run")[0] == 202
    evs = wait_for(base, "done")
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


def test_the_plan_waits_a_move_is_logged_and_runs_in_the_new_lane(tmp_path):
    srv, base = start(tmp_path, "http://127.0.0.1:9/v1")
    post(base, "/request", {"text": "update the plan doc"})
    wait_for(base, "plan")
    assert json.loads(urllib.request.urlopen(base + "/state").read())["phase"] == "awaiting_run"
    assert post(base, "/move", {"task": "t2", "lane": "frontier"})[0] == 200
    assert post(base, "/move", {"task": "t9", "lane": "local"})[0] == 409            # no such task
    post(base, "/run")
    evs = wait_for(base, "done")
    assert post(base, "/move", {"task": "t1", "lane": "local"})[0] == 409            # only before the run
    srv.shutdown()
    moved = [e for e in evs if e["kind"] == "lane_moved"]
    assert moved and moved[0]["text"] == "local → frontier"
    assert any(e["lane"] == "frontier" and e["kind"] == "task_start" and e["task"] == "t2" for e in evs)
    assert not any(e["lane"] == "local" and e.get("task") == "t2" for e in evs)
    rec = [json.loads(l) for l in (tmp_path / "live" / "lane-moves.jsonl").read_text().splitlines()]
    assert rec[0]["from"] == "local" and rec[0]["to"] == "frontier" and rec[0]["title"] == "draft note"


def test_reflect_proposes_an_edit_and_reject_leaves_the_memory_alone(tmp_path):
    srv, base = start(tmp_path, "http://127.0.0.1:9/v1")
    assert post(base, "/reflect", {"correction": "x"})[0] == 409                    # nothing finished yet
    post(base, "/request", {"text": "update the plan doc"}); wait_for(base, "plan")
    post(base, "/move", {"task": "t2", "lane": "frontier"}); post(base, "/run"); wait_for(base, "done")
    assert post(base, "/reflect", {"correction": "drafting goes local"})[0] == 202
    evs = wait_for(base, "proposal")
    p = [e for e in evs if e["kind"] == "proposal"][0]
    assert p["proposal"]["applied"] == ["add Lanes#draft_notes_locally"] and p["proposal"]["lint"] == []
    before = sorted(x.name for x in (tmp_path / "sandbox" / "memory").iterdir())
    assert post(base, "/proposal", {"id": p["task"], "accept": False})[0] == 200
    assert sorted(x.name for x in (tmp_path / "sandbox" / "memory").iterdir()) == before
    assert post(base, "/proposal", {"id": p["task"], "accept": True})[0] == 409      # already decided
    srv.shutdown()


AGENTVCS = os.environ.get("AGENTVCS") or __import__("shutil").which("agentvcs")


@__import__("pytest").mark.skipif(not AGENTVCS, reason="agentvcs CLI not available")
def test_accept_versions_the_edit_in_agentvcs_and_swaps_the_live_memory(tmp_path):
    srv, base = start(tmp_path, "http://127.0.0.1:9/v1", agentvcs=AGENTVCS)
    post(base, "/request", {"text": "update the plan doc"}); wait_for(base, "plan")
    post(base, "/run"); wait_for(base, "done")
    post(base, "/reflect", {"correction": "drafting goes local"})
    p = [e for e in wait_for(base, "proposal") if e["kind"] == "proposal"][0]
    code, out = post(base, "/proposal", {"id": p["task"], "accept": True})
    assert code == 200, out
    assert out["dimensions"] == ["memory.index", "memory.lanes"] and out["gate"]["lint_problems"] == 0
    assert (tmp_path / "sandbox" / "memory" / "Lanes.md").exists()
    st = json.loads(urllib.request.urlopen(base + "/state").read())
    assert st["memory_version"] == out["manifest_id"] and st["capabilities"]["statements"] == 2
    srv.shutdown()
