"""Plans that wait for the person (director/plans.py, EVOLUTION step 4): nothing is delegated before Run; a move goes
only to a lane allowed for the task, and is logged for the next reflection; the confidentiality rule still holds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import lanes as lns  # noqa: E402
from director import plans  # noqa: E402
from test_lanes import FAKE  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    (tmp_path / "lane.py").write_text(FAKE)
    secret = tmp_path / "clients"
    secret.mkdir()
    py = sys.executable
    sub = [py, "lane.py", "submit", "{id}", "--request", "{request_file}", "--input={input}"]
    c = {"confidential_paths": [str(secret)], "lanes": {
        "private-exec": {"description": "private", "confidential_ok": True, "cwd": str(tmp_path), "submit": sub},
        "cheap-cloud": {"description": "not for private data", "confidential_ok": False, "cwd": str(tmp_path),
                        "submit": sub}}}
    return c, tmp_path / ".director", secret


def test_a_plan_waits_and_nothing_is_delegated_before_run(cfg):
    c, st, _ = cfg
    rec = plans.propose(st, c, [{"id": "draft-note", "task": "draft the release note", "lane": "frontier"},
                                {"id": "sum-logs", "task": "summarise the logs", "lane": "cheap-cloud"}], "ship 0.6")
    assert rec["status"] == "waiting" and [p["id"] for p in plans.waiting(st)] == [rec["id"]]
    assert lns.listing(st) == []
    out = plans.run(st, c, rec["id"])
    assert [(r["id"], r["status"]) for r in out["results"]] == [("draft-note", "in_session"), ("sum-logs", "submitted")]
    assert plans.waiting(st) == [] and len(lns.listing(st)) == 2
    with pytest.raises(plans.PlanError, match="not waiting"):
        plans.run(st, c, rec["id"])


def test_a_confidential_task_cannot_be_planned_or_moved_to_an_uncleared_lane(cfg):
    c, st, secret = cfg
    doc = secret / "contract.md"
    doc.write_text("x")
    with pytest.raises(plans.PlanError, match="confidential"):
        plans.propose(st, c, [{"id": "read-it", "task": "summarise", "lane": "cheap-cloud", "inputs": [str(doc)]}])
    rec = plans.propose(st, c, [{"id": "read-it", "task": "summarise", "lane": "private-exec", "inputs": [str(doc)]}])
    assert rec["tasks"][0]["confidential"] is True
    with pytest.raises(plans.PlanError, match="no other lane"):          # frontier and cheap-cloud are not cleared
        plans.move(st, c, rec["id"], "read-it")
    with pytest.raises(plans.PlanError, match="refused"):
        plans.move(st, c, rec["id"], "read-it", "frontier")


def test_a_move_cycles_through_the_allowed_lanes_and_is_logged(cfg):
    c, st, _ = cfg
    rec = plans.propose(st, c, [{"id": "draft-note", "task": "draft the release note", "lane": "frontier"}])
    assert plans.move(st, c, rec["id"], "draft-note")["to"] == "private-exec"
    assert plans.move(st, c, rec["id"], "draft-note")["to"] == "cheap-cloud"
    assert plans.move(st, c, rec["id"], "draft-note")["to"] == "frontier"
    moves = [json.loads(line) for line in (st / "lane-moves.jsonl").read_text().splitlines()]
    assert [(m["from"], m["to"]) for m in moves] == [("frontier", "private-exec"), ("private-exec", "cheap-cloud"),
                                                     ("cheap-cloud", "frontier")]
    assert all(m["plan"] == rec["id"] and m["task"] == "draft-note" for m in moves)


def test_cancel_delegates_nothing_and_bad_input_is_refused(cfg):
    c, st, _ = cfg
    rec = plans.propose(st, c, [{"id": "a", "task": "x", "lane": "frontier"}])
    assert plans.cancel(st, rec["id"])["status"] == "cancelled" and plans.waiting(st) == []
    assert lns.listing(st) == []
    for bad, why in [([], "non-empty"), ([{"id": "Bad Id", "task": "x", "lane": "frontier"}], "task id"),
                     ([{"id": "a", "task": "x", "lane": "frontier"}, {"id": "a", "task": "y", "lane": "frontier"}],
                      "twice"),
                     ([{"id": "a", "task": "x", "lane": "nowhere"}], "no lane")]:
        with pytest.raises(plans.PlanError, match=why):
            plans.propose(st, c, bad)


def test_one_failing_task_does_not_stop_the_others(cfg):
    c, st, _ = cfg
    lns.delegate(st, c, task="earlier", lane="cheap-cloud", task_id="taken", inputs=[], confidential=False)
    rec = plans.propose(st, c, [{"id": "taken", "task": "x", "lane": "cheap-cloud"},
                                {"id": "fresh", "task": "y", "lane": "cheap-cloud"}])
    out = plans.run(st, c, rec["id"])
    assert "already exists" in out["results"][0]["error"] and out["results"][1]["status"] == "submitted"


def test_the_band_and_the_mcp_tools_reach_the_same_plans(cfg, tmp_path):
    c, _, _ = cfg
    proj = tmp_path / "proj"
    (proj / ".director" / "memory").mkdir(parents=True)
    (proj / ".director" / "memory" / "Index.md").write_text("# Index\n")
    lanes_file = tmp_path / "lanes.json"
    lanes_file.write_text(json.dumps(c))
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(proj), "DIRECTOR_LANES": str(lanes_file)}
    env.pop("DIRECTOR_MEMORY", None)

    def ui(*args):
        p = subprocess.run([sys.executable, str(ROOT / "director" / "ui_state.py"), *args], capture_output=True,
                           text=True, env=env, check=True)
        return json.loads(p.stdout)

    msgs = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "plan_propose", "arguments": {
                "request": "ship it", "tasks": [{"id": "draft-note", "task": "draft", "lane": "frontier"}]}}}]
    p = subprocess.run([sys.executable, str(ROOT / "director" / "mcp_server.py")], capture_output=True, text=True,
                       input="".join(json.dumps(m) + "\n" for m in msgs), env=env, check=True)
    proposed = json.loads(json.loads(p.stdout.splitlines()[1])["result"]["content"][0]["text"])
    [plan] = ui("status")["plans"]
    assert plan["id"] == proposed["id"] and plan["tasks"][0]["movable"] is True
    assert ui("plan-move", plan["id"], "draft-note")["to"] == "private-exec"
    assert ui("plan-run", plan["id"])["results"][0]["status"] == "submitted"
    assert ui("status")["plans"] == []
