"""Lanes (director/lanes.py): configured commands, and the confidentiality rule enforced in code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import lanes as lns  # noqa: E402
from director import mcp_server  # noqa: E402

# A fake lane: a script that echoes its arguments as the JSON the director expects.
FAKE = """
import json, sys
op, rest = sys.argv[1], sys.argv[2:]
if op == "submit":
    req = open(rest[2]).read()
    print("progress line")
    print(json.dumps({"id": rest[0], "request": req, "inputs": [a.split("=", 1)[1] for a in rest[3:]]}))
elif op == "status":
    print(json.dumps({"id": rest[0], "status": "running"}))
elif op == "fetch":
    print(json.dumps({"id": rest[0], "fetched_to": rest[1]}))
else:
    sys.exit(3)
"""


@pytest.fixture()
def cfg(tmp_path):
    (tmp_path / "lane.py").write_text(FAKE)
    secret = tmp_path / "clients"
    secret.mkdir()
    py = sys.executable
    c = {"confidential_paths": [str(secret)], "lanes": {
        "private-exec": {"description": "private", "confidential_ok": True, "cwd": str(tmp_path),
                         "submit": [py, "lane.py", "submit", "{id}", "--request", "{request_file}", "--input={input}"],
                         "status": [py, "lane.py", "status", "{id}"],
                         "fetch": [py, "lane.py", "fetch", "{id}", "{dest}"]},
        "cheap-cloud": {"description": "not for private data", "confidential_ok": False,
                        "submit": [py, "lane.py", "submit", "{id}", "--request", "{request_file}"]}}}
    return c, tmp_path, secret


def test_submit_status_fetch_through_the_configured_commands(cfg):
    c, tmp, _ = cfg
    doc = tmp / "chapter.md"
    doc.write_text("x")
    rec = lns.delegate(tmp / "state", c, task="correct chapter 1", lane="private-exec", task_id="ch-1",
                       inputs=[str(doc)], confidential=False)
    assert rec["submit"] == {"id": "ch-1", "request": "correct chapter 1", "inputs": [str(doc)]}
    assert lns.status(tmp / "state", c, "ch-1")["status"] == "running"
    assert lns.fetch(tmp / "state", c, "ch-1", str(tmp / "out"))["fetched_to"] == str(tmp / "out")
    assert [d["id"] for d in lns.listing(tmp / "state")] == ["ch-1"]
    assert lns.listing(tmp / "state")[0]["fetched"] == str(tmp / "out")     # the status line stops counting it


def test_a_confidential_task_is_refused_by_the_frontier_and_by_lanes_without_confidential_ok(cfg):
    c, tmp, _ = cfg
    for lane in ("frontier", "cheap-cloud"):
        with pytest.raises(lns.LaneError, match="refused: a confidential task"):
            lns.delegate(tmp / "state", c, task="t", lane=lane, task_id=f"x-{lane}", inputs=[], confidential=True)
    assert lns.listing(tmp / "state") == []


def test_an_input_under_a_confidential_folder_makes_the_task_confidential_whatever_the_flag(cfg):
    c, tmp, secret = cfg
    f = secret / "acme" / "contract.md"
    f.parent.mkdir()
    f.write_text("x")
    with pytest.raises(lns.LaneError, match="refused"):
        lns.delegate(tmp / "state", c, task="t", lane="frontier", task_id="y", inputs=[str(f)], confidential=False)
    rec = lns.delegate(tmp / "state", c, task="t", lane="private-exec", task_id="y", inputs=[str(f)],
                       confidential=False)
    assert rec["confidential"] and "is under" in rec["why_confidential"]


def test_public_work_can_stay_in_the_session(cfg):
    c, tmp, _ = cfg
    rec = lns.delegate(tmp / "state", c, task="draft the release note", lane="frontier", task_id="rn",
                       inputs=[], confidential=False)
    assert rec["status"] == "in_session"


def test_bad_ids_unknown_lanes_and_duplicates(cfg):
    c, tmp, _ = cfg
    with pytest.raises(lns.LaneError, match="id must be"):
        lns.delegate(tmp / "s", c, task="t", lane="frontier", task_id="Bad Id", inputs=[], confidential=False)
    with pytest.raises(lns.LaneError, match="no lane"):
        lns.delegate(tmp / "s", c, task="t", lane="nope", task_id="a", inputs=[], confidential=False)
    lns.delegate(tmp / "s", c, task="t", lane="frontier", task_id="a", inputs=[], confidential=False)
    with pytest.raises(lns.LaneError, match="already exists"):
        lns.delegate(tmp / "s", c, task="t", lane="frontier", task_id="a", inputs=[], confidential=False)


def test_the_mcp_tools(cfg, monkeypatch):
    c, tmp, _ = cfg
    (tmp / "lanes.json").write_text(json.dumps(c))
    mem = tmp / "memory"
    mem.mkdir()
    (mem / "Index.md").write_text("# Index\n")
    monkeypatch.setenv("DIRECTOR_LANES", str(tmp / "lanes.json"))
    monkeypatch.setenv("DIRECTOR_MEMORY", str(mem))

    def call(name, args):
        r = mcp_server.handle({"method": "tools/call", "id": 1, "params": {"name": name, "arguments": args}})
        return json.loads(r["result"]["content"][0]["text"])

    assert set(call("lanes_list", {})["lanes"]) == {"frontier", "private-exec", "cheap-cloud"}
    bad = call("delegate", {"task": "t", "lane": "frontier", "id": "c1", "confidential": True})
    assert not bad["ok"] and "refused" in bad["error"]
    ok = call("delegate", {"task": "t", "lane": "private-exec", "id": "c1", "confidential": True})
    assert ok["ok"] and ok["confidential"]
    assert call("lane_status", {"id": "c1"})["status"] == "running"
