"""The band's and status line's script (director/ui_state.py): what it reads, and that its buttons go through the same
gates as the MCP tools. The band itself is tested by `claude plugin test .` (hooks/ui.test.tsx)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from test_plugin import EDIT, ROOT, mcp, project_with_memory


def ui(proj: Path, *args: str) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("DIRECTOR_MEMORY", "AGENTVCS")}
    e.update(CLAUDE_PROJECT_DIR=str(proj), PATH="/usr/bin:/bin")
    p = subprocess.run([sys.executable, str(ROOT / "director" / "ui_state.py"), *args], capture_output=True, text=True,
                       env=e, check=True)
    return json.loads(p.stdout)


def test_without_a_memory_the_status_says_so(tmp_path):
    assert ui(tmp_path, "status") == {"memory": None}


def test_status_counts_statements_and_lists_a_waiting_proposal(tmp_path):
    proj = project_with_memory(tmp_path)
    s = ui(proj, "status")
    assert s["statements"] == 1 and s["accepted"] == 0 and s["proposals"] == [] and s["delegations"] == []
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "scrub paths"}))
    [p] = ui(proj, "status")["proposals"]
    assert p["id"] == prop["id"] and p["rationale"] == "scrub paths" and p["acceptable"] and p["check"] is None
    assert p["edits"] == ["add Docs#scrub_paths"]
    assert "+## #scrub_paths" in ui(proj, "diff", prop["id"])["diff"]


def test_accept_from_the_band_goes_through_the_gate(tmp_path):
    proj = project_with_memory(tmp_path)
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "r"}))
    d = proj / ".director" / "candidates" / prop["id"]
    (d / "check.json").write_text(json.dumps({"verdict": "REDUNDANT", "teaches": False}))
    [p] = ui(proj, "status")["proposals"]
    assert not p["acceptable"] and "REDUNDANT" in p["why_not"]
    assert "REDUNDANT" in ui(proj, "accept", prop["id"])["error"]
    assert "scrub_paths" not in (proj / ".director" / "memory" / "Docs.md").read_text()

    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "r"}))
    assert ui(proj, "accept", prop["id"])["status"] == "accepted"
    assert "## #scrub_paths" in (proj / ".director" / "memory" / "Docs.md").read_text()
    s = ui(proj, "status")
    assert s["accepted"] == 1 and s["proposals"] == [] and s["statements"] == 2


def test_reject_leaves_the_memory_and_clears_the_band(tmp_path):
    proj = project_with_memory(tmp_path)
    before = (proj / ".director" / "memory" / "Docs.md").read_text()
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "r"}))
    assert ui(proj, "reject", prop["id"])["status"] == "rejected"
    assert ui(proj, "status")["proposals"] == []
    assert "is rejected" in ui(proj, "accept", prop["id"])["error"]
    assert (proj / ".director" / "memory" / "Docs.md").read_text() == before


def test_delegations_not_brought_back_are_listed_and_the_frontier_is_not(tmp_path):
    proj = project_with_memory(tmp_path)
    d = proj / ".director" / "delegations"
    d.mkdir(parents=True)
    for rec in ({"id": "a", "lane": "local", "status": "done"}, {"id": "b", "lane": "frontier", "status": "in_session"},
                {"id": "c", "lane": "local", "status": "done", "fetched": "/x"}):
        (d / f"{rec['id']}.json").write_text(json.dumps(rec))
    assert ui(proj, "status")["delegations"] == [{"id": "a", "lane": "local", "status": "done", "done": True}]


def test_a_bad_call_is_an_error_not_a_crash(tmp_path):
    proj = project_with_memory(tmp_path)
    assert "usage" in ui(proj)["error"]
    assert "no proposal" in ui(proj, "accept", "p0")["error"]


def test_the_ui_module_and_its_types_are_wired():
    hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text())
    assert all((ROOT / "hooks" / m).is_file() for m in hooks["modules"])
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert (ROOT / manifest["types"]).is_file()
