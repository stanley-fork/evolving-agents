"""The episodic journal (director/memory/journal.py, EVOLUTION step 5): one append-only file the director writes as
things happen, read by the reflection since the last `reflected` mark, and never fatal to the work it records."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import lanes as lns  # noqa: E402
from director import plans  # noqa: E402
from director.memory import gate, journal  # noqa: E402
from test_plugin import EDIT, project_with_memory  # noqa: E402


def kinds(st: Path) -> list[str]:
    return [e["kind"] for e in journal.read(st)]


def test_append_read_and_since_reflection(tmp_path):
    journal.append(tmp_path, "note", text="a")
    journal.reflected(tmp_path, "learned x")
    journal.append(tmp_path, "correction", text="the note goes on the frontier")
    assert kinds(tmp_path) == ["note", "reflected", "correction"]
    assert [e["kind"] for e in journal.since_reflection(tmp_path)] == ["correction"]
    assert journal.read(tmp_path, kinds=["note"])[0]["text"] == "a"


def test_a_torn_line_is_skipped_and_a_write_failure_is_not_fatal(tmp_path):
    journal.append(tmp_path, "note", text="a")
    with journal.path(tmp_path).open("a") as f:
        f.write('{"t": "x", "kind": "no')                 # a line cut by a crash
    journal.append(tmp_path, "note", text="b")
    assert [e.get("text") for e in journal.read(tmp_path)] == ["a", "b"]   # the torn line is skipped; b is kept
    blocked = tmp_path / "file"
    blocked.write_text("")
    journal.append(blocked, "note", text="c")                          # state is a file: the write fails quietly


def test_the_gate_writes_what_happened_to_a_lesson(tmp_path):
    proj = project_with_memory(tmp_path)
    mem = proj / ".director" / "memory"
    a = gate.propose(mem, [EDIT], rationale="scrub paths")
    gate.reject(mem, a["id"])
    b = gate.propose(mem, [{**EDIT, "id": "other"}], rationale="r")
    gate.accept(mem, b["id"], True)
    st = proj / ".director"
    assert kinds(st) == ["lesson.proposed", "lesson.rejected", "lesson.proposed", "lesson.accepted"]
    assert journal.read(st)[0]["rationale"] == "scrub paths"


def test_plans_and_lanes_write_what_happened(tmp_path):
    cfg = {"confidential_paths": [], "lanes": {}}
    st = tmp_path / ".director"
    p = plans.propose(st, cfg, [{"id": "draft", "task": "draft the note", "lane": "frontier"}], "ship")
    plans.run(st, cfg, p["id"])
    q = plans.propose(st, cfg, [{"id": "other", "task": "x", "lane": "frontier"}])
    plans.cancel(st, q["id"])
    assert kinds(st) == ["plan.proposed", "lane.delegated", "plan.ran", "plan.proposed", "plan.cancelled"]
    assert journal.read(st, kinds=["lane.delegated"])[0]["status"] == "in_session"
    assert lns.listing(st)[0]["id"] == "draft"


def test_the_session_reads_and_notes_through_mcp(tmp_path):
    proj = project_with_memory(tmp_path)
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(proj)}
    env.pop("DIRECTOR_MEMORY", None)
    calls = [("journal_note", {"kind": "correction", "text": "public text goes to the frontier"}),
             ("journal_read", {}),
             ("journal_note", {"kind": "reflected", "text": "learned the lane"}),
             ("journal_read", {}),
             ("journal_note", {"kind": "plan.ran", "text": "forged"})]
    msgs = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}]
    msgs += [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": n, "arguments": a}}
             for i, (n, a) in enumerate(calls, 1)]
    p = subprocess.run([sys.executable, str(ROOT / "director" / "mcp_server.py")], capture_output=True, text=True,
                       input="".join(json.dumps(m) + "\n" for m in msgs), env=env, check=True)
    out = [json.loads(json.loads(line)["result"]["content"][0]["text"]) for line in p.stdout.splitlines()[1:]]
    assert [e["kind"] for e in out[1]["entries"]] == ["correction"]
    assert out[3]["entries"] == []                                      # nothing since the reflection
    assert not out[4]["ok"]                                             # the session cannot forge the director's kinds
