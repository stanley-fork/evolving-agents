"""The director with no person present (`director.headless`): a benchmark keeps one store between sessions, records
the verifier's verdicts, and consolidates by policy. Claude Code is replaced by a fake that proposes through the gate
the way `memory_propose` would and records the prompt it was given."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import headless  # noqa: E402
from director.memory import gate, journal  # noqa: E402
from director.memory.format import state_dir  # noqa: E402

TRACE = [
    {"type": "system", "subtype": "init"},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest -q"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "use make check, not pytest"}]}},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "make check"}}]}},
    {"type": "result", "result": "done: added the command", "num_turns": 7},
]

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
sys.path.insert(0, os.environ["DIRECTOR_ROOT"])
from pathlib import Path
from director.memory import gate
args = sys.argv[1:]
Path(os.environ["FAKE_LOG"]).write_text(json.dumps(args))
edit = {"op": "add", "page": "Checks", "about": "How this repository is checked.", "id": "make_check",
        "when": "about to run the tests", "do": "run make check, not bare pytest", "because": "pytest skips the lint",
        "links": [], "situations": ["about to run the tests"]}
gate.propose(Path(os.environ["DIRECTOR_MEMORY"]), [edit], rationale="checks")
print(json.dumps({"type": "result", "num_turns": 5, "total_cost_usd": 0.12, "is_error": False,
                  "usage": {"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 50}}))
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "claude-args.json"
    monkeypatch.setenv("DIRECTOR_ROOT", str(ROOT))
    monkeypatch.setenv("FAKE_LOG", str(log))
    return exe, log


def write_trace(path: Path) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in TRACE))
    return path


def test_init_makes_an_empty_memory_once_inside_the_store(tmp_path):
    store = tmp_path / "store"
    first, again = headless.init(store), headless.init(store)
    assert first["created"] and not again["created"]
    assert (store / ".director" / "memory" / "Index.md").is_file()
    assert state_dir(headless.memory_of(store)) == store / ".director"     # journal and proposals stay in the store


def test_outcome_records_the_verdict_and_a_digest_of_the_session(tmp_path):
    store = tmp_path / "store"
    headless.init(store)
    res = headless.outcome(store, "03", False, write_trace(tmp_path / "t.jsonl"))
    assert res == {"task": "03", "passed": False, "turns": 7, "errors": 1}
    [e] = journal.read(store / ".director", kinds=["outcome"])
    assert e["source"] == "verifier" and e["passed"] is False
    assert "Bash: pytest -q" in e["text"] and "use make check, not pytest" in e["text"] and "done:" in e["text"]


def test_verified_shows_the_reflection_only_the_passed_sessions_and_accepts_by_policy(tmp_path, fake_claude):
    exe, log = fake_claude
    store = tmp_path / "store"
    headless.init(store)
    headless.outcome(store, "01", True, write_trace(tmp_path / "a.jsonl"))
    headless.outcome(store, "02-secret-fail", False, write_trace(tmp_path / "b.jsonl"))
    res = headless.consolidate(store, "verified", claude=str(exe))
    sent = json.loads(log.read_text())
    assert "session 01" in sent[1] and "02-secret-fail" not in sent[1]
    assert "--plugin-dir" in sent and "memory_accept" not in sent[sent.index("--allowedTools") + 1]
    assert res["sessions"] == 1 and len(res["accepted"]) == 1 and res["refused"] == []
    assert res["input_tokens"] == 100 and res["cache_read_tokens"] == 900                 # director-bench's count
    assert res["output_tokens"] == 50 and res["cost_usd"] == 0.12
    assert "## #make_check" in (headless.memory_of(store) / "Checks.md").read_text()
    kept = journal.read(store / ".director", kinds=["lesson.accepted"])
    assert kept[-1]["decided_by"] == "policy:verified"
    again = headless.consolidate(store, "verified", claude=str(exe))       # nothing new since the reflection
    assert again["sessions"] == 0 and again["accepted"] == []


def test_all_lets_failed_sessions_reach_the_reflection(tmp_path, fake_claude):
    exe, log = fake_claude
    store = tmp_path / "store"
    headless.outcome(store, "02-fail", False, write_trace(tmp_path / "b.jsonl"))
    res = headless.consolidate(store, "all", claude=str(exe))
    assert "02-fail" in json.loads(log.read_text())[1] and res["sessions"] == 1


def test_the_persons_gate_is_unchanged_and_policy_accept_is_not_a_tool(tmp_path):
    store = tmp_path / "store"
    m = Path(headless.init(store)["memory"])
    edit = {"op": "add", "page": "Checks", "about": "Checks.", "id": "x", "when": "w", "do": "d", "because": "b",
            "links": [], "situations": ["about to run the tests"]}
    p = gate.propose(m, [edit], rationale="r")
    with pytest.raises(gate.GateError):
        gate.accept(m, p["id"], False)
    with pytest.raises(gate.GateError):
        gate.accept_by_policy(m, p["id"], "anything")
    with pytest.raises(gate.GateError):
        gate.accept_by_policy(m, p["id"], "verified", require_check=True)
    src = (ROOT / "director" / "mcp_server.py").read_text()
    assert "accept_by_policy" not in src


def test_the_cli_prints_one_json_line(tmp_path):
    store = tmp_path / "store"
    p = subprocess.run([sys.executable, "-m", "director.headless", "outcome", "--memory", str(store), "--task", "05",
                        "--passed", "true"], cwd=ROOT, capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert json.loads(p.stdout.strip().splitlines()[-1])["passed"] is True


def test_a_proposal_the_policy_refuses_is_closed_not_left_waiting(tmp_path, monkeypatch):
    store = tmp_path / "store"
    m = Path(headless.init(store)["memory"])
    headless.outcome(store, "01", True)
    bad = {"op": "add", "page": "Checks", "about": "Checks.", "id": "x", "when": "w", "do": "d", "because": "b",
           "links": ["Nowhere#missing"], "situations": ["about to run the tests"]}

    def fake_run(*a, **k):
        gate.propose(m, [bad], rationale="dangling link")
        return subprocess.CompletedProcess(a, 0, stdout="{}", stderr="")
    monkeypatch.setattr(headless.subprocess, "run", fake_run)
    res = headless.consolidate(store, "verified")
    assert res["accepted"] == [] and len(res["refused"]) == 1
    assert gate.waiting(m)[0] == []
