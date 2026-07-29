"""The hook is exercised as the SDK runs it: JSON on stdin, JSON on stdout.

Driving the script as a subprocess rather than importing it is the point — the
plugin's contract with the SDK is a process boundary, and a test that imports
`main()` would pass even if the file were unreadable or the shebang wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "avcs_hook.py"


def run_hook(event: dict) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    """A JSONL transcript shaped like the one the SDK writes."""
    path = tmp_path / "session.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "build the thing"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "on it"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "done"}},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_pre_compact_archives_the_transcript_before_it_is_summarised(
    tmp_path: Path, transcript: Path
) -> None:
    workdir = tmp_path / "repo"
    workdir.mkdir()

    code, stdout, stderr = run_hook(
        {
            "hook_event_name": "PreCompact",
            "session_id": "abc-123",
            "cwd": str(workdir),
            "transcript_path": str(transcript),
        }
    )

    assert code == 0, stderr
    archives = list((workdir / ".agentvcs/sdk/transcripts").glob("abc-123-*.jsonl"))
    assert len(archives) == 1, "expected exactly one archived transcript"
    assert archives[0].read_text(encoding="utf-8") == transcript.read_text(encoding="utf-8")

    # The user is told, because a silent archive is indistinguishable from none.
    assert json.loads(stdout)["systemMessage"].startswith("agentvcs archived 3 messages")


def test_the_ledger_records_what_was_archived(tmp_path: Path, transcript: Path) -> None:
    workdir = tmp_path / "repo"
    workdir.mkdir()
    run_hook(
        {
            "hook_event_name": "PreCompact",
            "session_id": "abc-123",
            "cwd": str(workdir),
            "transcript_path": str(transcript),
        }
    )

    entries = [
        json.loads(line)
        for line in (workdir / ".agentvcs/sdk/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["event"] == "PreCompact"
    assert entries[0]["messages"] == 3
    assert entries[0]["session"] == "abc-123"


def test_a_missing_transcript_is_recorded_not_fatal(tmp_path: Path) -> None:
    """A brand-new session may have no transcript on disk yet."""
    workdir = tmp_path / "repo"
    workdir.mkdir()

    code, stdout, _ = run_hook(
        {
            "hook_event_name": "PreCompact",
            "session_id": "nope",
            "cwd": str(workdir),
            "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
        }
    )

    assert code == 0
    assert stdout == "", "nothing was archived, so say nothing to the user"
    entry = json.loads((workdir / ".agentvcs/sdk/events.jsonl").read_text(encoding="utf-8"))
    assert entry["archived"] is None


def test_subagent_stop_records_the_swarm_dimension(tmp_path: Path) -> None:
    workdir = tmp_path / "repo"
    workdir.mkdir()

    code, _, _ = run_hook(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "abc-123",
            "cwd": str(workdir),
            "agent_id": "sub-1",
            "agent_type": "Explore",
        }
    )

    assert code == 0
    entry = json.loads((workdir / ".agentvcs/sdk/swarm.jsonl").read_text(encoding="utf-8"))
    assert entry["agent_type"] == "Explore"
    assert entry["agent_id"] == "sub-1"


@pytest.mark.parametrize(
    "payload",
    ["", "not json at all", '{"hook_event_name": "SomeEventWeDoNotHandle"}'],
    ids=["empty", "malformed", "unhandled-event"],
)
def test_the_hook_never_fails_a_session(payload: str) -> None:
    """Whatever arrives, exit 0 and stay quiet — the agent's work is not ours to break."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_the_hook_imports_nothing_outside_the_standard_library() -> None:
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported - set(sys.stdlib_module_names), (
        "the hook runs in whatever interpreter the user's agent runs under; "
        "it cannot assume anything is installed"
    )
