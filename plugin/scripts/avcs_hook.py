#!/usr/bin/env python3
"""One hook script, dispatching on `hook_event_name`.

The Agent SDK hands each hook a JSON object on stdin carrying at minimum
`session_id`, `cwd` and `hook_event_name`. We keep every branch cheap and
non-blocking: a hook that raises on `PreCompact` is logged and the session
continues, but a hook that *hangs* costs the user real time, so nothing here
does network I/O or waits on a lock.

Standard library only, deliberately. The plugin has to run in whatever
interpreter the user's agent happens to run under, and `packages/agentvcs`
already holds itself to the same bar (see its test_zero_dependencies).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

STORE = ".agentvcs/sdk"

# Where Claude Code and the Agent SDK both write session transcripts. The
# directory name is the absolute cwd with every non-alphanumeric byte replaced
# by a dash, which is why we can find a transcript from `cwd` alone.
PROJECTS = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"


def encode_cwd(cwd: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(Path(cwd).resolve()))


def find_transcript(event: dict) -> Path | None:
    """Locate the session's JSONL.

    Prefer whatever path the event handed us — the field name differs across
    events (`transcript_path` on most, `agent_transcript_path` inside a
    subagent), so try both before falling back to discovery.
    """
    for key in ("transcript_path", "agent_transcript_path"):
        raw = event.get(key)
        if raw and Path(raw).is_file():
            return Path(raw)

    session_id, cwd = event.get("session_id"), event.get("cwd")
    if not session_id or not cwd:
        return None
    candidate = PROJECTS / encode_cwd(cwd) / f"{session_id}.jsonl"
    return candidate if candidate.is_file() else None


def store_dir(event: dict) -> Path:
    root = Path(event.get("cwd") or ".") / STORE
    root.mkdir(parents=True, exist_ok=True)
    return root


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_ledger(event: dict, name: str, record: dict) -> None:
    record = {"at": stamp(), "session": event.get("session_id"), **record}
    with (store_dir(event) / name).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def on_pre_compact(event: dict) -> str | None:
    """Copy the transcript aside before compaction summarises it away.

    This is the one place where fidelity is lost and cannot be recovered: after
    compaction the earlier turns exist only as a summary. The SDK's own docs
    name this exact use case for the hook.
    """
    src = find_transcript(event)
    if src is None:
        append_ledger(event, "events.jsonl", {"event": "PreCompact", "archived": None})
        return None

    archives = store_dir(event) / "transcripts"
    archives.mkdir(exist_ok=True)
    dest = archives / f"{event.get('session_id', 'unknown')}-{stamp()}.jsonl"
    shutil.copy2(src, dest)

    lines = sum(1 for _ in dest.open(encoding="utf-8", errors="replace"))
    append_ledger(
        event,
        "events.jsonl",
        {"event": "PreCompact", "archived": str(dest), "messages": lines},
    )
    return f"agentvcs archived {lines} messages to {dest.name} before compaction"


def on_session_start(event: dict) -> None:
    append_ledger(
        event,
        "events.jsonl",
        {"event": "SessionStart", "cwd": event.get("cwd"), "source": event.get("source")},
    )


def on_subagent_stop(event: dict) -> None:
    """The swarm dimension: which subagents ran, under which parent."""
    append_ledger(
        event,
        "swarm.jsonl",
        {
            "event": "SubagentStop",
            "agent_id": event.get("agent_id"),
            "agent_type": event.get("agent_type"),
        },
    )


HANDLERS = {
    "PreCompact": on_pre_compact,
    "SessionStart": on_session_start,
    "SubagentStop": on_subagent_stop,
}


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0  # Malformed input is not worth failing a user's session over.

    handler = HANDLERS.get(event.get("hook_event_name", ""))
    if handler is None:
        return 0

    try:
        message = handler(event)
    except OSError as exc:
        # Disk full, read-only checkout, permissions — record nothing, say nothing,
        # let the agent keep working.
        print(f"agentvcs hook skipped: {exc}", file=sys.stderr)
        return 0

    if message:
        print(json.dumps({"systemMessage": message}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
