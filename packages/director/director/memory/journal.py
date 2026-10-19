"""The episodic journal: what happened, in one append-only file per project (`<state>/journal.jsonl`), EVOLUTION §4.

It holds evidence, not instruction. Nothing that plans reads it; only the reflection does, and what the reflection
draws from it reaches the memory through the gate like any other lesson.

    append(state, kind, **fields)     one entry: {"t": ..., "kind": ..., ...}; never rewrites an earlier one
    read(state, since=None, kinds=None)
    since_reflection(state)           the entries after the last `reflected` entry (all of them if none)
    reflected(state, note)            marks the point the next reflection starts from

Kinds written by the director itself:

    lesson.proposed / lesson.accepted / lesson.rejected / lesson.check_started   memory.gate
    plan.proposed / plan.moved / plan.cancelled / plan.ran                       plans
    lane.delegated / lane.status / lane.fetched                                  lanes
    correction / outcome / note                                                  the session, through `journal_note`
    reflected                                                                    the reflection, when it is done
"""

from __future__ import annotations

import json
import time
from pathlib import Path

NAME = "journal.jsonl"
SESSION_KINDS = ("correction", "outcome", "note")


def path(state: Path) -> Path:
    return Path(state) / NAME


def append(state: Path, kind: str, **fields) -> dict:
    """Append one entry. A failure to write is swallowed: the journal is evidence, and losing one line must never stop
    the work it records."""
    entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **fields}
    try:
        p = path(state)
        p.parent.mkdir(parents=True, exist_ok=True)
        torn = False
        if p.is_file() and p.stat().st_size:
            with p.open("rb") as f:
                f.seek(-1, 2)
                torn = f.read(1) != b"\n"                 # a line cut by a crash: start this entry on its own line
        with p.open("a", encoding="utf-8") as f:
            f.write(("\n" if torn else "") + json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return entry


def read(state: Path, since: str | None = None, kinds: tuple[str, ...] | list[str] | None = None) -> list[dict]:
    p = path(state)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue                                  # a torn line is skipped, never fatal
        if since is not None and e.get("t", "") <= since:
            continue
        if kinds is not None and e.get("kind") not in kinds:
            continue
        out.append(e)
    return out


def since_reflection(state: Path) -> list[dict]:
    entries = read(state)
    last = max((i for i, e in enumerate(entries) if e.get("kind") == "reflected"), default=-1)
    return entries[last + 1:]


def reflected(state: Path, note: str = "") -> dict:
    return append(state, "reflected", note=note)
