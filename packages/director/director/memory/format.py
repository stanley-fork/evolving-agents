"""Where the memory is, and its pages as data.

A memory is a directory holding `Index.md` (situations → entry statements) and one page per subject; a page holds
atomic statements, each a `## #id` heading with **When**, **Do**, **Because** and **Links** lines
(`docs/MEMORY-FORMAT.md`).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

STMT = re.compile(r"^## #(\w+)\s*$", re.M)
LINK = re.compile(r"\[\[([A-Za-z0-9_]+)#(\w+)\]\]")
_ID = re.compile(r"(?m)^## #([A-Za-z0-9_]+)\s*$")
NOT_PAGES = ("Index.md", "README.md")


def find(project: Path) -> Path | None:
    """The project's memory, first found wins: `$DIRECTOR_MEMORY`, `<project>/.director/memory`, `<project>/memory`.
    A directory counts only if it holds `Index.md`."""
    env = os.environ.get("DIRECTOR_MEMORY")
    for d in ([Path(env).expanduser()] if env else []) + [project / ".director" / "memory", project / "memory"]:
        d = d if d.is_absolute() else project / d
        if (d / "Index.md").is_file():
            return d
    return None


def state_dir(memory: Path) -> Path:
    """Where the director keeps its state beside a memory (candidates, checks, the agentvcs store, delegations)."""
    memory = Path(memory)
    return memory.parent if memory.parent.name == ".director" else memory.parent / ".director"


def version(memory: Path) -> str:
    """The memory's version: a hash of every page's path and bytes. Views compiled from the memory (the guard's
    baseline, the recipes) record the version they were built from, so a stale one is caught by lint, not by a bad
    plan."""
    memory, h = Path(memory), hashlib.sha256()
    for p in sorted(memory.rglob("*.md")):
        h.update(p.relative_to(memory).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def statements(page_text: str) -> dict[str, str]:
    """id -> the statement's block (from its `## #id` line up to the next one)."""
    out, ms = {}, list(STMT.finditer(page_text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(page_text)
        out[m.group(1)] = page_text[m.start():end].rstrip() + "\n"
    return out


def render(sid: str, e: dict) -> str:
    links = " · ".join(f"[[{l}]]" for l in e.get("links") or [])
    return (f"## #{sid}\n**When:** {e['when'].strip()}\n**Do:** {e['do'].strip()}\n"
            f"**Because:** {e['because'].strip()}\n**Links:** {links}\n")


def field(block: str, name: str) -> str:
    m = re.search(rf"^\*\*{name}:\*\*\s*(.*)$", block, re.M)
    return m.group(1).strip() if m else ""


def statement_ids(memory: Path) -> set[str]:
    """Every `Page#id` the memory's pages hold."""
    ids = set()
    for f in Path(memory).glob("*.md"):
        if f.name in NOT_PAGES:
            continue
        ids |= {f"{f.stem}#{s}" for s in _ID.findall(f.read_text(encoding="utf-8"))}
    return ids
