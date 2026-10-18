"""The plugin's SessionStart hook: put the walk (`DIRECTOR.md`) into the session's context when the project has a
director memory. Standard library only; silent when there is none, so the plugin costs nothing where it is not used.

Where the memory is, first found wins: `$DIRECTOR_MEMORY`, `<project>/.director/memory`, `<project>/memory`. A
directory counts only if it holds `Index.md`. The text is DIRECTOR.md's with `memory/` pointed at that directory —
byte for byte DIRECTOR.md when the memory is `./memory/` (P0's parity with D0's arm T).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

DIRECTOR_MD = Path(__file__).resolve().parent / "DIRECTOR.md"


def find_memory(project: Path) -> Path | None:
    env = os.environ.get("DIRECTOR_MEMORY")
    for d in ([Path(env).expanduser()] if env else []) + [project / ".director" / "memory", project / "memory"]:
        d = d if d.is_absolute() else project / d
        if (d / "Index.md").is_file():
            return d
    return None


def context(project: Path) -> str | None:
    mem = find_memory(project)
    if mem is None:
        return None
    try:
        rel = mem.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        rel = str(mem)
    text = DIRECTOR_MD.read_text()
    # One pass over every quoted path (`./memory/`, `memory/…`), so a replaced path is never replaced again.
    return text if rel == "memory" else re.sub(r"`(?:\./)?memory/", lambda _: f"`{rel}/", text)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = {}
    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd())
    text = context(project)
    if text:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
