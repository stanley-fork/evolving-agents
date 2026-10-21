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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run as a script by the hook

from director.memory import journal  # noqa: E402
from director.memory.format import find as find_memory, state_dir  # noqa: E402

DIRECTOR_MD = Path(__file__).resolve().parent / "DIRECTOR.md"


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


def nudge(mem: Path) -> str:
    n = journal.owed(state_dir(mem))
    return (f"\n\n**The journal holds {n} message{'s' if n != 1 else ''} or correction{'s' if n != 1 else ''} since the "
            f"last reflection.** Decisions the person made there (a choice, a cap, a rule, a preference) are not in this "
            f"memory until a reflection proposes them: run `/director:reflect` at the next natural break.") if n else ""


def capture(project: Path, prompt: str) -> bool:
    """UserPromptSubmit: the person's message goes to the journal as it is sent, keys scrubbed. Nothing is decided
    here and nothing reaches the memory; the reflection reads it later and proposes lessons through the gate."""
    mem = find_memory(project)
    text = (prompt or "").strip()
    if mem is None or not text or text.startswith("<"):
        return False
    journal.append(state_dir(mem), "person", text=journal.scrub(text)[:4000])
    return True


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = {}
    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd())
    if event.get("hook_event_name") == "UserPromptSubmit":
        capture(project, event.get("prompt", ""))
        return 0
    text = context(project)
    if text:
        text += nudge(find_memory(project))
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
