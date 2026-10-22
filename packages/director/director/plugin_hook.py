"""The plugin's SessionStart and UserPromptSubmit hook. SessionStart puts the walk (`DIRECTOR.md`) into the session's
context when the project has a director memory; UserPromptSubmit sends the person's message to the journal. Both say
when the director running is older than the one installed (restart Claude Code; `claude --resume` keeps the
conversation): SessionStart every time, UserPromptSubmit once per distinct installed version per session (recorded in
`<state>/stale_warned.json`), so an update made mid-session is noticed at the next prompt without repeating on every
one. Standard library only; silent when there is no memory, so the plugin costs nothing where it is not used.

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


def installed_newer(root: Path | None = None, registry: Path | None = None) -> tuple[str, str, str] | None:
    """`(running, installed, install path)` when the director running here is not the version Claude Code has
    installed (a session that loaded an older cached copy, or a plugin updated after the session started); else None.
    Two small JSON reads, cheap enough for every prompt."""
    root = Path(root or os.environ.get("CLAUDE_PLUGIN_ROOT") or Path(__file__).resolve().parents[1])
    registry = Path(registry or Path.home() / ".claude" / "plugins" / "installed_plugins.json")
    try:
        mine = json.loads((root / ".claude-plugin" / "plugin.json").read_text())["version"]
        reg = json.loads(registry.read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for key, installs in (reg.get("plugins", reg) or {}).items():
        if not key.startswith("director@"):
            continue
        for inst in installs if isinstance(installs, list) else [installs]:
            if not isinstance(inst, dict):
                continue
            path, version = inst.get("installPath"), inst.get("version")
            if path and Path(path).resolve() != root.resolve() and version and version != mine:
                return mine, version, path
    return None


def restart_message(mine: str, version: str) -> str:
    return (f"director {mine} is running, but {version} is installed: restart Claude Code to load it "
            f"(`claude --resume` keeps this conversation).")


def stale_install(root: Path | None = None, registry: Path | None = None) -> str:
    """SessionStart's paragraph when an older director is running than the one installed."""
    found = installed_newer(root, registry)
    if not found:
        return ""
    mine, version, path = found
    return (f"\n\n**This session runs director {mine}, but {version} is installed** ({path}). The band, the status "
            f"line and the tools here are {mine}'s until Claude Code restarts: tell the person to restart it "
            f"(`claude --resume` keeps the conversation) if they are waiting for a change.")


WARNED = "stale_warned.json"


def first_warning(state: Path, session: str, version: str) -> bool:
    """True the first time this session is warned about this installed version, and records it: the prompt-time
    warning fires once per distinct installed version per session, never on every prompt."""
    f = Path(state) / WARNED
    try:
        seen = json.loads(f.read_text())
        seen = seen if isinstance(seen, dict) else {}
    except (OSError, ValueError):
        seen = {}
    versions = seen.get(session) if isinstance(seen.get(session), list) else []
    if version in versions:
        return False
    seen.pop(session, None)
    seen[session] = versions + [version]
    seen = dict(list(seen.items())[-50:])          # the last sessions only; the file never grows
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(seen))
    except OSError:
        pass
    return True


def prompt_warning(project: Path, session: str, root: Path | None = None, registry: Path | None = None) -> str:
    """UserPromptSubmit: the restart message, when the installed director changed under this session and the
    session has not been told about that version yet; empty otherwise (and where there is no memory)."""
    mem = find_memory(project)
    found = installed_newer(root, registry) if mem is not None else None
    if not found or not first_warning(state_dir(mem), session, found[1]):
        return ""
    return restart_message(found[0], found[1])


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
    session = str(event.get("session_id") or "")
    if event.get("hook_event_name") == "UserPromptSubmit":
        capture(project, event.get("prompt", ""))
        warn = prompt_warning(project, session)
        if warn:
            print(json.dumps({"systemMessage": warn, "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": f"{warn} Until then the band, the status line and the director's tools are the "
                                     f"old version's; tell the person once."}}))
        return 0
    text = context(project)
    if text:
        mem = find_memory(project)
        text += nudge(mem) + stale_install()
        out = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}
        found = installed_newer()
        if found:
            first_warning(state_dir(mem), session, found[1])     # the first prompt need not repeat it
            out["systemMessage"] = restart_message(found[0], found[1])
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
