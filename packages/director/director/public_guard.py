"""The plugin's PreToolUse hook on Bash: what is private never reaches a public repository. Standard library only.

A rule that must hold is enforced in code, not only learned (EVOLUTION §2, principle 4): the same way a confidential
task is refused by a lane not cleared for it, a commit, a push or a pull-request text bound for a repository not
listed as private is refused when it carries a private term or something shaped like a key. The terms live in a private
file; this code knows none of them.

The private config, `$DIRECTOR_PUBLIC_GUARD` or `~/.config/director/public_guard.json`:

    {"terms_file": "~/.config/director/redact.json",       # {"terms": [...]}, case-insensitive, matched inside words
     "private_remotes": ["owner/repo", ...]}                # repositories the guard lets through

Without that file the guard is silent. What it reads before deciding:

    git commit …              the staged diff and its message (-m, -F)
    git push …                the commits not yet on the remote (`@{upstream}..HEAD`, else `origin/HEAD..HEAD`)
    gh pr|issue create|edit|comment …   its --title and --body, and any --body-file / -F file

The decision goes back as `permissionDecision: "deny"` with the count and the first match, so the session can fix the
text; nothing is logged elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run as a script by the hook

from director.memory.journal import SECRET  # noqa: E402

COMMIT = re.compile(r"(?:^|[;&|]\s*|\s)git\s+(?:-C\s+\S+\s+)?commit\b")
PUSH = re.compile(r"(?:^|[;&|]\s*|\s)git\s+(?:-C\s+\S+\s+)?push\b")
GH_TEXT = re.compile(r"(?:^|[;&|]\s*|\s)gh\s+(?:pr|issue)\s+(?:create|edit|comment)\b")
REMOTE = re.compile(r"github\.com[:/]+([^/\s]+/[^/\s]+?)(?:\.git)?/?$")


def config() -> dict | None:
    p = Path(os.environ.get("DIRECTOR_PUBLIC_GUARD") or "~/.config/director/public_guard.json").expanduser()
    if not p.is_file():
        return None
    c = json.loads(p.read_text())
    terms = json.loads(Path(c["terms_file"]).expanduser().read_text()).get("terms", [])
    return {"terms": [t for t in terms if t], "private": {r.lower() for r in c.get("private_remotes", [])}}


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else ""


def workdir(cwd: Path, command: str) -> Path:
    m = re.search(r"git\s+-C\s+(\S+)", command) or re.search(r"(?:^|&&|;)\s*cd\s+(\S+)", command)
    if m:
        d = Path(os.path.expanduser(m.group(1).strip("'\"")))
        return d if d.is_absolute() else cwd / d
    return cwd


def repo(cwd: Path, command: str) -> str | None:
    m = re.search(r"--repo[ =](\S+)|-R\s+(\S+)", command)
    if m:
        return (m.group(1) or m.group(2)).strip("'\"").lower()
    url = git(cwd, "remote", "get-url", "origin").strip()
    m = REMOTE.search(url)
    return m.group(1).lower() if m else None


HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\s*\1\b", re.S)
MESSAGE_FLAGS = {"-m", "--message", "-t", "--title", "-b", "--body"}
FILE_FLAGS = {"-F", "--file", "--body-file"}


def flag_values(command: str, cwd: Path) -> list[str]:
    """The texts a commit or a gh command publishes: the values of its message, title and body flags, and the files
    named by its file flags — never the rest of the command line (which may hold a search for the very terms)."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    out = []
    for i, w in enumerate(words):
        name, _, inline = w.partition("=")
        value = inline if inline else (words[i + 1] if i + 1 < len(words) else "")
        if name in MESSAGE_FLAGS:
            out.append(value)
        elif name in FILE_FLAGS and value and value != "-":
            f = Path(os.path.expanduser(value))
            f = f if f.is_absolute() else cwd / f
            if f.is_file():
                out.append(f.read_text(errors="ignore"))
    return out


def outgoing(cwd: Path, command: str) -> str:
    """Everything the command would publish, as text."""
    parts = []
    if COMMIT.search(command):
        parts.append(git(cwd, "diff", "--cached"))
    if PUSH.search(command):
        parts.append(git(cwd, "log", "-p", "@{upstream}..HEAD") or git(cwd, "log", "-p", "origin/HEAD..HEAD"))
    if COMMIT.search(command) or GH_TEXT.search(command):
        parts += flag_values(command, cwd)
        if re.search(r"(?:-F|--file|--body-file)\s+-(?:\s|$)", command):    # the text comes on stdin, from a heredoc
            parts += [m.group(2) for m in HEREDOC.finditer(command)]
    return "\n".join(parts)


def findings(text: str, terms: list[str]) -> list[str]:
    hits = [t for t in terms if re.search(re.escape(t), text, re.I)]
    if SECRET.search(text):
        hits.append("a key or token")
    return hits


def decide(event: dict) -> dict | None:
    """The hook's reply, or None to let the call through."""
    if event.get("tool_name") != "Bash":
        return None
    command = (event.get("tool_input") or {}).get("command", "")
    if not (COMMIT.search(command) or PUSH.search(command) or GH_TEXT.search(command)):
        return None
    cfg = config()
    if cfg is None:
        return None
    cwd = workdir(Path(event.get("cwd") or os.getcwd()), command)
    name = repo(cwd, command)
    if name and name in cfg["private"]:
        return None
    hits = findings(outgoing(cwd, command), cfg["terms"])
    if not hits:
        return None
    where = name or "a repository not listed as private"
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": (f"director public guard: this would publish {len(hits)} private item(s) to {where} "
                                     f"(first: {hits[0]}). Remove them from the diff, commits or text — a private name "
                                     f"includes paths and directory names — and try again.")}}


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    try:
        out = decide(event)
    except Exception as e:      # a broken guard must say so, never silently allow
        out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                      "permissionDecisionReason": f"director public guard failed ({e}); check by hand"}}
    if out:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
