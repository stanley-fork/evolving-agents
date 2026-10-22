#!/usr/bin/env python3
"""Fake `claude` for offline tests: scripted edits + Claude-Code-shaped stream-json.

Behaviour comes from the JSON file in $FAKE_CLAUDE_SPEC:

    {"rules": [{"match": "<substring of the prompt>", ...behaviour}], "default": {...}}

behaviour keys (all optional):
    overlay          dir copied over the cwd (e.g. a reference solution)
    writes           {"relative/path": "text"} written into the cwd
    run              shell commands run in the cwd after overlay/writes (e.g. a regen step)
    turns            number of assistant turns (default 2)
    usage            per-turn usage {"input_tokens", "output_tokens",
                     "cache_creation_input_tokens", "cache_read_input_tokens"}
    tool_errors      list of messages; turn i gets an is_error tool_result with message i
    sleep            seconds to sleep after each turn
    memory_append    text appended to the file named by $FAKE_MEMORY
    home_note        if true, append to $HOME/.claude/fake-notes.txt
    cost             total_cost_usd of the result line
    resume           behaviour used when this session is resumed (`--resume <id>`); default {}
    exit             exit code (default 0)

Every call is logged as one JSON line to $FAKE_CLAUDE_LOG (if set).
"""
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    argv = sys.argv[1:]
    if "--version" in argv:
        print("9.9.9 (Fake Claude)")
        return 0
    prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""
    spec = json.loads(Path(os.environ["FAKE_CLAUDE_SPEC"]).read_text()) if os.environ.get("FAKE_CLAUDE_SPEC") else {}
    cwd = Path.cwd()
    home = Path(os.environ.get("HOME", "/nonexistent"))
    state_dir = home / ".fake-claude-sessions"
    if "--resume" in argv:
        session_id = argv[argv.index("--resume") + 1]
        state = json.loads((state_dir / f"{session_id}.json").read_text())   # fails like claude on a bad id
        beh = state["rule"].get("resume", {})
    else:
        beh = next((r for r in spec.get("rules", []) if r.get("match", "\0") in prompt), spec.get("default", {}))
        session_id = str(uuid.uuid4())
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"{session_id}.json").write_text(json.dumps({"rule": beh}))
    mem_file = os.environ.get("FAKE_MEMORY")

    if os.environ.get("FAKE_CLAUDE_LOG"):
        with open(os.environ["FAKE_CLAUDE_LOG"], "a") as fh:
            fh.write(json.dumps({
                "argv": argv, "cwd": str(cwd), "home": str(home), "prompt": prompt,
                "resumed": "--resume" in argv,
                "claude_config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
                "files": sorted(p.relative_to(cwd).as_posix() for p in cwd.rglob("*")
                                if p.is_file() and ".git" not in p.parts),
                "memory": Path(mem_file).read_text() if mem_file and Path(mem_file).exists() else None,
                "home_notes": (home / ".claude" / "fake-notes.txt").read_text()
                if (home / ".claude" / "fake-notes.txt").exists() else None,
                "env_keys": sorted(os.environ),
            }) + "\n")

    if beh.get("overlay"):
        shutil.copytree(beh["overlay"], cwd, dirs_exist_ok=True)
    for rel, text in (beh.get("writes") or {}).items():
        (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
        (cwd / rel).write_text(text)
    for command in beh.get("run") or []:
        import subprocess
        subprocess.run(command, shell=True, cwd=cwd, check=False, capture_output=True)
    if beh.get("memory_append") and mem_file:
        Path(mem_file).parent.mkdir(parents=True, exist_ok=True)
        with open(mem_file, "a") as fh:
            fh.write(beh["memory_append"] + "\n")
    if beh.get("home_note"):
        (home / ".claude").mkdir(parents=True, exist_ok=True)
        with open(home / ".claude" / "fake-notes.txt", "a") as fh:
            fh.write(prompt[:40] + "\n")

    emit({"type": "system", "subtype": "init", "session_id": session_id, "cwd": str(cwd),
          "model": argv[argv.index("--model") + 1] if "--model" in argv else "fake", "tools": ["Bash", "Edit"]})
    turns = int(beh.get("turns", 2))
    usage = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, **(beh.get("usage") or {})}
    errors = beh.get("tool_errors") or []
    for i in range(turns):
        tid = f"toolu_{i}"
        msg = {"id": f"msg_{i}", "type": "message", "role": "assistant", "model": "fake",
               "content": [{"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "./tools/check"}}],
               "usage": usage}
        emit({"type": "assistant", "message": msg, "session_id": session_id})
        emit({"type": "assistant", "message": msg, "session_id": session_id})  # duplicate block line
        is_err = i < len(errors)
        emit({"type": "user", "session_id": session_id, "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "is_error": is_err,
             "content": errors[i] if is_err else "ok"}]}})
        if beh.get("sleep"):
            time.sleep(float(beh["sleep"]))
    total = {k: v * turns for k, v in usage.items()}
    emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": turns,
          "duration_ms": 1, "result": "done", "session_id": session_id,
          "total_cost_usd": beh.get("cost", 0.01), "usage": total})
    return int(beh.get("exit", 0))


if __name__ == "__main__":
    sys.exit(main())
