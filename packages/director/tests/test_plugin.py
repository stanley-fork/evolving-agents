"""The plugin's wiring: the SessionStart hook puts DIRECTOR.md into the context only where a memory exists (P0 measures
that channel, docs/BRIEF-P0.md), and the manifest and hooks point at files that exist."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director.plugin_hook import context  # noqa: E402

DIRECTOR_MD = (ROOT / "director" / "DIRECTOR.md").read_text()


def hook(cwd: Path, env: dict | None = None) -> str:
    e = {k: v for k, v in os.environ.items() if k not in ("DIRECTOR_MEMORY", "CLAUDE_PROJECT_DIR")}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(ROOT / "director" / "plugin_hook.py")], input=json.dumps({"cwd": str(cwd)}),
                       capture_output=True, text=True, env=e, check=True)
    return p.stdout


def test_the_hook_is_silent_without_a_memory(tmp_path):
    assert hook(tmp_path) == ""


def test_with_memory_at_dot_memory_the_text_is_director_md_byte_for_byte(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "Index.md").write_text("# Index\n")
    out = json.loads(hook(tmp_path))["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    assert out["additionalContext"] == DIRECTOR_MD


def test_a_dot_director_memory_wins_and_the_paths_follow_it(tmp_path):
    for d in ("memory", ".director/memory"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "Index.md").write_text("# Index\n")
    text = context(tmp_path)
    assert "`.director/memory/Index.md`" in text and "`./memory/`" not in text


def test_director_memory_env_points_anywhere(tmp_path):
    m = tmp_path / "elsewhere"
    m.mkdir()
    (m / "Index.md").write_text("# Index\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    assert "elsewhere" in json.loads(hook(proj, {"DIRECTOR_MEMORY": str(m)}))["hookSpecificOutput"]["additionalContext"]


def test_manifest_and_hooks_point_at_files_that_exist():
    assert json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())["name"] == "director"
    hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    for groups in hooks.values():
        for g in groups:
            for h in g["hooks"]:
                rel = h["command"].split('"${CLAUDE_PLUGIN_ROOT}"/')[1].split()[0]
                assert (ROOT / rel).is_file(), rel
