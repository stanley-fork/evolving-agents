"""Views stamped with the memory version they came from (EVOLUTION step 6): the guard's baseline and the recipes record
the memory's version, and a stale one is reported by lint and on the status line, not discovered by a bad plan."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import guard, recipes as rcp  # noqa: E402
from director.memory import gate, version  # noqa: E402
from test_plugin import EDIT, project_with_memory  # noqa: E402

RECIPES = [{"id": "docs", "trigger": "committing a document",
            "steps": [{"do": "mirror it", "cite": "Docs#mirror"}]}]


def test_the_version_follows_the_pages_and_ignores_everything_else(tmp_path):
    mem = project_with_memory(tmp_path) / ".director" / "memory"
    v = version(mem)
    (mem / "recipes.json").write_text(json.dumps(RECIPES))
    rcp.stamp(mem)
    assert version(mem) == v                              # the recipes and their stamp are not pages
    assert guard.memory_hash(mem) == v                    # the baseline is stored under the same version
    (mem / "Docs.md").write_text((mem / "Docs.md").read_text() + "\n")
    assert version(mem) != v


def test_recipes_go_from_unstamped_to_fresh_to_stale_when_a_lesson_is_accepted(tmp_path):
    mem = project_with_memory(tmp_path) / ".director" / "memory"
    (mem / "recipes.json").write_text(json.dumps(RECIPES))
    f = rcp.freshness(mem)
    assert f["unstamped"] and not f["stale"]
    rcp.stamp(mem)
    f = rcp.freshness(mem)
    assert not f["unstamped"] and not f["stale"] and f["recipes_built_from"] == f["memory_version"]
    p = gate.propose(mem, [EDIT], rationale="r")
    gate.accept(mem, p["id"], True)
    assert (mem / "recipes.json").is_file()               # the accept carries the recipes into the new memory
    assert rcp.freshness(mem)["stale"]
    rcp.stamp(mem)
    assert not rcp.freshness(mem)["stale"]


def test_lint_and_the_status_line_report_stale_recipes(tmp_path):
    proj = project_with_memory(tmp_path)
    mem = proj / ".director" / "memory"
    (mem / "recipes.json").write_text(json.dumps(RECIPES))
    rcp.stamp(mem)
    (mem / "Docs.md").write_text((mem / "Docs.md").read_text() + "\n")      # the memory moves on
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(proj)}
    env.pop("DIRECTOR_MEMORY", None)
    msgs = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "recipes_lint", "arguments": {}}}]
    p = subprocess.run([sys.executable, str(ROOT / "director" / "mcp_server.py")], capture_output=True, text=True,
                       input="".join(json.dumps(m) + "\n" for m in msgs), env=env, check=True)
    lint = json.loads(json.loads(p.stdout.splitlines()[1])["result"]["content"][0]["text"])
    assert lint["stale"] and "run /director:recipes again" in lint["next"] and lint["problems"] == []
    s = subprocess.run([sys.executable, str(ROOT / "director" / "ui_state.py"), "status"], capture_output=True,
                       text=True, env=env, check=True)
    assert json.loads(s.stdout)["recipes_stale"] is True
    cli = subprocess.run([sys.executable, "-m", "director.recipes", "lint", str(mem)], cwd=ROOT, capture_output=True,
                         text=True)
    assert json.loads(cli.stdout)["stale"] is True
