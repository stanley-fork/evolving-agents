"""The memory's boundary (docs/EVOLUTION.md §4 and §8, step 1): `director.memory` is the memory's own code — format,
editing, lint, the gate, versioning — and it never imports the agent's side (planning, reflection, lanes, the
interfaces), so it never calls a model. The agent's modules reach the memory only through it."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "director" / "memory"
AGENT = {"core", "learn", "lanes", "guard", "recipes", "server", "compile", "mcp_server", "ui_state", "plugin_hook",
         "plugin_check", "learn_gate"}


def test_the_memory_package_imports_nothing_from_the_agent():
    for f in MEMORY.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level >= 2 or mod.startswith("director."):   # `from ..x` or `from director.x`
                    name = mod.split(".")[-1] if node.level >= 2 else mod.split(".")[1]
                    assert name == "memory" or name not in AGENT, f"{f.name} imports {mod}"
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("director.") or a.name.startswith("director.memory"), f.name


def test_importing_the_memory_loads_no_agent_module():
    code = ("import sys; sys.path.insert(0, %r); import director.memory, director.memory.gate, director.memory.vcs; "
            "import json; print(json.dumps(sorted(m for m in sys.modules if m.startswith('director'))))" % str(ROOT))
    loaded = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)
    assert all(m == "director" or m.startswith("director.memory") for m in loaded), loaded


def test_the_old_names_still_point_at_the_memory():
    sys.path.insert(0, str(ROOT))
    from director import learn, memory, memory_vcs, recipes
    from director.memory import gate, vcs
    assert learn.apply_edits is memory.apply_edits and learn.lint is memory.lint and learn.leak is gate.leak
    assert memory_vcs.LiveStore is vcs.LiveStore and recipes.statement_ids is memory.statement_ids
    assert vcs.GATE.is_file()
