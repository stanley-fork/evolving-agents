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
    assert "in `.director/memory/`, as pages" in text and "`.director/memory/README.md`" in text
    assert ".director/.director" not in text


def test_an_absolute_memory_path_is_named_once(tmp_path):
    m = tmp_path / "elsewhere" / "memory"
    m.mkdir(parents=True)
    (m / "Index.md").write_text("# Index\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    text = json.loads(hook(proj, {"DIRECTOR_MEMORY": str(m)}))["hookSpecificOutput"]["additionalContext"]
    assert f"in `{m}/`, as pages" in text and f"`{m}/Index.md`" in text and f"{m}/{m}" not in text


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


# ------------------------------------------------------------------ the MCP server

AGENTVCS = os.environ.get("AGENTVCS") or __import__("shutil").which("agentvcs")


def mcp(proj: Path, *calls: tuple[str, dict], agentvcs: str | None = None) -> list[dict]:
    msgs = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"}]
    msgs += [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": n, "arguments": a}}
             for i, (n, a) in enumerate(calls, 1)]
    e = {k: v for k, v in os.environ.items() if k not in ("DIRECTOR_MEMORY", "AGENTVCS")}
    e.update(CLAUDE_PROJECT_DIR=str(proj), PATH="/usr/bin:/bin")     # no agentvcs on PATH: unversioned accept
    if agentvcs:
        e["AGENTVCS"] = agentvcs
    p = subprocess.run([sys.executable, str(ROOT / "director" / "mcp_server.py")], capture_output=True, text=True,
                       input="".join(json.dumps(m) + "\n" for m in msgs), env=e, check=True)
    out = [json.loads(l) for l in p.stdout.splitlines()]
    assert out[0]["result"]["serverInfo"]["name"] == "director"
    return [json.loads(r["result"]["content"][0]["text"]) for r in out[1:]]


def project_with_memory(tmp: Path) -> Path:
    m = tmp / ".director" / "memory"
    m.mkdir(parents=True)
    (m / "Index.md").write_text("# Index\n- about to commit a document → [[Docs#mirror]]\n")
    (m / "Docs.md").write_text("# Docs\nDocs.\n\n## #mirror\n**When:** a\n**Do:** b\n**Because:** c\n**Links:** \n")
    return tmp


EDIT = {"op": "add", "page": "Docs", "id": "scrub_paths", "when": "an image of a terminal is about to be published",
        "do": "crop or blur home paths", "because": "a screenshot showed one", "links": ["Docs#mirror"],
        "situations": ["about to commit a document"]}


def test_propose_touches_only_a_candidate_and_accept_needs_the_users_yes(tmp_path):
    proj = project_with_memory(tmp_path)
    live = proj / ".director" / "memory" / "Docs.md"
    before = live.read_text()
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "r"}))
    assert prop["ok"] and prop["applied"] == ["add Docs#scrub_paths"] and prop["lint"] == []
    assert "+## #scrub_paths" in prop["diff"] and live.read_text() == before
    no, yes, again = mcp(proj, ("memory_accept", {"id": prop["id"], "user_said_yes": False}),
                         ("memory_accept", {"id": prop["id"], "user_said_yes": True}),
                         ("memory_accept", {"id": prop["id"], "user_said_yes": True}))
    assert not no["ok"] and yes["ok"] and yes["versioned"] is False and not again["ok"]
    assert "## #scrub_paths" in live.read_text()
    assert (Path(yes["previous_memory"]) / "Docs.md").read_text() == before


def test_a_lint_problem_cannot_be_accepted(tmp_path):
    proj = project_with_memory(tmp_path)
    bad = {**EDIT, "links": ["Nowhere#x"]}
    [prop] = mcp(proj, ("memory_propose", {"edits": [bad], "rationale": "r"}))
    assert prop["lint"] and "cannot be accepted" in prop["next"]
    [acc] = mcp(proj, ("memory_accept", {"id": prop["id"], "user_said_yes": True}))
    assert not acc["ok"]


def test_a_check_that_did_not_teach_refuses_and_an_incomplete_one_decides_nothing(tmp_path):
    proj = project_with_memory(tmp_path)
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "r"}))
    d = proj / ".director" / "candidates" / prop["id"]
    (d / "check.json").write_text(json.dumps({"verdict": "INCOMPLETE", "teaches": False}))
    [acc] = mcp(proj, ("memory_accept", {"id": prop["id"], "user_said_yes": True}))
    assert not acc["ok"] and json.loads((d / "proposal.json").read_text())["status"] == "proposed"
    (d / "check.json").write_text(json.dumps({"verdict": "REDUNDANT", "teaches": False}))
    [acc] = mcp(proj, ("memory_accept", {"id": prop["id"], "user_said_yes": True}))
    assert not acc["ok"] and "REDUNDANT" in acc["error"]
    assert "scrub_paths" not in (proj / ".director" / "memory" / "Docs.md").read_text()


def test_without_a_memory_the_tools_say_how_to_get_one(tmp_path):
    [r] = mcp(tmp_path, ("memory_lint", {}))
    assert not r["ok"] and "/director:compile" in r["error"]


def test_the_accept_gate_reads_a_probe_only_check(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"dimensions": {
        "memory.index": {"content": {"template": "# Index\n- s → [[Docs#mirror]]\n"}},
        "memory.docs": {"content": {"template": "# Docs\nD.\n\n## #mirror\n**When:** a\n**Do:** b\n**Because:** c\n**Links:** \n"}}}}))
    g = tmp_path / "guard.json"
    g.write_text(json.dumps({"mode": "probe", "passed": True, "probe": {"verdict": "TEACHES", "teaches": True}}))
    p = subprocess.run([sys.executable, str(ROOT / "director" / "learn_gate.py"), str(g)], capture_output=True, text=True,
                       env={**os.environ, "AGENTVCS_MANIFEST_FILE": str(man)}, check=True)
    assert json.loads(p.stdout)["metrics"] == {"lint_problems": 0, "human_accepted": 1, "guard_passed": 1, "teaches": 1}


@__import__("pytest").mark.skipif(not AGENTVCS, reason="agentvcs CLI not available")
def test_with_agentvcs_an_accept_is_a_gated_applied_patch(tmp_path):
    proj = project_with_memory(tmp_path)
    [prop] = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "scrub paths"}), agentvcs=AGENTVCS)
    [acc] = mcp(proj, ("memory_accept", {"id": prop["id"], "user_said_yes": True}), agentvcs=AGENTVCS)
    assert acc["ok"] and acc["versioned"], acc
    assert acc["patch"]["dimensions"] == ["memory.docs", "memory.index"]
    assert acc["patch"]["gate"]["lint_problems"] == 0 and acc["patch"]["gate"]["human_accepted"] == 1


def test_two_proposals_in_the_same_second_keep_both(tmp_path):
    proj = project_with_memory(tmp_path)
    other = {**EDIT, "id": "second_lesson"}
    a, b = mcp(proj, ("memory_propose", {"edits": [EDIT], "rationale": "one"}),
               ("memory_propose", {"edits": [other], "rationale": "two"}))
    assert a["id"] != b["id"]
    cands = proj / ".director" / "candidates"
    assert json.loads((cands / a["id"] / "proposal.json").read_text())["rationale"] == "one"
    assert json.loads((cands / b["id"] / "proposal.json").read_text())["rationale"] == "two"
    assert "## #scrub_paths" in (cands / a["id"] / "memory" / "Docs.md").read_text()
    assert "## #second_lesson" in (cands / b["id"] / "memory" / "Docs.md").read_text()


def test_new_proposal_ids_skip_a_taken_one(tmp_path, monkeypatch):
    from director import mcp_server
    monkeypatch.setattr(mcp_server.time, "strftime", lambda fmt: "p20261018-120000")
    ids = [mcp_server.new_proposal(tmp_path)[0] for _ in range(3)]
    assert ids == ["p20261018-120000", "p20261018-120000-2", "p20261018-120000-3"]
