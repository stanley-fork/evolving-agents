"""0.2.4: a stale proposal is rebased onto the live memory when nothing it touches changed since it was made (and still
refused when something did); a situation that is a whole Index line is refused at propose; and a director updated
under a running session is announced at the next prompt, once per installed version."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import plugin_hook  # noqa: E402
from director.memory import gate, journal  # noqa: E402
from director.memory.format import find, state_dir  # noqa: E402
from test_plugin import EDIT, mcp, project_with_memory  # noqa: E402

# a lesson on another page, reached from another situation: touches nothing EDIT touches
ELSEWHERE = {"op": "add", "page": "Shell", "id": "quote_paths", "about": "Shell commands.",
             "when": "a path with spaces goes into a command", "do": "quote it", "because": "a copy went astray",
             "links": [], "situations": ["about to run a shell command"]}


# ------------------------------------------------------------------ 1. rebase

def test_a_stale_proposal_whose_statements_did_not_change_is_rebased_and_accepted(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [EDIT], rationale="old")
    other = gate.propose(mem, [ELSEWHERE], rationale="other")
    gate.accept(mem, other["id"], user_said_yes=True)
    rec = json.loads((gate.proposal_dir(mem, old["id"]) / "proposal.json").read_text())
    assert gate.stale(mem, rec)                                   # stale by the fingerprint, as before
    assert [w for w in gate.waiting(mem)[0] if w["id"] == old["id"]][0]["acceptable"] is True

    st = gate.status(mem, old["id"])
    assert st["rebased_from"] == rec["base"] and st["rebased_onto"] == gate.fingerprint(mem)
    assert st["accepted_since"] == [other["id"]] and "stale" not in st and "conflict" not in st
    assert "+## #scrub_paths" in st["diff"]
    changed = [l for l in st["diff"].splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    assert not any("quote_paths" in l for l in changed)            # the other lesson is the base now, not undone

    res = gate.accept(mem, old["id"], user_said_yes=True)
    assert res["status"] == "accepted" and res["rebased_from"] == rec["base"]
    assert "## #quote_paths" in (mem / "Shell.md").read_text()    # the lesson accepted in between survives
    assert "## #scrub_paths" in (mem / "Docs.md").read_text()
    index = (mem / "Index.md").read_text()
    assert "[[Shell#quote_paths]]" in index and "[[Docs#scrub_paths]]" in index
    assert [e["kind"] for e in journal.read(state_dir(mem))].count("lesson.rebased") == 1


def test_accept_rebases_by_itself_without_a_status_first(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [EDIT], rationale="old")
    gate.accept(mem, gate.propose(mem, [ELSEWHERE], rationale="other")["id"], user_said_yes=True)
    assert gate.accept(mem, old["id"], user_said_yes=True)["rebased_from"]
    assert "## #quote_paths" in (mem / "Shell.md").read_text()


def test_a_stale_proposal_whose_statement_changed_keeps_being_refused(tmp_path):
    mem = find(project_with_memory(tmp_path))
    mod = {"op": "modify", "page": "Docs", "id": "mirror", "do": "mirror it in the same commit"}
    old = gate.propose(mem, [mod], rationale="old")
    gate.accept(mem, gate.propose(mem, [{**mod, "do": "mirror it before the PR"}], rationale="new")["id"],
                user_said_yes=True)
    st = gate.status(mem, old["id"])
    assert st["conflict"] == ["Docs#mirror"] and "propose again" in st["stale"] and "rebased_from" not in st
    assert [w for w in gate.waiting(mem)[0] if w["id"] == old["id"]][0]["acceptable"] is False
    try:
        gate.accept(mem, old["id"], user_said_yes=True)
        raise AssertionError("a conflicting stale proposal was accepted")
    except gate.GateError as e:
        assert "Docs#mirror" in str(e)
    assert "before the PR" in (mem / "Docs.md").read_text()


def test_a_changed_index_line_it_touches_is_a_conflict(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [ELSEWHERE], rationale="old")
    # another lesson reaches the same situation line first: that line changed
    third = {**ELSEWHERE, "page": "Docs", "id": "third", "about": None}
    gate.accept(mem, gate.propose(mem, [third], rationale="new")["id"], user_said_yes=True)
    st = gate.status(mem, old["id"])
    assert st["conflict"] == ["Index: about to run a shell command"]
    assert gate.waiting(mem)[0][0]["acceptable"] is False


def test_memory_status_reports_the_rebase_through_the_tool(tmp_path):
    proj = project_with_memory(tmp_path)
    mem = find(proj)
    old = gate.propose(mem, [EDIT], rationale="old")
    gate.accept(mem, gate.propose(mem, [ELSEWHERE], rationale="other")["id"], user_said_yes=True)
    st, acc = mcp(proj, ("memory_status", {"id": old["id"]}),
                  ("memory_accept", {"id": old["id"], "user_said_yes": True}))
    assert st["ok"] and st["rebased_from"] and acc["ok"] and acc["status"] == "accepted"


# ------------------------------------------------------------------ 2. situations are situation text

def test_a_situation_that_is_an_index_line_is_refused(tmp_path):
    mem = find(project_with_memory(tmp_path))
    for bad in ("about to commit a document → [[Docs#mirror]]", "about to commit [[Docs#mirror]]"):
        prop = gate.propose(mem, [{**EDIT, "situations": [bad]}], rationale="r")
        assert prop["applied"] == [] and "situation text only" in prop["rejected"][0]["why"]
        assert prop["lint"] and "situation text only" in prop["lint"][0]
        try:
            gate.accept(mem, prop["id"], user_said_yes=True)
            raise AssertionError("an Index line passed as a situation was accepted")
        except gate.GateError:
            pass
    assert (mem / "Index.md").read_text().count("→") == 1           # no broken duplicate line was written


def test_the_situation_refusal_blocks_the_whole_proposal(tmp_path):
    mem = find(project_with_memory(tmp_path))
    prop = gate.propose(mem, [ELSEWHERE, {**EDIT, "situations": ["x → [[Docs#mirror]]"]}], rationale="r")
    assert prop["applied"] == ["add Shell#quote_paths"] and prop["lint"]
    assert gate.waiting(mem)[0][0]["acceptable"] is False


# ------------------------------------------------------------------ 3. the stale-install warning at the prompt

def _plugins(tmp_path: Path, running: str, installed: str) -> tuple[Path, Path]:
    def plugin(v):
        r = tmp_path / "cache" / v
        (r / ".claude-plugin").mkdir(parents=True, exist_ok=True)
        (r / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "director", "version": v}))
        return r
    root, inst = plugin(running), plugin(installed)
    reg = tmp_path / "installed_plugins.json"
    reg.write_text(json.dumps({"version": 2, "plugins": {"director@lab": [
        {"scope": "user", "installPath": str(inst), "version": installed}]}}))
    return root, reg


def test_the_prompt_warns_once_per_installed_version_per_session(tmp_path):
    proj = project_with_memory(tmp_path / "p")
    root, reg = _plugins(tmp_path, "0.2.3", "0.2.4")
    first = plugin_hook.prompt_warning(proj, "s1", root, reg)
    assert "0.2.4 is installed" in first and "restart Claude Code" in first and "claude --resume" in first
    assert plugin_hook.prompt_warning(proj, "s1", root, reg) == ""          # not on every prompt
    assert "0.2.4" in plugin_hook.prompt_warning(proj, "s2", root, reg)     # another session is told too
    _, reg5 = _plugins(tmp_path, "0.2.3", "0.2.5")
    assert "0.2.5 is installed" in plugin_hook.prompt_warning(proj, "s1", root, reg5)   # a newer install: once more
    root_ok, reg_ok = _plugins(tmp_path / "same", "0.2.4", "0.2.4")
    assert plugin_hook.prompt_warning(proj, "s3", root_ok, reg_ok) == ""   # running what is installed: silent
    assert plugin_hook.prompt_warning(tmp_path / "nomem", "s4", root, reg) == ""


def test_the_hook_says_it_to_the_person_at_the_prompt_and_still_journals(tmp_path):
    proj = project_with_memory(tmp_path / "p")
    root, _ = _plugins(tmp_path, "0.2.3", "0.2.4")
    home = tmp_path / "home"
    (home / ".claude" / "plugins").mkdir(parents=True)
    (tmp_path / "installed_plugins.json").rename(home / ".claude" / "plugins" / "installed_plugins.json")
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PROJECT_DIR", "DIRECTOR_MEMORY")}
    env.update(HOME=str(home), CLAUDE_PLUGIN_ROOT=str(root))

    def prompt(text):
        ev = {"hook_event_name": "UserPromptSubmit", "cwd": str(proj), "session_id": "abc", "prompt": text}
        return subprocess.run([sys.executable, str(ROOT / "director" / "plugin_hook.py")], input=json.dumps(ev),
                              capture_output=True, text=True, env=env, timeout=60).stdout

    out = json.loads(prompt("first"))
    assert "restart Claude Code" in out["systemMessage"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert prompt("second") == ""
    assert [e["text"] for e in journal.read(state_dir(find(proj)))] == ["first", "second"]


def test_session_start_warns_the_person_and_the_first_prompt_does_not_repeat_it(tmp_path):
    proj = project_with_memory(tmp_path / "p")
    root, reg = _plugins(tmp_path, "0.2.3", "0.2.4")
    home = tmp_path / "home"
    (home / ".claude" / "plugins").mkdir(parents=True)
    reg.rename(home / ".claude" / "plugins" / "installed_plugins.json")
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PROJECT_DIR", "DIRECTOR_MEMORY")}
    env.update(HOME=str(home), CLAUDE_PLUGIN_ROOT=str(root))
    run = lambda ev: subprocess.run([sys.executable, str(ROOT / "director" / "plugin_hook.py")],
                                    input=json.dumps({"cwd": str(proj), "session_id": "abc", **ev}),
                                    capture_output=True, text=True, env=env, timeout=60).stdout
    start = json.loads(run({"hook_event_name": "SessionStart"}))
    assert "claude --resume" in start["systemMessage"]
    assert "0.2.4 is installed" in start["hookSpecificOutput"]["additionalContext"]
    assert run({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}) == ""


def test_a_proposal_whose_check_is_running_is_not_rebased_under_it(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [EDIT], rationale="old")
    d = gate.proposal_dir(mem, old["id"])
    (d / "check.running").write_text("now")
    gate.accept(mem, gate.propose(mem, [ELSEWHERE], rationale="other")["id"], user_said_yes=True)
    before = (d / "memory" / "Index.md").read_text()
    st = gate.status(mem, old["id"])
    assert "check is still running" in st["stale"] and "rebased_from" not in st
    assert (d / "memory" / "Index.md").read_text() == before
    (d / "check.running").unlink()
    assert gate.status(mem, old["id"])["rebased_from"]
