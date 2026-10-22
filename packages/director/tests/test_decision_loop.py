"""The decision loop (docs/BRIEF-C1.md): the person's messages reach the journal by themselves, the status line and
the session say what a reflection still owes, Accept works through a linked memory, and the public guard refuses to
publish what is private."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import plugin_hook, public_guard  # noqa: E402
from director.memory import gate, journal  # noqa: E402
from director.memory.format import find, state_dir  # noqa: E402
from test_plugin import EDIT, project_with_memory  # noqa: E402

HOOK = ROOT / "director" / "plugin_hook.py"
GUARD = ROOT / "director" / "public_guard.py"


def run_hook(script: Path, event: dict, env: dict | None = None) -> str:
    e = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PROJECT_DIR", "DIRECTOR_MEMORY")}
    return subprocess.run([sys.executable, str(script)], input=json.dumps(event), capture_output=True, text=True,
                          env={**e, "HOME": str(Path(__file__).parent / "no-home"), **(env or {})},
                          timeout=60).stdout


# ------------------------------------------------------------------ capture, owed, nudge

def test_every_message_reaches_the_journal_with_keys_scrubbed(tmp_path):
    proj = project_with_memory(tmp_path)
    key = "AQ." + "x" * 40
    out = run_hook(HOOK, {"hook_event_name": "UserPromptSubmit", "cwd": str(proj),
                          "prompt": f"no gastes más de 50 dólares con esta key {key}"})
    assert out == ""                                  # capture says nothing to the session
    [e] = journal.read(state_dir(find(proj)))
    assert e["kind"] == "person" and "50 dólares" in e["text"] and key not in e["text"] and "[SECRET]" in e["text"]


def test_no_memory_no_capture_and_system_text_is_skipped(tmp_path):
    assert plugin_hook.capture(tmp_path, "hola") is False
    proj = project_with_memory(tmp_path / "p")
    assert plugin_hook.capture(proj, "<system-reminder>x</system-reminder>") is False
    assert plugin_hook.capture(proj, "   ") is False
    assert journal.read(state_dir(find(proj))) == []


def test_what_a_reflection_owes_is_counted_and_said_at_session_start(tmp_path):
    proj = project_with_memory(tmp_path)
    st = state_dir(find(proj))
    for t in ("usá gemma4:12b", "la UI la hace Claude Code"):
        plugin_hook.capture(proj, t)
    journal.append(st, "outcome", text="ran")             # not owed: an outcome is not the person's
    assert journal.owed(st) == 2
    ctx = json.loads(run_hook(HOOK, {"hook_event_name": "SessionStart", "cwd": str(proj)}))
    assert "2 messages or corrections since the last reflection" in ctx["hookSpecificOutput"]["additionalContext"]
    journal.reflected(st, "learned both")
    assert journal.owed(st) == 0 and plugin_hook.nudge(find(proj)) == ""


def test_the_status_snapshot_carries_what_is_owed(tmp_path):
    proj = project_with_memory(tmp_path)
    plugin_hook.capture(proj, "from now on, PRs only")
    r = subprocess.run([sys.executable, str(ROOT / "director" / "ui_state.py"), "status"], cwd=proj,
                       capture_output=True, text=True, timeout=60,
                       env={k: v for k, v in os.environ.items() if k != "DIRECTOR_MEMORY"} | {"CLAUDE_PROJECT_DIR": str(proj)})
    assert json.loads(r.stdout)["to_reflect"] == 1


# ------------------------------------------------------------------ Accept through a linked memory

def test_accept_replaces_a_linked_memory_at_its_target_and_keeps_the_link(tmp_path):
    proj = project_with_memory(tmp_path / "proj")
    home = tmp_path / "private" / "memory"
    home.parent.mkdir()
    (proj / ".director" / "memory").rename(home)
    (proj / ".director" / "memory").symlink_to(home)
    mem = find(proj)
    prop = gate.propose(mem, [EDIT], rationale="r")
    res = gate.accept(mem, prop["id"], user_said_yes=True)
    assert res["status"] == "accepted"
    assert (proj / ".director" / "memory").is_symlink()
    assert "## #scrub_paths" in (home / "Docs.md").read_text()
    assert "## #scrub_paths" not in (Path(res["previous_memory"]) / "Docs.md").read_text()


# ------------------------------------------------------------------ a proposal made before another was accepted

OTHER = {**EDIT, "id": "two_estimates", "when": "a centre can be estimated two ways", "do": "compute both"}


def test_a_proposal_older_than_an_accepted_one_cannot_undo_it(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [EDIT], rationale="old")
    new = gate.propose(mem, [EDIT, OTHER], rationale="new")
    gate.accept(mem, new["id"], user_said_yes=True)
    assert "stale" in gate.status(mem, old["id"])
    assert [w for w in gate.waiting(mem)[0] if w["id"] == old["id"]][0]["acceptable"] is False
    try:
        gate.accept(mem, old["id"], user_said_yes=True)
        raise AssertionError("a stale proposal was accepted")
    except gate.GateError as e:
        assert "propose again" in str(e)
    assert "## #two_estimates" in (mem / "Docs.md").read_text()
    assert gate.reject(mem, old["id"])["status"] == "rejected"


def test_a_proposal_without_a_base_is_stale_once_a_later_one_is_accepted(tmp_path):
    mem = find(project_with_memory(tmp_path))
    old = gate.propose(mem, [EDIT], rationale="made before the base existed")
    d = gate.proposal_dir(mem, old["id"])
    rec = json.loads((d / "proposal.json").read_text())
    del rec["base"]
    (d / "proposal.json").write_text(json.dumps(rec))
    assert gate.stale(mem, rec) == ""
    new = gate.propose(mem, [OTHER], rationale="new")
    gate.accept(mem, new["id"], user_said_yes=True)
    assert new["id"] in gate.stale(mem, rec)


def test_memory_reject_is_a_tool(tmp_path, monkeypatch):
    from director import mcp_server
    proj = project_with_memory(tmp_path)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(proj))
    pid = gate.propose(find(proj), [EDIT], rationale="r")["id"]
    assert mcp_server.TOOLS["memory_reject"][0]({"id": pid}) == {"id": pid, "status": "rejected"}


# ------------------------------------------------------------------ the public guard

def repo_with(tmp: Path, remote: str, staged: str) -> Path:
    r = tmp / "repo"
    r.mkdir()
    for args in (["init", "-q"], ["remote", "add", "origin", remote], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(r), *args], check=True, capture_output=True)
    (r / "f.md").write_text(staged)
    subprocess.run(["git", "-C", str(r), "add", "f.md"], check=True, capture_output=True)
    return r


def guard_env(tmp: Path, private: list[str] | None = None) -> dict:
    (tmp / "terms.json").write_text(json.dumps({"terms": ["Zorblax"]}))
    (tmp / "guard.json").write_text(json.dumps({"terms_file": str(tmp / "terms.json"),
                                                "private_remotes": private or []}))
    return {"DIRECTOR_PUBLIC_GUARD": str(tmp / "guard.json")}


def decide(tmp: Path, repo: Path, command: str, private: list[str] | None = None) -> dict | None:
    out = run_hook(GUARD, {"tool_name": "Bash", "cwd": str(repo), "tool_input": {"command": command}},
                   guard_env(tmp, private))
    return json.loads(out)["hookSpecificOutput"] if out.strip() else None


def test_a_private_term_in_the_diff_is_refused_for_a_public_repo(tmp_path):
    r = repo_with(tmp_path, "https://github.com/acme/public.git", "the Zorblax lane\n")
    d = decide(tmp_path, r, 'git commit -m "docs"')
    assert d and d["permissionDecision"] == "deny" and "acme/public" in d["permissionDecisionReason"]


def test_a_private_repo_passes_and_a_clean_diff_passes(tmp_path):
    r = repo_with(tmp_path, "git@github.com:acme/secret.git", "the Zorblax lane\n")
    assert decide(tmp_path, r, 'git commit -m "docs"', private=["acme/secret"]) is None
    (r / "f.md").write_text("nothing here\n")
    subprocess.run(["git", "-C", str(r), "add", "f.md"], check=True, capture_output=True)
    assert decide(tmp_path, r, 'git commit -m "docs"') is None


def test_the_command_line_itself_is_not_scanned_only_what_it_publishes(tmp_path):
    r = repo_with(tmp_path, "https://github.com/acme/public.git", "clean\n")
    check = "git diff --cached | grep -qiE 'Zorblax' && exit 1; git commit -m \"docs\""
    assert decide(tmp_path, r, check) is None


def test_messages_titles_bodies_and_heredocs_are_scanned(tmp_path):
    r = repo_with(tmp_path, "https://github.com/acme/public.git", "clean\n")
    assert decide(tmp_path, r, 'git commit -m "the Zorblax run"')["permissionDecision"] == "deny"
    assert decide(tmp_path, r, "git commit -q -F - <<'EOF'\nthe Zorblax run\n\nmore\nEOF")["permissionDecision"] == "deny"
    assert decide(tmp_path, r, 'gh pr create --title t --body "about Zorblax"')["permissionDecision"] == "deny"
    assert decide(tmp_path, r, "gh pr create --title t --body \"key AIza" + "y" * 30 + "\"")["permissionDecision"] == "deny"
    assert decide(tmp_path, r, 'gh pr create --title t --body "all public"') is None


def test_without_a_config_the_guard_is_silent(tmp_path):
    r = repo_with(tmp_path, "https://github.com/acme/public.git", "the Zorblax lane\n")
    out = run_hook(GUARD, {"tool_name": "Bash", "cwd": str(r), "tool_input": {"command": 'git commit -m x'}},
                   {"DIRECTOR_PUBLIC_GUARD": str(tmp_path / "missing.json")})
    assert out.strip() == ""
    assert public_guard.decide({"tool_name": "Read", "tool_input": {}}) is None


# ------------------------------------------------------------------ a session running an older copy than installed

def test_session_start_says_when_an_older_director_is_running(tmp_path):
    def plugin(v):
        r = tmp_path / v
        (r / ".claude-plugin").mkdir(parents=True)
        (r / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "director", "version": v}))
        return r
    old, new = plugin("0.2.1"), plugin("0.2.2")
    reg = tmp_path / "installed_plugins.json"
    reg.write_text(json.dumps({"version": 2, "plugins": {"director@lab": [
        {"scope": "user", "installPath": str(new), "version": "0.2.2"}]}}))
    assert "0.2.2 is installed" in plugin_hook.stale_install(old, reg)
    assert plugin_hook.stale_install(new, reg) == ""
    assert plugin_hook.stale_install(old, tmp_path / "missing.json") == ""
