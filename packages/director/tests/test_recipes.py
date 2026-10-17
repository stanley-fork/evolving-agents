"""Meta-recipes (director/recipes.py): the code runs the recipe, the model only picks and answers yes/no (D15)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director import recipes as rcp  # noqa: E402
from director import mcp_server  # noqa: E402

RECIPES = [
    {"id": "merge_pr", "trigger": "Committing, opening or merging a pull request",
     "steps": [
         {"do": "Merge only with this session's authorisation and green checks.", "cite": "Git#merge_is_users_call"},
         {"ask": "Does the request say a CI check failed?",
          "yes": [{"do": "Read the failing check's log first.", "cite": "Git#read_failing_log"}],
          "no": [{"do": "Wait for gh pr checks to exit 0.", "cite": "Git#checks_green"}]},
         {"do": "Merge only with this session's authorisation and green checks.", "cite": "Git#merge_is_users_call"},
     ]},
    {"id": "launch_run", "trigger": "Launching a benchmark or training run",
     "steps": [{"do": "Write the BRIEF first.", "cite": "Runs#brief_first"}]},
]


def memory(tmp: Path) -> Path:
    m = tmp / "memory"
    m.mkdir()
    (m / "Index.md").write_text("# Index\n- merging → [[Git#merge_is_users_call]]\n")
    (m / "Git.md").write_text("# Git\n\n## #merge_is_users_call\n**When:** x\n\n## #read_failing_log\n**When:** x\n\n"
                              "## #checks_green\n**When:** x\n")
    (m / "Runs.md").write_text("# Runs\n\n## #brief_first\n**When:** x\n")
    (m / "recipes.json").write_text(json.dumps(RECIPES))
    return m


class Scripted:
    def __init__(self, picks, answers):
        self.picks, self.answers, self.asked = picks, list(answers), []

    def pick(self, request, triggers):
        return self.picks

    def yes_no(self, request, question):
        self.asked.append(question)
        return self.answers.pop(0)


def test_the_branch_the_model_answers_is_the_one_taken_and_text_is_the_recipes():
    out = rcp.plan("PR's check FAILED, merge it", RECIPES, Scripted([0], [True]))
    assert out["recipes"] == ["merge_pr"]
    assert out["answers"] == [{"ask": "Does the request say a CI check failed?", "answer": "yes"}]
    assert out["plan"].splitlines() == [
        "1. Merge only with this session's authorisation and green checks. [[Git#merge_is_users_call]]",
        "2. Read the failing check's log first. [[Git#read_failing_log]]"]  # the repeated step appears once


def test_the_other_branch_and_two_recipes():
    out = rcp.plan("merge and then launch", RECIPES, Scripted([0, 1], [False]))
    assert out["cites"] == ["Git#merge_is_users_call", "Git#checks_green", "Runs#brief_first"]


def test_no_recipe_picked_is_an_abstention():
    out = rcp.plan("make me a sandwich", RECIPES, Scripted([], []))
    assert out["abstain"] and out["plan"] == ""


def test_lint_is_clean_on_a_good_file_and_catches_bad_cites_and_depth(tmp_path):
    m = memory(tmp_path)
    known = rcp.statement_ids(m)
    assert rcp.lint(RECIPES, known) == []
    deep = {"ask": "a?", "yes": [{"ask": "b?", "yes": [{"ask": "c?", "yes": [], "no": []}], "no": []}], "no": []}
    bad = [{"id": "x", "trigger": "t", "steps": [{"do": "y", "cite": "Git#nope"}, {"do": "z"}, deep]}]
    problems = rcp.lint(bad, known)
    assert any("does not exist: Git#nope" in p for p in problems)
    assert any("without cite" in p for p in problems)
    assert any("nested deeper" in p for p in problems)
    assert rcp.cited(RECIPES) == known


def test_the_mcp_tools_list_and_lint(tmp_path, monkeypatch):
    m = memory(tmp_path)
    monkeypatch.setenv("DIRECTOR_MEMORY", str(m))
    names = {t["name"] for t in mcp_server.handle({"method": "tools/list", "id": 1})["result"]["tools"]}
    assert {"recipes_lint", "plan_local"} <= names
    r = mcp_server.handle({"method": "tools/call", "id": 2, "params": {"name": "recipes_lint", "arguments": {}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["ok"] and payload["problems"] == [] and payload["statements_cited"] == payload["statements"] == 4


def test_plan_local_uses_the_answerer(tmp_path, monkeypatch):
    m = memory(tmp_path)
    monkeypatch.setenv("DIRECTOR_MEMORY", str(m))
    monkeypatch.setattr(rcp, "Ollama", lambda model: type("F", (Scripted,), {"model": model, "url": "-", "calls": 2,
                                                                              "failures": 0})([1], []))
    r = mcp_server.handle({"method": "tools/call", "id": 3, "params": {"name": "plan_local",
                                                                       "arguments": {"request": "launch a run"}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["ok"] and payload["recipes"] == ["launch_run"] and "[[Runs#brief_first]]" in payload["plan"]
