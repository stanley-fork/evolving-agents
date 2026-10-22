"""End to end through the CLI: C0 on ledgerkit with the fake claude, then report and audit."""
import json
import sys

from conftest import FAKE, LEDGERKIT, read_jsonl
from runner.cli import main
from runner.config import load_family

FAM = load_family(LEDGERKIT)
REPEATED = "Exit code 1\nModuleNotFoundError: No module named 'ledgerkit._compat'"


def smoke_spec(fail=("03",), fix_on_review=("07",)):
    rules = []
    for t in FAM.tasks:
        first_line = FAM.statement(t).splitlines()[0]
        rule = {"match": first_line, "turns": 4, "tool_errors": [REPEATED],
                "usage": {"input_tokens": 1200, "output_tokens": 300, "cache_creation_input_tokens": 500,
                          "cache_read_input_tokens": 8000}}
        solution = {"overlay": str(LEDGERKIT / "reference" / t), "run": [f"{sys.executable} tools/regen"]}
        if t in fix_on_review:   # first attempt: only the code a reader of the repo would write
            mods = sorted((LEDGERKIT / "reference" / t / "src/ledgerkit/commands").glob("*.py"))
            rule["writes"] = {f"src/ledgerkit/commands/{m.name}": m.read_text() for m in mods}
            rule["run"] = solution["run"]
            rule["resume"] = {"turns": 2, **solution}
        elif t not in fail:
            rule.update(solution)
        rules.append(rule)
    return {"rules": rules}


def test_cli_smoke_c0(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(smoke_spec()))
    results = tmp_path / "results"
    rc = main(["--results", str(results), "run", "--family", "ledgerkit", "--config", "C0", "--seed", "1",
               "--model", "fake-model", "--claude-bin", str(FAKE), "--run-id", "smoke",
               "--sandbox", str(tmp_path / "sbx"), "--env", f"FAKE_CLAUDE_SPEC={spec}"])
    assert rc == 0
    out = results / "smoke"
    tasks = read_jsonl(out / "tasks.jsonl")
    assert len(tasks) == 10 and [t["task"] for t in tasks][0] == "01"
    assert {t["task"]: t["passed"] for t in tasks} == {t: t != "03" for t in FAM.tasks}
    rec = {t["task"]: t for t in tasks}
    assert (rec["07"]["first_pass"], rec["07"]["rounds_used"]) == (False, 1)
    assert rec["03"]["rounds_used"] == 1 and not rec["03"]["passed"]
    assert all(rec[t]["first_pass"] for t in FAM.tasks if t not in ("03", "07"))
    feedback = (out / rec["07"]["feedback"]).read_text()
    assert "CI LK-R01: docs/commands.md has no row for 'tag'" in feedback
    assert "CI LK-R04: review" in feedback and "behaviour test" not in feedback
    for marker in FAM.leak_markers:
        assert marker not in feedback
    assert not (out / "consolidations.jsonl").exists()           # C0 never consolidates
    capsys.readouterr()
    assert main(["--results", str(results), "report", "--run", "smoke", "--n-boot", "100"]) == 0
    text = capsys.readouterr().out
    assert "== C0" in text and "rer" in text and "90.0%" in text and "first_pass_rate" in text
    assert main(["--results", str(results), "audit", "--run", "smoke"]) == 0


def test_cli_refuses_todo_config(tmp_path, capsys):
    rc = main(["--results", str(tmp_path), "run", "--family", "ledgerkit", "--config", "C1", "--seed", "1",
               "--model", "m", "--claude-bin", str(FAKE)])
    assert rc == 2 and "TODO" in capsys.readouterr().err
