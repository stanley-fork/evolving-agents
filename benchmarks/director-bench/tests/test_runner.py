import json
from pathlib import Path

import pytest

from conftest import FAKE, read_jsonl, write_config
from runner.config import BenchError, load_config
from runner.run import task_order

ROOT = Path(__file__).resolve().parent.parent


def solve_all(extra=None):
    rules = [{"match": f"TASK-{t}", "writes": {f"done-{t}.txt": "ok"}, "memory_append": f"learned {t}",
              "home_note": True} for t in ("01", "02", "03", "04")]
    return {"rules": rules, **(extra or {})}


def test_full_run_records_and_persists_memory(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    runner, log = make_run(tiny_family, cfg, solve_all())
    out = runner.run()
    tasks = read_jsonl(out / "tasks.jsonl")
    assert [t["task"] for t in tasks] == runner.order
    assert all(t["passed"] and t["verifier_passed"] for t in tasks)
    assert {t["kind"] for t in tasks} == {"main", "trap", "retention"}
    calls = read_jsonl(log)
    # memory persists across tasks...
    assert calls[0]["memory"] is None
    assert calls[-1]["memory"].count("learned") == 3
    # ...while the workdir is reset to the task snapshot each time
    for call, task in zip(calls, runner.order):
        assert call["files"] == ["README.md"], call["files"]
        assert call["cwd"] == calls[0]["cwd"]          # stable path (native memory keys on cwd)
    # hooks saw the BENCH_* context; post-session order and passed flag
    mem = out / "memory" / "final" / "store"
    lines = (mem / "outcomes.txt").read_text().splitlines()
    assert lines[0] == f"1 {runner.order[0]} true 01-{runner.order[0]}.jsonl"
    assert (mem / "setup.txt").exists()
    pins = json.loads((out / "pins.json").read_text())
    assert pins["claude_version"].startswith("9.9.9") and pins["memory_system_version"] == "mem-1"
    assert pins["model"] == "fake-model"
    assert json.loads((out / "run.json").read_text())["status"] == "complete"
    assert (out / "traces" / f"01-{runner.order[0]}.jsonl").read_text().count('"type": "result"') == 1
    assert not runner.sandbox.exists()                 # sandbox cleaned


def test_token_accounting_dedupes_message_ids(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    spec = solve_all({"default": {}})
    for r in spec["rules"]:
        r.update({"turns": 3, "usage": {"input_tokens": 10, "output_tokens": 5,
                                        "cache_creation_input_tokens": 2, "cache_read_input_tokens": 100}})
    runner, _ = make_run(tiny_family, cfg, spec)
    s = read_jsonl(runner.run() / "tasks.jsonl")[0]["session"]
    assert (s["input_tokens"], s["output_tokens"], s["cache_creation_tokens"], s["cache_read_tokens"]) == (30, 15, 6, 300)
    assert s["tokens"] == 51 and s["turns"] == 3 and s["cost_usd"] == 0.01


def test_budget_kill_on_tokens(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    spec = {"default": {"turns": 50, "sleep": 0.05, "writes": {"done-01.txt": "ok"},
                        "usage": {"input_tokens": 30000, "output_tokens": 0}}}
    runner, _ = make_run(tiny_family, cfg, spec, max_tasks=1)
    rec = read_jsonl(runner.run() / "tasks.jsonl")[0]
    s = rec["session"]
    assert s["status"] == "budget_tokens"
    assert 100000 < s["tokens"] <= 130000          # killed right after crossing 100k
    assert s["wall_seconds"] < 2
    assert rec["verifier_passed"] is True and rec["passed"] is False   # over budget = fail


def test_verifier_failure_recorded_with_signature(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    runner, _ = make_run(tiny_family, cfg, {"default": {"tool_errors": ["Error: /tmp/abc123/x.py:12: NameError: name 'y'"]}})
    out = runner.run()
    tasks = read_jsonl(out / "tasks.jsonl")
    assert not any(t["passed"] for t in tasks)
    t0 = tasks[0]
    assert t0["verifier_exit"] == 1
    assert t0["verifier_signatures"] == ["verify:hidden|done-01.txt|check|missing at done-01.txt"]
    assert t0["tool_error_signatures"] == ["tool:Bash|<tmp>/x.py|NameError|Error: <tmp>/x.py:<n>: NameError: name 'y'"]
    assert "BENCH-FAIL" in (out / "verify" / f"01-{t0['task']}.log").read_text()


def test_consolidation_cadence_and_proration(tiny_family, tmp_path, make_run):
    from metrics.core import load_run, run_metrics
    cfg = write_config(tmp_path)
    runner, _ = make_run(tiny_family, cfg, solve_all())
    out = runner.run()
    cons = read_jsonl(out / "consolidations.jsonl")
    # 4 tasks, every 2 -> after task 2 only (no consolidation after the last task)
    assert [c["after_position"] for c in cons] == [2]
    assert cons[0]["covers"] == [1, 2] and cons[0]["cost_usd"] == 0.5
    assert (out / "memory" / "final" / "store" / "cons.txt").read_text() == "consolidated 1,2\n"
    m = run_metrics(load_run(out), {"trap_task": "04"})
    assert m.consolidation_tokens == 1200
    assert [t["consolidation_tokens"] for t in m.per_task] == [600, 600, 0, 0]
    assert m.tokens == m.session_tokens + 1200


def test_cadence_three(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, consolidate={"command": "echo '{\"input_tokens\": 3}'", "every": 1})
    runner, _ = make_run(tiny_family, cfg, solve_all())
    cons = read_jsonl(runner.run() / "consolidations.jsonl")
    assert [c["covers"] for c in cons] == [[1], [2], [3]]


def test_home_isolation_task_scope(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, name="H", home_scope="task")
    runner, log = make_run(tiny_family, cfg, solve_all())
    runner.run()
    calls = read_jsonl(log)
    assert all(c["home_notes"] is None for c in calls)          # HOME wiped every session
    assert all(c["claude_config_dir"] == c["home"] + "/.claude" for c in calls)
    assert calls[-1]["memory"].count("learned") == 3            # memory dir still persists


def test_home_persists_in_run_scope_but_not_across_runs(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, name="R", home_scope="run")
    r1, log1 = make_run(tiny_family, cfg, solve_all(), seed=1)
    r1.run()
    r2, log2 = make_run(tiny_family, cfg, solve_all(), seed=2)
    r2.run()
    c1, c2 = read_jsonl(log1), read_jsonl(log2)
    assert c1[-1]["home_notes"].count("\n") == 3                # carried within the run
    assert c2[0]["home_notes"] is None and c2[0]["memory"] is None   # nothing shared across runs
    assert c1[0]["home"] != c2[0]["home"]
    assert not str(Path(c1[0]["home"])).startswith(str(Path.home()) + "/.claude")


def test_session_env_is_allowlisted(tiny_family, tmp_path, make_run, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_HOST_VAR", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = write_config(tmp_path)
    runner, log = make_run(tiny_family, cfg, solve_all(), max_tasks=1)
    runner.run()
    keys = read_jsonl(log)[0]["env_keys"]
    assert "ANTHROPIC_API_KEY" in keys and "FAKE_MEMORY" in keys
    assert "SOME_SECRET_HOST_VAR" not in keys and "BENCH_TASK_ID" not in keys


def test_claude_command_line(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, session={"extra_args": ["--plugin-dir", "${BENCH_MEMORY_DIR}/plug"],
                                          "settings": {"autoMemoryEnabled": False}})
    runner, log = make_run(tiny_family, cfg, solve_all(), max_tasks=1)
    runner.run()
    argv = read_jsonl(log)[0]["argv"]
    assert argv[:2] == ["-p", "TASK-01: write done-01.txt"]
    for flag in ("--output-format", "--verbose", "--max-turns", "--model", "--settings", "--plugin-dir"):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--max-turns") + 1] == "10"
    assert argv[argv.index("--plugin-dir") + 1].endswith("/memory/plug")


def test_order_permuted_by_seed_and_shared_across_configs(tiny_family):
    from runner.config import load_family
    fam = load_family(ROOT / "families" / "ledgerkit")
    orders = {s: task_order(fam, s) for s in range(1, 6)}
    assert all(o[0] == "01" and o[-2:] == ["09", "10"] and sorted(o) == fam.tasks for o in orders.values())
    assert len({tuple(o) for o in orders.values()}) > 1
    assert task_order(fam, 3) == task_order(fam, 3)


def test_refuses_todo_configs(tiny_family, tmp_path, make_run):
    for name in ("C1", "C3"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.todos()
        with pytest.raises(BenchError, match="TODO"):
            make_run(tiny_family, cfg, solve_all())
    for name in ("C0", "C2", "A1"):
        assert not load_config(ROOT / "configs" / f"{name}.yaml").todos()


def test_director_configs_use_given_interface():
    c2 = load_config(ROOT / "configs" / "C2.yaml")
    a1 = load_config(ROOT / "configs" / "A1.yaml")
    assert c2.env["DIRECTOR_MEMORY"] == "${BENCH_MEMORY_DIR}/.director/memory"
    assert c2.extra_args == ["--plugin-dir", "${DIRECTOR_PLUGIN_DIR}"]
    assert "director.headless init --memory \"${BENCH_MEMORY_DIR}\"" in c2.setup[0]
    assert "outcome --memory \"${BENCH_MEMORY_DIR}\" --task \"${BENCH_TASK_ID}\" --passed \"${BENCH_PASSED}\" --trace \"${BENCH_TRACE}\"" in c2.post_session[0]
    assert c2.consolidate[0].endswith("--policy verified") and a1.consolidate[0].endswith("--policy all")
    assert c2.every == a1.every == 2


def test_director_config_runs_against_stub(tiny_family, tmp_path, make_run):
    """C2.yaml end to end with a stub `director.headless` module (not director itself)."""
    stub = tmp_path / "stub" / "director"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "headless.py").write_text(
        "import json, sys, pathlib\n"
        "a = sys.argv[1:]\n"
        "mem = pathlib.Path(a[a.index('--memory') + 1])\n"
        "(mem / '.director' / 'memory').mkdir(parents=True, exist_ok=True)\n"
        "with open(mem / 'calls.txt', 'a') as fh: fh.write(' '.join(a) + '\\n')\n"
        "if a[0] == 'consolidate': print(json.dumps({'input_tokens': 7, 'output_tokens': 3, 'cost_usd': 0.1}))\n")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    cfg = load_config(ROOT / "configs" / "C2.yaml")
    import sys
    runner, log = make_run(tiny_family, cfg, solve_all(), extra_env={
        "DIRECTOR_PYTHON": sys.executable, "DIRECTOR_PLUGIN_DIR": str(plugin),
        "DIRECTOR_VERSION": "stub-1", "PYTHONPATH": str(tmp_path / "stub")})
    out = runner.run()
    calls = (out / "memory" / "final" / "store" / "calls.txt").read_text().splitlines()
    assert calls[0].startswith("init --memory ")
    assert sum(c.startswith("outcome ") for c in calls) == 4
    assert sum(c.startswith("consolidate ") and c.endswith("--policy verified") for c in calls) == 1
    first = read_jsonl(log)[0]
    assert first["argv"][first["argv"].index("--plugin-dir") + 1] == str(plugin)
    assert json.loads((out / "pins.json").read_text())["memory_system_version"] == "stub-1"
    assert read_jsonl(out / "consolidations.jsonl")[0]["input_tokens"] == 7


def test_missing_director_env_fails_loudly(tiny_family, tmp_path, make_run, monkeypatch):
    monkeypatch.delenv("DIRECTOR_VERSION", raising=False)
    cfg = load_config(ROOT / "configs" / "C2.yaml")
    runner, _ = make_run(tiny_family, cfg, solve_all())
    with pytest.raises(BenchError, match="version_cmd"):
        runner.run()


def test_trap_marker_and_audit_leak(tiny_family, tmp_path, make_run):
    from runner.audit import audit
    cfg = write_config(tmp_path)
    spec = solve_all()
    spec["rules"][0]["memory_append"] = "always edit OLD-CONVENTION file"
    spec["rules"][3].pop("writes")               # trap task fails
    runner, _ = make_run(tiny_family, cfg, spec)
    out = runner.run()
    trap = [t for t in read_jsonl(out / "tasks.jsonl") if t["kind"] == "trap"][0]
    assert trap["passed"] is False and trap["obsolete_marker_in_memory"] is True
    leaks, warnings = audit(out, tiny_family.dir)
    assert leaks == [] and warnings == []

    spec["rules"][1]["memory_append"] = "the oracle says SECRET-ORACLE-02"   # planted leak
    runner2, _ = make_run(tiny_family, cfg, spec, seed=2)
    out2 = runner2.run()
    leaks, _ = audit(out2, tiny_family.dir)
    assert leaks and all("SECRET-ORACLE" in x for x in leaks)


def test_audit_cli_exit_code(tiny_family, tmp_path, make_run):
    from runner.cli import main
    cfg = write_config(tmp_path)
    spec = solve_all()
    spec["rules"][0]["memory_append"] = "SECRET-ORACLE-01 leaked"
    runner, _ = make_run(tiny_family, cfg, spec)
    out = runner.run()
    assert main(["audit", "--run", str(out), "--family", str(tiny_family.dir)]) == 1


def test_existing_run_id_refused(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    runner, _ = make_run(tiny_family, cfg, solve_all(), run_id="fixed", max_tasks=1)
    runner.run()
    with pytest.raises(BenchError, match="already exist"):
        make_run(tiny_family, cfg, solve_all(), run_id="fixed")


def test_fake_claude_is_executable():
    import os
    assert os.access(FAKE, os.X_OK)


def test_a_session_that_never_starts_stops_the_run(tiny_family, tmp_path, make_run):
    # e.g. Claude Code refusing bypassPermissions as root: no turns, exit 1. That is not a failed task.
    cfg = write_config(tmp_path)
    runner, _ = make_run(tiny_family, cfg, {"rules": [], "default": {"turns": 0, "exit": 1}})
    with pytest.raises(BenchError, match="did not start"):
        runner.run()


def test_is_sandbox_reaches_the_session():
    from runner.run import SESSION_ENV_ALLOW
    assert "IS_SANDBOX" in SESSION_ENV_ALLOW


def _fix_on_review(task="01", extra=None):
    spec = solve_all()
    rule = spec["rules"][int(task) - 1]
    rule.pop("writes")
    rule.update({"turns": 4, "resume": {"turns": 3, "writes": {f"done-{task}.txt": "ok"}}, **(extra or {})})
    return spec


def test_feedback_round_runs_only_when_ci_fails(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, hooks={"post_session": [
        'echo "$BENCH_TASK_ID $BENCH_FIRST_PASS $BENCH_ROUNDS $BENCH_PASSED $(cat "$BENCH_FEEDBACK" | wc -l)"'
        ' >> "$BENCH_MEMORY_DIR/fb.txt"']})
    runner, log = make_run(tiny_family, cfg, _fix_on_review())
    out = runner.run()
    recs = {t["task"]: t for t in read_jsonl(out / "tasks.jsonl")}
    r = recs["01"]
    assert (r["first_pass"], r["passed"], r["rounds_used"]) == (False, True, 1)
    assert r["ci_feedback_lines"] == ["CI LK-R01: review: done-01.txt is missing"]
    assert r["ci_feedback_signatures"] == ["ci||LK-R01|review: done-01.txt is missing"]
    assert r["session"]["turns"] == 7 and len(r["rounds"]) == 2
    assert all(t["rounds_used"] == 0 and t["first_pass"] for k, t in recs.items() if k != "01")
    calls = read_jsonl(log)
    resumed = [c for c in calls if c["resumed"]]
    assert len(resumed) == 1 and len(calls) == 5
    first = calls[0]
    trace = (out / "traces" / "01-01.jsonl").read_text().splitlines()
    sid = json.loads(trace[0])["session_id"]
    argv = resumed[0]["argv"]
    assert argv[argv.index("--resume") + 1] == sid
    assert argv[argv.index("--max-turns") + 1] == "6"              # 10 - 4 turns already used
    assert "CI LK-R01: review: done-01.txt is missing" in resumed[0]["prompt"]
    assert resumed[0]["cwd"] == first["cwd"] and resumed[0]["home"] == first["home"]
    assert sum('"type": "result"' in ln for ln in trace) == 2       # both rounds in one trace
    fb_line = [json.loads(ln) for ln in trace if "bench_feedback_round" in ln]
    assert fb_line and fb_line[0]["message"]["content"][0]["text"] == resumed[0]["prompt"]
    feedback = (out / "feedback" / "01-01.txt").read_text()
    assert feedback == resumed[0]["prompt"]
    for secret in ("SECRET-ORACLE", "BENCH-FAIL", "/done-01.txt"):
        assert secret not in feedback
    assert (out / "verify" / "01-01-r1.log").exists()
    hooks = (out / "memory" / "final" / "store" / "fb.txt").read_text().splitlines()
    assert hooks[0] == "01 false 1 true 4"
    assert hooks[1].startswith(f"{runner.order[1]} true 0 true")


def test_feedback_round_shares_the_budget(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path)
    spec = _fix_on_review(extra={"usage": {"input_tokens": 20000, "output_tokens": 0}})
    spec["rules"][0]["resume"].update({"turns": 5, "sleep": 0.05, "usage": {"input_tokens": 15000, "output_tokens": 0}})
    runner, _ = make_run(tiny_family, cfg, spec, max_tasks=1)
    rec = read_jsonl(runner.run() / "tasks.jsonl")[0]
    # 4 x 20k = 80k in round 0; 20k left: killed in round 1 after crossing it
    assert rec["rounds_used"] == 1 and rec["session"]["status"] == "budget_tokens"
    assert 100000 < rec["session"]["tokens"] <= 115000
    assert rec["verifier_passed"] is True and rec["passed"] is False


def test_no_feedback_round_when_rounds_is_zero(tiny_family, tmp_path, make_run):
    tiny_family.feedback_rounds = 0
    cfg = write_config(tmp_path)
    runner, log = make_run(tiny_family, cfg, _fix_on_review())
    recs = {t["task"]: t for t in read_jsonl(runner.run() / "tasks.jsonl")}
    assert (recs["01"]["first_pass"], recs["01"]["passed"], recs["01"]["rounds_used"]) == (False, False, 0)
    assert not any(c["resumed"] for c in read_jsonl(log))


def test_feedback_rounds_from_family_yaml():
    from runner.config import load_family
    assert load_family(ROOT / "families" / "ledgerkit").feedback_rounds == 1


def test_final_feedback_reaches_hooks_only_when_the_task_still_fails(tiny_family, tmp_path, make_run):
    cfg = write_config(tmp_path, hooks={"post_session": [
        'echo "$BENCH_TASK_ID $(wc -l < "$BENCH_FINAL_FEEDBACK") $(cat "$BENCH_FINAL_FEEDBACK")" >> "$BENCH_MEMORY_DIR/final.txt"']})
    spec = _fix_on_review(extra={"resume": {"turns": 1}})            # the round fixes nothing
    runner, _ = make_run(tiny_family, cfg, spec)
    out = runner.run()
    recs = {t["task"]: t for t in read_jsonl(out / "tasks.jsonl")}
    assert recs["01"]["passed"] is False
    assert recs["01"]["final_ci_feedback_lines"] == ["CI LK-R01: review: done-01.txt is missing"]
    final = (out / "feedback" / "01-01.final.txt").read_text()
    assert final == "CI LK-R01: review: done-01.txt is missing"
    for secret in ("SECRET-ORACLE", "BENCH-FAIL"):
        assert secret not in final
    assert all(not t["final_ci_feedback_lines"] for k, t in recs.items() if k != "01")
