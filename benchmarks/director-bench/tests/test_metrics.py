import json

import pytest

from metrics import criteria
from metrics.core import family_metrics, run_metrics
from metrics.stats import bootstrap, ols_slope, percentile_ci
from runner.config import BenchError
from runner.report import check_pins, report

INFO = {"trap_task": "T", "retention_task": "R", "retention_of": "a"}


def task(pos, tid, passed, tool=(), verify=(), turns=5, tokens=1000, kind="main", marker=None):
    rec = {"position": pos, "task": tid, "kind": kind, "passed": passed,
           "tool_errors": [{"tool": "Bash", "file": "", "message": m} for m in tool],
           "verify_fail_lines": list(verify),
           "session": {"turns": turns, "tokens": tokens, "cost_usd": 0.01, "status": "ok"}}
    if marker is not None:
        rec["obsolete_marker_in_memory"] = marker
    return rec


def run(config, seed, tasks, family="f", cons=(), pins=None):
    meta = {"run_id": f"{family}-{config}-s{seed}", "family": family, "config": config, "seed": seed,
            "budget": {"max_turns": 60}, "sandbox": "/sbx", "status": "complete"}
    p = {"model": "m", "claude_version": "1", "permission_mode": "x", "bench_version": "0.1.0",
         "memory_system_version": "v", "config": config, "family": family, "config_hash": "c",
         "family_hash": "h", **(pins or {})}
    return {"meta": meta, "pins": p, "tasks": tasks, "consolidations": list(cons)}


def test_rer_counts_repeats_across_sessions_only():
    lint = "BENCH-FAIL type=lint file=ledgerkit/commands/x.py symbol=LK-L006 msg=needs tests/test_x.py built with make_ledger"
    r = run("C0", 1, [
        task(1, "a", False, tool=["Exit code 1\nNameError: q", "Exit code 1\nNameError: q"], verify=[lint]),
        task(2, "b", True, tool=["Exit code 1\nNameError: q"]),                  # repeat of a tool error
        task(3, "c", False, verify=[lint.replace("x.py", "y.py")]),             # different file: new
        task(4, "d", False, verify=[lint]),                                      # repeat
    ])
    m = run_metrics(r, INFO)
    # session1: {tool, lint-x} (dup within session counted once); s2: {tool}=rep; s3: {lint-y}; s4: {lint-x}=rep
    assert (m.rer_repeated, m.rer_total) == (2, 5)
    assert family_metrics([m])["rer"] == pytest.approx(0.4)


def test_passed_task_verifier_lines_are_not_failures():
    r = run("C0", 1, [task(1, "a", True, verify=["BENCH-FAIL type=x file=- symbol=- msg=stale"])])
    assert run_metrics(r, INFO).rer_total == 0


def test_slope_turns_censoring_and_retention():
    r = run("C0", 1, [task(1, "a", True, turns=20), task(2, "b", True, turns=15), task(3, "c", False, turns=9),
                      task(4, "T", False, kind="trap", marker=True), task(5, "R", True, turns=10, kind="retention")])
    m = run_metrics(r, INFO)
    assert m.turns_censored == {1: 20, 2: 15, 3: 60, 4: 60, 5: 10}
    assert m.slope == pytest.approx(ols_slope([1, 2, 3], [20, 15, 60]))
    assert m.retention_success_delta == 0.0 and m.retention_turns_delta == -10
    assert m.memory_induced is True and m.memory_induced_source == "heuristic"


def test_ols_slope():
    assert ols_slope([1, 2, 3, 4], [10, 8, 6, 4]) == pytest.approx(-2)
    assert ols_slope([1], [3]) is None and ols_slope([2, 2], [1, 5]) is None


def test_label_overrides_heuristic(tmp_path):
    (tmp_path / "f-C2-s1.json").write_text(json.dumps({"trap_memory_induced": False, "labeler": "llm-judge"}))
    r = run("C2", 1, [task(1, "T", False, kind="trap", marker=True)])
    m = run_metrics(r, INFO, tmp_path)
    assert m.memory_induced is False and m.memory_induced_source == "llm-judge"


def test_regression_vs_c0_same_seed():
    c0 = run_metrics(run("C0", 1, [task(1, "a", True), task(2, "b", True), task(3, "c", False)]), INFO)
    x = run_metrics(run("C2", 1, [task(1, "a", True), task(2, "b", False), task(3, "c", True)]), INFO)
    assert family_metrics([x], {1: c0})["regression"] == pytest.approx(0.5)
    assert family_metrics([x])["regression"] is None


def test_tokens_per_success_includes_prorated_consolidation():
    r = run("C2", 1, [task(1, "a", True, tokens=1000), task(2, "b", False, tokens=1000)],
            cons=[{"after_position": 2, "covers": [1, 2], "input_tokens": 300, "output_tokens": 100, "cost_usd": 1.0}])
    m = run_metrics(r, INFO)
    assert [t["consolidation_tokens"] for t in m.per_task] == [200, 200]
    assert family_metrics([m])["tokens_per_success"] == 2400


def test_bootstrap_is_deterministic_and_paired():
    seeds = {"f": [1, 2, 3, 4, 5], "g": [1, 2, 3]}
    a = bootstrap(lambda s: sum(s["f"]) + 10 * sum(s["g"]), seeds, 200, boot_seed=7)
    b = bootstrap(lambda s: sum(s["f"]) + 10 * sum(s["g"]), seeds, 200, boot_seed=7)
    c = bootstrap(lambda s: sum(s["f"]) + 10 * sum(s["g"]), seeds, 200, boot_seed=8)
    assert a == b and a != c
    lo, hi = percentile_ci(a)
    assert lo <= sum(seeds["f"]) + 10 * sum(seeds["g"]) <= hi


def _table(c1_rer_fail, x_fail_tool, x_success_drop=False, x_tokens=1000, a1=None, x3_tool=None):
    """Build config->family->seed->RunMetrics with controlled RER."""
    table = {}

    def add(cfg, seed, tasks):
        m = run_metrics(run(cfg, seed, tasks), INFO)
        table.setdefault(cfg, {}).setdefault("f", {})[seed] = m

    for s in range(1, 6):
        err = [f"Exit code 1\nNameError: e{s}"]
        add("C0", s, [task(1, "a", True), task(2, "b", True, tool=err), task(3, "c", True, tool=err)])
        # C1 repeats its error in sessions 2 and 3: RER = 2/3 (+ jitter by seed)
        add("C1", s, [task(1, "a", True, tool=err), task(2, "b", True, tool=err),
                      task(3, "c", True, tool=err if c1_rer_fail else [f"Exit code 1\nOther{s}Error: z"]),
                      task(4, "T", True, kind="trap", marker=False)])
        x_tasks = [task(1, "a", True, tool=err, tokens=x_tokens), task(2, "b", True, tool=x_fail_tool(s), tokens=x_tokens),
                   task(3, "c", not x_success_drop, tool=(x3_tool or (lambda s: [f"Exit code 1\nFresh{s}Error: n"]))(s),
                        tokens=x_tokens),
                   task(4, "T", True, kind="trap", marker=False)]
        add("C2", s, x_tasks)
        if a1 is not None:
            add("A1", s, a1(s, err))
    return table


def test_win_table_all_pass_and_go():
    t = _table(True, lambda s: [f"Exit code 1\nNew{s}Error: z"],
               a1=lambda s, err: [task(1, "a", True, tool=err), task(2, "b", True, tool=err),
                                  task(3, "c", True, tool=err), task(4, "T", True, kind="trap")])
    w = criteria.win_table(t, "C2", n_boot=300, boot_seed=1)
    status = [r["pass"] for r in w["rows"]]
    assert status == [True, True, True, True, True], w["rows"]
    assert w["rows"][0]["value"] == pytest.approx(1.0)        # C2 never repeats
    assert w["verdict"].startswith("GO")


def test_win_table_rer_fail_is_pivot():
    same = lambda s: [f"Exit code 1\nNameError: e{s}"]       # C2 repeats like C1 does
    t = _table(True, same, x3_tool=same)
    w = criteria.win_table(t, "C2", n_boot=300, boot_seed=1)
    assert w["rows"][0]["pass"] is False and w["verdict"].startswith("PIVOT")


def test_win_table_success_and_tokens_fail_is_iterate():
    t = _table(True, lambda s: [f"Exit code 1\nNew{s}Error: z"], x_success_drop=True, x_tokens=5000)
    w = criteria.win_table(t, "C2", n_boot=300, boot_seed=1)
    names = {r["condition"].split()[0]: r["pass"] for r in w["rows"]}
    assert names["RER"] is True and names["success"] is False and names["tokens"] is False
    assert names["regression"] is False                      # C2 lost task c that C0 solved
    assert w["verdict"].startswith("ITERATE")


def test_check_pins_refuses_mixing():
    a = run("C0", 1, [])
    b = run("C1", 1, [], pins={"claude_version": "2"})
    with pytest.raises(BenchError, match="claude_version"):
        check_pins([a, b])
    c = run("C1", 2, [], pins={"memory_system_version": "w"})
    d = run("C1", 3, [])
    with pytest.raises(BenchError, match="memory_system_version"):
        check_pins([c, d])
    with pytest.raises(BenchError, match="duplicate"):
        check_pins([run("C0", 1, []), run("C0", 1, [])])
    check_pins([run("C0", 1, []), run("C1", 1, [])])          # same pins: fine


def test_report_from_disk_prints_win_table(tmp_path):
    for cfg, rep in (("C0", True), ("C1", True), ("C2", False)):
        for s in (1, 2, 3):
            r = run(cfg, s, [task(1, "a", True, tool=["Exit code 1\nNameError: x"]),
                             task(2, "b", True, tool=["Exit code 1\nNameError: x"] if rep else [])])
            d = tmp_path / r["meta"]["run_id"]
            d.mkdir()
            (d / "run.json").write_text(json.dumps(r["meta"]))
            (d / "pins.json").write_text(json.dumps(r["pins"]))
            (d / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in r["tasks"]))
    text, data = report([tmp_path], n_boot=200, boot_seed=3, root=tmp_path)
    assert "== win criteria: C2 vs C1" in text
    assert "[PASS] RER relative reduction" in text
    text2, _ = report([tmp_path], n_boot=200, boot_seed=3, root=tmp_path)
    assert text == text2                                      # deterministic given the bootstrap seed
    assert data["summary"]["C1"]["point"]["rer"] == pytest.approx(0.5)


def test_rer_counts_first_attempt_ci_feedback_and_rates():
    ci = ["CI LK-R01: docs/commands.md has no row for 'x'"]
    t1 = task(1, "a", True)
    t1.update({"first_pass": False, "rounds_used": 1, "ci_feedback_lines": ci})
    t2 = task(2, "b", True)
    t2.update({"first_pass": False, "rounds_used": 1, "ci_feedback_lines": [ci[0].replace("'x'", "'y'")]})
    t3 = task(3, "c", True)
    t3.update({"first_pass": False, "rounds_used": 1, "ci_feedback_lines": ci})
    m = run_metrics(run("C0", 1, [t1, t2, t3, task(4, "d", True)]), INFO)
    assert (m.rer_repeated, m.rer_total) == (1, 3)
    fm = family_metrics([m])
    assert fm["success"] == 1.0 and fm["first_pass_rate"] == 0.25 and fm["rounds_per_task"] == 0.75
