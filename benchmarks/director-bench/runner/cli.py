"""bench CLI: run / report / audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runner.config import BenchError, bench_root, load_config, load_family, resolve


def cmd_run(a: argparse.Namespace) -> int:
    from runner.run import RunOptions, Runner

    root = Path(a.root) if a.root else bench_root()
    fam = load_family(resolve(root, "families", a.family))
    cfg = load_config(resolve(root, "configs", a.config))
    extra = dict(kv.split("=", 1) for kv in a.env)
    opts = RunOptions(
        family=fam, config=cfg, seed=a.seed, model=a.model,
        results_dir=Path(a.results) if a.results else root / "results",
        claude_bin=a.claude_bin, permission_mode=a.permission_mode, run_id=a.run_id,
        sandbox_dir=Path(a.sandbox) if a.sandbox else None, allow_todo=a.allow_todo,
        pass_env=a.pass_env, extra_env=extra, keep_sandbox=a.keep_sandbox, max_tasks=a.max_tasks,
    )
    runner = Runner(opts)
    print(f"run {runner.run_id}: order {' '.join(runner.order)}", file=sys.stderr)
    out = runner.run()
    passed = sum(json.loads(ln)["passed"] for ln in (out / "tasks.jsonl").read_text().splitlines() if ln)
    print(f"done: {passed}/{len(runner.order)} passed -> {out}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from runner.report import report, write_json

    root = Path(a.root) if a.root else bench_root()
    results = Path(a.results) if a.results else root / "results"
    paths = [Path(r) if Path(r).exists() else results / r for r in a.run] or [results]
    text, data = report(paths, n_boot=a.n_boot, boot_seed=a.boot_seed, root=root,
                        labels_dir=Path(a.labels) if a.labels else None)
    print(text, end="")
    if a.json:
        write_json(data, Path(a.json))
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    from runner.audit import audit

    root = Path(a.root) if a.root else bench_root()
    results = Path(a.results) if a.results else root / "results"
    run_dir = Path(a.run) if Path(a.run).exists() else results / a.run
    meta = json.loads((run_dir / "run.json").read_text())
    fam_dir = Path(a.family) if a.family else resolve(root, "families", meta["family"])
    leaks, warnings = audit(run_dir, fam_dir)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if leaks:
        print(f"LEAK DETECTED in run {meta['run_id']}: memory store saw verifier/solution content", file=sys.stderr)
        for leak in leaks:
            print(f"  {leak}", file=sys.stderr)
        return 1
    print(f"audit ok: no leak markers in {run_dir / 'memory'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench", description="director-bench: memory across Claude Code sessions")
    p.add_argument("--root", help="bench root (default: $BENCH_ROOT or the checkout)")
    p.add_argument("--results", help="results dir (default: <root>/results)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one chained-session run")
    r.add_argument("--family", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--model", default=os.environ.get("BENCH_MODEL"),
                   help="pinned model API string (or $BENCH_MODEL)")
    r.add_argument("--claude-bin", default=os.environ.get("BENCH_CLAUDE_BIN", "claude"))
    r.add_argument("--permission-mode", default="bypassPermissions")
    r.add_argument("--run-id")
    r.add_argument("--sandbox", help="where HOME/memory/workdir live (default: $TMPDIR/director-bench-sandbox)")
    r.add_argument("--allow-todo", action="store_true", help="run a config with TODO(confirm...) markers")
    r.add_argument("--pass-env", action="append", default=[], help="host env var to forward to claude")
    r.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for session and hooks")
    r.add_argument("--keep-sandbox", action="store_true")
    r.add_argument("--max-tasks", type=int, help="stop after N tasks (smoke tests only)")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="metrics, CIs and win criteria over runs")
    rp.add_argument("--run", action="append", default=[], help="run id or dir (repeatable); default: all results")
    rp.add_argument("--n-boot", type=int, default=2000)
    rp.add_argument("--boot-seed", type=int, default=0)
    rp.add_argument("--labels", help="labels dir (default: <root>/labels)")
    rp.add_argument("--json", help="also write the full report as JSON here")
    rp.set_defaults(fn=cmd_report)

    au = sub.add_parser("audit", help="fail if any memory snapshot contains verifier/solution content")
    au.add_argument("--run", required=True)
    au.add_argument("--family", help="family dir (default: from the run)")
    au.set_defaults(fn=cmd_audit)

    a = p.parse_args(argv)
    if a.cmd == "run" and not a.model:
        p.error("--model (or $BENCH_MODEL) is required: the model string is pinned per run")
    try:
        return a.fn(a)
    except BenchError as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
