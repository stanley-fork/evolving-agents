"""`bench report`: metrics per family/config with bootstrap CIs, and the win table."""
from __future__ import annotations

import json
from pathlib import Path

from metrics import criteria
from metrics.core import METRICS, RunMetrics, across_families, family_metrics, load_run, run_metrics
from metrics.signatures import SIGNATURE_VERSION
from metrics.stats import bootstrap, percentile_ci
from runner.config import BenchError, bench_root, load_family

GLOBAL_PINS = ("model", "claude_version", "permission_mode", "bench_version")


def find_runs(paths: list[Path]) -> list[Path]:
    """Each path is a run dir or a directory containing run dirs."""
    out = []
    for p in paths:
        if (p / "run.json").exists():
            out.append(p)
        elif p.is_dir():
            out += sorted(d for d in p.iterdir() if (d / "run.json").exists())
        else:
            raise BenchError(f"not a run or results dir: {p}")
    return out


def check_pins(runs: list[dict]) -> None:
    """Refuse to mix results with different pins in one report."""
    problems = []
    for key in GLOBAL_PINS:
        vals = {r["pins"].get(key) for r in runs}
        if len(vals) > 1:
            problems.append(f"{key} differs across runs: {sorted(map(str, vals))}")
    groups: dict[tuple, set] = {}
    for r in runs:
        p = r["pins"]
        groups.setdefault(("memory_system_version", p["config"]), set()).add(p.get("memory_system_version"))
        groups.setdefault(("config_hash", p["config"]), set()).add(p.get("config_hash"))
        groups.setdefault(("family_hash", p["family"]), set()).add(p.get("family_hash"))
    for (key, name), vals in sorted(groups.items()):
        if len(vals) > 1:
            problems.append(f"{key} differs within {name}: {sorted(map(str, vals))}")
    seen = {}
    for r in runs:
        k = (r["meta"]["config"], r["meta"]["family"], r["meta"]["seed"])
        if k in seen:
            problems.append(f"duplicate run for config/family/seed {k}: {seen[k]} and {r['meta']['run_id']}")
        seen[k] = r["meta"]["run_id"]
        if r["meta"].get("status") != "complete":
            problems.append(f"run {r['meta']['run_id']} is not complete ({r['meta'].get('status')})")
    if problems:
        raise BenchError("refusing to report mixed or incomplete results:\n  " + "\n  ".join(problems))


def family_info(name: str, root: Path) -> dict:
    try:
        fam = load_family(root / "families" / name)
    except (OSError, KeyError, BenchError):
        return {}
    return {"trap_task": fam.trap_task, "retention_task": fam.retention_task, "retention_of": fam.retention_of,
            "signature_terms": {str(k): [str(x) for x in v] for k, v in (fam.raw.get("signature_terms") or {}).items()}}


def build_table(run_dirs: list[Path], root: Path | None = None, labels_dir: Path | None = None):
    root = root or bench_root()
    runs = [load_run(d) for d in run_dirs]
    if not runs:
        raise BenchError("no runs found")
    check_pins(runs)
    infos: dict[str, dict] = {}
    table: criteria.Table = {}
    for r in runs:
        fam = r["meta"]["family"]
        infos.setdefault(fam, family_info(fam, root))
        m = run_metrics(r, infos[fam], labels_dir if labels_dir else root / "labels")
        table.setdefault(m.config, {}).setdefault(m.family, {})[m.seed] = m
    return runs, table


def summarize(table: criteria.Table, n_boot: int, boot_seed: int) -> dict:
    out = {}
    for cfg in sorted(table):
        seeds = {f: sorted(s) for f, s in table[cfg].items()}
        c0 = table.get("C0", {}) if cfg != "C0" else {}   # regression is "vs C0": n/a for C0 itself

        def stat(sample, cfg=cfg):
            return across_families({f: family_metrics([table[cfg][f][s] for s in ss], c0.get(f))
                                    for f, ss in sample.items()})
        point = stat(seeds)
        boots = bootstrap(stat, seeds, n_boot, boot_seed)
        per_family = {f: family_metrics([table[cfg][f][s] for s in ss], c0.get(f)) for f, ss in seeds.items()}
        out[cfg] = {"seeds": seeds, "point": point, "per_family": per_family,
                    "ci95": {k: percentile_ci([b[k] for b in boots]) for k in point}}
    return out


def fmt(v, pct=False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, tuple):
        return f"[{fmt(v[0], pct)}, {fmt(v[1], pct)}]"
    return f"{100 * v:.1f}%" if pct else (f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}")


PCT = {"rer", "success", "first_pass_rate", "memory_induced", "regression", "retention_success_delta"}


def render(runs: list[dict], summary: dict, wins: list[dict]) -> str:
    pins = runs[0]["pins"]
    lines = [f"director-bench report  (signature v{SIGNATURE_VERSION}, {len(runs)} runs)",
             f"pins: model={pins['model']}  claude={pins['claude_version']}  "
             f"permission_mode={pins['permission_mode']}", ""]
    for cfg, s in summary.items():
        fams = ", ".join(f"{f}: seeds {v}" for f, v in s["seeds"].items())
        lines.append(f"== {cfg}  ({fams})")
        for k in METRICS + ["cost_per_success"]:
            lines.append(f"  {k:<26} {fmt(s['point'].get(k), k in PCT):>10}  95% CI {fmt(s['ci95'].get(k), k in PCT)}")
        lines.append("")
    for w in wins:
        lines.append(f"== win criteria: {w['config']} vs C1")
        if not w["rows"]:
            lines.append(f"  {w.get('note')}")
            continue
        for r in w["rows"]:
            mark = {True: "PASS", False: "FAIL", None: "N/A "}[r["pass"]]
            lines.append(f"  [{mark}] {r['condition']:<52} value {fmt(r['value'])}  CI {fmt(r['ci95'])}")
        lines.append(f"  verdict (advisory): {w['verdict']}")
        lines.append("")
    mech = wins[0].get("mechanism") if wins else None
    if mech:
        lines.append("== mechanism: A1 - C2 (positive = A1 worse)")
        for k in ("rer", "memory_induced"):
            lines.append(f"  {k:<16} {fmt(mech[k]['a1_minus_c2'])}  CI {fmt(mech[k]['ci95'])}  "
                         f"{'significant' if mech[k]['significant'] else 'not significant'}")
        lines.append(f"  [{'PASS' if mech['pass'] else 'FAIL'}] A1 significantly worse than C2")
    return "\n".join(lines).rstrip() + "\n"


def report(paths: list[Path], n_boot: int = 2000, boot_seed: int = 0, root: Path | None = None,
           labels_dir: Path | None = None) -> tuple[str, dict]:
    runs, table = build_table(find_runs(paths), root, labels_dir)
    summary = summarize(table, n_boot, boot_seed)
    wins = []
    if "C1" in table:
        mech = criteria.mechanism(table, n_boot, boot_seed)
        for cfg in sorted(c for c in table if c not in ("C0", "C1")):
            w = criteria.win_table(table, cfg, n_boot, boot_seed)
            w["mechanism"] = mech
            wins.append(w)
    data = {"summary": summary, "win_criteria": wins,
            "runs": {r["meta"]["run_id"]: [t for t in _per_task(table, r)] for r in runs}}
    return render(runs, summary, wins), data


def _per_task(table: criteria.Table, run: dict) -> list[dict]:
    m: RunMetrics = table[run["meta"]["config"]][run["meta"]["family"]][int(run["meta"]["seed"])]
    return m.per_task


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
