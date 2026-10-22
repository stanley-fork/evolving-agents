"""Per-run metrics and their aggregation per family / across families.

Definitions (see README "Metrics"):
- RER: failures = unique signatures per task session (first-attempt CI feedback +
  final verifier failures of failed tasks + tool errors of every round); a failure is *repeated* if the same signature
  appeared in an earlier session of the same run. RER = repeated / failures,
  pooled over the seeds of a family.
- tokens: input + cache_creation + output of the sessions plus the consolidation
  tokens (each consolidation prorated over the tasks it covers).
- turns are censored at the family's max_turns for failed tasks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from metrics.signatures import ci_signatures, mask_terms, tool_error_signature, verifier_signatures
from metrics.stats import mean, ols_slope


@dataclass
class RunMetrics:
    run_id: str
    family: str
    config: str
    seed: int
    n: int = 0
    successes: int = 0
    first_passes: int = 0
    rounds: int = 0
    rer_repeated: int = 0
    rer_total: int = 0
    session_tokens: int = 0
    consolidation_tokens: int = 0
    session_cost: float = 0.0
    consolidation_cost: float = 0.0
    turns_censored: dict[int, int] = field(default_factory=dict)   # position -> turns
    turns_to_green: float | None = None
    slope: float | None = None
    retention_success_delta: float | None = None
    retention_turns_delta: float | None = None
    trap_failed: bool | None = None
    memory_induced: bool | None = None
    memory_induced_source: str = ""
    passes: dict[str, bool] = field(default_factory=dict)          # task -> passed
    per_task: list[dict] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.session_tokens + self.consolidation_tokens

    @property
    def cost(self) -> float:
        return self.session_cost + self.consolidation_cost


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "dir": run_dir, "meta": meta,
        "pins": json.loads((run_dir / "pins.json").read_text(encoding="utf-8")),
        "tasks": sorted(read_jsonl(run_dir / "tasks.jsonl"), key=lambda r: r["position"]),
        "consolidations": read_jsonl(run_dir / "consolidations.jsonl"),
    }


def load_label(labels_dir: Path | None, run_id: str) -> dict | None:
    """Optional labels/<run-id>.json: {"trap_memory_induced": bool, "labeler": "...", ...}."""
    if not labels_dir:
        return None
    p = Path(labels_dir) / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_metrics(run: dict, family_info: dict | None = None, labels_dir: Path | None = None) -> RunMetrics:
    meta, tasks = run["meta"], run["tasks"]
    info = family_info or {}
    max_turns = int(meta.get("budget", {}).get("max_turns", 60))
    sandbox = meta.get("sandbox", "")
    roots = [f"{sandbox}/work/{meta['family']}", sandbox] if sandbox else []
    m = RunMetrics(meta["run_id"], meta["family"], meta["config"], int(meta["seed"]), n=len(tasks))

    # consolidation proration
    share: dict[int, int] = {}
    share_cost: dict[int, float] = {}
    for c in run["consolidations"]:
        covers = c.get("covers") or [c["after_position"]]
        tok = int(c.get("input_tokens", 0)) + int(c.get("output_tokens", 0))
        for pos in covers:
            share[pos] = share.get(pos, 0) + tok / len(covers)
            share_cost[pos] = share_cost.get(pos, 0.0) + float(c.get("cost_usd") or 0.0) / len(covers)
        m.consolidation_tokens += tok
        m.consolidation_cost += float(c.get("cost_usd") or 0.0)

    seen: set[str] = set()
    by_task = {}
    for rec in tasks:
        s = rec["session"]
        passed = bool(rec["passed"])
        sigs = set(tool_error_signature(e, roots) for e in rec.get("tool_errors", []))
        if not passed:
            sigs |= set(verifier_signatures(rec.get("verify_fail_lines", []), roots))
        sigs |= set(ci_signatures(rec.get("ci_feedback_lines", []), roots))
        terms = (info.get("signature_terms") or {}).get(str(rec["task"]), [])
        sigs = {mask_terms(x, terms) for x in sigs}
        m.first_passes += bool(rec.get("first_pass", passed))
        m.rounds += int(rec.get("rounds_used", 0))
        repeated = sigs & seen
        m.rer_total += len(sigs)
        m.rer_repeated += len(repeated)
        seen |= sigs
        m.successes += passed
        m.session_tokens += int(s["tokens"])
        m.session_cost += float(s.get("cost_usd") or 0.0)
        turns = int(s["turns"]) if passed else max_turns
        m.turns_censored[rec["position"]] = turns
        m.passes[rec["task"]] = passed
        by_task[rec["task"]] = rec
        m.per_task.append({"position": rec["position"], "task": rec["task"], "kind": rec["kind"],
                           "passed": passed, "turns": s["turns"], "tokens": s["tokens"],
                           "consolidation_tokens": round(share.get(rec["position"], 0.0), 1),
                           "first_pass": bool(rec.get("first_pass", passed)),
                           "rounds": int(rec.get("rounds_used", 0)),
                           "failures": len(sigs), "repeated": len(repeated), "status": s["status"]})

    m.turns_to_green = mean(m.turns_censored.values())
    main = [r for r in tasks if r["kind"] == "main"]
    m.slope = ols_slope([r["position"] for r in main], [m.turns_censored[r["position"]] for r in main])

    ret, of = info.get("retention_task"), info.get("retention_of")
    if ret in by_task and of in by_task:
        m.retention_success_delta = float(m.passes[ret]) - float(m.passes[of])
        m.retention_turns_delta = (m.turns_censored[by_task[ret]["position"]]
                                   - m.turns_censored[by_task[of]["position"]])

    trap = info.get("trap_task")
    if trap in by_task:
        rec = by_task[trap]
        m.trap_failed = not rec["passed"]
        label = load_label(labels_dir, m.run_id)
        if label and "trap_memory_induced" in label:
            m.memory_induced = bool(label["trap_memory_induced"])
            m.memory_induced_source = str(label.get("labeler", "label"))
        else:
            m.memory_induced = m.trap_failed and bool(rec.get("obsolete_marker_in_memory"))
            m.memory_induced_source = "heuristic"
    return m


def regression_counts(run: RunMetrics, c0: RunMetrics | None) -> tuple[int, int] | None:
    """(# tasks C0 solved that this run did not, # tasks C0 solved), same seed."""
    if c0 is None:
        return None
    solved = [t for t, ok in c0.passes.items() if ok]
    return sum(1 for t in solved if not run.passes.get(t, False)), len(solved)


METRICS = ["rer", "success", "first_pass_rate", "rounds_per_task", "tokens_per_success", "turns_to_green",
           "slope",
           "retention_success_delta", "retention_turns_delta", "memory_induced", "regression"]


def family_metrics(runs: list[RunMetrics], c0: dict[int, RunMetrics] | None = None) -> dict:
    """Aggregate a (possibly resampled, with repeats) list of runs of one config and family."""
    rep = sum(r.rer_repeated for r in runs)
    tot = sum(r.rer_total for r in runs)
    succ = sum(r.successes for r in runs)
    n = sum(r.n for r in runs)
    reg = [regression_counts(r, (c0 or {}).get(r.seed)) for r in runs] if c0 else []
    reg = [x for x in reg if x is not None]
    reg_den = sum(d for _, d in reg)
    return {
        "rer": rep / tot if tot else None,
        "success": succ / n if n else None,
        "first_pass_rate": sum(r.first_passes for r in runs) / n if n else None,
        "rounds_per_task": sum(r.rounds for r in runs) / n if n else None,
        "tokens_per_success": sum(r.tokens for r in runs) / succ if succ else None,
        "cost_per_success": sum(r.cost for r in runs) / succ if succ else None,
        "turns_to_green": mean(r.turns_to_green for r in runs),
        "slope": mean(r.slope for r in runs),
        "retention_success_delta": mean(r.retention_success_delta for r in runs),
        "retention_turns_delta": mean(r.retention_turns_delta for r in runs),
        "memory_induced": mean(None if r.memory_induced is None else float(r.memory_induced) for r in runs),
        "regression": sum(k for k, _ in reg) / reg_den if reg_den else None,
    }


def across_families(per_family: dict[str, dict]) -> dict:
    """Unweighted mean over families of each metric (families with None skipped)."""
    keys = set().union(*[set(v) for v in per_family.values()]) if per_family else set()
    return {k: mean(v.get(k) for v in per_family.values()) for k in sorted(keys)}
