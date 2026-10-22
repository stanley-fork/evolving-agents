"""Pre-registered win criteria vs C1 (see DECISION.md). Thresholds must not change
after the first run that counts."""
from __future__ import annotations

from metrics.core import RunMetrics, across_families, family_metrics
from metrics.stats import bootstrap, percentile_ci

RER_MIN_REL_REDUCTION = 0.30
SUCCESS_MAX_DROP = 0.02           # absolute (2 pp)
REGRESSION_MAX_EXCESS = 0.01      # absolute (1 pp)
TOKENS_MAX_RATIO = 1.10

Table = dict[str, dict[str, dict[int, RunMetrics]]]   # config -> family -> seed -> metrics


def common_seeds(table: Table, *configs: str) -> dict[str, list[int]]:
    fams = set.intersection(*[set(table.get(c, {})) for c in configs])
    out = {}
    for f in sorted(fams):
        seeds = sorted(set.intersection(*[set(table[c][f]) for c in configs]))
        if seeds:
            out[f] = seeds
    return out


def config_summary(table: Table, config: str, sample: dict[str, list[int]]) -> dict:
    per_family = {}
    for fam, seeds in sample.items():
        runs = [table[config][fam][s] for s in seeds]
        per_family[fam] = family_metrics(runs, table.get("C0", {}).get(fam))
    return across_families(per_family)


def _rel_reduction(base, x):
    return None if base in (None, 0) or x is None else (base - x) / base


def _diff(a, b):
    return None if a is None or b is None else a - b


def _ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


def _row(name, value, ci, threshold, ok) -> dict:
    return {"condition": name, "value": value, "ci95": ci, "threshold": threshold,
            "pass": bool(ok) if ok is not None else None}


def win_table(table: Table, config: str, n_boot: int = 2000, boot_seed: int = 0) -> dict:
    """Five conditions of `config` vs C1 (+ the mechanism condition if A1 and C2 exist)."""
    seeds = common_seeds(table, "C1", config)
    if not seeds:
        return {"config": config, "rows": [], "note": "no (family, seed) pairs shared with C1"}

    def both(sample):
        return config_summary(table, "C1", sample), config_summary(table, config, sample)

    point_c1, point_x = both(seeds)
    boots = [both(s) for s in _samples(seeds, n_boot, boot_seed)]

    def stat(fn):
        val = fn(point_c1, point_x)
        return val, percentile_ci([fn(b1, bx) for b1, bx in boots])

    rows = []
    v, ci = stat(lambda c1, x: _rel_reduction(c1["rer"], x["rer"]))
    rows.append(_row("RER relative reduction >= 30% (95% CI excludes 0)", v, ci, RER_MIN_REL_REDUCTION,
                     None if v is None or ci is None else v >= RER_MIN_REL_REDUCTION and ci[0] > 0))
    v, ci = stat(lambda c1, x: _diff(x["success"], c1["success"]))
    rows.append(_row("success rate not worse than -2 pp", v, ci, -SUCCESS_MAX_DROP,
                     None if v is None else v >= -SUCCESS_MAX_DROP - 1e-12))
    v, ci = stat(lambda c1, x: _diff(x["regression"], c1["regression"]))
    rows.append(_row("regression rate <= C1 + 1 pp (needs C0)", v, ci, REGRESSION_MAX_EXCESS,
                     None if v is None else v <= REGRESSION_MAX_EXCESS + 1e-12))
    v, ci = stat(lambda c1, x: _diff(x["memory_induced"], c1["memory_induced"]))
    rows.append(_row("memory-induced error on trap <= C1", v, ci, 0.0, None if v is None else v <= 1e-12))
    v, ci = stat(lambda c1, x: _ratio(x["tokens_per_success"], c1["tokens_per_success"]))
    rows.append(_row("tokens per successful task <= C1 + 10%", v, ci, TOKENS_MAX_RATIO,
                     None if v is None else v <= TOKENS_MAX_RATIO + 1e-12))
    return {"config": config, "seeds": seeds, "c1": point_c1, "x": point_x, "rows": rows,
            "verdict": verdict(rows, mechanism(table, n_boot, boot_seed))}


def mechanism(table: Table, n_boot: int = 2000, boot_seed: int = 0) -> dict | None:
    """A1 (no verifier gating) significantly worse than C2 in RER or memory-induced error."""
    seeds = common_seeds(table, "A1", "C2")
    if not seeds:
        return None
    out = {}
    for key in ("rer", "memory_induced"):
        def fn(sample, key=key):
            return _diff(config_summary(table, "A1", sample)[key], config_summary(table, "C2", sample)[key])
        point = fn(seeds)
        ci = percentile_ci([fn(s) for s in _samples(seeds, n_boot, boot_seed)])
        out[key] = {"a1_minus_c2": point, "ci95": ci, "significant": bool(ci and ci[0] > 0)}
    out["pass"] = out["rer"]["significant"] or out["memory_induced"]["significant"]
    return out


def verdict(rows: list[dict], mech: dict | None) -> str:
    """Go / Iterate / Pivot per DECISION.md (advisory; the decision log is written by people)."""
    if not rows or rows[0]["pass"] is None:
        return "undetermined (RER not computable)"
    if not rows[0]["pass"]:
        return "PIVOT (RER does not improve vs C1)"
    if all(r["pass"] for r in rows) and mech and mech["pass"]:
        return "GO (all five conditions + mechanism)"
    if all(r["pass"] for r in rows) and mech is None:
        return "ITERATE (five conditions pass; mechanism needs A1 and C2 runs)"
    return "ITERATE (RER improves, another condition fails)"


def _samples(seeds: dict[str, list[int]], n_boot: int, boot_seed: int) -> list[dict[str, list[int]]]:
    return bootstrap(lambda s: s, seeds, n_boot, boot_seed)
