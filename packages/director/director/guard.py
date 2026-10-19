"""The D0 guard behind Accept: before an edit is kept, the candidate memory re-plans D0's 14 cases (Sonnet 5.5
through `claude -p`, read-only tools, the blind judge) and must hold `min` — the D2 rule that caught nothing broken
(0.964 against T's 0.929). It reuses D0's runner; the private config (`config`) supplies the placeholders.

    "guard": {"enabled": true, "k": 1, "min": 0.86, "config": "~/.config/director/d0.json", "cap_usd": 15}

k = 1 costs about US$6.5 and ten minutes; k = 2 is D0's own sampling.

`"mode": "paired"` (redesign 1 of D4, after an absolute guard at k = 1 refused an edit on cases that split between
samples anyway): the current memory is run on the same cases too — cached by its content hash, so it is paid once per
memory version — and the candidate is refused only if it loses `max_net_losses` (4) or more cases than it wins:
above the three cases D0 measured splitting between samples of one memory.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def memory_hash(memory: Path) -> str:
    """The memory's version (`director.memory.format.version`): the baseline is stored under it."""
    from .memory.format import version
    return version(memory)


def paired_verdict(cand: dict, base: dict, slack: float | None = None, max_net_losses: int = 4) -> dict:
    """cand/base: case -> score. The candidate is refused if it loses `max_net_losses` or more cases than it wins (or,
    with `slack`, if its mean falls below the base's by more than it). Both must cover all 14 cases. The default, 4,
    sits above the spread D0 measured: three cases (c06, c10, c12) split between samples of the same memory."""
    common = set(cand) & set(base)
    wins = sum(cand[k] > base[k] for k in common)
    losses = sum(cand[k] < base[k] for k in common)
    mc = sum(cand.values()) / len(cand) if cand else None
    mb = sum(base.values()) / len(base) if base else None
    ok = (len(cand) == 14 and len(base) == 14 and losses - wins < max_net_losses
          and (slack is None or mc >= mb - slack))
    return {"wins": wins, "losses": losses, "passed": bool(ok)}


def _arm(R, c, arm: str, memory: Path, retries: int = 2) -> dict:
    """One arm on D0's cases. A plan that errored or a verdict that did not parse is retried (`R.run` and `R.judge`
    resume what is missing) up to `retries` times; what is still missing leaves the arm incomplete, never scored as a
    failure. The arm's directory must be fresh: `R.run` resumes whatever it finds there."""
    d = R.arm_dir(c, arm)
    assert not (d / "plans.jsonl").exists(), f"stale arm directory {d}: a guard never resumes another guard's plans"
    shutil.copytree(memory, d / "sandbox" / "memory")
    shutil.copy(HERE / "DIRECTOR.md", d / "system.md")
    for _ in range(retries + 1):
        R.run(c, arm)
        R.judge(c, arm)
        s = R.score(c, arm)
        if s["n_cases"] == 14 and s["samples"] == 14 * R.K:
            break
    return s


def run_guard(candidate: Path, workdir: Path, g: dict, progress=print, current: Path | None = None) -> dict:
    sys.path.insert(0, str(HERE.parent / "d0"))
    import run_d0 as R
    c = R.cfg(g["config"])
    c["private_dir"] = str(workdir)
    R.EXPERIMENT, R.K, R.CAP = ".", int(g.get("k", 1)), float(g.get("cap_usd", 15))
    R.CASES_FILE = HERE.parent / "d0" / "cases.jsonl"
    R.print = lambda *a, **k: progress(" ".join(str(x) for x in a))     # the runner's progress lines, as events
    if g.get("mode") == "targeted":
        return _targeted(R, c, Path(candidate), Path(current), g, progress)
    s = _arm(R, c, "G", candidate)
    mean, floor = s["mean"], float(g.get("min", 0.86))
    complete = s["n_cases"] == 14
    out = {"mode": g.get("mode", "absolute"), "mean": mean, "min": floor, "k": R.K, "n_cases": s["n_cases"],
           "samples": s["samples"], "below_one": {k: v for k, v in s["cases"].items() if v < 1},
           "cost_usd": R.spent(c)}
    if g.get("mode") == "paired" and current is not None:
        cache = Path(workdir).parent / "cache" / f"{memory_hash(current)}-k{R.K}.json"
        if cache.exists():
            base = json.loads(cache.read_text())
            progress(f"[guard] current memory {cache.stem}: cached, mean {base['mean']}")
        else:
            progress("[guard] current memory: measuring it on the same cases (paid once per memory version)")
            b = _arm(R, c, "C", current)
            base = {"mean": b["mean"], "cases": b["cases"], "n_cases": b["n_cases"]}
            if b["n_cases"] == 14:                       # an incomplete measurement is never cached
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(base))
        v = paired_verdict(s["cases"], base["cases"], g.get("slack"), int(g.get("max_net_losses", 4)))
        out.update(current_mean=base["mean"], current_n_cases=base["n_cases"], wins=v["wins"], losses=v["losses"],
                   max_net_losses=int(g.get("max_net_losses", 4)), cost_usd=R.spent(c), passed=v["passed"],
                   incomplete=not (complete and base["n_cases"] == 14))
    else:
        out["passed"] = bool(complete and mean is not None and mean >= floor)
    return out


# ------------------------------------------------------------------ targeted guard (D6, docs/BRIEF-D6.md)

CITE = __import__("re").compile(r"\[\[([A-Za-z0-9_]+)#(\w+)\]\]")


def cited_statements(plans: list[dict]) -> dict[str, set[str]]:
    """case -> the statements its plans cite, over every sample of a baseline."""
    out: dict[str, set[str]] = {}
    for r in plans:
        if r.get("plan"):
            out.setdefault(r["case"], set()).update(f"{p}#{i}" for p, i in CITE.findall(r["plan"]))
    return out


def changed_statements(current: Path, candidate: Path) -> set[str]:
    """The statements the candidate removes or whose When, Do or Because it modifies (added ones are not here: a
    positive probe checks those)."""
    from .memory.format import statements
    out = set()
    for p in Path(current).glob("*.md"):
        if p.name in ("Index.md", "README.md"):
            continue
        before = statements(p.read_text())
        q = Path(candidate) / p.name
        after = statements(q.read_text()) if q.exists() else {}
        out |= {f"{p.stem}#{sid}" for sid, block in before.items()
                if sid not in after or _substance(after[sid]) != _substance(block)}
    return out


def _substance(block: str) -> tuple[str, str, str]:
    """When, Do and Because — what a plan follows. A change to Links alone is navigation (a dangling link is the
    lint's business), so it does not make the statements that carried the link relevant."""
    from .memory.format import field
    return field(block, "When"), field(block, "Do"), field(block, "Because")


def relevant_cases(changed: set[str], cites: dict[str, set[str]]) -> list[str]:
    return sorted(case for case, cs in cites.items() if cs & changed)


def targeted_verdict(base: dict[str, list[float]], cand: dict[str, list[float]], reliable: float = 0.75,
                     fail: float = 1 / 3) -> dict:
    """Refuse iff some relevant case was reliable before (baseline mean >= `reliable`) and fails now (candidate mean
    <= `fail`). Every relevant case must have candidate samples."""
    mean = lambda xs: sum(xs) / len(xs)
    broke = [k for k in cand if cand[k] and mean(base[k]) >= reliable and mean(cand[k]) <= fail + 1e-9]
    complete = all(cand.get(k) for k in cand)
    return {"broke": broke, "passed": complete and not broke, "complete": complete,
            "per_case": {k: [round(mean(base[k]), 3), round(mean(cand[k]), 3) if cand[k] else None] for k in cand}}


def baseline_path(memory: Path) -> Path:
    return Path(memory).parent / "guards" / "baseline" / f"{memory_hash(memory)}.json"


def _targeted(R, c, candidate: Path, current: Path, g: dict, progress) -> dict:
    """D6's guard (VALIDATED, docs/BRIEF-D6.md): re-plan only the D0 cases whose baseline plans cite a statement the
    edit removes or changes in substance, k = 3; refuse iff one of them was reliable (>= 0.75) and now fails (<= 1/3).
    The baseline is per memory version; an accepted edit carries it forward, the relevant cases' samples replaced."""
    bp = baseline_path(current)
    if not bp.exists():
        progress(f"[guard] no baseline for this memory ({bp.name}); build one before Accept can be guarded")
        return {"mode": "targeted", "incomplete": True, "passed": False, "reason": "no baseline"}
    base = json.loads(bp.read_text())
    changed = changed_statements(current, candidate)
    rel = relevant_cases(changed, {k: set(v["cites"]) for k, v in base.items()})
    out = {"mode": "targeted", "changed": sorted(changed), "relevant": rel, "k": int(g.get("k", 3))}
    if not rel:
        progress("[guard] no statement a D0 plan relied on was changed: nothing to re-plan (an added statement is "
                 "checked by its own probe, not here)")
        _carry(base, {}, {}, candidate)
        return {**out, "passed": True, "incomplete": False, "per_case": {}, "broke": [], "cost_usd": 0.0,
                "mean": None}
    progress(f"[guard] re-planning {len(rel)} case(s) that relied on what changed: {', '.join(rel)}")
    R.K = int(g.get("k", 3))
    d = R.arm_dir(c, "T")
    d.mkdir(parents=True, exist_ok=True)
    cf = d / "cases.jsonl"
    cf.write_text("\n".join(l for l in (HERE.parent / "d0" / "cases.jsonl").read_text().splitlines()
                            if json.loads(l)["id"] in rel) + "\n")
    R.CASES_FILE = cf
    shutil.copytree(candidate, d / "sandbox" / "memory")
    shutil.copy(HERE / "DIRECTOR.md", d / "system.md")
    for _ in range(3):
        R.run(c, "T")
        R.judge(c, "T")
        got = _samples(R.rows(d / "verdicts.jsonl"))
        if all(len(got.get(k, [])) == R.K for k in rel):
            break
    cand = {k: got.get(k, []) for k in rel}
    v = targeted_verdict({k: base[k]["samples"] for k in rel}, cand)
    if v["complete"]:
        _carry(base, cand, cited_statements(R.rows(d / "plans.jsonl")), candidate)
    return {**out, **v, "incomplete": not v["complete"], "cost_usd": R.spent(c),
            "mean": round(sum(sum(x) / len(x) for x in cand.values() if x) / len(cand), 4)}


def _samples(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("items"):
            out.setdefault(r["case"], []).append(sum(x["met"] for x in r["items"]) / len(r["items"]))
    return out


def _carry(base: dict, cand: dict, cites: dict, candidate: Path) -> None:
    """The candidate memory's baseline: the current one, with the re-planned cases' samples and citations replaced."""
    new = {k: dict(v) for k, v in base.items()}
    for k, xs in cand.items():
        new[k] = {"samples": xs, "cites": sorted(cites.get(k, set()))}
    bp = baseline_path(candidate)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(new, indent=1))


# ------------------------------------------------------------------ positive probe (D7, docs/BRIEF-D7.md)

def positive_probe(probes: list[dict], current: Path, candidate: Path, workdir: Path, g: dict, progress=print) -> dict:
    """Run the edit's own probes on the current memory and the candidate (D0's runner and blind judge, k = 2).
    TEACHES: candidate >= 0.67 and candidate - current >= 0.33; REDUNDANT: current already >= 0.67; else NO EFFECT."""
    sys.path.insert(0, str(HERE.parent / "d0"))
    import run_d0 as R
    if not probes:
        return {"verdict": "NO PROBE", "teaches": False}
    c = R.cfg(g["config"])
    c["private_dir"] = str(workdir)
    R.EXPERIMENT, R.K, R.CAP = ".", int(g.get("probe_k", 2)), float(g.get("cap_usd", 15))
    R.print = lambda *a, **k: progress(" ".join(str(x) for x in a))
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    cf = root / "probes.jsonl"
    cf.write_text("".join(json.dumps({"id": f"probe-{i}", "situation": p["situation"],
                                      "checklist": [{"id": f"probe-{i}/{j}", "text": t}
                                                    for j, t in enumerate(p["checklist"], 1)]}) + "\n"
                          for i, p in enumerate(probes, 1)))
    R.CASES_FILE = cf
    means = {}
    for arm, mem in (("PC", current), ("PN", candidate)):
        d = R.arm_dir(c, arm)
        if not (d / "sandbox").exists():
            shutil.copytree(mem, d / "sandbox" / "memory")
            shutil.copy(HERE / "DIRECTOR.md", d / "system.md")
        for _ in range(3):
            R.run(c, arm)
            R.judge(c, arm)
            got = _samples(R.rows(d / "verdicts.jsonl"))
            if all(len(got.get(f"probe-{i}", [])) == R.K for i in range(1, len(probes) + 1)):
                break
        xs = [x for v in got.values() for x in v]
        means[arm] = round(sum(xs) / len(xs), 4) if xs else None
    cur, cand = means["PC"], means["PN"]
    if cur is None or cand is None:
        v = "INCOMPLETE"
    elif cur >= 0.67:
        v = "REDUNDANT"
    elif cand >= 0.67 and cand - cur >= 0.33:
        v = "TEACHES"
    else:
        v = "NO EFFECT"
    return {"verdict": v, "teaches": v == "TEACHES", "current": cur, "candidate": cand, "cost_usd": R.spent(c)}
