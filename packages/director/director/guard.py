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

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def memory_hash(memory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(memory).rglob("*.md")):
        h.update(p.relative_to(memory).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


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
