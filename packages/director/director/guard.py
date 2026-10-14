"""The D0 guard behind Accept: before an edit is kept, the candidate memory re-plans D0's 14 cases (Sonnet 5.5
through `claude -p`, read-only tools, the blind judge) and must hold `min` — the D2 rule that caught nothing broken
(0.964 against T's 0.929). It reuses D0's runner; the private config (`config`) supplies the placeholders.

    "guard": {"enabled": true, "k": 1, "min": 0.86, "config": "~/.config/director/d0.json", "cap_usd": 15}

k = 1 costs about US$6.5 and ten minutes; k = 2 is D0's own sampling.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_guard(candidate: Path, workdir: Path, g: dict, progress=print) -> dict:
    sys.path.insert(0, str(HERE.parent / "d0"))
    import run_d0 as R
    c = R.cfg(g["config"])
    c["private_dir"] = str(workdir)
    R.EXPERIMENT, R.K, R.CAP = ".", int(g.get("k", 1)), float(g.get("cap_usd", 15))
    R.CASES_FILE = HERE.parent / "d0" / "cases.jsonl"
    R.print = lambda *a, **k: progress(" ".join(str(x) for x in a))     # the runner's progress lines, as events
    d = R.arm_dir(c, "G")
    if (d / "sandbox").exists():
        shutil.rmtree(d / "sandbox")
    shutil.copytree(candidate, d / "sandbox" / "memory")
    shutil.copy(HERE / "DIRECTOR.md", d / "system.md")
    R.run(c, "G")
    R.judge(c, "G")
    s = R.score(c, "G")
    mean, floor = s["mean"], float(g.get("min", 0.86))
    complete = s["n_cases"] == 14
    return {"mean": mean, "min": floor, "k": R.K, "n_cases": s["n_cases"], "samples": s["samples"],
            "below_one": {k: v for k, v in s["cases"].items() if v < 1}, "cost_usd": R.spent(c),
            "passed": bool(complete and mean is not None and mean >= floor)}
