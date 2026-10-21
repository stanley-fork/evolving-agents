"""F1O runner (docs/BRIEF-F1O.md): F1's arm A, byte for byte, planned and judged by Opus 5.5 (headroom only).
Standard library only; D0's runner plans, judges and scores, J0's judge re-judges.

    python run_f1o.py prepare --config ~/.config/director/d0.json   # copy C0's and C1's sandboxes, hashes checked
    python run_f1o.py smoke --config ...                            # one plan per arm (c01 s0), costs printed
    python run_f1o.py run --arm AO --config ...                     # resumable; the smoke plan is reused
    python run_f1o.py judge --arm AO | gemini --arm AO | report --config ...
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("d0", "c0"):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parent))
import run_d0   # noqa: E402
import rejudge  # noqa: E402
import run_c0   # noqa: E402  (sets run_d0's globals on import; set here last)

run_d0.EXPERIMENT = "f1o"
run_d0.CASES_FILE = HERE.parent / "f1" / "cases.jsonl"
run_d0.CALL_BUDGET = 2.50
run_d0.CAP = 10.0            # until the person sets the cap from the smoke test
OPUS = "opus"
# Every claude call in this experiment — plans and the judge — goes to Opus 5.5 unless a call names a model.
_claude = run_d0.claude
run_d0.claude = functools.partial(lambda *a, model=OPUS, **k: _claude(*a, model=model, **k))
ARMS = {"AO": ("f1", "A"), "TO": ("f1", "T")}


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "f1o"


def prepare(c: dict) -> dict:
    out = {}
    for arm, (exp, src_arm) in ARMS.items():
        src, d = Path(c["private_dir"]) / exp / f"arm-{src_arm}", run_d0.arm_dir(c, arm)
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(src / "sandbox", d / "sandbox")
        shutil.copy(src / "system.md", d / "system.md")
        assert run_c0.digest(d / "sandbox") == run_c0.digest(src / "sandbox"), f"parity: {arm}"
        out[arm] = {"from": f"{exp}/arm-{src_arm}", "sandbox_sha": run_c0.digest(d / "sandbox")}
    root(c).mkdir(parents=True, exist_ok=True)
    (root(c) / "prepare.json").write_text(json.dumps(out, indent=1))
    return out


def smoke(c: dict) -> dict:
    first = run_d0.CASES_FILE.read_text().splitlines()[0]
    tmp = root(c) / "smoke-case.jsonl"
    tmp.write_text(first + "\n")
    saved_file, saved_k = run_d0.CASES_FILE, run_d0.K
    run_d0.CASES_FILE, run_d0.K = tmp, 1
    try:
        for arm in ARMS:
            run_d0.run(c, arm)
    finally:
        run_d0.CASES_FILE, run_d0.K = saved_file, saved_k
    out = {}
    for arm in ARMS:
        rows = run_d0.rows(run_d0.arm_dir(c, arm) / "plans.jsonl")
        r = rows[-1]
        out[arm] = {"case": r["case"], "cost_usd": r["cost_usd"], "turns": r["turns"], "wall_s": r["wall_s"],
                    "error": bool(r["error"] or not r["plan"]),
                    "full_arm_estimate_usd": round((r["cost_usd"] or 0) * 20, 2)}
    return out


def report(c: dict) -> dict:
    out = {}
    for arm in ARMS:
        s_ = run_d0.score(c, arm)
        g = rejudge.score(run_d0.rows(rejudge.out_dir(c, arm) / "verdicts.jsonl"))
        plans = run_d0.rows(run_d0.arm_dir(c, arm) / "plans.jsonl")
        ok = [r for r in plans if r.get("plan")]
        out[arm] = {"opus": s_["mean"], "gemini": g["mean"], "cases": s_["cases"], "samples": s_["samples"],
                    "cost_per_plan": round(sum(r.get("cost_usd") or 0 for r in plans) / max(len(plans), 1), 4),
                    "turns_mean": round(sum(r.get("turns") or 0 for r in ok) / max(len(ok), 1), 2),
                    "errors": sum(1 for r in plans if r.get("error") or not r.get("plan"))}
    a = out["AO"]
    if a["opus"] is not None:
        out["verdict"] = ("STOP — no headroom with Opus" if a["opus"] >= run_d0.A_THRESHOLD
                          else "ROOM — TO needs the person's cap")
    out["spent_claude_usd"] = run_d0.spent(c)
    out["spent_gemini_run_usd"] = round(rejudge.run_spent(c), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "smoke", "run", "judge", "gemini", "report"])
    ap.add_argument("--arm", default="AO", choices=list(ARMS))
    ap.add_argument("--cap", type=float, default=None, help="the person's cap for the whole experiment")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    if a.cap is not None:
        run_d0.CAP = a.cap
    if a.cmd in ("prepare", "smoke"):
        print(json.dumps({"prepare": prepare, "smoke": smoke}[a.cmd](c), indent=1))
    elif a.cmd == "run":
        run_d0.run(c, a.arm)
    elif a.cmd == "judge":
        run_d0.judge(c, a.arm)
    elif a.cmd == "gemini":
        rejudge.judge(c, a.arm)
    else:
        rep = report(c)
        (root(c) / "report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
