"""D2 runner (docs/BRIEF-D2.md). Reuses D0's runner for the arms and the blind judge, with D2's probes.

    python run_d2.py t0 --config ~/.config/director/d0.json        # headroom: today's memory on the probes
    python run_d2.py reflect --config ...                          # episodes -> edits -> lint -> memory v1
    python run_d2.py t1 --config ...                               # the edited memory on the probes
    python run_d2.py guard --config ...                            # the edited memory on D0's cases
    python run_d2.py report --config ...

Everything private lives under <private_dir>/d2/; the report holds scores, counts and lint results only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d0"))
sys.path.insert(0, str(HERE.parent))

import run_d0 as R  # noqa: E402
from director.core import Log  # noqa: E402
from director.learn import apply_edits, new_problems, reflect  # noqa: E402

R.EXPERIMENT, R.CAP = "d2", 45.0          # the brief's cap for D2, raised 40 -> 45 by the user for redesign 1
PROBES, D0_CASES = HERE / "probes.jsonl", HERE.parent / "d0" / "cases.jsonl"
T0_THRESHOLD, LEARN_DELTA, GUARD_MIN = 0.85, 0.30, 0.86


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "d2"


def make_arm(c: dict, arm: str, memory: Path) -> None:
    d = R.arm_dir(c, arm)
    if (d / "sandbox").exists():
        shutil.rmtree(d / "sandbox")
    shutil.copytree(memory, d / "sandbox" / "memory")
    shutil.copy(HERE.parent / "director" / "DIRECTOR.md", d / "system.md")


def run_arm(c: dict, arm: str, cases_file: Path) -> dict:
    R.CASES_FILE = cases_file
    R.run(c, arm)
    R.judge(c, arm)
    return R.score(c, arm)


def t0(c: dict) -> dict:
    make_arm(c, "T0", Path(c["private_dir"]) / "d0" / "arm-T" / "sandbox" / "memory")
    s = run_arm(c, "T0", PROBES)
    s["verdict"] = "STOP — no headroom" if s["mean"] >= T0_THRESHOLD else "GO: reflect"
    return s


def do_reflect(c: dict, episodes: Path = HERE / "episodes.jsonl", v1_name: str = "memory-v1",
               tag: str = "") -> dict:
    """Episodes in order; each reflects on the memory the previous ones left. An episode whose edits introduce a
    lint problem is dropped whole. The result is memory v1."""
    v0 = R.arm_dir(c, "T0") / "sandbox" / "memory"
    cur, log = v0, Log(root(c) / "reflect-events.jsonl")
    record = []
    for i, line in enumerate(Path(episodes).read_text().splitlines(), 1):
        ep = json.loads(line)
        sb = root(c) / "reflect-sandbox"          # the reflection sees ./memory and nothing else (no probes)
        if sb.exists():
            shutil.rmtree(sb)
        shutil.copytree(cur, sb / "memory")
        r = reflect(ep["episode"], sb / "memory", log)
        nxt = root(c) / f"memory-after-{tag}{ep['lesson']}"
        res = apply_edits(cur, r["edits"] or [], nxt)
        probs = new_problems(cur, nxt)
        kept = bool(r["edits"]) and not probs and res["applied"]
        record.append({"lesson": ep["lesson"], "proposed": len(r["edits"] or []), "applied": res["applied"],
                       "rejected": res["rejected"], "lint": probs, "kept": bool(kept), "cost_usd": r["cost"],
                       "turns": r["turns"]})
        with (root(c) / "reflect.jsonl").open("a") as f:      # counted by R.spent()
            f.write(json.dumps({"lesson": ep["lesson"], "cost_usd": r["cost"]}) + "\n")
        print(f"[d2] reflect {ep['lesson']}: {len(r['edits'] or [])} proposed · applied {res['applied']} · "
              f"rejected {len(res['rejected'])} · lint {len(probs)} · {'KEPT' if kept else 'DROPPED'} · "
              f"${r['cost'] or 0:.2f}", flush=True)
        if kept:
            cur = nxt
    v1 = root(c) / v1_name
    if v1.exists():
        shutil.rmtree(v1)
    shutil.copytree(cur, v1)
    (root(c) / f"reflect-record{tag}.json").write_text(json.dumps(record, indent=1))
    return {"episodes": len(record), "kept": sum(x["kept"] for x in record),
            "edits_applied": sum(len(x["applied"]) for x in record)}


def t1(c: dict) -> dict:
    make_arm(c, "T1", root(c) / "memory-v1")
    return run_arm(c, "T1", PROBES)


def guard(c: dict) -> dict:
    make_arm(c, "T1guard", root(c) / "memory-v1")
    return run_arm(c, "T1guard", D0_CASES)


def report(c: dict) -> dict:
    R.CASES_FILE = PROBES
    a, b = R.score(c, "T0"), R.score(c, "T1")
    common = sorted(set(a["cases"]) & set(b["cases"]))
    wins = sum(b["cases"][k] > a["cases"][k] for k in common)
    losses = sum(b["cases"][k] < a["cases"][k] for k in common)
    n = wins + losses
    p = sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0
    R.CASES_FILE = D0_CASES
    g = R.score(c, "T1guard")
    out = {"T0": a, "T1": b, "delta": round((b["mean"] or 0) - (a["mean"] or 0), 4), "wins": wins, "losses": losses,
           "sign_p": round(p, 4), "guard": g, "spent_usd": R.spent(c)}
    rec = root(c) / "reflect-record.json"
    if rec.exists():
        out["reflect"] = [{k: x[k] for k in ("lesson", "proposed", "applied", "rejected", "lint", "kept")}
                          for x in json.loads(rec.read_text())]
    learned = out["delta"] >= LEARN_DELTA and wins > losses
    held = g["mean"] is not None and g["mean"] >= GUARD_MIN
    out["verdict"] = ("LEARNS" if learned and held else "LEARNS, BREAKS" if learned else "DOES NOT LEARN")
    return out


# ------------------------------------------------------------------ redesign 1 (docs/BRIEF-D2.md)

PROBES2, KEPT_MAX = HERE / "probes2.jsonl", 0.5


def kept_file(c: dict) -> Path:
    return root(c) / "kept-probes.jsonl"


def screen(c: dict) -> dict:
    """S: today's memory on the 10 probes, k = 1; keep the ones at or under 0.5."""
    make_arm(c, "S", Path(c["private_dir"]) / "d0" / "arm-T" / "sandbox" / "memory")
    R.K = 1
    s = run_arm(c, "S", PROBES2)
    keep = {k for k, v in s["cases"].items() if v <= KEPT_MAX}
    lines = [l for l in PROBES2.read_text().splitlines() if json.loads(l)["id"] in keep]
    kept_file(c).write_text("\n".join(lines) + "\n")
    s.update({"kept": sorted(keep), "verdict": "GO" if len(keep) >= 4 else "STOP — fewer than 4 probes with headroom"})
    return s


def t0c(c: dict) -> dict:
    make_arm(c, "T0c", Path(c["private_dir"]) / "d0" / "arm-T" / "sandbox" / "memory")
    return run_arm(c, "T0c", kept_file(c))


def reflect2(c: dict) -> dict:
    return do_reflect(c, HERE / "episodes2.jsonl", "memory-v1b", tag="r1-")


def t1b(c: dict) -> dict:
    make_arm(c, "T1b", root(c) / "memory-v1b")
    return run_arm(c, "T1b", kept_file(c))


def guard2(c: dict) -> dict:
    make_arm(c, "T1bguard", root(c) / "memory-v1b")
    return run_arm(c, "T1bguard", D0_CASES)


def report2(c: dict) -> dict:
    R.CASES_FILE = kept_file(c)
    a, b = R.score(c, "T0c"), R.score(c, "T1b")
    common = sorted(set(a["cases"]) & set(b["cases"]))
    wins = sum(b["cases"][k] > a["cases"][k] for k in common)
    losses = sum(b["cases"][k] < a["cases"][k] for k in common)
    n = wins + losses
    p = sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0
    R.CASES_FILE = D0_CASES
    g = R.score(c, "T1bguard")
    need = round(0.5 * (1 - (a["mean"] or 0)), 4)
    delta = round((b["mean"] or 0) - (a["mean"] or 0), 4)
    out = {"S": R.score(c, "S") if (R.arm_dir(c, "S") / "verdicts.jsonl").exists() else None, "T0c": a, "T1b": b,
           "delta": delta, "required": need, "wins": wins, "losses": losses, "sign_p": round(p, 4), "guard": g,
           "spent_usd": R.spent(c)}
    rec = root(c) / "reflect-recordr1-.json"
    if rec.exists():
        out["reflect"] = [{k: x[k] for k in ("lesson", "proposed", "applied", "rejected", "lint", "kept")}
                          for x in json.loads(rec.read_text())]
    learned = delta >= need and wins > losses
    held = g["mean"] is not None and g["mean"] >= GUARD_MIN
    out["verdict"] = "LEARNS" if learned and held else "LEARNS, BREAKS" if learned else "DOES NOT LEARN"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["t0", "reflect", "t1", "guard", "report",
                                    "screen", "t0c", "reflect2", "t1b", "guard2", "report2"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = R.cfg(a.config)
    root(c).mkdir(parents=True, exist_ok=True)
    out = {"t0": t0, "reflect": do_reflect, "t1": t1, "guard": guard, "report": report, "screen": screen,
           "t0c": t0c, "reflect2": reflect2, "t1b": t1b, "guard2": guard2, "report2": report2}[a.cmd](c)
    out["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
