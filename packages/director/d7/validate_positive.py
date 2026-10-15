"""D7 (docs/BRIEF-D7.md): validate the positive probe on D2's five real episodes, reflected again (with probe
generation) on the memory before D2's edits. Resumable per lesson (a finished lesson's record is reused).

    python validate_positive.py --config ~/.config/director/d0.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from director.core import Log  # noqa: E402
from director.guard import positive_probe  # noqa: E402
from director.learn import apply_edits, leak, new_problems, reflect  # noqa: E402

EXPECT = {"A": "TEACHES", "B": "TEACHES", "I": "TEACHES", "F": "REDUNDANT", "E": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).expanduser().read_text())
    priv = Path(cfg["private_dir"]).expanduser()
    v0 = priv / "d0" / "arm-T" / "sandbox" / "memory"
    root = priv / "d7"
    root.mkdir(parents=True, exist_ok=True)
    g = {"config": a.config, "probe_k": 2, "cap_usd": 12}
    out = {}
    for line in (HERE.parent / "d2" / "episodes2.jsonl").read_text().splitlines():
        ep = json.loads(line)
        L = ep["lesson"]
        rec_file = root / f"{L}.json"
        if rec_file.exists():
            out[L] = json.loads(rec_file.read_text())
            print(f"[d7] {L}: reused {out[L]['probe']['verdict']}", flush=True)
            continue
        sb = root / f"{L}-reflect" / "memory"
        if sb.parent.exists():
            shutil.rmtree(sb.parent)
        shutil.copytree(v0, sb)
        r = reflect(ep["episode"], sb, Log(root / f"{L}-reflect-events.jsonl"))
        cand = root / f"{L}-candidate" / "memory"
        res = apply_edits(v0, r["edits"] or [], cand)
        lint = new_problems(v0, cand)
        leaks = [leak(ep["episode"], p["situation"]) for p in r["probes"]]
        pr = positive_probe(r["probes"], v0, cand, root / f"{L}-probe", g,
                            lambda line: print(f"  {line}", flush=True) if line.startswith("[") else None)
        rec = {"lesson": L, "applied": res["applied"], "rejected": res["rejected"], "lint": lint,
               "probes": r["probes"], "leaks": leaks, "reflect_cost": r["cost"], "probe": pr,
               "expected": EXPECT[L]}
        rec_file.write_text(json.dumps(rec, indent=1))
        out[L] = rec
        print(f"[d7] {L}: {pr['verdict']} (current {pr.get('current')} → candidate {pr.get('candidate')}) · "
              f"edits {res['applied']} · leaks {leaks} · ${(r['cost'] or 0) + (pr.get('cost_usd') or 0):.2f}", flush=True)
    teaches = all(out[L]["probe"]["verdict"] == "TEACHES" for L in ("A", "B", "I"))
    redundant = out["F"]["probe"]["verdict"] == "REDUNDANT"
    spent = sum((x["reflect_cost"] or 0) + (x["probe"].get("cost_usd") or 0) for x in out.values())
    report = {"by_lesson": {L: {"verdict": x["probe"]["verdict"], "expected": x["expected"], "current": x["probe"].get("current"),
                                "candidate": x["probe"].get("candidate"), "leaks": x["leaks"]} for L, x in out.items()},
              "spent_usd": round(spent, 2),
              "verdict": "VALIDATED" if teaches and redundant else "MISSES" if not teaches else "NOT SPECIFIC",
              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
