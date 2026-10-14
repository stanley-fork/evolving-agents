"""D3 runner (docs/BRIEF-D3.md): one isolated director — a copy of the live memory, its own log and agentvcs
store — planned on the routing probes before and after one lane move and one reflection. Lanes are read from
the plan's JSON; no judge. Needs the local lane online (the user's llama.cpp) for the planner to see it.

    python run_d3.py --config ~/.config/director/director.json
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

from director.core import Director, local_online  # noqa: E402

K, CAP = 2, 15.0
EPISODE = ("Draft the release note for agentvcs's new bisect golden cases from these changes: bisect accepts an audit "
           "bundle; 14 conformance cases pin the search, the probe's environment and its failures; a real probe "
           "re-executes the toy pipeline from its checkpoint. Two short paragraphs for the GitHub release.")
CORRECTION = ("Public-facing text — release notes, READMEs, the website, announcements — is drafted on the frontier "
              "even when it is text-only; private, internal text stays on the local lane.")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def spent(root: Path) -> float:
    return round(sum(r.get("cost") or 0 for r in rows(root / "costs.jsonl")), 4)


def charge(root: Path, what: str, cost) -> None:
    with (root / "costs.jsonl").open("a") as f:
        f.write(json.dumps({"what": what, "cost": cost}) + "\n")


def plan_lanes(d: Director, request: str) -> dict:
    assert d.claim("planning", ("idle", "awaiting_run"))
    d.handle(request)
    p = d.plan
    d.phase = "idle"                       # a probe's plan is read, never run
    return {"lanes": [t["lane"] for t in p["tasks"]], "cites": [c for t in p["tasks"] for c in t.get("cites") or []],
            "cost": p.get("cost")}


def arm(d: Director, root: Path, name: str) -> None:
    out = root / f"{name}.jsonl"
    done = {(r["probe"], r["sample"]) for r in rows(out)}
    for line in (HERE / "probes.jsonl").read_text().splitlines():
        pr = json.loads(line)
        for s in range(K):
            if (pr["id"], s) in done:
                continue
            if spent(root) >= CAP:
                raise SystemExit(f"cap reached (${CAP})")
            r = plan_lanes(d, pr["request"])
            score = sum(l == pr["expected_lane"] for l in r["lanes"]) / max(len(r["lanes"]), 1)
            rec = {"probe": pr["id"], "sample": s, "expected": pr["expected_lane"], **r, "score": round(score, 4)}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            charge(root, f"{name} {pr['id']} s{s}", r["cost"])
            print(f"[d3] {name} {pr['id']} s{s} · lanes {r['lanes']} · expected {pr['expected_lane']} · "
                  f"{score:.2f} · ${r['cost'] or 0:.3f}", flush=True)


def summary(root: Path, name: str) -> dict:
    rs = rows(root / f"{name}.jsonl")
    pub = [r["score"] for r in rs if r["expected"] == "frontier"]
    ctl = [r["score"] for r in rs if r["expected"] == "local"]
    return {"public_to_frontier": round(sum(pub) / len(pub), 4) if pub else None,
            "internal_kept_local": round(sum(ctl) / len(ctl), 4) if ctl else None, "samples": len(rs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    live = json.loads(Path(a.config).expanduser().read_text())
    root = Path(live["log"]).expanduser().parent.parent / "d3"
    root.mkdir(parents=True, exist_ok=True)
    mem = root / "sandbox" / "memory"
    if not mem.exists():                   # an isolated copy: D3 never touches the live memory
        shutil.copytree(Path(live["memory_dir"]).expanduser(), mem)
    cfg = {**live, "memory_dir": str(mem), "log": str(root / "events.jsonl"), "auto_run": False}
    if not local_online(cfg.get("local_base_url")):
        raise SystemExit("the local lane is offline: start the user's llama.cpp first (the planner must see it online)")
    d = Director(cfg)

    arm(d, root, "R0")
    r0 = summary(root, "R0")
    print("[d3] R0", json.dumps(r0), flush=True)
    if r0["public_to_frontier"] >= 0.5:
        print(json.dumps({"verdict": "STOP — no headroom", "R0": r0, "spent_usd": spent(root)}, indent=1))
        return 0

    if not (root / "episode.json").exists():
        assert d.claim("planning", ("idle",))
        d.handle(EPISODE)
        charge(root, "episode plan", d.plan.get("cost"))
        local = [t["id"] for t in d.plan["tasks"] if t["lane"] == "local"]
        for t in local:                     # the user's move, scripted: the release note goes to the frontier
            d.move(t, "frontier")
        assert d.claim("running", ("awaiting_run",))
        d.execute()
        ends = [e for e in d.log.since(0) if e["kind"] == "task_end" and e["lane"] == "frontier"]
        charge(root, "episode run", sum(e.get("cost") or 0 for e in ends))
        assert d.claim("reflecting", ("idle",))
        d.reflect(CORRECTION)
        pid, prop = sorted(d.proposals.items())[-1]
        charge(root, "reflect", prop.get("cost"))
        res = d.decide(pid, True) if not prop["lint"] and prop["applied"] else {"status": "not acceptable"}
        ep = {"planned_local": local, "moved": [m["task"] for m in d.plan["moves"]], "applied": prop["applied"],
              "rejected": prop["rejected"], "lint": prop["lint"], "decision": res.get("status"),
              "patch": res.get("patch_id"), "dimensions": res.get("dimensions")}
        (root / "episode.json").write_text(json.dumps(ep, indent=1))
        print("[d3] episode", json.dumps(ep), flush=True)
    ep = json.loads((root / "episode.json").read_text())

    arm(d, root, "R1")
    r1 = summary(root, "R1")
    learned = r1["public_to_frontier"] >= 0.8
    kept = r1["internal_kept_local"] >= 0.75
    verdict = ("LEARNS THE LANE" if learned and kept else "OVER-GENERALISES" if learned else "DOES NOT LEARN")
    out = {"R0": r0, "R1": r1, "episode": ep, "spent_usd": spent(root), "verdict": verdict,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
