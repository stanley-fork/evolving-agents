"""F1 runner (docs/BRIEF-F1.md): D0's frozen arms on situations from after the memory was compiled.
Standard library only; D0's runner does the planning, judging and scoring.

    python run_f1.py prepare --config ~/.config/director/d0.json        # copy D0's arms, assert byte parity
    python run_f1.py write --config ~/.config/director/d0.json          # a fresh Opus writes 18 candidates
    python run_f1.py screen --config ~/.config/director/d0.json         # code screen -> f1/cases.jsonl
    python run_f1.py run --arm A --config ...                           # D0's run, on F1's cases
    python run_f1.py judge --arm A --config ...                         # Sonnet, D0's judge
    python run_f1.py gemini --arm A --config ...                        # gemini-4-argon, J0's judge
    python run_f1.py report --config ...

Private: the record, the sandboxes, the writer's workspace, plans and verdicts live under <private_dir>/f1/.
Public: f1/cases.jsonl (placeholders only) and the scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d0"))
import run_d0   # noqa: E402
import rejudge  # noqa: E402

D0 = "d0"
run_d0.EXPERIMENT = "f1"
run_d0.CASES_FILE = HERE / "cases.jsonl"
run_d0.CAP = 30.0                       # the brief's cap for everything through claude -p
WRITER, WRITER_BUDGET, WRITER_TURNS = "opus", 6.0, 80
N_CANDIDATES, N_BANK, N_MIN = 18, 14, 12
FREEZE = "2026-10-15"
JACCARD_MAX = 0.5


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "f1"


def digest(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def prepare(c: dict) -> dict:
    out = {}
    for arm in ("A", "T"):
        src = Path(c["private_dir"]) / D0 / f"arm-{arm}"
        dst = run_d0.arm_dir(c, arm)
        if (dst / "sandbox").exists():
            shutil.rmtree(dst / "sandbox")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "sandbox", dst / "sandbox")
        shutil.copy(src / "system.md", dst / "system.md")
        assert digest(src / "sandbox") == digest(dst / "sandbox"), f"parity: {arm}'s sandbox differs from D0's"
        assert (src / "system.md").read_bytes() == (dst / "system.md").read_bytes(), f"parity: {arm}'s prompt differs"
        out[arm] = {"files": len(digest(dst / "sandbox")), "sandbox_sha": hashlib.sha256(
            json.dumps(digest(dst / "sandbox"), sort_keys=True).encode()).hexdigest()[:16]}
    (root(c) / "parity.json").write_text(json.dumps(out, indent=1))
    return out


def write(c: dict) -> dict:
    w = root(c) / "writer"
    if w.exists():
        shutil.rmtree(w)
    (w / "out").mkdir(parents=True)
    shutil.copytree(run_d0.arm_dir(c, "A") / "sandbox", w / "sources")
    shutil.copytree(root(c) / "record", w / "record")
    d0 = [json.loads(l) for l in (HERE.parent / "d0" / "cases.jsonl").read_text().splitlines() if l.strip()]
    (w / "d0-situations.md").write_text("\n\n".join(f"- {cs['situation']}" for cs in d0) + "\n")
    r = run_d0.claude((HERE / "write.md").read_text(), w, None, "Read,Grep,Glob,Write", WRITER_TURNS, WRITER_BUDGET,
                      model=WRITER)
    with (w / "write.jsonl").open("a") as f:     # counted by run_d0.spent()
        f.write(json.dumps({"cost_usd": r.get("total_cost_usd"), "turns": r.get("num_turns"),
                            "error": r.get("is_error"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    cand = w / "out" / "candidates.jsonl"
    n = sum(1 for l in cand.read_text().splitlines() if l.strip()) if cand.exists() else 0
    return {"candidates": n, "cost_usd": r.get("total_cost_usd"), "turns": r.get("num_turns"), "reply": r.get("result")}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", s.lower())}


def screen(c: dict) -> dict:
    w = root(c) / "writer"
    redact, leaks = rejudge.redactor(c)
    record = {}
    for f in (root(c) / "record").glob("*.json"):
        for pr in json.loads(f.read_text()):
            record[(f.stem, pr["number"])] = pr["createdAt"]
    d0 = [words(json.loads(l)["situation"]) for l in (HERE.parent / "d0" / "cases.jsonl").read_text().splitlines() if l.strip()]
    kept, report = [], []
    for line in (w / "out" / "candidates.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        cs = json.loads(line)
        why = []
        prov = cs.get("provenance", {})
        for it in cs["checklist"]:
            src = prov.get("items", {}).get(it["id"])
            f = w / "sources" / src["file"] if src else None
            if not src or not f.is_file() or norm(src["quote"]) not in norm(f.read_text()):
                why.append(f"parity: {it['id']}")
        created = record.get((prov.get("repo"), prov.get("pr")))
        if not created or created[:10] < FREEZE:
            why.append("not in the record after the freeze")
        if leaks(json.dumps(cs, ensure_ascii=False)):
            why.append("private name")
        overlap = max((len(words(cs["situation"]) & d) / len(words(cs["situation"]) | d) for d in d0), default=0)
        if overlap >= JACCARD_MAX:
            why.append(f"overlaps a D0 case ({overlap:.2f})")
        if not 2 <= len(cs["checklist"]) <= 3:
            why.append("checklist size")
        report.append({"id": cs["id"], "kept": not why and len(kept) < N_BANK, "why": why})
        if not why and len(kept) < N_BANK:
            kept.append({"id": cs["id"], "situation": cs["situation"], "checklist": cs["checklist"],
                         "provenance": prov})
    run_d0.CASES_FILE.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    (root(c) / "screen.json").write_text(json.dumps(report, indent=1))
    passing = sum(1 for r in report if not r["why"])
    return {"candidates": len(report), "passing": passing, "bank": len(kept),
            "verdict": "OK" if len(kept) >= N_MIN else "TOO FEW — rerun the writer once with the failures",
            "failures": [r for r in report if r["why"]]}


def report(c: dict) -> dict:
    out = {"arms": {}}
    for arm in ("A", "T"):
        s = run_d0.score(c, arm)
        if s["mean"] is None:
            continue
        g = rejudge.score(run_d0.rows(rejudge.out_dir(c, arm) / "verdicts.jsonl"))
        plans = run_d0.rows(run_d0.arm_dir(c, arm) / "plans.jsonl")
        out["arms"][arm] = {"sonnet": s["mean"], "gemini": g["mean"], "cases": s["n_cases"], "samples": s["samples"],
                            "agreement": rejudge.agreement(c, arm), "sonnet_cases": s["cases"],
                            "cost_per_plan": round(sum(r.get("cost_usd") or 0 for r in plans) / max(len(plans), 1), 4),
                            "errors": sum(1 for r in plans if r.get("error") or not r.get("plan"))}
    a, t = out["arms"].get("A"), out["arms"].get("T")
    if a and a["sonnet"] >= run_d0.A_THRESHOLD:
        out["verdict"] = "STOP — no headroom"
    if a and t:
        ks = sorted(set(a["sonnet_cases"]) & set(t["sonnet_cases"]))
        gs = round(t["sonnet"] - a["sonnet"], 4)
        gg = round(t["gemini"] - a["gemini"], 4) if a["gemini"] is not None and t["gemini"] is not None else None
        ok = [x is not None and x >= run_d0.PASS_DELTA for x in (gs, gg)]
        out.update({"T_minus_A_sonnet": gs, "T_minus_A_gemini": gg,
                    "T_wins": sum(t["sonnet_cases"][k] > a["sonnet_cases"][k] for k in ks),
                    "A_wins": sum(t["sonnet_cases"][k] < a["sonnet_cases"][k] for k in ks),
                    "verdict": "HOLDS ON THE FUTURE" if all(ok) else "SPLIT" if any(ok)
                    else "DOES NOT HOLD ON THE FUTURE"})
    out["spent_claude_usd"] = run_d0.spent(c)
    out["spent_gemini_run_usd"] = round(rejudge.run_spent(c), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "write", "screen", "run", "judge", "gemini", "report"])
    ap.add_argument("--arm", default="A", choices=["A", "T"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    if a.cmd == "prepare":
        print(json.dumps(prepare(c), indent=1))
    elif a.cmd == "write":
        print(json.dumps(write(c), indent=1))
    elif a.cmd == "screen":
        print(json.dumps(screen(c), indent=1))
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
