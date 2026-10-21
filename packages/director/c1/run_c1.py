"""C1 runner (docs/BRIEF-C1.md): C0's frozen session plus the director, after one reflection over the thread.
Standard library only; D0's runner plans, judges and scores, J0's judge re-judges, the memory's code applies edits.

    python run_c1.py prepare --config ~/.config/director/d0.json   # C0's sandbox + the live memory, hashed
    python run_c1.py reflect --config ...                          # one unattended reflection over the journal
    python run_c1.py run | judge | gemini | report --config ...

Private: the sandbox, the journal, the reflection's workspace, plans and verdicts, under <private_dir>/c1/.
Public: the scores and the counts.
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
sys.path.insert(0, str(HERE.parent / "c0"))
sys.path.insert(0, str(HERE.parent))
import run_d0   # noqa: E402
import rejudge  # noqa: E402
import run_c0   # noqa: E402  (sets EXPERIMENT to c0 on import; reset below)
from director import plugin_hook  # noqa: E402
from director.memory.edit import apply_edits, new_problems  # noqa: E402

run_d0.EXPERIMENT = "c1"
run_d0.CASES_FILE = HERE.parent / "c0" / "cases.jsonl"   # C0's probes, unchanged
run_d0.CAP = 20.0
ARM = "T"
REFLECTOR, REFLECT_BUDGET, REFLECT_TURNS = "opus", 8.0, 120
PASS = run_d0.PASS_DELTA


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "c1"


def digest(d: Path) -> str:
    return run_c0.digest(d)


def prepare(c: dict) -> dict:
    src, d = Path(c["private_dir"]) / "c0" / "arm-A", run_d0.arm_dir(c, ARM)
    freeze = json.loads((Path(c["private_dir"]) / "c0" / "freeze.json").read_text())
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src / "sandbox", d / "sandbox")
    assert digest(d / "sandbox") == freeze["sandbox_sha"], "parity: C0's sandbox changed since its freeze"
    live = Path(c.get("live_memory", "~/.config/director/live/memory")).expanduser()
    shutil.copytree(live, root(c) / "memory-before")
    rec = {"c0_sandbox_sha": freeze["sandbox_sha"], "memory_before_sha": digest(root(c) / "memory-before"),
           "prepared": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root(c) / "prepare.json").write_text(json.dumps(rec, indent=1))
    return rec


def journal_text(c: dict) -> str:
    return "\n".join(f"## {ts[:10]} {ts[11:16]} · person\n\n{text[:3000]}\n" for ts, text in run_c0.messages(c))


def reflect(c: dict) -> dict:
    w = root(c) / "reflect"
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    shutil.copytree(root(c) / "memory-before", w / "memory")
    (w / "journal.md").write_text(journal_text(c))
    r = run_d0.claude((HERE / "reflect.md").read_text(), w, None, "Read,Grep,Glob", REFLECT_TURNS, REFLECT_BUDGET,
                      model=REFLECTOR)
    with (w / "reflect.jsonl").open("a") as f:      # counted by run_d0.spent()
        f.write(json.dumps({"cost_usd": r.get("total_cost_usd"), "turns": r.get("num_turns"),
                            "error": r.get("is_error"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    m = re.findall(r"```json\s*(\{.*?\})\s*```", r.get("result") or "", re.S)
    edits = json.loads(m[-1]).get("edits", []) if m else []
    # The gate's first check, edit by edit: keep an edit only if it adds no lint problem to what is kept so far.
    kept, refused, cur = [], [], root(c) / "memory-before"
    for i, e in enumerate(edits):
        trial = w / f"trial-{i}"
        res = apply_edits(cur, [e], trial)
        if res["applied"] and not new_problems(cur, trial):
            kept.append(e)
            cur = trial
        else:
            refused.append({"edit": f"{e.get('op')} {e.get('page')}#{e.get('id')}", "rejected": res["rejected"],
                            "lint": new_problems(cur, trial)[:3]})
    after = root(c) / "memory-after"
    if after.exists():
        shutil.rmtree(after)
    shutil.copytree(cur, after)
    (w / "edits.json").write_text(json.dumps({"proposed": edits, "kept": kept, "refused": refused}, indent=1,
                                             ensure_ascii=False))
    # The director as the plugin puts it into a session: .director/memory/ and the walk after C0's system prompt.
    sb = run_d0.arm_dir(c, ARM) / "sandbox"
    if (sb / ".director").exists():
        shutil.rmtree(sb / ".director")
    shutil.copytree(after, sb / ".director" / "memory")
    walk = plugin_hook.context(sb)
    assert walk and ".director/memory/" in walk, "the walk does not point at .director/memory/"
    c0_system = (Path(c["private_dir"]) / "c0" / "arm-A" / "system.md").read_text()
    (run_d0.arm_dir(c, ARM) / "system.md").write_text(c0_system + "\n\n" + walk)
    return {"proposed": len(edits), "kept": len(kept), "refused": len(refused), "cost_usd": r.get("total_cost_usd"),
            "turns": r.get("num_turns"), "memory_after_sha": digest(after),
            "kept_ids": [f"{e.get('op')} {e.get('page')}#{e.get('id')}" for e in kept]}


def scores(c: dict, experiment: str, arm: str) -> tuple[dict, dict]:
    saved = run_d0.EXPERIMENT
    run_d0.EXPERIMENT = experiment
    try:
        s = run_d0.score(c, arm)
        g = rejudge.score(run_d0.rows(rejudge.out_dir(c, arm) / "verdicts.jsonl"))
    finally:
        run_d0.EXPERIMENT = saved
    return s, g


def report(c: dict) -> dict:
    s0, g0 = scores(c, "c0", "A")
    s1, g1 = scores(c, "c1", ARM)
    plans = run_d0.rows(run_d0.arm_dir(c, ARM) / "plans.jsonl")
    out = {"C0": {"sonnet": s0["mean"], "gemini": g0["mean"]}, "C1": {"sonnet": s1["mean"], "gemini": g1["mean"]},
           "C1_cases": s1["cases"], "C0_cases": s0["cases"], "samples": s1["samples"],
           "agreement": rejudge.agreement(c, ARM), "cost_per_plan": round(sum(r.get("cost_usd") or 0 for r in plans)
                                                                          / max(len(plans), 1), 4),
           "errors": sum(1 for r in plans if r.get("error") or not r.get("plan")),
           "citing": sum(1 for r in plans if re.search(r"\[\[[^\]]+#", r.get("plan") or "")),
           "spent_claude_usd": run_d0.spent(c), "spent_gemini_run_usd": round(rejudge.run_spent(c), 4)}
    if s1["mean"] is not None and g1["mean"] is not None:
        ds, dg = round(s1["mean"] - s0["mean"], 4), round(g1["mean"] - g0["mean"], 4)
        ks = sorted(set(s0["cases"]) & set(s1["cases"]))
        drop = [k for k in ks if not k.startswith("c01")]
        out.update({"C1_minus_C0_sonnet": ds, "C1_minus_C0_gemini": dg,
                    "C1_wins": sum(s1["cases"][k] > s0["cases"][k] for k in ks),
                    "C0_wins": sum(s1["cases"][k] < s0["cases"][k] for k in ks),
                    "without_c01_sonnet": {"C0": round(sum(s0["cases"][k] for k in drop) / len(drop), 4),
                                           "C1": round(sum(s1["cases"][k] for k in drop) / len(drop), 4)},
                    "verdict": "THE LOOP WORKS" if ds >= PASS and dg >= PASS else "SPLIT" if max(ds, dg) >= PASS
                    else "THE LOOP DOES NOT WORK YET"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "reflect", "run", "judge", "gemini", "report"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    root(c).mkdir(parents=True, exist_ok=True)
    if a.cmd in ("prepare", "reflect"):
        print(json.dumps({"prepare": prepare, "reflect": reflect}[a.cmd](c), indent=1, ensure_ascii=False))
    elif a.cmd == "run":
        run_d0.run(c, ARM)
    elif a.cmd == "judge":
        run_d0.judge(c, ARM)
    elif a.cmd == "gemini":
        rejudge.judge(c, ARM)
    else:
        rep = report(c)
        (root(c) / "report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
