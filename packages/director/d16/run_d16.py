"""D16: does a directory of lanes, as data, route as well as routing rules in prose? See docs/BRIEF-D16.md.

    python d16/run_d16.py prepare            # arms P and D from the live memory and the private inputs; lint both
    python d16/run_d16.py parity             # a fresh agent checks every fact is in both arms; stops on a gap
    python d16/run_d16.py plans --arm P      # Sonnet, k = 2, resumable (also --arm D)
    python d16/run_d16.py follow             # measurement 3: arm D re-planned with `local`, then `tracker`, down
    python d16/run_d16.py d0                 # measurement 4: D0's 14 situations through arm D's memory, judged blind
    python d16/run_d16.py report

Everything private lives in ~/.config/director/d16/: the fact sheet (`facts.md`), the bank (`cases.jsonl`, written by a
fresh agent from the facts with BANK_PROMPT, then read once with REVIEW_PROMPT), and the arms' inputs (`inputs/`:
each arm's `Task_Lanes.md`, `directory.json`, `P_lanes.md`). Only scores, counts and costs are published.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d0"))
sys.path.insert(0, str(HERE.parent))
import run_d0 as d0  # noqa: E402
from director.memory.edit import lint  # noqa: E402

PRIV = Path.home() / ".config" / "director"
OUT = PRIV / "d16"
INPUTS = OUT / "inputs"
LIVE = PRIV / "live" / "memory"
K, CAP, TURNS, CALL_BUDGET = 2, 75.0, 12, 0.60
LANES = ["frontier", "isolated", "local", "tracker", "private-exec", "cloud-agent"]
INDEX_OLD = re.compile(r"^- assigning a task to the local or frontier lane while planning → .*$", re.M)
INDEX_NEW = "- assigning a task to a lane while planning → "

# the bank and its read-through, as they were run (Opus 5.5, `claude -p --restricted`, tools Read and Write, the
# fact sheet the only file in the directory)
BANK_PROMPT = """Read `facts.md` in this directory: the lanes a lab can run a planned task on. Write a bank of **18 routing cases** from those facts alone.

Each case is a situation a coding agent is in — a task in flight, in the second person ("The user asks you to…", "You have just…"), concrete (name files, folders, tools, what the output is for) — that the agent must plan, deciding on which lane each step runs. Make each case **one unit of work**, so that a single set of acceptable lanes applies to every step of its plan. Do not say which lane to use, do not quote the fact sheet, and do not name the facts' numbers in the situation.

Exactly three cases of each kind:
1. `public`: public-facing text;
2. `internal`: internal text;
3. `confidential`: confidential material — a client's file, or an input under the confidential folder (call it `~/clients/` or "the confidential folder");
4. `specialist`: work in a specialist's domain;
5. `reliability`: a lane whose reliability claim should keep the task away from it;
6. `bulk`: bulk or long work where cost or a sandbox decides.

For each case give its acceptable lanes (more than one where the facts allow it; lane ids exactly as in the fact sheet), whether its input is confidential, and one line naming the facts (by number) that decide it.

Write the bank to `cases.jsonl`, one JSON object per line:
{"id": "<kind>-<n>", "kind": "<kind>", "situation": "...", "confidential": true|false, "acceptable": ["<lane>", ...], "decided_by": "F.., F..: <why, one line>"}
"""
REVIEW_PROMPT = """Read `facts.md` (the lanes a lab can run a planned task on) and `cases.jsonl` (routing cases written from those facts by someone else). For each case, decide whether the facts alone **decide** its answer: an agent who knows only these facts, reading the situation, would agree that the case's `acceptable` lanes are exactly the lanes the facts allow — no lane listed that the facts rule out, no lane left out that the facts allow equally well, and no reliance on judgement the facts do not settle (compare cases against each other too: two cases that resolve the same tension in opposite ways cannot both be decided).

Do not rewrite cases. Write `review.jsonl`, one line per case: {"id": "...", "keep": true|false, "why": "<one line>"}.
"""

PARITY_PROMPT = """Three files are in this directory. `facts.md` is a fact sheet about the lanes a lab can run a planned task on, numbered F01–F30. `arm-P.md` and `arm-D.md` are the same facts as two planners will see them: arm P as memory statements in prose plus a list of lane names; arm D as a lane directory (JSON) plus memory statements that cite it.

For each fact, check that it is carried — in substance, not wording — by arm P and by arm D. Then check that neither arm carries a fact about the lanes that the fact sheet lacks, or that the other arm lacks.

Write `parity.json`: {"missing_from_P": ["F..: why"], "missing_from_D": ["F..: why"], "extra_in_P": ["..."], "extra_in_D": ["..."]} — empty lists where nothing is wrong. Be strict: a fact only implied, or weakened, is missing.
"""

ASK = ("{situation}\n\nWrite your plan: the concrete next actions you will take, in order, and on which lane each one "
       "runs. Do not execute anything; you may read files in this directory first.\n\nEnd your reply with one fenced "
       "JSON block and nothing after it, one entry per task you will do or delegate (not the message to the user):\n"
       "```json\n{{\"tasks\": [{{\"task\": \"<what>\", \"lane\": \"<one of: " + ", ".join(LANES) + ">\", "
       "\"confidential\": true or false}}]}}\n```")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def cases() -> list[dict]:
    return rows(OUT / "cases.jsonl")


def spent() -> float:
    return round(sum(x.get("cost_usd") or 0 for f in OUT.rglob("*.jsonl") for x in rows(f)), 4)


def directory() -> dict:
    return json.loads((INPUTS / "directory.json").read_text())


def render_directory(d: dict) -> str:
    return ("## Lane directory\n\nEvery task runs on one entry of this directory (`id`). It is data: read it when you "
            "pick a lane.\n\n```json\n" + json.dumps(d, ensure_ascii=False, indent=1) + "\n```\n")


def lanes_section(arm: str, d: dict | None = None) -> str:
    if arm == "P":
        return "## Lanes\n\nEvery task runs on one of these lanes:\n\n" + (INPUTS / "P_lanes.md").read_text()
    return render_directory(d or directory())


def arm_dir(arm: str) -> Path:
    return OUT / f"arm-{arm}"


# ------------------------------------------------------------------ prepare and parity

def build(arm: str, d: dict | None = None) -> dict:
    a = arm_dir(arm)
    if a.exists():
        shutil.rmtree(a)
    mem = a / "sandbox" / "memory"
    shutil.copytree(LIVE, mem, ignore=shutil.ignore_patterns("recipes*.json", ".director"))
    src = INPUTS / f"{arm[0]}_Task_Lanes.md"
    (mem / "Task_Lanes.md").write_text(src.read_text())
    ids = re.findall(r"^## #(\w+)", src.read_text(), re.M)
    index = (mem / "Index.md").read_text()
    assert len(INDEX_OLD.findall(index)) == 1, "the live Index no longer has the lane situation line"
    index = INDEX_OLD.sub(INDEX_NEW + ", ".join(f"[[Task_Lanes#{i}]]" for i in ids), index)
    (mem / "Index.md").write_text(index)
    section = lanes_section(arm[0], d)
    (a / "system.md").write_text((HERE.parent / "director" / "DIRECTOR.md").read_text() + "\n" + section)
    problems = lint(mem)
    return {"arm": arm, "statements_in_Task_Lanes": len(ids), "lint": problems,
            "lane_prompt_words": len(section.split()), "lane_memory_words": len(src.read_text().split())}


def prepare() -> None:
    out = [build("P"), build("D")]
    (OUT / "prepare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if any(x["lint"] for x in out):
        raise SystemExit("lint problems: fix the inputs before planning")


def parity() -> None:
    w = OUT / "parity"
    if w.exists():
        shutil.rmtree(w)
    w.mkdir()
    shutil.copy(OUT / "facts.md", w / "facts.md")
    (w / "arm-P.md").write_text(lanes_section("P") + "\n## Memory statements\n\n" + (INPUTS / "P_Task_Lanes.md").read_text())
    (w / "arm-D.md").write_text(lanes_section("D") + "\n## Memory statements\n\n" + (INPUTS / "D_Task_Lanes.md").read_text())
    r = d0.claude(PARITY_PROMPT, w, None, "Read,Write", 10, 3.0, model="opus")
    with (OUT / "parity.jsonl").open("a") as f:
        f.write(json.dumps({"cost_usd": r.get("total_cost_usd"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    p = json.loads((w / "parity.json").read_text())
    print(json.dumps(p, indent=1, ensure_ascii=False))
    if any(p.get(k) for k in ("missing_from_P", "missing_from_D")):
        raise SystemExit("parity: a fact is missing from an arm; the run stops until it is fixed (BRIEF-D16)")


# ------------------------------------------------------------------ plans

def parse_lanes(text: str) -> list[dict] | None:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", re.S)
    try:
        tasks = json.loads(m[-1])["tasks"] if m else None
    except (ValueError, KeyError):
        return None
    if not isinstance(tasks, list):
        return None
    return [{"lane": t.get("lane"), "confidential": bool(t.get("confidential")), "task": str(t.get("task", ""))[:200]}
            for t in tasks if isinstance(t, dict)]


def plan_arm(arm: str, todo: list[tuple[dict, int]]) -> None:
    a, out = arm_dir(arm), arm_dir(arm) / "plans.jsonl"
    done = {(x["case"], x["sample"]) for x in rows(out) if x.get("tasks") is not None}
    todo = [(c, s) for c, s in todo if (c["id"], s) not in done]
    print(f"[d16] {arm}: {len(done)} done, {len(todo)} to go · spent ${spent():.2f} of ${CAP:.0f}", flush=True)
    for i, (c, s) in enumerate(todo, 1):
        if spent() >= CAP:
            raise SystemExit(f"[d16] cap reached (${CAP:.0f})")
        r = d0.claude(ASK.format(situation=c["situation"]), a / "sandbox", a / "system.md", "Read,Grep,Glob",
                      TURNS, CALL_BUDGET)
        if r.get("is_error") and re.search(r"hit your (session|usage) limit|rate limit", str(r.get("result") or ""), re.I):
            raise SystemExit(f"[d16] stopped: {r.get('result')}")
        tasks = parse_lanes(r.get("result") or "")
        rec = {"arm": arm, "case": c["id"], "sample": s, "tasks": tasks, "plan": r.get("result"),
               "error": r.get("is_error"), "cost_usd": r.get("total_cost_usd"), "turns": r.get("num_turns"),
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lanes = [t["lane"] for t in tasks or []]
        print(f"[d16] {arm} {i}/{len(todo)} {c['id']} s{s} · {lanes if tasks is not None else 'UNPARSED'} · "
              f"{rec['turns']} turns · ${rec['cost_usd'] or 0:.3f}", flush=True)


def plans(arm: str) -> None:
    plan_arm(arm, [(c, s) for c in cases() for s in range(K)])


DOWN = {"Dlocal": "local", "Dtracker": "tracker"}


def follow() -> None:
    """Measurement 3: one fact changed in the directory — a lane's health — and nothing else."""
    for arm, lane in DOWN.items():
        d = copy.deepcopy(directory())
        for e in d["entries"]:
            if e["id"] == lane:
                e["health"] = "down"
        if not (arm_dir(arm) / "system.md").exists():
            build(arm, d)
        plan_arm(arm, [(c, s) for c in cases() if lane in c["acceptable"] for s in range(K)])


def run_d0() -> None:
    """Measurement 4: D0's situations, D0's ask, through arm D's memory and prompt; D0's blind judge."""
    a = arm_dir("D0D")
    if not a.exists():
        shutil.copytree(arm_dir("D") / "sandbox", a / "sandbox")
        shutil.copy(arm_dir("D") / "system.md", a / "system.md")
    c = d0.cfg(str(PRIV / "d0.json"))
    c["private_dir"] = str(PRIV)
    d0.EXPERIMENT, d0.CAP = "d16", CAP          # d0.spent() then sums this experiment's spend: one cap for D16
    d0.run(c, "D0D")
    d0.judge(c, "D0D")


# ------------------------------------------------------------------ report

def sign_p(w: int, l: int) -> float:
    n = w + l
    return sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0


def cleared() -> set[str]:
    return {e["id"] for e in directory()["entries"] if e.get("confidential_ok")}


def accuracy(arm: str) -> tuple[dict, dict]:
    by = {c["id"]: c for c in cases()}
    per, viol, unparsed = {}, 0, 0
    for x in rows(arm_dir(arm) / "plans.jsonl"):
        c = by.get(x["case"])
        if c is None:
            continue
        ts = x.get("tasks")
        if not ts:
            unparsed += ts is None
            per.setdefault(x["case"], []).append(0.0)
            continue
        per.setdefault(x["case"], []).append(sum(t["lane"] in c["acceptable"] for t in ts) / len(ts))
        viol += sum((c["confidential"] or t["confidential"]) and t["lane"] not in cleared() for t in ts)
    means = {k: sum(v) / len(v) for k, v in sorted(per.items())}
    return means, {"violations": viol, "unparsed_or_empty": unparsed}


def expected_after(lane_down: str, t: dict, case: dict) -> set[str]:
    e = {x["id"]: x for x in directory()["entries"]}
    fb = e[lane_down].get("fallback")
    conf = case["confidential"] or t["confidential"]
    if fb and not (conf and fb not in cleared()):
        return {fb}
    return {x for x in cleared() if x != lane_down and e[x]["health"] == "up"} if conf else set()


def following() -> dict:
    by = {c["id"]: c for c in cases()}
    base = {(x["case"], x["sample"]): x["tasks"] or [] for x in rows(arm_dir("D") / "plans.jsonl")}
    moved = affected = stayed = unaffected = 0
    for arm, lane in DOWN.items():
        for x in rows(arm_dir(arm) / "plans.jsonl"):
            b, n, c = base.get((x["case"], x["sample"]), []), x.get("tasks") or [], by[x["case"]]
            for i, t in enumerate(b):          # tasks aligned by position; a re-plan's extra tasks are not scored
                new = n[i]["lane"] if i < len(n) else None
                if t["lane"] == lane:
                    affected += 1
                    moved += new in expected_after(lane, t, c)
                else:
                    unaffected += 1
                    stayed += new == t["lane"]
    return {"affected": affected, "moved_to_fallback": round(moved / affected, 3) if affected else None,
            "unaffected": unaffected, "stayed": round(stayed / unaffected, 3) if unaffected else None}


def report() -> dict:
    P, pv = accuracy("P")
    D, dv = accuracy("D")
    mp, md = (round(sum(m.values()) / len(m), 4) if m else None for m in (P, D))
    common = sorted(set(P) & set(D))
    w, l = sum(D[k] > P[k] for k in common), sum(D[k] < P[k] for k in common)
    by = {c["id"]: c for c in cases()}
    kinds = {}
    for arm, m in (("P", P), ("D", D)):
        for k, v in m.items():
            kinds.setdefault(by[k]["kind"], {}).setdefault(arm, []).append(v)
    kinds = {k: {a: round(sum(v) / len(v), 3) for a, v in d.items()} for k, d in kinds.items()}
    d0.EXPERIMENT = "d16"
    s0 = d0.score({"private_dir": str(PRIV), "placeholders": {}}, "D0D") if (arm_dir("D0D") / "verdicts.jsonl").exists() else {}
    fol = following()
    cost = {a: round(sum(x.get("cost_usd") or 0 for x in rows(arm_dir(a) / "plans.jsonl")), 2)
            for a in ("P", "D", "Dlocal", "Dtracker", "D0D")}
    n_plans = {a: len(rows(arm_dir(a) / "plans.jsonl")) for a in ("P", "D")}
    holds = (md is not None and mp is not None and md >= mp - 0.05 and dv["violations"] <= pv["violations"]
             and (s0.get("mean") or 0) >= 0.899 and (fol["moved_to_fallback"] or 0) >= 0.8 and (fol["stayed"] or 0) >= 0.9)
    better = holds and md >= mp + 0.10 and w > l
    verdict = "DIRECTORY ROUTES BETTER" if better else "DIRECTORY HOLDS" if holds else "DIRECTORY LOSES"
    prep = json.loads((OUT / "prepare.json").read_text()) if (OUT / "prepare.json").exists() else []
    out = {"cases": len(cases()), "lane_accuracy": {"P": mp, "D": md}, "D_minus_P": round(md - mp, 4) if md is not None and mp is not None else None,
           "headroom_P_below_0.95": mp is not None and mp < 0.95,
           "case_wins": {"D": w, "P": l, "ties": len(common) - w - l, "sign_p": round(sign_p(w, l), 4)},
           "violations": {"P": pv["violations"], "D": dv["violations"]},
           "unparsed_or_empty": {"P": pv["unparsed_or_empty"], "D": dv["unparsed_or_empty"]},
           "following_the_data": fol, "d0_through_D": {k: s0.get(k) for k in ("mean", "n_cases", "samples")},
           "per_kind": kinds, "per_case": {k: [round(P.get(k, -1), 3), round(D.get(k, -1), 3)] for k in sorted(set(P) | set(D))},
           "cost_usd": {**cost, "total_d16": spent()}, "cost_per_plan": {a: round(cost[a] / n_plans[a], 3) if n_plans[a] else None for a in ("P", "D")},
           "lane_words": {x["arm"]: {"prompt": x["lane_prompt_words"], "memory": x["lane_memory_words"]} for x in prep},
           "verdict": verdict}
    (OUT / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "parity", "plans", "follow", "d0", "report"])
    ap.add_argument("--arm", choices=["P", "D"], default="P")
    a = ap.parse_args()
    {"prepare": prepare, "parity": parity, "plans": lambda: plans(a.arm), "follow": follow, "d0": run_d0,
     "report": report}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
