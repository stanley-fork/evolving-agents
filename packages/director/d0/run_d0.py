"""D0 runner (docs/BRIEF-D0.md): build the arms' sandboxes, run each case through `claude -p`, judge blind,
report. Standard library only.

Everything private — the private config, the sandboxes (copies of the lab's memory), the plans and the
judge's verdicts — lives under the private directory named by the config, never in this repository. The
report is the only output meant to be published: scores, costs and counts, no plan text.

    python run_d0.py prepare-a --config ~/.config/director/d0.json
    python run_d0.py run --arm A --config ~/.config/director/d0.json        # resumable, one line per call
    python run_d0.py judge --arm A --config ~/.config/director/d0.json
    python run_d0.py report --arm A --config ~/.config/director/d0.json

The private config (JSON): {"private_dir": "...", "workspace": "~/evolvingagents",
"memory_dir": "~/.claude/projects/<project>/memory", "placeholders": {"PRIVATE_PROJECT": "..."}}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "sonnet"            # Claude Sonnet 5.5 — the brief's model for both arms and the judge
K = 2                       # samples per case
TURNS, CALL_BUDGET, CAP = 12, 0.60, 25.0
A_THRESHOLD, PASS_DELTA = 0.85, 0.15

ASK = ("{situation}\n\nWrite your plan: the concrete next actions you will take, in order, and what you will "
       "tell the user. Do not execute anything; you may read files in this directory first.")

# What Claude Code puts in a session's context about its auto-memory, in short: the index is loaded, the files
# are one Read away. Arm A gets the lab's real index under this header.
A_MEMORY_HEADER = ("# Memory\n\nYou have a persistent, file-based memory in ./memory/ (one fact per file). Its index, "
                   "memory/MEMORY.md, is below; read a file when it is relevant to the task.\n\n"
                   "Contents of memory/MEMORY.md:\n\n")


def cfg(path: str) -> dict:
    c = json.loads(Path(path).expanduser().read_text())
    for k in ("private_dir", "workspace", "memory_dir"):
        c[k] = str(Path(c[k]).expanduser())
    return c


def cases(c: dict) -> list[dict]:
    out = []
    for line in (HERE / "cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        fill = lambda s: re.sub(r"\{([A-Z_]+)\}", lambda m: c["placeholders"][m.group(1)], s)
        case["situation"] = fill(case["situation"])
        for it in case["checklist"]:
            it["text"] = fill(it["text"])
        out.append(case)
    return out


def arm_dir(c: dict, arm: str) -> Path:
    return Path(c["private_dir"]) / "d0" / f"arm-{arm}"


# ------------------------------------------------------------------ arm A: the session as we run it today

def prepare_a(c: dict) -> dict:
    """The sandbox Claude Code would see working in the workspace: the workspace CLAUDE.md (auto-loaded),
    lora-kernel's CLAUDE.md and its agents and skills, and the memory files. The list is the record of what
    arm T's compiler may read (information parity)."""
    ws, d = Path(c["workspace"]), arm_dir(c, "A") / "sandbox"
    if d.exists():
        shutil.rmtree(d)
    (d / "lora-kernel").mkdir(parents=True)
    shutil.copy(ws / "CLAUDE.md", d / "CLAUDE.md")
    shutil.copy(ws / "lora-kernel" / "CLAUDE.md", d / "lora-kernel" / "CLAUDE.md")
    for sub in ("agents", "skills"):
        shutil.copytree(ws / "lora-kernel" / ".claude" / sub, d / "lora-kernel" / ".claude" / sub,
                        ignore=shutil.ignore_patterns("deprecated"))
    shutil.copytree(c["memory_dir"], d / "memory")
    files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    sysprompt = A_MEMORY_HEADER + (d / "memory" / "MEMORY.md").read_text()
    (arm_dir(c, "A") / "system.md").write_text(sysprompt)
    rec = {"arm": "A", "sandbox": str(d), "files": files, "prepared": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (arm_dir(c, "A") / "sources.json").write_text(json.dumps(rec, indent=1))
    return {"files": len(files), "sandbox": str(d)}


# ------------------------------------------------------------------ arm T: the same sources, compiled and walked

COMPILER = "opus"           # the frontier refines the memory (the director's design); Claude Opus 5.5


def prepare_t(c: dict) -> dict:
    """Compile arm A's sandbox — and nothing else — into atomic-statement pages, then build T's sandbox:
    the pages under memory/ (with the format as memory/README.md) and DIRECTOR.md as the system prompt.
    Information parity is asserted: the compiler's input list is A's."""
    src_a, t = arm_dir(c, "A") / "sandbox", arm_dir(c, "T")
    comp = t / "compile"
    if comp.exists():
        shutil.rmtree(comp)
    shutil.copytree(src_a, comp / "sources")
    (comp / "out").mkdir()
    inputs = sorted(str(p.relative_to(comp / "sources")) for p in (comp / "sources").rglob("*") if p.is_file())
    assert inputs == json.loads((arm_dir(c, "A") / "sources.json").read_text())["files"], "parity: inputs differ from A"
    r = claude((HERE / "compile.md").read_text(), comp, None, "Read,Grep,Glob,Write", 200, 10.0, model=COMPILER)
    pages = sorted((comp / "out").glob("*.md"))
    statements = sum(len(re.findall(r"^## #", p.read_text(), re.M)) for p in pages)
    sb = t / "sandbox"
    if sb.exists():
        shutil.rmtree(sb)
    shutil.copytree(comp / "out", sb / "memory")
    shutil.copy(HERE.parent / "docs" / "MEMORY-FORMAT.md", sb / "memory" / "README.md")
    shutil.copy(HERE.parent / "director" / "DIRECTOR.md", t / "system.md")
    rec = {"arm": "T", "compiler": COMPILER, "inputs": inputs, "pages": len(pages), "statements": statements,
           "compile_cost_usd": r.get("total_cost_usd"), "compile_turns": r.get("num_turns"),
           "compile_reply": r.get("result"), "error": r.get("is_error"), "prepared": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (t / "sources.json").write_text(json.dumps(rec, indent=1))
    with (t / "compile.jsonl").open("w") as f:          # counted by spent()
        f.write(json.dumps({"cost_usd": r.get("total_cost_usd")}) + "\n")
    return {k: rec[k] for k in ("pages", "statements", "compile_cost_usd", "compile_turns", "error")}


# ------------------------------------------------------------------ calls

def claude(prompt: str, cwd: Path, system_file: Path | None, tools: str, turns: int, budget: float,
           model: str = MODEL) -> dict:
    argv = ["claude", "-p", prompt, "--output-format", "json", "--restricted", "--no-session-persistence",
            "--permission-mode", "dontAsk", "--tools", tools, "--model", model,
            "--max-turns", str(turns), "--max-budget-usd", str(budget)]
    if tools:
        argv += ["--allowedTools", tools]
    if system_file is not None:
        argv += ["--append-system-prompt", system_file.read_text()]
    t0 = time.time()
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = {"is_error": True, "result": None, "stderr": p.stderr[-500:], "stdout": p.stdout[-500:]}
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def spent(c: dict) -> float:
    tot = 0.0
    for f in (Path(c["private_dir"]) / "d0").rglob("*.jsonl"):
        tot += sum(r.get("cost_usd") or 0 for r in rows(f))
    return round(tot, 4)


def run(c: dict, arm: str) -> None:
    d = arm_dir(c, arm)
    out = d / "plans.jsonl"
    done = {(r["case"], r["sample"]) for r in rows(out) if r.get("plan")}
    todo = [(cs, s) for cs in cases(c) for s in range(K) if (cs["id"], s) not in done]
    print(f"[d0] arm {arm}: {len(done)} done, {len(todo)} to go · spent ${spent(c):.2f} of ${CAP:.0f}", flush=True)
    for i, (cs, s) in enumerate(todo, 1):
        if spent(c) >= CAP:
            print(f"[d0] cap reached (${CAP:.0f}); stopping", flush=True)
            return
        r = claude(ASK.format(situation=cs["situation"]), d / "sandbox", d / "system.md", "Read,Grep,Glob",
                   TURNS, CALL_BUDGET)
        rec = {"arm": arm, "case": cs["id"], "sample": s, "plan": r.get("result"), "error": r.get("is_error"),
               "cost_usd": r.get("total_cost_usd"), "turns": r.get("num_turns"), "wall_s": r["wall_s"],
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[d0] {arm} {i}/{len(todo)} {cs['id']} s{s} · {rec['turns']} turns · ${rec['cost_usd'] or 0:.3f} · "
              f"{rec['wall_s']} s{' · ERROR' if rec['error'] or not rec['plan'] else ''}", flush=True)


def judge(c: dict, arm: str) -> None:
    d = arm_dir(c, arm)
    by = {cs["id"]: cs for cs in cases(c)}
    tmpl = (HERE / "judge.md").read_text()
    out = d / "verdicts.jsonl"
    done = {(r["case"], r["sample"]) for r in rows(out) if r.get("items")}
    todo = [r for r in rows(d / "plans.jsonl") if r.get("plan") and (r["case"], r["sample"]) not in done]
    empty = d / "judge-cwd"
    empty.mkdir(exist_ok=True)
    for i, r in enumerate(todo, 1):
        cs = by[r["case"]]
        checklist = "\n".join(f"- `{it['id']}`: {it['text']}" for it in cs["checklist"])
        prompt = (tmpl.replace("{situation}", cs["situation"]).replace("{checklist}", checklist)
                  .replace("{plan}", r["plan"]))
        v = claude(prompt, empty, None, "", 1, 0.30)
        items = parse_items(v.get("result") or "", [it["id"] for it in cs["checklist"]])
        rec = {"arm": arm, "case": r["case"], "sample": r["sample"], "items": items,
               "cost_usd": v.get("total_cost_usd"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        met = sum(x["met"] for x in items or [])
        print(f"[judge] {arm} {i}/{len(todo)} {r['case']} s{r['sample']} · "
              f"{met}/{len(cs['checklist'])}" + ("" if items else " · UNPARSED"), flush=True)


def parse_items(text: str, ids: list[str]):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        items = json.loads(m.group(0))["items"] if m else None
    except (ValueError, KeyError):
        return None
    if not items or sorted(x.get("id") for x in items) != sorted(ids):
        return None
    return [{"id": x["id"], "met": bool(x["met"]), "evidence": x.get("evidence")} for x in items]


def score(c: dict, arm: str) -> dict:
    per = {}
    for r in rows(arm_dir(c, arm) / "verdicts.jsonl"):
        if r.get("items"):
            per.setdefault(r["case"], []).append(sum(x["met"] for x in r["items"]) / len(r["items"]))
    cases_ = {k: round(sum(v) / len(v), 4) for k, v in sorted(per.items())}
    mean = round(sum(cases_.values()) / len(cases_), 4) if cases_ else None
    return {"arm": arm, "cases": cases_, "n_cases": len(cases_), "samples": sum(len(v) for v in per.values()),
            "mean": mean}


def report(c: dict, arm: str) -> dict:
    s = score(c, arm)
    plans = rows(arm_dir(c, arm) / "plans.jsonl")
    s["cost_usd"] = round(sum(r.get("cost_usd") or 0 for r in plans), 4)
    s["turns_mean"] = round(sum(r.get("turns") or 0 for r in plans) / max(len(plans), 1), 2)
    s["errors"] = sum(1 for r in plans if r.get("error") or not r.get("plan"))
    if arm == "A" and s["mean"] is not None:
        s["verdict"] = "STOP — no headroom" if s["mean"] >= A_THRESHOLD else "GO to T"
    if arm == "T" and s["mean"] is not None:
        a = score(c, "A")
        common = sorted(set(a["cases"]) & set(s["cases"]))
        wins = sum(s["cases"][k] > a["cases"][k] for k in common)
        losses = sum(s["cases"][k] < a["cases"][k] for k in common)
        n = wins + losses
        from math import comb
        p = sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0   # exact one-sided sign test
        delta = round(s["mean"] - a["mean"], 4)
        s.update({"A_mean": a["mean"], "delta": delta, "wins": wins, "losses": losses, "sign_p": round(p, 4),
                  "verdict": "T PASSES" if delta >= PASS_DELTA and wins > losses and p < 0.05 else "T FAILS"})
        cites = [r for r in plans if r.get("plan")]
        s["plans_citing_statements"] = sum(bool(re.search(r"\[\[[^\]]+#[^\]]+\]\]", r["plan"])) for r in cites)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare-a", "prepare-t", "run", "judge", "report"])
    ap.add_argument("--arm", default="A", choices=["A", "T"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = cfg(a.config)
    if a.cmd == "prepare-a":
        print(json.dumps(prepare_a(c)))
    elif a.cmd == "prepare-t":
        print(json.dumps(prepare_t(c)))
    elif a.cmd == "run":
        run(c, a.arm)
    elif a.cmd == "judge":
        judge(c, a.arm)
    else:
        print(json.dumps(report(c, a.arm), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
