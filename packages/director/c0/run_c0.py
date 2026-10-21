"""C0 runner (docs/BRIEF-C0.md): does today's session already keep a multi-day thread's decisions?
Standard library only; D0's runner plans, judges and scores, J0's judge re-judges.

    python run_c0.py prepare --config ~/.config/director/d0.json   # freeze today's session (D0's arm A, rebuilt)
    python run_c0.py extract --config ...                          # the person's messages, 2026-10-14..21, private
    python run_c0.py write --config ...                            # a fresh Opus writes 14 candidate probes
    python run_c0.py screen --config ...                           # code screen -> c0/cases.jsonl
    python run_c0.py run --config ... / judge / gemini / report

Private: the thread, the frozen sandbox, the writer's workspace, plans and verdicts, under <private_dir>/c0/.
Public: c0/cases.jsonl (placeholders only) and the scores.
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

run_d0.EXPERIMENT = "c0"
run_d0.CASES_FILE = HERE / "cases.jsonl"
run_d0.CAP = 10.0
ARM = "A"                                   # C0 is D0's arm A, rebuilt from today's files
FROM, TO = "2026-10-14", "2026-10-21"
WRITER, WRITER_BUDGET, WRITER_TURNS = "opus", 6.0, 60
N_BANK, N_MIN, MAX_WHY = 10, 8, 4
JACCARD_MAX = 0.5
# Anything shaped like an API key or token is replaced before the thread is written or read (the person pasted a key
# into the thread on 2026-10-20); a probe that still carries one is refused by the screen.
SECRET = re.compile(r"\b(?:AQ\.[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "c0"


def digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def prepare(c: dict) -> dict:
    r = run_d0.prepare_a(c)
    rec = {**r, "sandbox_sha": digest(run_d0.arm_dir(c, ARM) / "sandbox"),
           "system_sha": hashlib.sha256((run_d0.arm_dir(c, ARM) / "system.md").read_bytes()).hexdigest()[:16],
           "frozen": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root(c) / "freeze.json").write_text(json.dumps(rec, indent=1))
    return rec


def messages(c: dict) -> list[tuple[str, str]]:
    """The person's messages in the session transcripts of the projects the private config names
    (`"c0_projects"`: Claude Code project directory names), dated, keys scrubbed."""
    out = []
    for proj in c["c0_projects"]:
        for f in sorted((Path.home() / ".claude" / "projects" / proj).glob("*.jsonl")):
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") != "user":
                    continue
                cnt = r.get("message", {}).get("content")
                text = cnt if isinstance(cnt, str) else "".join(
                    x.get("text", "") for x in cnt if isinstance(x, dict) and x.get("type") == "text") \
                    if isinstance(cnt, list) else ""
                ts = r.get("timestamp", "")
                if text.strip() and not text.lstrip().startswith("<") and FROM <= ts[:10] <= TO:
                    out.append((ts, SECRET.sub("[SECRET]", text.strip())))
    return sorted(set(out))


def extract(c: dict) -> dict:
    ms = messages(c)
    lines = [f"## {ts[:10]} {ts[11:16]}\n\n{text[:3000]}\n" for ts, text in ms]
    root(c).mkdir(parents=True, exist_ok=True)
    (root(c) / "thread.md").write_text("\n".join(lines))
    return {"messages": len(ms), "days": sorted({ts[:10] for ts, _ in ms})}


def write(c: dict) -> dict:
    w = root(c) / "writer"
    if w.exists():
        shutil.rmtree(w)
    (w / "out").mkdir(parents=True)
    shutil.copy(root(c) / "thread.md", w / "thread.md")
    avoid = [json.loads(l)["situation"] for f in ("d0", "f1")
             for l in (HERE.parent / f / "cases.jsonl").read_text().splitlines() if l.strip()]
    (w / "avoid.md").write_text("\n\n".join(f"- {s}" for s in avoid) + "\n")
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
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def screen(c: dict) -> dict:
    _, leaks = rejudge.redactor(c)
    by_day: dict[str, str] = {}
    for ts, text in messages(c):
        by_day[ts[:10]] = by_day.get(ts[:10], "") + "\n" + norm(text)
    avoid = [words(json.loads(l)["situation"]) for f in ("d0", "f1")
             for l in (HERE.parent / f / "cases.jsonl").read_text().splitlines() if l.strip()]
    kept, why_n, report = [], 0, []
    for line in (root(c) / "writer" / "out" / "candidates.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        cs = json.loads(line)
        why, dec = [], cs.get("decision", {})
        if not dec.get("quote") or norm(dec["quote"]) not in by_day.get(dec.get("date", ""), ""):
            why.append("quote not found in the person's messages of that date")
        if not cs.get("day") or not dec.get("date") or cs["day"] <= dec["date"]:
            why.append("probe not set after the decision")
        if leaks(json.dumps(cs, ensure_ascii=False)):
            why.append("private name")
        if SECRET.search(json.dumps(cs, ensure_ascii=False)):
            why.append("secret")
        ws = words(cs["situation"])
        overlap = max((len(ws & a) / len(ws | a) for a in avoid if ws | a), default=0)
        if overlap >= JACCARD_MAX:
            why.append(f"overlaps an earlier case ({overlap:.2f})")
        if not 2 <= len(cs.get("checklist", [])) <= 3:
            why.append("checklist size")
        take = not why and len(kept) < N_BANK and (cs.get("kind") != "why" or why_n < MAX_WHY)
        report.append({"id": cs["id"], "kind": cs.get("kind"), "kept": take, "why": why})
        if take:
            why_n += cs.get("kind") == "why"
            kept.append({"id": cs["id"], "kind": cs.get("kind"), "day": cs["day"],
                         "situation": f"Today is {cs['day']}. {cs['situation']}", "checklist": cs["checklist"],
                         "decision": dec})
    run_d0.CASES_FILE.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    (root(c) / "screen.json").write_text(json.dumps(report, indent=1))
    return {"candidates": len(report), "passing": sum(1 for r in report if not r["why"]), "bank": len(kept),
            "verdict": "OK" if len(kept) >= N_MIN else "TOO FEW — rerun the writer once with the failures",
            "failures": [r for r in report if r["why"]]}


def report(c: dict) -> dict:
    s = run_d0.score(c, ARM)
    g = rejudge.score(run_d0.rows(rejudge.out_dir(c, ARM) / "verdicts.jsonl"))
    plans = run_d0.rows(run_d0.arm_dir(c, ARM) / "plans.jsonl")
    out = {"C0_sonnet": s["mean"], "C0_gemini": g["mean"], "cases": s["cases"], "samples": s["samples"],
           "agreement": rejudge.agreement(c, ARM),
           "errors": sum(1 for r in plans if r.get("error") or not r.get("plan")),
           "spent_claude_usd": run_d0.spent(c), "spent_gemini_run_usd": round(rejudge.run_spent(c), 4)}
    if s["mean"] is not None:
        out["verdict"] = "STOP — no headroom" if s["mean"] >= run_d0.A_THRESHOLD else "GO to 2b"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "extract", "write", "screen", "run", "judge", "gemini", "report"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    root(c).mkdir(parents=True, exist_ok=True)
    if a.cmd in ("prepare", "extract", "write", "screen"):
        print(json.dumps({"prepare": prepare, "extract": extract, "write": write, "screen": screen}[a.cmd](c), indent=1))
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
