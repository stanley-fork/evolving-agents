"""PG runner (docs/BRIEF-PG.md): F1's arms given to a planner from another provider (gemini-4-argon, no tools).
Standard library only; D0's runner judges and scores, J0's judge re-judges.

    python run_pg.py prepare --config ~/.config/director/d0.json    # copy F1's A and T, hashes checked
    python run_pg.py run --arm GA --config ...                      # one Gemini call per plan, resumable
    python run_pg.py judge --arm GA | gemini --arm GA | report --config ...

Private: the prompts' sources, plans and verdicts under <private_dir>/pg/. Public: the scores.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("d0", "c0"):
    sys.path.insert(0, str(HERE.parent / sub))
import run_d0   # noqa: E402
import rejudge  # noqa: E402
import run_c0   # noqa: E402  (sets run_d0's globals on import; set here last)

run_d0.EXPERIMENT = "pg"
run_d0.CASES_FILE = HERE.parent / "f1" / "cases.jsonl"
PLANNER = "gemini-4-argon"
RUN_CAP_GEMINI = 15.0
ARMS = {"GA": ("f1", "A"), "GT": ("f1", "T"), "GF": (None, None)}
# The Opus judge: every claude call here (only the judge's) goes to Opus 5.5.
_claude = run_d0.claude
run_d0.claude = functools.partial(lambda *a, model="opus", **k: _claude(*a, model=model, **k))
T_PREFACE = ("The memory's pages are given below, each under its path, instead of on disk: walking it means reading "
             "them here, in the order the walk says.\n\n")
ASK = ("{situation}\n\nWrite your plan: the concrete next actions you will take, in order, and what you will tell the "
       "user. Do not execute anything.")


def root(c: dict) -> Path:
    return Path(c["private_dir"]) / "pg"


def prepare(c: dict) -> dict:
    out = {}
    for arm, (exp, src_arm) in ARMS.items():
        d = run_d0.arm_dir(c, arm)
        if d.exists():
            shutil.rmtree(d)
        if exp is None:
            (d / "sandbox").mkdir(parents=True)
            (d / "system.md").write_text("")
            out[arm] = {"from": None}
            continue
        src = Path(c["private_dir"]) / exp / f"arm-{src_arm}"
        shutil.copytree(src / "sandbox", d / "sandbox")
        shutil.copy(src / "system.md", d / "system.md")
        assert run_c0.digest(d / "sandbox") == run_c0.digest(src / "sandbox"), f"parity: {arm}"
        out[arm] = {"from": f"{exp}/arm-{src_arm}", "sandbox_sha": run_c0.digest(d / "sandbox")}
    root(c).mkdir(parents=True, exist_ok=True)
    (root(c) / "prepare.json").write_text(json.dumps(out, indent=1))
    return out


def context(c: dict, arm: str) -> tuple[str, str]:
    """(system prompt, files) for an arm, before redaction."""
    d = run_d0.arm_dir(c, arm)
    system = (d / "system.md").read_text()
    if arm == "GT":
        system = T_PREFACE + system
    files = "\n\n".join(f"### FILE: {p.relative_to(d / 'sandbox')}\n\n{p.read_text(errors='ignore')}"
                        for p in sorted((d / "sandbox").rglob("*.md")))
    return system, files


def run_spent(c: dict) -> float:
    return round(sum(r.get("cost_usd") or 0 for a in ARMS for r in run_d0.rows(run_d0.arm_dir(c, a) / "plans.jsonl")), 4)


def plan(key: str, system: str, user: str) -> tuple[str, dict]:
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}]}
    if system.strip():
        body["systemInstruction"] = {"parts": [{"text": system}]}
    req = urllib.request.Request(rejudge.URL.format(model=PLANNER), data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            raise
    u = d.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in (d.get("candidates") or [{}])[0].get("content", {}).get("parts", []))
    return text, {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}


def run(c: dict, arm: str) -> None:
    key = rejudge.api_key(c)
    redact, leaks = rejudge.redactor(c)
    secret = run_c0.SECRET
    d = run_d0.arm_dir(c, arm)
    out = d / "plans.jsonl"
    done = {(r["case"], r["sample"]) for r in run_d0.rows(out) if r.get("plan")}
    todo = [(cs, s) for cs in run_d0.cases(c) for s in range(run_d0.K) if (cs["id"], s) not in done]
    system, files = context(c, arm)
    system, files = redact(system), redact(files)
    print(f"[pg] {arm}: {len(done)} done, {len(todo)} to go · system {len(system)} chars, files {len(files)} chars",
          flush=True)
    for i, (cs, s) in enumerate(todo, 1):
        if rejudge.ledger_total() >= rejudge.KEY_CAP or run_spent(c) >= RUN_CAP_GEMINI:
            sys.exit(f"[pg] cap reached: key US${rejudge.ledger_total():.2f}/{rejudge.KEY_CAP}, run US${run_spent(c):.2f}")
        user = (files + "\n\n---\n\n" if files else "") + ASK.format(situation=redact(cs["situation"]))
        whole = system + "\n" + user
        if leaks(whole) or secret.search(whole):
            sys.exit(f"[pg] LEAK in {arm} {cs['id']}: nothing sent")
        t0 = time.time()
        text, tok = plan(key, system, user)
        cost = round(tok["in"] * rejudge.PRICE_IN + tok["out"] * rejudge.PRICE_OUT, 6)
        with rejudge.LEDGER.open("a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "experiment": "pg-plan", "model": PLANNER,
                                "tokens": tok, "cost_usd": cost}) + "\n")
        rec = {"arm": arm, "case": cs["id"], "sample": s, "plan": text or None, "error": not text, "cost_usd": cost,
               "tokens": tok, "turns": 1, "wall_s": round(time.time() - t0, 1), "planner": PLANNER,
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[pg] {arm} {i}/{len(todo)} {cs['id']} s{s} · in {tok['in']} out {tok['out']} · ${cost:.3f} · "
              f"run ${run_spent(c):.2f} · key ${rejudge.ledger_total():.2f}{' · EMPTY' if not text else ''}", flush=True)


def report(c: dict) -> dict:
    out = {}
    for arm in ARMS:
        s_ = run_d0.score(c, arm)
        g = rejudge.score(run_d0.rows(rejudge.out_dir(c, arm) / "verdicts.jsonl"))
        plans = run_d0.rows(run_d0.arm_dir(c, arm) / "plans.jsonl")
        n = max(len(plans), 1)
        out[arm] = {"opus": s_["mean"], "gemini": g["mean"], "cases": s_["cases"], "samples": s_["samples"],
                    "cost_per_plan": round(sum(r.get("cost_usd") or 0 for r in plans) / n, 4),
                    "input_tokens_per_plan": round(sum((r.get("tokens") or {}).get("in", 0) for r in plans) / n),
                    "errors": sum(1 for r in plans if r.get("error") or not r.get("plan"))}
    a, t = out["GA"], out["GT"]
    if a["opus"] is not None and a["opus"] >= run_d0.A_THRESHOLD:
        out["verdict"] = "STOP — no headroom"
    elif a["opus"] is not None and t["opus"] is not None:
        d = round(t["opus"] - a["opus"], 4)
        ks = sorted(set(a["cases"]) & set(t["cases"]))
        out.update({"GT_minus_GA_opus": d,
                    "GT_minus_GA_gemini": round(t["gemini"] - a["gemini"], 4) if None not in (a["gemini"], t["gemini"]) else None,
                    "GT_wins": sum(t["cases"][k] > a["cases"][k] for k in ks),
                    "GA_wins": sum(t["cases"][k] < a["cases"][k] for k in ks),
                    "verdict": "THE MEMORY PORTS" if d >= run_d0.PASS_DELTA else "IT DOES NOT PORT"
                    if d <= -run_d0.PASS_DELTA else "PARITY"})
    out["spent_gemini_plans_usd"] = run_spent(c)
    out["spent_gemini_judge_usd"] = round(rejudge.run_spent(c), 4)
    out["spent_claude_judge_usd"] = run_d0.spent(c) - run_spent(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "run", "judge", "gemini", "report"])
    ap.add_argument("--arm", default="GA", choices=list(ARMS))
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    if a.cmd == "prepare":
        print(json.dumps(prepare(c), indent=1))
    elif a.cmd == "run":
        run(c, a.arm)
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
