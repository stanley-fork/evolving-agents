"""J0 (docs/BRIEF-J0.md): D0's plans, already on disk, judged again by a model from another provider.
Standard library only.

Nothing is re-planned. Each plan D0's arms wrote is sent, with its situation and checklist, through D0's own
`judge.md`, to a Gemini model instead of Claude. Before anything leaves the machine, every private term listed in a
private file is replaced by a fixed marker in the situation, the checklist and the plan, and the prompt is checked
for any term left; a leak stops the run. Every call's tokens and cost are appended to a private ledger, and no call
is made once the ledger reaches the key's cap.

    python rejudge.py judge --arm A --config ~/.config/director/d0.json     # resumable, one line per call
    python rejudge.py report --config ~/.config/director/d0.json            # scores only, no plan text

The private config is D0's, plus optionally "redact_file" (default ~/.config/director/redact.json:
{"terms": [...]}) and "gemini_env" (default ~/.config/director/gemini.env, a line GEMINI_API_KEY=...).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import run_d0

JUDGE_MODEL = "gemini-4-argon"            # the user's choice, 2026-10-20: Gemini 4 if it costs no more than 3.1 Pro
PRICE_IN, PRICE_OUT = 2.00e-6, 10.00e-6   # US$ per token through 2026-11-30 (4.00 / 20.00 from 2026-12-01); output includes thinking
KEY_CAP = 50.0                            # the user's cap on everything spent with this key, 2026-10-20
RUN_CAP = 10.0                            # this experiment's cap, inside KEY_CAP
PASS_DELTA = run_d0.PASS_DELTA            # D0's own bar: T − A ≥ +0.15
LEDGER = Path("~/.config/director/gemini-spend.jsonl").expanduser()
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def out_dir(c: dict, arm: str) -> Path:
    # Beside the experiment's directory, not in it: run_d0.spent() sums every cost under it. F1 reuses this
    # judge by setting run_d0.EXPERIMENT.
    return Path(c["private_dir"]) / f"{run_d0.EXPERIMENT}-rejudge" / JUDGE_MODEL / f"arm-{arm}"


def api_key(c: dict) -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(c.get("gemini_env", "~/.config/director/gemini.env")).expanduser()
    for line in env.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no GEMINI_API_KEY")


def redactor(c: dict):
    terms = json.loads(Path(c.get("redact_file", "~/.config/director/redact.json")).expanduser().read_text())["terms"]
    terms = sorted(set(terms), key=len, reverse=True)
    pats = [(re.compile(re.escape(t), re.I), f"[PRIVATE-{i}]") for i, t in enumerate(terms, 1)]

    def redact(s: str) -> str:
        for p, mark in pats:
            s = p.sub(mark, s)
        return s

    def leaks(s: str) -> list[str]:
        return [t for t in terms if re.search(re.escape(t), s, re.I)]

    return redact, leaks


def ledger_total() -> float:
    return sum(json.loads(l).get("cost_usd", 0) for l in LEDGER.read_text().splitlines() if l.strip()) \
        if LEDGER.exists() else 0.0


def run_spent(c: dict) -> float:
    return sum(r.get("cost_usd") or 0 for f in (Path(c["private_dir"]) / f"{run_d0.EXPERIMENT}-rejudge").rglob("verdicts.jsonl")
               for r in run_d0.rows(f))


def gemini(key: str, prompt: str) -> tuple[str, dict]:
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(URL.format(model=JUDGE_MODEL), data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    u = d.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in (d.get("candidates") or [{}])[0].get("content", {}).get("parts", []))
    return text, {"in": u.get("promptTokenCount", 0),
                  "out": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}


def judge(c: dict, arm: str) -> None:
    key = api_key(c)
    redact, leaks = redactor(c)
    by = {cs["id"]: cs for cs in run_d0.cases(c)}
    tmpl = (run_d0.HERE / "judge.md").read_text()
    d = out_dir(c, arm)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "verdicts.jsonl"
    done = {(r["case"], r["sample"]) for r in run_d0.rows(out) if r.get("items")}
    todo = [r for r in run_d0.rows(run_d0.arm_dir(c, arm) / "plans.jsonl")
            if r.get("plan") and (r["case"], r["sample"]) not in done]
    for i, r in enumerate(todo, 1):
        if ledger_total() >= KEY_CAP or run_spent(c) >= RUN_CAP:
            sys.exit(f"[j0] cap reached: key US${ledger_total():.2f}/{KEY_CAP}, run US${run_spent(c):.2f}/{RUN_CAP}")
        cs = by[r["case"]]
        checklist = "\n".join(f"- `{it['id']}`: {redact(it['text'])}" for it in cs["checklist"])
        prompt = (tmpl.replace("{situation}", redact(cs["situation"])).replace("{checklist}", checklist)
                  .replace("{plan}", redact(r["plan"])))
        if leaks(prompt):
            sys.exit(f"[j0] LEAK in {r['case']} s{r['sample']}: {len(leaks(prompt))} private term(s); nothing sent")
        ids = [it["id"] for it in cs["checklist"]]
        items, cost, tries = None, 0.0, 0
        while items is None and tries < 2:   # one retry on an unparsed verdict
            tries += 1
            text, tok = gemini(key, prompt)
            call = tok["in"] * PRICE_IN + tok["out"] * PRICE_OUT
            cost += call
            with LEDGER.open("a") as f:
                f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "experiment": "j0",
                                    "model": JUDGE_MODEL, "tokens": tok, "cost_usd": round(call, 6)}) + "\n")
            items = run_d0.parse_items(text, ids)
        rec = {"arm": arm, "case": r["case"], "sample": r["sample"], "items": items, "judge": JUDGE_MODEL,
               "cost_usd": round(cost, 6), "tries": tries, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        met = sum(x["met"] for x in items or [])
        print(f"[j0] {arm} {i}/{len(todo)} {r['case']} s{r['sample']} · {met}/{len(ids)}"
              f"{'' if items else ' · UNPARSED'} · ${cost:.3f} · run ${run_spent(c):.2f} · key ${ledger_total():.2f}",
              flush=True)


def score(rows: list[dict]) -> dict:
    per = {}
    for r in rows:
        if r.get("items"):
            per.setdefault(r["case"], []).append(sum(x["met"] for x in r["items"]) / len(r["items"]))
    cases_ = {k: round(sum(v) / len(v), 4) for k, v in sorted(per.items())}
    return {"cases": cases_, "mean": round(sum(cases_.values()) / len(cases_), 4) if cases_ else None,
            "samples": sum(len(v) for v in per.values())}


def agreement(c: dict, arm: str) -> dict:
    """Item by item, Gemini's verdict against Sonnet's on the same plan: raw agreement and Cohen's kappa."""
    son = {(r["case"], r["sample"], x["id"]): x["met"] for r in run_d0.rows(run_d0.arm_dir(c, arm) / "verdicts.jsonl")
           if r.get("items") for x in r["items"]}
    gem = {(r["case"], r["sample"], x["id"]): x["met"] for r in run_d0.rows(out_dir(c, arm) / "verdicts.jsonl")
           if r.get("items") for x in r["items"]}
    keys = sorted(set(son) & set(gem))
    if not keys:
        return {}
    n = len(keys)
    agree = sum(son[k] == gem[k] for k in keys) / n
    ps, pg = sum(son[k] for k in keys) / n, sum(gem[k] for k in keys) / n
    pe = ps * pg + (1 - ps) * (1 - pg)
    return {"items": n, "agree": round(agree, 4), "kappa": round((agree - pe) / (1 - pe), 4) if pe < 1 else None,
            "sonnet_met": round(ps, 4), "gemini_met": round(pg, 4)}


def report(c: dict) -> dict:
    rep = {"judge": JUDGE_MODEL, "arms": {}}
    for arm in ("A", "T", "Aplus", "P"):
        g = score(run_d0.rows(out_dir(c, arm) / "verdicts.jsonl"))
        if g["mean"] is None:
            continue
        s = score(run_d0.rows(run_d0.arm_dir(c, arm) / "verdicts.jsonl"))
        rep["arms"][arm] = {"gemini": g["mean"], "sonnet": s["mean"], "samples": g["samples"],
                            "agreement": agreement(c, arm), "gemini_cases": g["cases"]}
    a, t = rep["arms"].get("A"), rep["arms"].get("T")
    if a and t:
        common = sorted(set(a["gemini_cases"]) & set(t["gemini_cases"]))
        gap = round(t["gemini"] - a["gemini"], 4)
        rep.update({"T_minus_A_gemini": gap, "T_minus_A_sonnet": round(t["sonnet"] - a["sonnet"], 4),
                    "T_wins": sum(t["gemini_cases"][k] > a["gemini_cases"][k] for k in common),
                    "A_wins": sum(t["gemini_cases"][k] < a["gemini_cases"][k] for k in common),
                    "verdict": "HOLDS" if gap >= PASS_DELTA else "SELF-PREFERENCE IS PART OF THE EFFECT"})
    rep["spent_run_usd"], rep["spent_key_usd"] = round(run_spent(c), 4), round(ledger_total(), 4)
    for arm in rep["arms"].values():
        arm.pop("gemini_cases")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "report"])
    ap.add_argument("--arm", choices=["A", "T", "Aplus", "P"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = run_d0.cfg(a.config)
    if a.cmd == "judge":
        judge(c, a.arm)
    else:
        rep = report(c)
        out = Path(c["private_dir"]) / f"{run_d0.EXPERIMENT}-rejudge" / JUDGE_MODEL / "report.json"
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
