"""D11: plan quality on a fresh bank. See docs/BRIEF-D11.md.

    python d11/run_d11.py plans --arms N,B,S9,S10     # local, US$0; resumable
    python d11/run_d11.py plans --arms T              # Sonnet via claude -p; can run alongside the local arms
    python d11/run_d11.py judge                       # Sonnet, blind
    python d11/run_d11.py report

The bank is private: ~/.config/director/d11/cases.jsonl (written by a fresh agent, BRIEF-D11).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d0"))
sys.path.insert(0, str(HERE.parent / "d8"))
sys.path.insert(0, str(HERE.parent / "d10"))
import run_d0 as d0  # noqa: E402
import run_d8 as r  # noqa: E402
import run_d10 as s10  # noqa: E402

PRIV = Path.home() / ".config" / "director"
OUT = PRIV / "d11"
CASES = OUT / "cases.jsonl"
MEMORY = PRIV / "d9" / "memory"
T_DIR = PRIV / "d0" / "arm-T"
MODEL = "gemma4:12b-mlx"
K, CAP, TOP_B = 2, 25.0, 15

ASK_12B = ("{situation}\n\nWrite your plan: the concrete next actions you will take, in order, and what you will "
           "tell the user. Do not execute anything.")
CON_MEMORIA = ("You are a coding agent in this lab. These statements from the lab's operational memory were found "
               "for this task. Follow the ones that apply, cite `[[Page#id]]` on each action that follows one, and "
               "if a statement conflicts with what the user asked, say so to the user instead of silently "
               "choosing.\n\n{enunciados}\n\n")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def casos() -> list[dict]:
    return rows(CASES)


def gastado() -> float:
    return round(sum(x.get("cost_usd") or 0 for f in OUT.glob("*.jsonl") for x in rows(f)), 4)


def generar(prompt: str) -> str:
    cuerpo = {"model": MODEL, "stream": False, "think": False,
              "messages": [{"role": "user", "content": prompt}],
              "options": {"temperature": 0.3, "num_ctx": 16384}}
    req = urllib.request.Request(f"{r.OLLAMA}/api/chat", data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"]


def plans(arms: list[str]) -> None:
    situaciones, enunciados = r.cargar_memoria(MEMORY)
    trayectorias, viejas = s10.partir_index(MEMORY, situaciones)
    for arm in arms:
        out = OUT / f"plans-{arm}.jsonl"
        hechos = {(x["case"], x["sample"]) for x in rows(out) if x.get("plan")}
        m = r.Modelo(MODEL)
        for c in casos():
            for s in range(K):
                if (c["id"], s) in hechos:
                    continue
                t0, costo, sel = time.time(), None, []
                if arm == "T":
                    if gastado() >= CAP:
                        raise SystemExit(f"cap US${CAP} reached")
                    res = d0.claude(d0.ASK.format(situation=c["situation"]), T_DIR / "sandbox", T_DIR / "system.md",
                                    "Read,Grep,Glob", d0.TURNS, d0.CALL_BUDGET)
                    if res.get("is_error") and re.search(r"limit", str(res.get("result") or ""), re.I):
                        raise SystemExit(f"stopped: {res.get('result')}")
                    plan, costo = res.get("result"), res.get("total_cost_usd")
                else:
                    if arm == "B":
                        sel = r.bm25(c["situation"], enunciados, TOP_B)
                    elif arm == "S9":
                        sel = r.recorrer(m, c["situation"], situaciones, enunciados)
                    elif arm == "S10":
                        sel = s10.recorrer_s10(m, c["situation"], trayectorias, viejas, enunciados)
                    prefijo = ""
                    if arm != "N":
                        lista = "\n".join(f"- [[{x}]] When: {enunciados[x]['When']} Do: {enunciados[x]['Do']} "
                                          f"Because: {enunciados[x]['Because']}" for x in sel) or "(none found)"
                        prefijo = CON_MEMORIA.format(enunciados=lista)
                    plan = generar(prefijo + ASK_12B.format(situation=c["situation"]))
                fila = {"arm": arm, "case": c["id"], "sample": s, "plan": plan, "statements": sel,
                        "cost_usd": costo, "wall_s": round(time.time() - t0), "at": time.strftime("%H:%M:%S")}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                print(f"[{arm}] {c['id']} s{s} · {len(sel)} statements · {fila['wall_s']} s"
                      + (f" · ${costo:.3f}" if costo else "") + ("" if plan else " · NO PLAN"), flush=True)


def judge() -> None:
    by = {c["id"]: c for c in casos()}
    tmpl = (HERE.parent / "d0" / "judge.md").read_text()
    out = OUT / "verdicts.jsonl"
    hechos = {(x["arm"], x["case"], x["sample"]) for x in rows(out) if x.get("items")}
    vacio = OUT / "judge-cwd"
    vacio.mkdir(exist_ok=True)
    for f in sorted(OUT.glob("plans-*.jsonl")):
        for p in rows(f):
            if not p.get("plan") or (p["arm"], p["case"], p["sample"]) in hechos:
                continue
            if gastado() >= CAP:
                raise SystemExit(f"cap US${CAP} reached")
            c = by[p["case"]]
            lista = "\n".join(f"- `{it['id']}`: {it['text']}" for it in c["checklist"])
            v = d0.claude(tmpl.replace("{situation}", c["situation"]).replace("{checklist}", lista)
                          .replace("{plan}", p["plan"]), vacio, None, "", 1, 0.30)
            items = d0.parse_items(v.get("result") or "", [it["id"] for it in c["checklist"]])
            with out.open("a", encoding="utf-8") as g:
                g.write(json.dumps({"arm": p["arm"], "case": p["case"], "sample": p["sample"], "items": items,
                                    "cost_usd": v.get("total_cost_usd")}, ensure_ascii=False) + "\n")
            met = sum(x["met"] for x in items or [])
            print(f"[judge] {p['arm']} {p['case']} s{p['sample']} · {met}/{len(c['checklist'])}"
                  + ("" if items else " · UNPARSED"), flush=True)


def signo(g: int, p: int) -> float:
    n = g + p
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(g, p) + 1)) / 2 ** n)


def report() -> dict:
    por: dict[str, dict[str, list[float]]] = {}
    for v in rows(OUT / "verdicts.jsonl"):
        if v.get("items"):
            por.setdefault(v["arm"], {}).setdefault(v["case"], []).append(
                sum(x["met"] for x in v["items"]) / len(v["items"]))
    caso = {a: {c: sum(s) / len(s) for c, s in d.items()} for a, d in por.items()}
    media = {a: round(sum(d.values()) / len(d), 3) for a, d in caso.items()}

    def par(a: str, b: str) -> dict:
        com = sorted(set(caso.get(a, {})) & set(caso.get(b, {})))
        g = sum(caso[a][c] > caso[b][c] for c in com)
        p = sum(caso[a][c] < caso[b][c] for c in com)
        return {"wins": g, "losses": p, "ties": len(com) - g - p, "p": round(signo(g, p), 4),
                "delta": round(media[a] - media[b], 3)}

    inf: dict = {"means": media, "per_case": {a: {c: round(x, 3) for c, x in sorted(d.items())}
                                              for a, d in caso.items()},
                 "cost_usd": gastado()}
    if {"S9", "S10", "N", "T"} <= media.keys():
        walk = "S10" if media["S10"] >= media["S9"] else "S9"
        vsn = par(walk, "N")
        ayuda = vsn["delta"] >= 0.15 and vsn["wins"] > vsn["losses"]
        inf |= {"walk": walk, "walk_vs_N": vsn, "walk_vs_T": par(walk, "T"), "S10_vs_S9": par("S10", "S9"),
                "walk_vs_B": par(walk, "B") if "B" in media else None,
                "verdict": ("LOCAL PLANNING WORKS" if ayuda and media[walk] >= media["T"] - 0.10 else
                            "MEMORY HELPS, GAP REMAINS" if ayuda else "NO HELP")}
    (OUT / "report.json").write_text(json.dumps(inf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in inf.items() if k != "per_case"}, ensure_ascii=False))
    return inf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plans", "judge", "report"])
    ap.add_argument("--arms", default="N,B,S9,S10")
    a = ap.parse_args()
    {"plans": lambda: plans(a.arms.split(",")), "judge": judge, "report": report}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
