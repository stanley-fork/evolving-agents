"""D12: a plan composed by code from closed answers. See docs/BRIEF-D12.md.

    python d12/run_d12.py plans --arms C,F     # local, US$0; resumable
    python d12/run_d12.py plans --arms T       # Sonnet via claude -p; can run alongside
    python d12/run_d12.py judge
    python d12/run_d12.py report

Arm F is D11's S10 arm (walk + free-written plan), run on D12's bank. Bank: ~/.config/director/d12/cases.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d11"))
import run_d11 as d11  # noqa: E402

r, s10 = d11.r, d11.s10
d11.OUT = d11.PRIV / "d12"
d11.CASES = d11.OUT / "cases.jsonl"
d11.CAP = 20.0
MAX_RESTO = 5


def responder(prompt: str, esquema: dict) -> dict | None:
    cuerpo = {"model": d11.MODEL, "stream": False, "think": False, "format": esquema,
              "messages": [{"role": "user", "content": prompt}],
              "options": {"temperature": 0.3, "num_ctx": 16384}}
    for _ in range(2):
        req = urllib.request.Request(f"{r.OLLAMA}/api/chat", data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"])
        except Exception:  # noqa: BLE001 — counted by the caller
            continue
    return None


POR_ENUNCIADO = {"type": "object", "required": ["applies", "conflicts", "step"], "properties": {
    "applies": {"type": "boolean"}, "conflicts": {"type": "boolean"}, "step": {"type": "string"}}}
RESTO = {"type": "object", "required": ["steps", "tell_user"], "properties": {
    "steps": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_RESTO},
    "tell_user": {"type": "string"}}}


def componer(situacion: str, sel: list[str], enunciados: dict) -> tuple[str, dict]:
    pasos, conflictos, fallas = [], [], 0
    for sid in sel:
        e = enunciados[sid]
        a = responder(
            f"Situation:\n{situacion}\n\nA rule of this lab:\n[[{sid}]]\nWhen: {e['When']}\nDo: {e['Do']}\n"
            f"Because: {e['Because']}\n\nAnswer in JSON:\n"
            "- applies: true if the rule's When matches this situation.\n"
            "- conflicts: true if what the user asked (or the obvious way to do it) goes against the rule's Do.\n"
            "- step: the concrete action that carries out the rule's Do in this situation, in one sentence, keeping "
            "every requirement of the Do (files, commands, conditions, order).", POR_ENUNCIADO)
        if a is None:
            fallas += 1
            continue
        if a.get("applies"):
            pasos.append((sid, a.get("step", "").strip()))
            if a.get("conflicts"):
                conflictos.append((sid, e["Do"]))
    lista = "\n".join(f"- {p} [[{sid}]]" for sid, p in pasos) or "(none)"
    resto = responder(
        f"Situation:\n{situacion}\n\nThese steps are already decided, from the lab's rules:\n{lista}\n\n"
        f"Answer in JSON:\n- steps: at most {MAX_RESTO} remaining steps the task needs that the steps above do not "
        "cover. They must not contradict the steps above.\n- tell_user: what you will tell the user.", RESTO) or {}
    if not resto:
        fallas += 1
    partes = []
    if conflictos:
        partes.append("**First, tell the user:**\n" + "\n".join(
            f"- What you asked conflicts with the lab's rule [[{sid}]]: {do} I will follow the rule unless you "
            "confirm otherwise." for sid, do in conflictos))
    if pasos:
        partes.append("**Steps:**\n" + "\n".join(f"{i}. {p} [[{sid}]]" for i, (sid, p) in enumerate(pasos, 1)))
    if resto.get("steps"):
        partes.append("**Then:**\n" + "\n".join(f"- {p}" for p in resto["steps"]))
    if resto.get("tell_user"):
        partes.append("**What I will tell the user:** " + resto["tell_user"])
    return "\n\n".join(partes), {"applies": len(pasos), "conflicts": len(conflictos), "json_failures": fallas}


def plans_c() -> None:
    situaciones, enunciados = r.cargar_memoria(d11.MEMORY)
    trayectorias, viejas = s10.partir_index(d11.MEMORY, situaciones)
    out = d11.OUT / "plans-C.jsonl"
    hechos = {(x["case"], x["sample"]) for x in d11.rows(out) if x.get("plan")}
    m = r.Modelo(d11.MODEL)
    for c in d11.casos():
        for s in range(d11.K):
            if (c["id"], s) in hechos:
                continue
            t0 = time.time()
            sel = s10.recorrer_s10(m, c["situation"], trayectorias, viejas, enunciados)
            plan, info = componer(c["situation"], sel, enunciados)
            fila = {"arm": "C", "case": c["id"], "sample": s, "plan": plan, "statements": sel, **info,
                    "cost_usd": None, "wall_s": round(time.time() - t0), "at": time.strftime("%H:%M:%S")}
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            print(f"[C] {c['id']} s{s} · {len(sel)} statements · applies {info['applies']} · conflicts "
                  f"{info['conflicts']} · {fila['wall_s']} s", flush=True)


def report() -> dict:
    inf = d11.report()  # means, per_case, cost (it also writes a D11-shaped report.json; replaced below)
    caso, media = inf["per_case"], inf["means"]

    def par(a: str, b: str) -> dict:
        com = sorted(set(caso.get(a, {})) & set(caso.get(b, {})))
        g = sum(caso[a][c] > caso[b][c] for c in com)
        p = sum(caso[a][c] < caso[b][c] for c in com)
        return {"wins": g, "losses": p, "ties": len(com) - g - p, "p": round(d11.signo(g, p), 4),
                "delta": round(media[a] - media[b], 3)}

    out = {"means": media, "per_case": caso, "cost_usd": inf["cost_usd"]}
    if {"C", "S10", "T"} <= media.keys():
        cf = par("C", "S10")
        out |= {"C_vs_F": cf, "C_vs_T": par("C", "T"), "F_vs_T": par("S10", "T"),
                "verdict": ("COMPOSITION CLOSES THE GAP" if media["C"] >= media["T"] - 0.10 and cf["wins"] > cf["losses"]
                            else "COMPOSITION HELPS" if cf["delta"] >= 0.10 and cf["wins"] > cf["losses"]
                            else "NO GAIN")}
    (d11.OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_case"}, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plans", "judge", "report"])
    ap.add_argument("--arms", default="C,F")
    a = ap.parse_args()
    if a.cmd == "plans":
        for arm in a.arms.split(","):
            plans_c() if arm == "C" else d11.plans(["S10" if arm == "F" else arm])
    elif a.cmd == "judge":
        d11.judge()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
