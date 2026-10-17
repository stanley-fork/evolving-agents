"""D15: meta-recipes written by Claude, run by code; the model only picks recipes and answers yes/no. BRIEF-D15.

    python d15/run_d15.py plans --arms G12,Q4,Q2     # local, US$0; resumable
    python d15/run_d15.py plans --arms RS,T          # Sonnet (paid); can run alongside
    python d15/run_d15.py judge
    python d15/run_d15.py report

Private inputs: ~/.config/director/d15/{recipes.json,cases.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d11"))
import run_d11 as d11  # noqa: E402

d0, r = d11.d0, d11.r
d11.OUT = d11.PRIV / "d15"
d11.CASES = d11.OUT / "cases.jsonl"
d11.CAP = 30.0
OUT = d11.OUT
MAX_RECETAS = 2
MODELOS = {"G12": "gemma4:12b-mlx", "Q4": "qwen3.5:4b", "Q2": "qwen3.5:2b"}


class Local:
    def __init__(self, modelo: str):
        self.modelo, self.llamadas, self.fallas = modelo, 0, 0

    def _json(self, prompt: str, esquema: dict):
        cuerpo = {"model": self.modelo, "stream": False, "think": False, "format": esquema,
                  "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0.3, "num_ctx": 8192}}
        for _ in range(2):
            self.llamadas += 1
            req = urllib.request.Request(f"{r.OLLAMA}/api/chat", data=json.dumps(cuerpo).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                return json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"])
            except Exception:  # noqa: BLE001 — counted
                continue
        self.fallas += 1
        return None

    def elegir(self, pedido: str, triggers: list[str]) -> list[int]:
        lista = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triggers))
        a = self._json(f"Request:\n{pedido}\n\nWhich of these types of request is it? Pick at most {MAX_RECETAS}.\n"
                       f"{lista}\n\nReturn {{\"elegidas\": [numbers]}}.",
                       {"type": "object", "required": ["elegidas"], "properties": {"elegidas": {
                           "type": "array", "items": {"type": "integer", "minimum": 1, "maximum": len(triggers)},
                           "maxItems": MAX_RECETAS}}})
        return list(dict.fromkeys(i - 1 for i in (a or {}).get("elegidas", []) if 1 <= i <= len(triggers)))[:MAX_RECETAS]

    def si_no(self, pedido: str, pregunta: str) -> bool:
        a = self._json(f"Request:\n{pedido}\n\nQuestion about the request: {pregunta}\nAnswer yes or no.",
                       {"type": "object", "required": ["answer"], "properties": {"answer": {"enum": ["yes", "no"]}}})
        return (a or {}).get("answer") == "yes"


class Sonnet(Local):
    def __init__(self):
        super().__init__("sonnet")
        self.costo = 0.0

    def _texto(self, prompt: str) -> str:
        self.llamadas += 1
        res = d0.claude(prompt, OUT / "cwd", None, "", 1, 0.30)
        self.costo += res.get("total_cost_usd") or 0
        return res.get("result") or ""

    def elegir(self, pedido, triggers):
        lista = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triggers))
        t = self._texto(f"Request:\n{pedido}\n\nWhich of these types of request is it? Pick at most {MAX_RECETAS}.\n"
                        f"{lista}\n\nReply with JSON only: {{\"elegidas\": [numbers]}}")
        m = re.search(r"\[([\d,\s]*)\]", t)
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
        if not m:
            self.fallas += 1
        return list(dict.fromkeys(i - 1 for i in nums if 1 <= i <= len(triggers)))[:MAX_RECETAS]

    def si_no(self, pedido, pregunta):
        t = self._texto(f"Request:\n{pedido}\n\nQuestion about the request: {pregunta}\n"
                        "Reply with exactly one word: yes or no.").strip().lower()
        if not t.startswith(("yes", "no")):
            self.fallas += 1
        return t.startswith("yes")


def ejecutar(pasos: list[dict], pedido: str, quien: Local, log: list) -> list[tuple[str, str]]:
    salida = []
    for p in pasos:
        if "do" in p:
            salida.append((p["do"], p.get("cite", "")))
        elif "ask" in p:
            si = quien.si_no(pedido, p["ask"])
            log.append({"ask": p["ask"], "answer": "yes" if si else "no"})
            salida += ejecutar(p.get("yes" if si else "no", []), pedido, quien, log)
    return salida


def plan_por_receta(pedido: str, recetas: list[dict], quien: Local) -> tuple[str, dict]:
    idx = quien.elegir(pedido, [x["trigger"] for x in recetas])
    log: list = []
    pasos: list[tuple[str, str]] = []
    for i in idx:
        pasos += ejecutar(recetas[i]["steps"], pedido, quien, log)
    vistos, lineas = set(), []
    for do, cite in pasos:
        if (do, cite) in vistos:
            continue
        vistos.add((do, cite))
        lineas.append(f"{len(lineas) + 1}. {do}" + (f" [[{cite}]]" if cite else ""))
    plan = "\n".join(lineas) or "(no recipe chosen)"
    return plan, {"recipes": [recetas[i]["id"] for i in idx], "answers": log,
                  "cites": [c for _, c in pasos if c]}


def plans(arms: list[str]) -> None:
    recetas = json.loads((OUT / "recipes.json").read_text())
    (OUT / "cwd").mkdir(exist_ok=True)
    for arm in arms:
        if arm == "T":
            d11.plans(["T"])
            continue
        out = OUT / f"plans-{arm}.jsonl"
        hechos = {(x["case"], x["sample"]) for x in d11.rows(out) if x.get("plan")}
        for c in d11.casos():
            for s in range(d11.K):
                if (c["id"], s) in hechos:
                    continue
                if arm == "RS" and d11.gastado() >= d11.CAP:
                    raise SystemExit("cap reached")
                quien = Sonnet() if arm == "RS" else Local(MODELOS[arm])
                t0 = time.time()
                plan, info = plan_por_receta(c["situation"], recetas, quien)
                fila = {"arm": arm, "case": c["id"], "sample": s, "plan": plan, **info,
                        "calls": quien.llamadas, "failures": quien.fallas,
                        "cost_usd": getattr(quien, "costo", None), "wall_s": round(time.time() - t0)}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                print(f"[{arm}] {c['id']} s{s} · {info['recipes']} · {len(info['answers'])} asks · "
                      f"{fila['wall_s']} s" + (f" · ${fila['cost_usd']:.3f}" if fila["cost_usd"] else ""), flush=True)


def report() -> dict:
    inf = d11.report()
    media, caso = inf["means"], inf["per_case"]
    casos = {c["id"]: c for c in d11.casos()}
    alcance, acuerdo = {}, {}
    rs = {(x["case"], x["sample"]): x for x in d11.rows(OUT / "plans-RS.jsonl")}
    for arm in ("G12", "Q4", "Q2", "RS", "T"):
        filas = [x for x in d11.rows(OUT / f"plans-{arm}.jsonl") if x.get("plan")]
        if not filas:
            continue
        cita = lambda x: set(x.get("cites") or d11.r.CITA.findall(x["plan"]))  # noqa: E731
        alcance[arm] = round(sum(len(set(casos[x["case"]]["grounds"]) & cita(x)) / len(casos[x["case"]]["grounds"])
                                 for x in filas) / len(filas), 3)
        if arm in ("G12", "Q4", "Q2") and rs:
            same_rec = sum(set(x["recipes"]) == set(rs[(x["case"], x["sample"])]["recipes"])
                           for x in filas if (x["case"], x["sample"]) in rs)
            acuerdo[arm] = {"same_recipes_as_RS": f"{same_rec}/{len(filas)}"}
    t = media.get("T", 0)
    pequeños = [media[a] for a in ("Q4", "Q2") if a in media]
    veredicto = ("RECIPES WORK, EVEN SMALL" if pequeños and max(pequeños) >= t - 0.10 else
                 "RECIPES WORK" if media.get("G12", 0) >= t - 0.10 else
                 "PARTIAL" if media.get("G12", 0) >= t - 0.20 else "NOT YET")

    def par(a, b):
        com = sorted(set(caso.get(a, {})) & set(caso.get(b, {})))
        g = sum(caso[a][c] > caso[b][c] for c in com)
        p = sum(caso[a][c] < caso[b][c] for c in com)
        return {"wins": g, "losses": p, "ties": len(com) - g - p, "p": round(d11.signo(g, p), 4)}

    out = {"means": media, "grounds_reached": alcance, "agreement": acuerdo, "cost_usd": inf["cost_usd"],
           "vs_T": {a: par(a, "T") for a in media if a != "T"}, "per_case": caso, "verdict": veredicto}
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_case"}, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plans", "judge", "report"])
    ap.add_argument("--arms", default="G12,Q4,Q2")
    a = ap.parse_args()
    {"plans": lambda: plans(a.arms.split(",")), "judge": d11.judge, "report": report}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
