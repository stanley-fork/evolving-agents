"""Diagnosis after a verdict: replay a walk with logging of every step. Not a redesign; see BRIEF-D9.

    python d8/diagnose_walk.py --memory ~/.config/director/d9/memory --out ~/.config/director/d9/diagnosis.jsonl

Same code path as run_d8.recorrer (temperature 0), with each closed question recorded: the options shown, which the
12B picked, and which of them were in the case's core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_d8 as r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:12b-mlx")
    ap.add_argument("--memory", default=str(r.MEMORY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    situaciones, enunciados = r.cargar_memoria(Path(a.memory).expanduser())
    ref = r.referencia()
    casos = [json.loads(l) for l in r.CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.cases:
        casos = [c for c in casos if c["id"].split("-")[0] in a.cases.split(",")]
    m = r.Modelo(a.model)
    original = m.elegir
    pasos: list[dict] = []

    def elegir(situacion, opciones, tope, que):
        sel = original(situacion, opciones, tope, que)
        pasos.append({"que": que, "opciones": opciones, "elegidas": sel})
        return sel

    m.elegir = elegir
    with Path(a.out).expanduser().open("w", encoding="utf-8") as f:
        for c in casos:
            pasos.clear()
            camino = r.recorrer(m, c["situation"], situaciones, enunciados)
            nucleo, _ = ref[c["id"]]
            fila = {"case": c["id"], "picked": camino, "core": sorted(nucleo), "steps": []}
            for i, p in enumerate(pasos):
                if i == 0:
                    elegidas = [situaciones[j][0] for j in p["elegidas"]]
                    alcanzable = {s for j in p["elegidas"] for s in situaciones[j][1]}
                    fila["steps"].append({"step": "situations", "picked": elegidas,
                                          "core_reachable": len(alcanzable & nucleo)})
                else:
                    ids = [r.CITA.search(o).group(1) for o in p["opciones"]]
                    en_nucleo = [s for s in ids if s in nucleo]
                    fila["steps"].append({"step": f"statements{i}", "shown": len(ids), "picked": len(p["elegidas"]),
                                          "core_shown": len(en_nucleo),
                                          "core_picked": sum(ids[j] in nucleo for j in p["elegidas"])})
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            print(c["id"], json.dumps(fila["steps"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
