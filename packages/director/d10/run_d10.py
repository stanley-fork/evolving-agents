"""D10: a walk strategy that fits a strict filter. See docs/BRIEF-D10.md.

    python d10/run_d10.py [--memory ~/.config/director/d9/memory] [--out ~/.config/director/d10] [--k 3]

Reuses D8's model wrapper, BM25, reference and metrics. Arm S9 is D8's `recorrer` unchanged; arm S10 is `recorrer_s10`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d8"))
import run_d8 as r  # noqa: E402

TRAYECTORIAS = 3
SITUACIONES = 2
LOTE = 10


def partir_index(d: Path, situaciones):
    """(trajectories, old situations): the lines after `## Trajectories` are the trajectories."""
    texto = (d / "Index.md").read_text(encoding="utf-8").split("## Trajectories", 1)
    n = sum(1 for l in texto[1].splitlines() if l.startswith("- ") and "→" in l) if len(texto) > 1 else 0
    return situaciones[len(situaciones) - n:], situaciones[: len(situaciones) - n]


def recorrer_s10(m: r.Modelo, situacion: str, trayectorias, viejas, enunciados) -> list[str]:
    it = m.elegir(situacion, [s for s, _ in trayectorias], TRAYECTORIAS,
                  "Which of these types of task does it belong to?")
    iv = m.elegir(situacion, [s for s, _ in viejas], SITUACIONES,
                  "Which of these specific situations add something to it?")
    entrada = [sid for i in it for sid in trayectorias[i][1]] + [sid for i in iv for sid in viejas[i][1]]
    entrada = [s for s in dict.fromkeys(entrada) if s in enunciados]
    elegidos: list[str] = []
    for j in range(0, len(entrada), LOTE):
        lote = entrada[j: j + LOTE]
        fuera = set(m.elegir(situacion, [r.opcion(c, enunciados[c]) for c in lote], None,
                             "Which of these statements do NOT apply to it?"))
        elegidos += [c for i, c in enumerate(lote) if i not in fuera]
    vistos = set(entrada)
    candidatos = [l for c in elegidos for l in enunciados[c]["links"] if l in enunciados]
    for _ in range(r.SALTOS):
        candidatos = [c for c in dict.fromkeys(candidatos) if c not in vistos]
        if not candidatos:
            break
        vistos.update(candidatos)
        sel = m.elegir(situacion, [r.opcion(c, enunciados[c]) for c in candidatos], None,
                       "Which of these statements apply to it?")
        nuevos = [candidatos[i] for i in sel]
        elegidos += nuevos
        candidatos = [l for c in nuevos for l in enunciados[c]["links"] if l in enunciados]
    return list(dict.fromkeys(elegidos))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:12b-mlx")
    ap.add_argument("--memory", default=str(Path.home() / ".config" / "director" / "d9" / "memory"))
    ap.add_argument("--out", default=str(Path.home() / ".config" / "director" / "d10"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default="S10,S9")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mem = Path(a.memory).expanduser()
    situaciones, enunciados = r.cargar_memoria(mem)
    trayectorias, viejas = partir_index(mem, situaciones)
    assert len(trayectorias) == 20 and len(viejas) == 87, (len(trayectorias), len(viejas))
    ref = r.referencia()
    casos = [json.loads(l) for l in r.CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    hechos = set()
    walks = out / "walks.jsonl"
    if walks.exists():  # resumable
        hechos = {(w["arm"], w["rep"], w["case"]) for w in map(json.loads, walks.read_text().splitlines())}
    with walks.open("a", encoding="utf-8") as f:
        for rep in range(a.k):
            for arm in a.arms.split(","):
                m = r.Modelo(a.model)
                for c in casos:
                    if (arm, rep, c["id"]) in hechos:
                        continue
                    t0 = time.time()
                    camino = (recorrer_s10(m, c["situation"], trayectorias, viejas, enunciados) if arm == "S10"
                              else r.recorrer(m, c["situation"], situaciones, enunciados))
                    base = r.bm25(c["situation"], enunciados, max(1, len(camino)))
                    nucleo, amplio = ref[c["id"]]
                    fila = {"arm": arm, "rep": rep, "case": c["id"], "picked": camino, "bm25": base,
                            "recall_core": round(r.recall(camino, nucleo), 3),
                            "recall_core_bm25": round(r.recall(base, nucleo), 3),
                            "recall_broad": round(r.recall(camino, amplio), 3),
                            "precision_broad": round(len(set(camino) & amplio) / len(camino), 3) if camino else 0.0,
                            "json_failures": m.fallas, "wall_s": round(time.time() - t0)}
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"rep{rep} {arm} {c['id'][:3]}: picked {len(camino)} core {fila['recall_core']} "
                          f"(BM25 {fila['recall_core_bm25']}) prec {fila['precision_broad']} {fila['wall_s']} s",
                          flush=True)
    informe(walks, out)
    return 0


def informe(walks: Path, out: Path) -> None:
    filas = [json.loads(l) for l in walks.read_text().splitlines()]
    casos = sorted({f["case"] for f in filas})
    media = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    por = {}
    for arm in sorted({f["arm"] for f in filas}):
        pc = {c: [f for f in filas if f["arm"] == arm and f["case"] == c] for c in casos}
        m = {c: {k: media([f[k] for f in fs]) for k in ("recall_core", "recall_core_bm25", "precision_broad")}
             | {"picked": media([len(f["picked"]) for f in fs]), "reps": len(fs),
                "distinct": len({tuple(sorted(f["picked"])) for f in fs})} for c, fs in pc.items()}
        gana = sum(v["recall_core"] > v["recall_core_bm25"] for v in m.values())
        pierde = sum(v["recall_core"] < v["recall_core_bm25"] for v in m.values())
        rc, pr = media([v["recall_core"] for v in m.values()]), media([v["precision_broad"] for v in m.values()])
        falla = rc < 0.40 or gana <= pierde or pr < 0.70
        por[arm] = {"per_case": m, "mean_recall_core": round(rc, 3),
                    "mean_recall_core_bm25": round(media([v["recall_core_bm25"] for v in m.values()]), 3),
                    "mean_precision_broad": round(pr, 3),
                    "mean_picked": round(media([v["picked"] for v in m.values()]), 1),
                    "vs_bm25": {"wins": gana, "losses": pierde, "p": round(r.signo(gana, pierde), 4)},
                    "cases_varying_across_reps": sum(v["distinct"] > 1 for v in m.values()),
                    "verdict": "NOT NAVIGABLE" if falla else "NAVIGABLE" if rc >= 0.70 else "PARTIAL"}
    if {"S10", "S9"} <= por.keys():
        a, b = por["S10"]["per_case"], por["S9"]["per_case"]
        g = sum(a[c]["recall_core"] > b[c]["recall_core"] for c in casos)
        p = sum(a[c]["recall_core"] < b[c]["recall_core"] for c in casos)
        por["S10_vs_S9"] = {"wins": g, "losses": p, "ties": len(casos) - g - p, "p": round(r.signo(g, p), 4)}
    (out / "report.json").write_text(json.dumps(por, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_case"} if isinstance(v, dict) else v)
                      for k, v in por.items()}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
