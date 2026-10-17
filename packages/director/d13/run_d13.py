"""D13: retrieval by code (BM25, local embeddings, fusion) and the 12B as a short-list filter. See docs/BRIEF-D13.md.

    python d13/run_d13.py            # US$0, local Ollama (embeddinggemma + gemma4:12b-mlx)
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d8"))
import run_d8 as r  # noqa: E402

PRIV = Path.home() / ".config" / "director"
MEMORY = PRIV / "d9" / "memory"
OUT = PRIV / "d13"
BANKS = {"d11": PRIV / "d11", "d12": PRIV / "d12"}
EMBED = "embeddinggemma"
K, POOL, RRF, LOTE = 20, 40, 60, 10


def embed(textos: list[str]) -> list[list[float]]:
    req = urllib.request.Request(f"{r.OLLAMA}/api/embed", data=json.dumps({"model": EMBED, "input": textos}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["embeddings"]


def coseno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def fusion(*rankings: list[str]) -> list[str]:
    puntos: dict[str, float] = {}
    for rk in rankings:
        for i, sid in enumerate(rk):
            puntos[sid] = puntos.get(sid, 0) + 1 / (RRF + i + 1)
    return sorted(puntos, key=puntos.get, reverse=True)


def filtrar(m: r.Modelo, situacion: str, lista: list[str], enunciados: dict) -> list[str]:
    kept = []
    for j in range(0, len(lista), LOTE):
        lote = lista[j: j + LOTE]
        sel = m.elegir(situacion, [r.opcion(c, enunciados[c]) for c in lote], None,
                       "Which of these statements apply to it?")
        kept += [lote[i] for i in sel]
    return kept


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, enunciados = r.cargar_memoria(MEMORY)
    ids = list(enunciados)
    vec = dict(zip(ids, embed([f"title: none | text: When: {enunciados[s]['When']} Do: {enunciados[s]['Do']}"
                               for s in ids])))
    m = r.Modelo("gemma4:12b-mlx")
    informe, filas = {}, []
    for banco, d in BANKS.items():
        casos = [json.loads(l) for l in (d / "cases.jsonl").read_text().splitlines() if l.strip()]
        qv = embed([f"task: search result | query: {c['situation']}" for c in casos])
        acc: dict[str, list[float]] = {}
        for c, q in zip(casos, qv):
            g = set(c["grounds"])
            bm = r.bm25(c["situation"], enunciados, len(ids))
            em = sorted(ids, key=lambda s: coseno(q, vec[s]), reverse=True)
            hy = fusion(bm, em)
            fil = filtrar(m, c["situation"], hy[:POOL], enunciados)
            g40 = g & set(hy[:POOL])
            rec = lambda xs: len(g & set(xs)) / len(g)  # noqa: E731
            fila = {"bank": banco, "case": c["id"], "BM25": rec(bm[:K]), "EMB": rec(em[:K]), "HYB": rec(hy[:K]),
                    "HYB40": rec(hy[:POOL]), "HYB_F": rec(fil), "F_size": len(fil),
                    "F_keeps_of_HYB40": len(g40 & set(fil)) / len(g40) if g40 else None}
            filas.append(fila)
            for k, v in fila.items():
                if isinstance(v, (int, float)) and v is not None:
                    acc.setdefault(k, []).append(v)
            print(json.dumps(fila), flush=True)
        informe[banco] = {k: round(sum(v) / len(v), 3) for k, v in acc.items()}
    hyb = [informe[b]["HYB"] for b in BANKS]
    keeps = [x["F_keeps_of_HYB40"] for x in filas if x["F_keeps_of_HYB40"] is not None]
    keep_mean = sum(keeps) / len(keeps)
    removed = 1 - sum(x["F_size"] for x in filas) / (POOL * len(filas))
    informe["verdict_retrieval"] = ("READY" if min(hyb) >= 0.85 else "PARTIAL" if min(hyb) >= 0.70 else "NOT READY")
    informe["filter"] = {"keeps_of_HYB40_grounds": round(keep_mean, 3), "removed": round(removed, 3),
                         "verdict": "USEFUL" if keep_mean >= 0.90 and removed >= 0.30 else "NOT USEFUL"}
    informe["json_failures"] = m.fallas
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(f) for f in filas) + "\n")
    (OUT / "report.json").write_text(json.dumps(informe, indent=2))
    print(json.dumps(informe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
