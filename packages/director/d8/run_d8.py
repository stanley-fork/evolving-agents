"""D8: a local 12B walks the memory when code drives the walk. See docs/BRIEF-D8.md.

    python d8/run_d8.py [--model gemma4:12b-mlx] [--out ~/.config/director/d8] [--cases c01,c02]

Standard library only, like the rest of the director. Ollama at OLLAMA_URL (default http://127.0.0.1:11434).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE.parent / "d0" / "cases.jsonl"
D0 = Path.home() / ".config" / "director" / "d0" / "arm-T"
MEMORY = D0 / "sandbox" / "memory"
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CITA = re.compile(r"\[\[([A-Za-z0-9_]+#[A-Za-z0-9_]+)\]\]")
MAX_SITUACIONES = 4
SALTOS = 2
MAX_FALLAS_JSON = 5

SISTEMA = (
    "You help navigate a team's operational memory. You are given a situation and a numbered list of options. "
    "Pick only options that clearly apply to the situation. Answer only with the JSON requested."
)


# ── memory ─────────────────────────────────────────────────────────────────


def cargar_memoria(d: Path) -> tuple[list[tuple[str, list[str]]], dict[str, dict]]:
    """(Index situations with their entry statements, statements by `Page#id`)."""
    situaciones = []
    for linea in (d / "Index.md").read_text(encoding="utf-8").splitlines():
        if linea.startswith("- ") and "→" in linea:
            texto, destinos = linea[2:].split("→", 1)
            situaciones.append((texto.strip(), CITA.findall(destinos)))
    enunciados: dict[str, dict] = {}
    for f in sorted(d.glob("*.md")):
        if f.name in ("Index.md", "README.md"):
            continue
        pagina = f.stem
        for bloque in re.split(r"(?m)^## #", f.read_text(encoding="utf-8"))[1:]:
            sid, _, cuerpo = bloque.partition("\n")
            campos = {k: (re.search(rf"\*\*{k}:\*\*\s*(.+)", cuerpo) or [None, ""])[1].strip()
                      for k in ("When", "Do", "Because", "Links")}
            enunciados[f"{pagina}#{sid.strip()}"] = {**campos, "links": CITA.findall(campos["Links"])}
    return situaciones, enunciados


# ── the 12B, constrained ───────────────────────────────────────────────────


class Modelo:
    def __init__(self, nombre: str):
        self.nombre, self.fallas, self.llamadas = nombre, 0, 0

    def elegir(self, situacion: str, opciones: list[str], tope: int | None, que: str) -> list[int]:
        lista = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(opciones))
        esquema = {"type": "object", "required": ["elegidas"], "properties": {"elegidas": {
            "type": "array", "items": {"type": "integer", "minimum": 1, "maximum": len(opciones)},
            **({"maxItems": tope} if tope else {})}}}
        usuario = (f"Situation:\n{situacion}\n\n{que}\n{lista}\n\n"
                   f"Return {{\"elegidas\": [option numbers]}}"
                   + (f" with at most {tope} numbers." if tope else ".") + " An empty list is allowed.")
        for intento in range(2):
            self.llamadas += 1
            cuerpo = {"model": self.nombre, "stream": False, "think": False, "format": esquema,
                      "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}],
                      "options": {"temperature": 0, "num_ctx": 16384}}
            req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(cuerpo).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=900).read())
                elegidas = json.loads(r["message"]["content"])["elegidas"]
                ok = [i - 1 for i in elegidas if isinstance(i, int) and 1 <= i <= len(opciones)]
                return list(dict.fromkeys(ok))[: tope or None]
            except Exception:  # noqa: BLE001 — se cuenta, no se oculta
                if intento == 1:
                    self.fallas += 1
                    if self.fallas > MAX_FALLAS_JSON:
                        raise SystemExit(f"abort: more than {MAX_FALLAS_JSON} JSON failures (BRIEF-D8)")
        return []


def opcion(sid: str, e: dict) -> str:
    return f"[[{sid}]] When: {e['When']} Do: {e['Do']}"


def recorrer(m: Modelo, situacion: str, situaciones, enunciados) -> list[str]:
    idx = m.elegir(situacion, [s for s, _ in situaciones], MAX_SITUACIONES,
                   "Which of these situations from the memory's index match it?")
    vistos: set[str] = set()
    candidatos = [sid for i in idx for sid in situaciones[i][1] if sid in enunciados]
    elegidos: list[str] = []
    for _ in range(1 + SALTOS):
        candidatos = [c for c in dict.fromkeys(candidatos) if c not in vistos]
        if not candidatos:
            break
        vistos.update(candidatos)
        sel = m.elegir(situacion, [opcion(c, enunciados[c]) for c in candidatos], None,
                       "Which of these statements apply to it?")
        nuevos = [candidatos[i] for i in sel]
        elegidos += nuevos
        candidatos = [l for c in nuevos for l in enunciados[c]["links"] if l in enunciados]
    return list(dict.fromkeys(elegidos))


# ── the baseline without a model ───────────────────────────────────────────


def bm25(consulta: str, enunciados: dict[str, dict], k: int, k1: float = 1.5, b: float = 0.75) -> list[str]:
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())  # noqa: E731
    docs = {sid: tok(" ".join((e["When"], e["Do"], e["Because"]))) for sid, e in enunciados.items()}
    n, prom = len(docs), sum(map(len, docs.values())) / max(1, len(docs))
    df = Counter(w for d in docs.values() for w in set(d))
    q = tok(consulta)

    def puntaje(d: list[str]) -> float:
        tf = Counter(d)
        return sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                   * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / prom)) for w in q if w in tf)

    return sorted(docs, key=lambda s: puntaje(docs[s]), reverse=True)[:k]


# ── reference and metrics ──────────────────────────────────────────────────


def referencia() -> dict[str, tuple[set[str], set[str]]]:
    por_caso: dict[str, list[set[str]]] = {}
    for linea in (D0 / "plans.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(linea)
        por_caso.setdefault(p["case"], []).append(set(CITA.findall(p.get("plan") or "")))
    return {c: (set.intersection(*s), set.union(*s)) for c, s in por_caso.items()}


def recall(elegidos: list[str], ref: set[str]) -> float:
    return len(set(elegidos) & ref) / len(ref) if ref else 1.0


def signo(gana: int, pierde: int) -> float:
    n = gana + pierde
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gana, pierde) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:12b-mlx")
    ap.add_argument("--out", default=str(Path.home() / ".config" / "director" / "d8"))
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    situaciones, enunciados = cargar_memoria(MEMORY)
    ref = referencia()
    casos = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.cases:
        casos = [c for c in casos if c["id"].split("-")[0] in a.cases.split(",")]
    m = Modelo(a.model)
    filas = []
    with (out / "walks.jsonl").open("a", encoding="utf-8") as f:
        for c in casos:
            t0 = time.time()
            camino = recorrer(m, c["situation"], situaciones, enunciados)
            base = bm25(c["situation"], enunciados, max(1, len(camino)))
            nucleo, amplio = ref[c["id"]]
            fila = {"case": c["id"], "picked": camino, "bm25": base,
                    "core": sorted(nucleo), "broad_n": len(amplio),
                    "recall_core": round(recall(camino, nucleo), 3),
                    "recall_core_bm25": round(recall(base, nucleo), 3),
                    "recall_broad": round(recall(camino, amplio), 3),
                    "precision_broad": round(len(set(camino) & amplio) / len(camino), 3) if camino else 0.0,
                    "wall_s": round(time.time() - t0)}
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            filas.append(fila)
            print(f"{c['id']}: picked {len(camino)} · core recall {fila['recall_core']} "
                  f"(BM25 {fila['recall_core_bm25']}) · broad {fila['recall_broad']} · "
                  f"precision {fila['precision_broad']} · {fila['wall_s']} s", flush=True)
    n = len(filas)
    media = sum(r["recall_core"] for r in filas) / n
    gana = sum(r["recall_core"] > r["recall_core_bm25"] for r in filas)
    pierde = sum(r["recall_core"] < r["recall_core_bm25"] for r in filas)
    veredicto = ("NAVIGABLE" if media >= 0.70 and gana > pierde else
                 "NOT NAVIGABLE" if media < 0.40 or gana <= pierde else "PARTIAL")
    informe = {"model": a.model, "cases": n, "mean_recall_core": round(media, 3),
               "mean_recall_core_bm25": round(sum(r["recall_core_bm25"] for r in filas) / n, 3),
               "mean_recall_broad": round(sum(r["recall_broad"] for r in filas) / n, 3),
               "mean_precision_broad": round(sum(r["precision_broad"] for r in filas) / n, 3),
               "mean_picked": round(sum(len(r["picked"]) for r in filas) / n, 1),
               "vs_bm25": {"wins": gana, "losses": pierde, "ties": n - gana - pierde, "p": round(signo(gana, pierde), 4)},
               "json_failures": m.fallas, "calls": m.llamadas,
               "missed_core": sorted({s for r in filas for s in set(r["core"]) - set(r["picked"])}),
               "verdict": veredicto}
    (out / "report.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in informe.items() if k != "missed_core"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
