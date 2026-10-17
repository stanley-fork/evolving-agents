"""D14: local retrieval + Claude writes the plan (no tools), against Sonnet walking with tools. See docs/BRIEF-D14.md.

    python d14/run_d14.py --list HYB        # or HYB_F, per D13's filter verdict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("d0", "d8", "d11", "d13"):
    sys.path.insert(0, str(HERE.parent / sub))
import run_d0 as d0  # noqa: E402
import run_d11 as d11  # noqa: E402
import run_d13 as d13  # noqa: E402

r = d13.r
OUT = d13.PRIV / "d14"
CAP = 15.0


def listas(cual: str, enunciados: dict) -> dict[str, list[str]]:
    """Each case's list, recomputed as D13 computed it (same code, same settings)."""
    ids = list(enunciados)
    vec = dict(zip(ids, d13.embed([f"title: none | text: When: {enunciados[s]['When']} Do: {enunciados[s]['Do']}"
                                   for s in ids])))
    m = r.Modelo("gemma4:12b-mlx")
    out = {}
    for banco, d in d13.BANKS.items():
        casos = [json.loads(l) for l in (d / "cases.jsonl").read_text().splitlines() if l.strip()]
        qv = d13.embed([f"task: search result | query: {c['situation']}" for c in casos])
        for c, q in zip(casos, qv):
            bm = r.bm25(c["situation"], enunciados, len(ids))
            em = sorted(ids, key=lambda s: d13.coseno(q, vec[s]), reverse=True)
            hy = d13.fusion(bm, em)
            out[c["id"]] = (hy[: d13.K] if cual == "HYB" else d13.filtrar(m, c["situation"], hy[: d13.POOL], enunciados))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="HYB", choices=["HYB", "HYB_F"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, enunciados = r.cargar_memoria(d13.MEMORY)
    casos = {json.loads(l)["id"]: json.loads(l) for d in d13.BANKS.values()
             for l in (d / "cases.jsonl").read_text().splitlines() if l.strip()}
    cache = OUT / f"lists-{a.list}.json"
    if not cache.exists():
        cache.write_text(json.dumps(listas(a.list, enunciados)))
    lst = json.loads(cache.read_text())
    plans, verd = OUT / "plans-R.jsonl", OUT / "verdicts.jsonl"
    rows = d11.rows
    gastado = lambda: sum(x.get("cost_usd") or 0 for f in (plans, verd) for x in rows(f))  # noqa: E731
    vacio = OUT / "cwd"
    vacio.mkdir(exist_ok=True)
    hechos = {(x["case"], x["sample"]) for x in rows(plans) if x.get("plan")}
    for cid, c in casos.items():
        for s in range(2):
            if (cid, s) in hechos:
                continue
            if gastado() >= CAP:
                raise SystemExit("cap reached")
            texto = "\n".join(f"- [[{x}]] When: {enunciados[x]['When']} Do: {enunciados[x]['Do']} "
                              f"Because: {enunciados[x]['Because']}" for x in lst[cid])
            prompt = (d11.CON_MEMORIA.format(enunciados=texto) + d11.ASK_12B.format(situation=c["situation"]))
            t0 = time.time()
            res = d0.claude(prompt, vacio, None, "", 1, 0.60)
            with plans.open("a") as f:
                f.write(json.dumps({"arm": "R", "case": cid, "sample": s, "plan": res.get("result"),
                                    "statements": lst[cid], "cost_usd": res.get("total_cost_usd"),
                                    "wall_s": round(time.time() - t0)}) + "\n")
            print(f"[R] {cid} s{s} ${res.get('total_cost_usd') or 0:.3f}", flush=True)
    tmpl = (HERE.parent / "d0" / "judge.md").read_text()
    juzgados = {(x["case"], x["sample"]) for x in rows(verd) if x.get("items")}
    for p in rows(plans):
        if not p.get("plan") or (p["case"], p["sample"]) in juzgados:
            continue
        c = casos[p["case"]]
        cl = "\n".join(f"- `{it['id']}`: {it['text']}" for it in c["checklist"])
        v = d0.claude(tmpl.replace("{situation}", c["situation"]).replace("{checklist}", cl)
                      .replace("{plan}", p["plan"]), vacio, None, "", 1, 0.30)
        items = d0.parse_items(v.get("result") or "", [it["id"] for it in c["checklist"]])
        with verd.open("a") as f:
            f.write(json.dumps({"arm": "R", "case": p["case"], "sample": p["sample"], "items": items,
                                "cost_usd": v.get("total_cost_usd")}) + "\n")
    # report: R (new) vs T (D11 + D12 verdicts, reused)
    def puntajes(fuente: Path, arm: str) -> dict[str, float]:
        por: dict[str, list[float]] = {}
        for v in rows(fuente):
            if v.get("items") and v["arm"] == arm:
                por.setdefault(v["case"], []).append(sum(x["met"] for x in v["items"]) / len(v["items"]))
        return {k: sum(x) / len(x) for k, x in por.items()}
    R = puntajes(verd, "R")
    T = {**puntajes(d13.PRIV / "d11" / "verdicts.jsonl", "T"), **puntajes(d13.PRIV / "d12" / "verdicts.jsonl", "T")}
    com = sorted(set(R) & set(T))
    mr, mt = sum(R[c] for c in com) / len(com), sum(T[c] for c in com) / len(com)
    g, l = sum(R[c] > T[c] for c in com), sum(R[c] < T[c] for c in com)
    costo_t = [x.get("cost_usd") or 0 for b in ("d11", "d12") for x in rows(d13.PRIV / b / "plans-T.jsonl") if x.get("plan")]
    costo_r = [x.get("cost_usd") or 0 for x in rows(plans) if x.get("plan")]
    inf = {"list": a.list, "R": round(mr, 3), "T": round(mt, 3),
           "by_bank": {b: {"R": round(sum(R[c] for c in com if c.startswith(p)) / 12, 3),
                           "T": round(sum(T[c] for c in com if c.startswith(p)) / 12, 3)}
                       for b, p in (("d11", "n"), ("d12", "m"))},
           "R_vs_T": {"wins": g, "losses": l, "ties": len(com) - g - l, "p": round(d11.signo(g, l), 4)},
           "cost_per_plan": {"R": round(sum(costo_r) / len(costo_r), 3), "T": round(sum(costo_t) / len(costo_t), 3)},
           "total_cost_usd": round(gastado(), 2),
           "per_case": {c: {"R": round(R[c], 3), "T": round(T[c], 3)} for c in com},
           "verdict": "SPLIT WORKS" if mr >= mt - 0.05 else "SPLIT LOSES"}
    (OUT / "report.json").write_text(json.dumps(inf, indent=2))
    print(json.dumps({k: v for k, v in inf.items() if k != "per_case"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
