"""D6 (docs/BRIEF-D6.md): validate the targeted guard. Baseline = D5's four arms of the unchanged live memory (reused,
checked by content hash). Candidates: B1 (subtree rule removed), B2 (mirror rule removed), A'1 and A'2 (unchanged), Bc
(brief-before-run rule removed, descriptive). Each re-plans only its relevant cases, k = 3.

    python validate_targeted.py --config ~/.config/director/d0.json --memory ~/.config/director/live/memory
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "d0"))
sys.path.insert(0, str(HERE.parent / "d5"))

import run_d0 as R  # noqa: E402
from director.guard import (changed_statements, cited_statements, memory_hash, relevant_cases,  # noqa: E402
                            targeted_verdict)
from validate_guard import broken_copy  # noqa: E402

SUBTREE, MIRROR, BRIEF = ("Workspace_Rules#ai_base_byte_identical", "Documentation#mirror_same_commit",
                          "Experiment_Briefing#write_brief_first")
EXPECTED = {"B1": ["c13-edit-the-subtree"],
            "B2": ["c04-private-name-in-public", "c08-baseline-choice", "c09-doc-without-mirror"],
            "Bc": ["c02-heavy-local-run", "c03-launch-without-brief", "c06-demo-outside-measured-condition",
                   "c08-baseline-choice", "c10-grid-overnight", "c12-long-run-through-tail", "c14-baseline-at-ceiling"]}
APRIME = sorted(set(EXPECTED["B1"]) | set(EXPECTED["B2"]))


def samples(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("items"):
            out.setdefault(r["case"], []).append(sum(x["met"] for x in r["items"]) / len(r["items"]))
    return out


def arm(c, name: str, memory: Path, cases: list[str]) -> dict[str, list[float]]:
    d = R.arm_dir(c, name)
    d.mkdir(parents=True, exist_ok=True)
    cf = d / "cases.jsonl"
    all_cases = (HERE.parent / "d0" / "cases.jsonl").read_text().splitlines()
    cf.write_text("\n".join(l for l in all_cases if json.loads(l)["id"] in cases) + "\n")
    R.CASES_FILE = cf
    if not (d / "sandbox").exists():
        shutil.copytree(memory, d / "sandbox" / "memory")
        shutil.copy(HERE.parent / "director" / "DIRECTOR.md", d / "system.md")
    for _ in range(3):
        R.run(c, name)
        R.judge(c, name)
        got = samples(R.rows(d / "verdicts.jsonl"))
        if all(len(got.get(k, [])) == R.K for k in cases):
            break
    print(f"[d6] {name}: {({k: got.get(k) for k in cases})} · spent ${R.spent(c):.2f}", flush=True)
    return {k: got.get(k, []) for k in cases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--memory", required=True)
    a = ap.parse_args()
    c = R.cfg(a.config)
    d5 = Path(c["private_dir"]) / "d5"
    frozen = d5 / "memory-A"
    assert memory_hash(Path(a.memory).expanduser()) == memory_hash(frozen), "the live memory changed since D5's baseline"
    # the baseline: D5's four arms, k = 4 per case
    R.EXPERIMENT = "d5"
    base_rows, plans = [], []
    for x in ("A1", "A2", "A3", "A4"):
        base_rows += R.rows(R.arm_dir(c, x) / "verdicts.jsonl")
        plans += R.rows(R.arm_dir(c, x) / "plans.jsonl")
    base, cites = samples(base_rows), cited_statements(plans)
    assert all(len(v) == 4 for v in base.values()) and len(base) == 14
    R.EXPERIMENT, R.K, R.CAP = "d6", 3, 35.0
    root = Path(c["private_dir"]) / "d6"
    root.mkdir(parents=True, exist_ok=True)
    mems = {"B1": [SUBTREE], "B2": [MIRROR], "Bc": [BRIEF]}
    out = {"baseline": "D5 A1–A4 (k = 4)", "arms": {}}
    for name, remove in mems.items():
        m = root / f"memory-{name}"
        broken_copy(frozen, m, remove)
        changed = changed_statements(frozen, m)
        rel = relevant_cases(changed, cites)
        assert changed == set(remove) and rel == EXPECTED[name], (name, changed, rel)
        cand = arm(c, name, m, rel)
        out["arms"][name] = {"changed": sorted(changed), "relevant": rel, **targeted_verdict({k: base[k] for k in rel}, cand)}
    for name in ("A'1", "A'2"):
        cand = arm(c, name.replace("'", "p"), frozen, APRIME)
        out["arms"][name] = {"relevant": APRIME, **targeted_verdict({k: base[k] for k in APRIME}, cand)}
    A = out["arms"]
    detected = (not A["B1"]["passed"]) + (not A["B2"]["passed"])
    false_ref = (not A["A'1"]["passed"]) + (not A["A'2"]["passed"])
    out.update(detected=f"{detected}/2", false_refusals=f"{false_ref}/2", spent_usd=R.spent(c),
               verdict=("VALIDATED" if detected == 2 and false_ref == 0 else "TOO STRICT" if false_ref else "BLIND"),
               at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (root / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
