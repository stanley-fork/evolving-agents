"""D2 step 5: version the learning in agentvcs (docs/BRIEF-D2.md).

Memory v0 (compiled) and v1b (after reflection) become manifests; a run records the control's probe scores
under v0, then the v0 -> v1b patch is proposed with the lessons as rationale, gated by a suite that reads the
D2 report (probe gain against the half-headroom bar, guard against 0.86), applied only if the gate passes, and
the edited memory's probe scores are recorded under v1b. Then `verify` and `blame` on the probe metric.

    python version.py --config ~/.config/director/d0.json        # store under <private_dir>/vcs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "d0"))

import run_d0 as R  # noqa: E402
from director.memory_vcs import avcs, manifest, step  # noqa: E402


def samples(c: dict, arm: str) -> list[float]:
    R.EXPERIMENT = "d2"
    out = []
    for r in R.rows(R.arm_dir(c, arm) / "verdicts.jsonl"):
        if r.get("items"):
            out.append(sum(x["met"] for x in r["items"]) / len(r["items"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = R.cfg(a.config)
    priv = Path(c["private_dir"])
    store = priv / "vcs"
    if store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True)
    report = json.loads((priv / "d2" / "report2.json").read_text())
    avcs(store, "init")
    ids = {}
    for tag, mem in (("v0", priv / "d0" / "arm-T" / "sandbox" / "memory"), ("v1b", priv / "d2" / "memory-v1b")):
        f = store / f"memory-{tag}.json"
        f.write_text(json.dumps(manifest(mem, f"director memory {tag}")))
        ids[tag] = avcs(store, "snapshot", str(f))["manifest_id"]
    avcs(store, "run", "start", "--manifest", ids["v0"], "--run-id", "d2-learn")
    for v in samples(c, "T0c"):
        avcs(store, "step", "record", "d2-learn", stdin=step("probe", v))
    rationale = ("D2 lessons A, B, E, F, I (episodes2.jsonl): " +
                 "; ".join(f"{x['lesson']}: {', '.join(x['applied'])}" for x in report.get("reflect", [])))
    pid = avcs(store, "patch", "propose", "d2-learn", "--from", ids["v0"], "--to", ids["v1b"],
               "--rationale", rationale)["patch_id"]
    # The gate's evidence is the D2 report, already scored on held-out probes and on D0's cases: the suite
    # reads it rather than re-spending on the same measurement.
    suite = store / "d2-suite.yaml"
    suite.write_text(
        "command: |-\n"
        f"  python3 -c \"import json; d=json.load(open('{priv / 'd2' / 'report2.json'}')); "
        "print(json.dumps({'metrics': {'probe_gain_over_bar': d['delta'] - d['required'], "
        "'guard': d['guard']['mean']}}))\"\n"
        "thresholds:\n"
        "  probe_gain_over_bar: {op: \">=\", value: 0}\n"
        "  guard: {op: \">=\", value: 0.86}\n")
    gate = avcs(store, "gate", "run", pid, "--suite", str(suite))["gate_result"]
    applied = None
    if gate["passed"]:
        applied = avcs(store, "patch", "apply", pid)
        for v in samples(c, "T1b"):
            avcs(store, "step", "record", "d2-learn", stdin=step("probe", v))
    avcs(store, "export", "audit", "d2-learn", "-o", str(store / "d2-learn.audit.json"))
    ver = avcs(store, "verify", str(store / "d2-learn.audit.json"))
    blame = avcs(store, "blame", str(store / "d2-learn.audit.json"), "--metric", "probe")
    out = {"manifests": ids, "patch": pid, "gate": {"passed": gate["passed"], "metrics": gate["metrics"]},
           "applied": bool(applied), "verify": ver.get("valid"),
           "blame": [{"patches": x["patches"], "delta": x["delta"]} for x in blame.get("attributions", [])],
           "dimensions_changed": sorted(k for k in manifest(priv / "d2" / "memory-v1b", "").get("dimensions")
                                        if manifest(priv / "d2" / "memory-v1b", "")["dimensions"][k]
                                        != manifest(priv / "d0" / "arm-T" / "sandbox" / "memory", "")["dimensions"].get(k))}
    (priv / "d2" / "version.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
