"""Seed the targeted guard's baseline for the live memory from D5's four arms (the same memory, by content hash):
case -> its sample scores and the statements its plans cite. Written to <log dir>/guards/baseline/<hash>.json.

    python build_baseline.py --config ~/.config/director/d0.json --memory ~/.config/director/live/memory
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "d0")); sys.path.insert(0, str(HERE))
import run_d0 as R  # noqa: E402
from director.guard import cited_statements, memory_hash  # noqa: E402
from validate_targeted import samples  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--memory", required=True)
a = ap.parse_args()
c = R.cfg(a.config); mem = Path(a.memory).expanduser()
frozen = Path(c["private_dir"]) / "d5" / "memory-A"
assert memory_hash(mem) == memory_hash(frozen), "the live memory is not D5's"
R.EXPERIMENT = "d5"
rows, plans = [], []
for x in ("A1", "A2", "A3", "A4"):
    rows += R.rows(R.arm_dir(c, x) / "verdicts.jsonl"); plans += R.rows(R.arm_dir(c, x) / "plans.jsonl")
s, cites = samples(rows), cited_statements(plans)
out = {k: {"samples": s[k], "cites": sorted(cites.get(k, []))} for k in sorted(s)}
dst = mem.parent / "guards" / "baseline" / f"{memory_hash(mem)}.json"
dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(out, indent=1))
print(dst, len(out), "cases")
