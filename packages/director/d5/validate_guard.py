"""D5 (docs/BRIEF-D5.md): validate the paired guard. Four arms of the unchanged live memory (A1–A4) and one arm with
four rules removed (B), D0's 14 cases, k = 1; then false refusals over the A/A pairs and detection of B.

    python validate_guard.py --config ~/.config/director/d0.json --memory ~/.config/director/live/memory

Arms live under <private_dir>/d5/; a complete arm is reused on a rerun of this script (resume), never across scripts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "d0"))

import run_d0 as R  # noqa: E402
from director.guard import paired_verdict  # noqa: E402
from director.learn import STMT, statements  # noqa: E402

REMOVE = ["Workspace_Rules#ai_base_byte_identical", "Documentation#mirror_same_commit",
          "Experiment_Briefing#write_brief_first", "Confidentiality#never_name_private_in_public"]
FALSE_MAX, DETECT_MIN, NET = 1, 3, 4


def broken_copy(src: Path, dst: Path, remove: list[str] | None = None) -> list[str]:
    """`src` with the `remove` statements (default: REMOVE) deleted and every link or index entry to them dropped."""
    REMOVE = remove or globals()["REMOVE"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    removed = []
    for ref in REMOVE:
        page, sid = ref.split("#")
        p = dst / f"{page}.md"
        text = p.read_text()
        block = statements(text).get(sid)
        assert block, f"{ref} not found"
        start = text.index(f"## #{sid}\n")
        text = text[:start] + text[start + len(block):]
        p.write_text(re.sub(r"\n{3,}", "\n\n", text))
        removed.append(ref)
    for p in dst.glob("*.md"):
        t = p.read_text()
        for ref in REMOVE:
            # horizontal whitespace only: `\s` would eat the newline and glue the next `## #id` onto this line
            t = re.sub(rf"[ \t]*(?:·|,)[ \t]*\[\[{re.escape(ref)}\]\]", "", t)
            t = re.sub(rf"\[\[{re.escape(ref)}\]\][ \t]*(?:·|,)?[ \t]*", "", t)
        t = re.sub(r"^- [^\n]*?→[ \t]*$\n?", "", t, flags=re.M)     # an index line left with no target
        p.write_text(t)
    count = lambda d: sum(len(statements(x.read_text())) for x in d.glob("*.md") if x.name not in ("Index.md", "README.md"))
    left = [r for r in REMOVE if f"[[{r}]]" in "".join(x.read_text() for x in dst.glob("*.md"))]
    assert not left, left
    assert count(dst) == count(src) - len(REMOVE), (count(src), count(dst))   # exactly these, nothing else
    return removed


def arm(c, name: str, memory: Path) -> dict:
    d = R.arm_dir(c, name)
    if d.exists() and R.score(c, name)["n_cases"] == 14:
        print(f"[d5] {name}: complete, reused", flush=True)
        return R.score(c, name)
    if not (d / "sandbox").exists():              # an incomplete arm is completed (R.run / R.judge resume), not redone
        shutil.copytree(memory, d / "sandbox" / "memory")
        shutil.copy(HERE.parent / "director" / "DIRECTOR.md", d / "system.md")
    for _ in range(3):
        R.run(c, name)
        R.judge(c, name)
        s = R.score(c, name)
        if s["n_cases"] == 14:
            break
    print(f"[d5] {name}: mean {s['mean']} · {s['n_cases']} cases · spent ${R.spent(c):.2f}", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--memory", required=True)
    a = ap.parse_args()
    c = R.cfg(a.config)
    R.EXPERIMENT, R.K, R.CAP = "d5", 1, 40.0
    R.CASES_FILE = HERE.parent / "d0" / "cases.jsonl"
    root = Path(c["private_dir"]) / "d5"
    root.mkdir(parents=True, exist_ok=True)
    live = Path(a.memory).expanduser()
    frozen = root / "memory-A"                     # the memory under test, frozen at the start
    if not frozen.exists():
        shutil.copytree(live, frozen)
    removed = broken_copy(frozen, root / "memory-B")
    arms = {f"A{i}": arm(c, f"A{i}", frozen) for i in range(1, 5)}
    arms["B"] = arm(c, "B", root / "memory-B")
    A = [f"A{i}" for i in range(1, 5)]
    aa = []
    for i, j in itertools.combinations(A, 2):
        v = paired_verdict(arms[i]["cases"], arms[j]["cases"], None, NET)
        aa.append({"candidate": i, "current": j, **v})
    det = [{"current": j, **paired_verdict(arms["B"]["cases"], arms[j]["cases"], None, NET)} for j in A]
    false_ref = sum(not x["passed"] for x in aa)
    detected = sum(not x["passed"] for x in det)
    flips = {k: [arms[x]["cases"].get(k) for x in A] for k in sorted(arms["A1"]["cases"])}
    out = {"removed": removed,
           "means": {k: v["mean"] for k, v in arms.items()},
           "absolute_0.86_refuses": {k: v["mean"] < 0.86 for k, v in arms.items()},
           "aa_pairs": aa, "false_refusals": false_ref,
           "detection_pairs": det, "detected": detected,
           "B_cases_below_A": {k: [arms["B"]["cases"][k], flips[k]] for k in flips
                               if arms["B"]["cases"][k] < min(flips[k])},
           "flipping_cases": {k: v for k, v in flips.items() if len(set(v)) > 1},
           "spent_usd": R.spent(c),
           "verdict": ("VALIDATED" if false_ref <= FALSE_MAX and detected >= DETECT_MIN
                       else "TOO STRICT" if false_ref > FALSE_MAX else "BLIND"),
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
