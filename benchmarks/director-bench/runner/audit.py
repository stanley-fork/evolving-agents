"""`bench audit`: grep every memory-store snapshot of a run for verifier/solution content."""
from __future__ import annotations

import json
from pathlib import Path

from runner.config import BenchError, load_family


def audit(run_dir: Path, family_dir: Path) -> tuple[list[str], list[str]]:
    """Return (leaks, warnings). A leak is a leak marker found in a memory snapshot."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fam = load_family(family_dir)
    if fam.name != meta["family"]:
        raise BenchError(f"run is for family {meta['family']}, not {fam.name}")
    markers = fam.leak_markers
    warnings = []
    if not markers:
        warnings.append(f"family {fam.name} declares no leak_markers: audit cannot detect leaks")
    # A marker that appears in no verifier/reference file detects nothing.
    secret_files = [p for d in ("verify", "reference") if (fam.dir / d).exists()
                    for p in (fam.dir / d).rglob("*") if p.is_file()]
    blob = b"".join(p.read_bytes() for p in secret_files)
    for mk in markers:
        if mk.encode() not in blob:
            warnings.append(f"leak marker {mk!r} does not occur in verify/ or reference/")
    leaks = []
    mem = run_dir / "memory"
    if not mem.exists():
        warnings.append("run has no memory snapshots")
    for p in sorted(mem.rglob("*")) if mem.exists() else []:
        if not p.is_file():
            continue
        data = p.read_bytes()
        for mk in markers:
            if mk.encode() in data:
                leaks.append(f"{p.relative_to(run_dir)}: contains leak marker {mk!r}")
    return leaks, warnings
