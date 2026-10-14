"""The memory as an agentvcs manifest: one `prompt` dimension per page, so every learned edit is a versioned,
gated patch — `blame` says which edit moved a score, `bisect` finds the one that broke it.

    memory.<page>   {"kind": "prompt", "content": {"template": <the page's text>, "variables": []}}

`README.md` (the format) is not memory and is left out. `AGENTVCS` overrides the agentvcs executable.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def manifest(memory_dir: Path, name: str) -> dict:
    dims = {}
    for p in sorted(Path(memory_dir).glob("*.md")):
        if p.name == "README.md":
            continue
        dims[f"memory.{p.stem.lower()}"] = {"kind": "prompt", "content": {"template": p.read_text(), "variables": []}}
    return {"protocol": "agentvcs/0.1", "type": "harness_manifest", "name": name, "dimensions": dims}


def avcs(store: Path, *args: str, stdin: str | None = None) -> dict:
    exe = os.environ.get("AGENTVCS", "agentvcs")
    p = subprocess.run([exe, *args, "--json"], cwd=store, input=stdin, capture_output=True, text=True)
    out = json.loads(p.stdout)
    if not out.get("ok", True) and p.returncode:
        raise RuntimeError(f"agentvcs {' '.join(args)}: {out.get('error')}")
    return out


def step(metric: str, value: float, agent: str = "director") -> str:
    t = "2026-10-14T00:00:00Z"
    return json.dumps({"agent_id": agent, "inputs": [], "outputs": [], "started_at": t, "ended_at": t,
                       "tokens": {"in": 0, "out": 0}, "latency_ms": 0, "metrics": {metric: value}})
