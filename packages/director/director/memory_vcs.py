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


def avcs(store: Path, *args: str, stdin: str | None = None, exe: str | None = None) -> dict:
    exe = exe or os.environ.get("AGENTVCS", "agentvcs")
    p = subprocess.run([exe, *args, "--json"], cwd=store, input=stdin, capture_output=True, text=True)
    out = json.loads(p.stdout)
    if not out.get("ok", True) and p.returncode:
        raise RuntimeError(f"agentvcs {' '.join(args)}: {out.get('error')}")
    return out


def step(metric: str, value: float, agent: str = "director") -> str:
    t = "2026-10-14T00:00:00Z"
    return json.dumps({"agent_id": agent, "inputs": [], "outputs": [], "started_at": t, "ended_at": t,
                       "tokens": {"in": 0, "out": 0}, "latency_ms": 0, "metrics": {metric: value}})


GATE = Path(__file__).resolve().parent / "learn_gate.py"


class LiveStore:
    """The live memory's history: an agentvcs store with one run, `director-live`, whose active manifest is the
    memory the director plans with. Each accepted edit is a patch proposed from it, gated (`learn_gate.py`: the
    candidate lints clean, and a person accepted it) and applied."""

    def __init__(self, root: Path, memory: Path, exe: str | None = None):
        import sys
        self.root, self.exe = Path(root), exe
        self.python = sys.executable
        state = self.root / "head.json"
        if state.exists():
            self.head = json.loads(state.read_text())["head"]
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._avcs("init")
        self.head = self._snapshot(memory, "director memory (live)")
        self._avcs("run", "start", "--manifest", self.head, "--run-id", "director-live")
        state.write_text(json.dumps({"head": self.head}))

    def _avcs(self, *args: str, stdin: str | None = None) -> dict:
        return avcs(self.root, *args, stdin=stdin, exe=self.exe)

    def _snapshot(self, memory: Path, name: str) -> str:
        f = self.root / "candidate.json"
        f.write_text(json.dumps(manifest(memory, name)))
        return self._avcs("snapshot", str(f))["manifest_id"]

    def commit(self, candidate: Path, rationale: str, applied: list[str], guard: dict | None = None) -> dict:
        before = json.loads((self.root / ".agentvcs" / "manifests" / (self.head.split(":", 1)[1] + ".json")).read_text())["dimensions"]
        new = self._snapshot(candidate, "director memory (live)")
        pid = self._avcs("patch", "propose", "director-live", "--from", self.head, "--to", new,
                         "--rationale", f"{rationale} — edits: {'; '.join(applied)}",
                         "--author", "human:director-ui")["patch_id"]
        suite = self.root / "accept-suite.yaml"
        # absolute paths: the gate runs its command in the store's own working directory
        # with a D0 guard result, the gate also requires it: the command re-reads it from the candidate's guard.json
        gfile = Path(candidate).parent / "guard.json"
        cmd = f"{self.python} {GATE}" + (f" {gfile}" if guard is not None else "")
        extra = f"  guard: {{op: \">=\", value: {guard['min']}}}\n" if guard is not None else ""
        suite.write_text(f"command: |-\n  {cmd}\nthresholds:\n"
                         "  lint_problems: {op: \"<=\", value: 0}\n  human_accepted: {op: \">=\", value: 1}\n" + extra)
        gate = self._avcs("gate", "run", pid, "--suite", str(suite))["gate_result"]
        if not gate["passed"]:
            raise ValueError(f"the accept gate did not pass: {gate['metrics']}")
        self._avcs("patch", "apply", pid)
        after = manifest(candidate, "")["dimensions"]
        changed = sorted(k for k in after if after[k]["content"] != before.get(k, {}).get("content"))
        self.head = new
        (self.root / "head.json").write_text(json.dumps({"head": self.head}))
        return {"patch_id": pid, "manifest_id": new, "dimensions": changed, "gate": gate["metrics"]}
