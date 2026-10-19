"""Lanes: where a planned task runs. The session itself is the frontier lane; other lanes are commands named in a
private config, so the public code knows nothing about what they run. Standard library only.

Config (`$DIRECTOR_LANES`, default `~/.config/director/lanes.json`), private:

    {"confidential_paths": ["~/clients"],
     "lanes": {"local-exec": {
        "description": "a private executor on the LAN",
        "confidential_ok": true,
        "cwd": "~/somewhere",
        "submit": ["python3", "lane.py", "submit", "{id}", "--request", "{request_file}", "--input={input}"],
        "status": ["python3", "lane.py", "status", "{id}"],
        "fetch":  ["python3", "lane.py", "fetch", "{id}", "{dest}"]}}}

A template element holding `{input}` is repeated once per input file. Each command prints one JSON object on its last
line of output.

**The rule, enforced here and not only learned:** a task marked confidential, or with any input under a
`confidential_paths` folder, may only go to a lane with `"confidential_ok": true`. The frontier lane never has it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .memory import journal

FRONTIER = "frontier"
ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class LaneError(Exception):
    pass


def config_path() -> Path:
    return Path(os.environ.get("DIRECTOR_LANES") or Path.home() / ".config" / "director" / "lanes.json").expanduser()


def load_config(path: Path | None = None) -> dict:
    p = path or config_path()
    if not p.is_file():
        return {"lanes": {}, "confidential_paths": []}
    c = json.loads(p.read_text(encoding="utf-8"))
    c.setdefault("lanes", {})
    c.setdefault("confidential_paths", [])
    return c


def lanes(cfg: dict) -> dict[str, dict]:
    out = {FRONTIER: {"description": "this Claude Code session (frontier model)", "confidential_ok": False}}
    for name, lane in cfg["lanes"].items():
        if name == FRONTIER:
            raise LaneError("'frontier' is the session itself; a configured lane cannot take that name")
        out[name] = {"description": lane.get("description", ""), "confidential_ok": bool(lane.get("confidential_ok"))}
    return out


def is_confidential(flag: bool, inputs: list[str], cfg: dict) -> tuple[bool, str]:
    if flag:
        return True, "marked confidential"
    roots = [Path(p).expanduser().resolve() for p in cfg["confidential_paths"]]
    for i in inputs:
        pi = Path(i).expanduser().resolve()
        for r in roots:
            if pi == r or r in pi.parents:
                return True, f"input {i} is under {r}"
    return False, ""


def check(lane: str, confidential: bool, cfg: dict) -> None:
    known = lanes(cfg)
    if lane not in known:
        raise LaneError(f"no lane {lane!r}; configured: {', '.join(known)}")
    if confidential and not known[lane]["confidential_ok"]:
        raise LaneError(f"refused: a confidential task cannot go to the {lane} lane; "
                        f"use one of: {', '.join(n for n, l in known.items() if l['confidential_ok']) or 'none configured'}")


def _expand(template: list[str], values: dict, inputs: list[str]) -> list[str]:
    out = []
    for part in template:
        if "{input}" in part:
            out += [part.replace("{input}", i) for i in inputs]
        else:
            out.append(part.format(**values))
    return [os.path.expanduser(x) if x.startswith("~") else x for x in out]


def _run(lane_cfg: dict, op: str, values: dict, inputs: list[str] | None = None, timeout: int = 900) -> dict:
    if op not in lane_cfg:
        raise LaneError(f"the lane has no {op!r} command")
    argv = _expand(lane_cfg[op], values, inputs or [])
    cwd = Path(lane_cfg["cwd"]).expanduser() if lane_cfg.get("cwd") else None
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    if p.returncode != 0 or not lines:
        raise LaneError(f"{op} failed (exit {p.returncode}): {(p.stderr or p.stdout).strip()[-400:]}")
    try:
        return json.loads(lines[-1])
    except ValueError:
        raise LaneError(f"{op} did not end with a JSON line: {lines[-1][:200]}") from None


# ------------------------------------------------------------------ delegations (state beside the memory)

def _record(state: Path, task_id: str) -> Path:
    return state / "delegations" / f"{task_id}.json"


def delegate(state: Path, cfg: dict, *, task: str, lane: str, task_id: str, inputs: list[str],
             confidential: bool) -> dict:
    if not ID.fullmatch(task_id or "") or len(task_id) > 60:
        raise LaneError("id must be lowercase letters, digits and hyphens (≤ 60)")
    for i in inputs:
        if not Path(i).expanduser().is_file():
            raise LaneError(f"no input file {i}")
    conf, why = is_confidential(confidential, inputs, cfg)
    check(lane, conf, cfg)
    rec_path = _record(state, task_id)
    if rec_path.exists():
        raise LaneError(f"a delegation {task_id} already exists")
    rec = {"id": task_id, "lane": lane, "task": task, "inputs": inputs, "confidential": conf, "why_confidential": why,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "submitted"}
    if lane == FRONTIER:
        rec["status"] = "in_session"
        rec["note"] = "the frontier lane is this session: do the task here"
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(task)
        try:
            rec["submit"] = _run(cfg["lanes"][lane], "submit", {"id": task_id, "request_file": f.name}, inputs)
        finally:
            os.unlink(f.name)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    journal.append(state, "lane.delegated", id=task_id, lane=lane, confidential=conf, status=rec["status"],
                   task=task[:200])
    return rec


def _load(state: Path, task_id: str) -> dict:
    p = _record(state, task_id)
    if not ID.fullmatch(task_id or "") or not p.is_file():
        raise LaneError(f"no delegation {task_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def status(state: Path, cfg: dict, task_id: str) -> dict:
    rec = _load(state, task_id)
    if rec["lane"] == FRONTIER:
        return rec
    rec["last_status"] = _run(cfg["lanes"][rec["lane"]], "status", {"id": task_id}, timeout=120)
    rec["status"] = rec["last_status"].get("status", rec["status"])
    _record(state, task_id).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    journal.append(state, "lane.status", id=task_id, lane=rec["lane"], status=rec["status"])
    return rec


def fetch(state: Path, cfg: dict, task_id: str, dest: str | None = None) -> dict:
    rec = _load(state, task_id)
    if rec["lane"] == FRONTIER:
        raise LaneError("the frontier lane's output is in this session")
    d = Path(dest).expanduser() if dest else state / "delegations" / task_id
    out = _run(cfg["lanes"][rec["lane"]], "fetch", {"id": task_id, "dest": str(d)}, timeout=1800)
    journal.append(state, "lane.fetched", id=task_id, lane=rec["lane"], dest=str(d))
    return out


def listing(state: Path) -> list[dict]:
    d = state / "delegations"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
