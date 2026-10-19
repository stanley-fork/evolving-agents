"""A lesson's way into the memory: propose → (check) → accept or reject. Every interface (the MCP tools, the
plugin's band, the web interface's Accept) goes through these functions, so the gate is the same wherever a
person decides.

    propose(memory, edits, probes, rationale, episode)  the edits applied to a candidate copy, linted; a proposal id
    start_check(memory, pid, command)    the positive probe (D7), run in the background by the caller's command;
                                         the gate only starts it and reads its verdict
    status(memory, pid)                  the proposal, and the check's verdict once it lands
    accept(memory, pid, user_said_yes)   a person's yes, a clean lint and — if checked — TEACHES; the candidate
                                         becomes the live memory, versioned when agentvcs is installed
    reject(memory, pid)                  marked rejected; the memory is unchanged
    waiting(memory)                      the proposals that wait for a decision, and how many were accepted

A proposal lives in `<state>/candidates/<id>/`: `proposal.json`, the candidate `memory/`, and the check's files.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .edit import apply_edits, new_problems
from .format import state_dir


class GateError(Exception):
    """A refusal the person should read: the gate said no, and why."""


def leak(episode: str, situation: str, n: int = 8) -> bool:
    """A probe whose situation shares an n-word run with the episode it came from tests recall of the incident, not
    the lesson."""
    w = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    e, s = w(episode), w(situation)
    runs = {tuple(e[i:i + n]) for i in range(len(e) - n + 1)}
    return any(tuple(s[i:i + n]) in runs for i in range(len(s) - n + 1))


def diff(before: Path, after: Path, limit: int = 12000) -> str:
    out = []
    for p in sorted({x.name for x in before.glob("*.md")} | {x.name for x in after.glob("*.md")}):
        a = (before / p).read_text().splitlines() if (before / p).exists() else []
        b = (after / p).read_text().splitlines() if (after / p).exists() else []
        out += difflib.unified_diff(a, b, f"a/{p}", f"b/{p}", lineterm="", n=1)
    text = "\n".join(out)
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def new_proposal(root: Path) -> tuple[str, Path]:
    """A fresh proposal id and its directory, reserved atomically: two proposals in the same second get `-2`, `-3`, …
    rather than one overwriting the other's candidate."""
    root.mkdir(parents=True, exist_ok=True)
    base = time.strftime("p%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        pid = base if n == 1 else f"{base}-{n}"
        try:
            (root / pid).mkdir()
        except FileExistsError:
            continue
        return pid, root / pid
    raise GateError("could not reserve a proposal id")


def proposal_dir(memory: Path, pid: str) -> Path:
    if not pid or not pid.replace("-", "").isalnum():
        raise GateError(f"bad proposal id {pid!r}")
    d = state_dir(memory) / "candidates" / pid
    if not (d / "proposal.json").is_file():
        raise GateError(f"no proposal {pid}")
    return d


def _read(d: Path) -> dict:
    return json.loads((d / "proposal.json").read_text())


def _write(d: Path, rec: dict) -> None:
    (d / "proposal.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))


def _check(d: Path) -> dict | None:
    return json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None


def propose(memory: Path, edits: list[dict], probes: list[dict] | None = None, rationale: str = "",
            episode: str = "") -> dict:
    """The proposal record, plus its diff against the live memory. The live memory is never touched."""
    if not isinstance(edits, list) or not edits:
        raise GateError("edits: a non-empty list of edits (see the tool's schema)")
    probes = [p for p in probes or [] if isinstance(p, dict) and p.get("situation") and p.get("checklist")][:2]
    pid, d = new_proposal(state_dir(memory) / "candidates")
    res = apply_edits(memory, edits, d / "memory")
    rec = {"id": pid, "rationale": rationale or "", "edits": edits, "applied": res["applied"],
           "rejected": res["rejected"], "lint": new_problems(memory, d / "memory"), "probes": probes,
           "leaks": [i for i, p in enumerate(probes, 1) if episode and leak(episode, p["situation"])],
           "status": "proposed", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write(d, rec)
    return {**rec, "diff": diff(memory, d / "memory")}


def start_check(memory: Path, pid: str, command: list[str], cwd: Path | None = None) -> dict:
    """Start the caller's check in the background: `command + [memory, proposal dir]`. It must write `check.json`
    (with `verdict` and `teaches`) and remove `check.running` when it ends."""
    d = proposal_dir(memory, pid)
    rec = _read(d)
    if not rec["probes"]:
        raise GateError("this proposal has no probes to check; propose again with two probes")
    if (d / "check.running").exists():
        return {"id": rec["id"], "check": "already running", "probes": len(rec["probes"])}
    (d / "check.json").unlink(missing_ok=True)       # a rerun replaces an earlier (e.g. incomplete) verdict
    (d / "check.running").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log = (d / "check.log").open("w")
    subprocess.Popen([*command, str(memory), str(d)], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return {"id": rec["id"], "check": "started", "probes": len(rec["probes"])}


def status(memory: Path, pid: str) -> dict:
    d = proposal_dir(memory, pid)
    rec = _read(d)
    out = {k: rec[k] for k in ("id", "status", "applied", "rejected", "lint", "rationale")}
    if (d / "check.json").exists():
        out["check"] = _check(d)
    elif (d / "check.running").exists():
        tail = (d / "check.log").read_text().splitlines()[-3:] if (d / "check.log").exists() else []
        out["check"] = {"verdict": "RUNNING", "progress": tail}
    return out


def accept(memory: Path, pid: str, user_said_yes: bool) -> dict:
    d = proposal_dir(memory, pid)
    rec = _read(d)
    if rec["status"] != "proposed":
        raise GateError(f"proposal {rec['id']} is {rec['status']}")
    if rec["lint"] or not rec["applied"]:
        raise GateError("this proposal has lint problems or no edit applied; it cannot be accepted")
    if user_said_yes is not True:
        raise GateError("accept only on the user's explicit yes to this diff (user_said_yes: true)")
    check = _check(d)
    if (d / "check.running").exists() and check is None:
        raise GateError("its check is still running; wait for memory_status to show the verdict")
    if check is not None and check.get("verdict") == "INCOMPLETE":
        raise GateError("its check could not measure the probes (nothing was decided); memory_check again")
    if check is not None and not check.get("teaches"):
        rec["status"] = check.get("verdict", "not teaching").lower().replace(" ", "_")
        _write(d, rec)
        raise GateError(f"not kept — the check says {check.get('verdict')}; the memory is unchanged")
    state, patch = state_dir(memory), None
    exe = os.environ.get("AGENTVCS") or shutil.which("agentvcs")
    if exe:
        from .vcs import LiveStore
        guard = None
        if check is not None:
            guard = {"mode": "probe", "passed": True, "probe": check}
            (d / "guard.json").write_text(json.dumps(guard))
        patch = LiveStore(state / "vcs", memory, exe).commit(d / "memory", rec["rationale"] or "accepted in session",
                                                             rec["applied"], guard)
    backup = state / "previous" / rec["id"]
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(memory, backup)
    shutil.rmtree(memory)
    shutil.copytree(d / "memory", memory)
    rec.update(status="accepted", checked=check is not None, patch=patch)
    _write(d, rec)
    return {"id": rec["id"], "status": "accepted", "checked": check is not None,
            "versioned": patch is not None, "patch": patch, "previous_memory": str(backup)}


def reject(memory: Path, pid: str) -> dict:
    d = proposal_dir(memory, pid)
    rec = _read(d)
    if rec["status"] != "proposed":
        raise GateError(f"proposal {pid} is {rec['status']}")
    rec["status"] = "rejected"
    _write(d, rec)
    return {"id": pid, "status": "rejected"}


def waiting(memory: Path) -> tuple[list[dict], int]:
    """The proposals still `proposed`, each with its check's verdict and whether Accept would pass now; and the count
    of accepted ones."""
    out, accepted = [], 0
    for f in sorted((state_dir(memory) / "candidates").glob("*/proposal.json")):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        if rec.get("status") == "accepted":
            accepted += 1
        if rec.get("status") != "proposed":
            continue
        d = f.parent
        check = None
        if (d / "check.json").is_file():
            try:
                check = json.loads((d / "check.json").read_text()).get("verdict")
            except ValueError:
                check = "INCOMPLETE"
        elif (d / "check.running").exists():
            check = "RUNNING"
        if rec.get("lint") or not rec.get("applied"):
            why = "lint problems or no edit applied: propose again"
        elif check == "RUNNING":
            why = "its check is still running"
        elif check == "INCOMPLETE":
            why = "its check could not measure the probes: run memory_check again"
        elif check not in (None, "TEACHES"):
            why = f"its check says {check}"
        else:
            why = ""
        out.append({"id": rec["id"], "rationale": rec.get("rationale", ""),
                    "edits": [f"{e.get('op')} {e.get('page')}#{e.get('id')}" for e in rec.get("edits", [])],
                    "lint": len(rec.get("lint") or []), "check": check, "probes": len(rec.get("probes") or []),
                    "acceptable": not why, "why_not": why})
    return out, accepted
