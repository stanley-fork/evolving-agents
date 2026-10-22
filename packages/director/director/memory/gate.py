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
    stale(memory, rec)                   why a proposal no longer applies as it is: the live memory changed after it
                                         was made (its candidate is a whole copy, so accepting it would undo what came
                                         since)
    rebase(memory, pid)                  a stale proposal re-applied onto the live memory, when nothing it touches
                                         changed since it was made; otherwise the conflicts that keep it stale
    waiting(memory)                      the proposals that wait for a decision, and how many were accepted

A proposal lives in `<state>/candidates/<id>/`: `proposal.json`, the candidate `memory/`, and the check's files.

Rebasing. A proposal records, when it is made, what it touches as it was then (`touched`: each edit's statement block
and each Index line its situations match). When another lesson is accepted after it, the proposal is stale; if every
touched block and line is still the same in the live memory, its edits are re-applied onto the live memory, in place:
same id, same probes and check verdict (they are about the proposal's own statements, which are unchanged), a new
candidate and `base`, relinted, and `rebased_from` (the old base) recorded. If any of them changed, it stays stale and
`conflict` names them: accepting would overwrite another lesson's work, so it must be proposed again. `status` and
`accept` rebase; `waiting` (the band, polled) only says whether a rebase would succeed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import journal
from .edit import apply_edits, new_problems, situation_key
from .format import state_dir, statements


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


def fingerprint(memory: Path) -> str:
    """The live memory's pages, hashed: what a proposal's candidate was copied from."""
    h = hashlib.sha256()
    for f in sorted(Path(memory).glob("*.md")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def stale(memory: Path, rec: dict) -> str:
    """Empty when the proposal still applies to the live memory; otherwise why not. A proposal made before this check
    existed has no `base`: it is stale when a proposal created after it has been accepted."""
    if rec.get("base"):
        return "" if rec["base"] == fingerprint(memory) else \
            "the live memory changed since this proposal was made; accepting it would undo that: propose again"
    later = [f.parent.name for f in (state_dir(memory) / "candidates").glob("*/proposal.json")
             if f.parent.name > rec["id"] and json.loads(f.read_text()).get("status") == "accepted"]
    return f"{min(later)} was accepted after this proposal was made; accepting it would undo that: propose again" \
        if later else ""


def touched(memory: Path, edits: list[dict]) -> dict:
    """What a proposal's edits touch in `memory`, as it is there: `Page#id` -> the statement's block, and
    `Index: <situation>` -> the Index line that situation matches (None where there is none yet)."""
    memory, out = Path(memory), {}
    index = (memory / "Index.md").read_text().splitlines() if (memory / "Index.md").is_file() else []
    for e in edits or []:
        if not isinstance(e, dict):
            continue
        page, sid = e.get("page"), e.get("id")
        if isinstance(page, str) and isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_]+", page):
            p = memory / f"{page}.md"
            out[f"{page}#{sid}"] = statements(p.read_text()).get(sid) if p.is_file() else None
        for sit in e.get("situations") or []:
            if not isinstance(sit, str):
                continue
            key = situation_key(sit)
            out[f"Index: {key}"] = next((l for l in index if l.startswith("- ")
                                         and situation_key(l[2:]) == key), None)
    return out


def _accepted_since(memory: Path, rec: dict) -> list[str]:
    out = []
    for f in (state_dir(memory) / "candidates").glob("*/proposal.json"):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if r.get("status") == "accepted" and r.get("id") != rec["id"] and \
                (r.get("accepted", "") >= rec.get("created", "") if r.get("accepted") else r["id"] > rec["id"]):
            out.append(r["id"])
    return sorted(out)


_STALE = "the live memory changed since this proposal was made"


def _try_rebase(memory: Path, d: Path, rec: dict, dst: Path) -> tuple[list[str], str, dict | None]:
    """Re-apply the proposal's edits onto the live memory into `dst`. Returns (the touched statements and Index lines
    that changed, why it cannot be rebased, the rebased record — None when it cannot)."""
    if "touched" not in rec:            # made before proposals recorded what they touch: the old refusal
        return [], stale(memory, rec), None
    if rec.get("lint") or not rec.get("applied"):
        return [], "lint problems or no edit applied: propose again", None
    if (d / "check.running").exists():
        return [], f"{_STALE} and its check is still running; it is rebased once the verdict lands", None
    now = touched(memory, rec["edits"])
    conflicts = sorted(k for k in rec["touched"] if now.get(k) != rec["touched"][k])
    if conflicts:
        return conflicts, (f"{_STALE} and what it touches changed too ({', '.join(conflicts)}); accepting it would "
                           "undo that: propose again"), None
    res = apply_edits(memory, rec["edits"], dst)
    if res["applied"] != rec["applied"] or [r["edit"] for r in res["rejected"]] != [r["edit"] for r in rec["rejected"]]:
        return [], f"{_STALE} and its edits no longer apply the same way ({res['rejected']}): propose again", None
    problems = new_problems(memory, dst)
    if problems:
        return [], f"{_STALE} and, rebased, it does not lint ({'; '.join(problems)}): propose again", None
    return [], "", {**rec, "lint": problems, "base": fingerprint(memory), "rebased_from": rec["base"],
                "rebased_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "accepted_since": _accepted_since(memory, rec)}


def rebase(memory: Path, pid: str) -> dict:
    """A stale proposal brought up to the live memory when it can be. Returns `{"id", "rebased": bool, ...}`: with
    `rebased_from`/`diff` when it was rebased now, or `conflict`/`stale` when it could not be. A fresh proposal is
    left as it is."""
    d = proposal_dir(memory, pid)
    rec = _read(d)
    if rec["status"] != "proposed" or not stale(memory, rec):
        return {"id": rec["id"], "rebased": False}
    tmp = d / f"memory.rebase-{os.getpid()}"
    try:
        conflicts, why, new = _try_rebase(memory, d, rec, tmp)
        if new is None:
            return {"id": rec["id"], "rebased": False, "stale": why, **({"conflict": conflicts} if conflicts else {})}
        old = d / f"memory.old-{os.getpid()}"
        (d / "memory").rename(old)
        tmp.rename(d / "memory")
        shutil.rmtree(old)
        new["touched"] = touched(memory, rec["edits"])
        _write(d, new)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    journal.append(state_dir(memory), "lesson.rebased", id=rec["id"], onto=new["base"],
                   accepted_since=new["accepted_since"])
    return {"id": rec["id"], "rebased": True, "rebased_from": new["rebased_from"], "rebased_onto": new["base"],
            "accepted_since": new["accepted_since"], "diff": diff(memory, d / "memory")}


def propose(memory: Path, edits: list[dict], probes: list[dict] | None = None, rationale: str = "",
            episode: str = "") -> dict:
    """The proposal record, plus its diff against the live memory. The live memory is never touched."""
    if not isinstance(edits, list) or not edits:
        raise GateError("edits: a non-empty list of edits (see the tool's schema)")
    probes = [p for p in probes or [] if isinstance(p, dict) and p.get("situation") and p.get("checklist")][:2]
    pid, d = new_proposal(state_dir(memory) / "candidates")
    res = apply_edits(memory, edits, d / "memory")
    # a situation that is a whole Index line blocks the proposal, not only its edit: dropping it would leave a lesson
    # that no situation reaches the way the proposer meant
    bad = [f"{r['edit']}: {r['why']}" for r in res["rejected"] if r["why"].startswith("situations must be")]
    rec = {"id": pid, "rationale": rationale or "", "edits": edits, "applied": res["applied"],
           "rejected": res["rejected"], "lint": bad + new_problems(memory, d / "memory"), "probes": probes,
           "leaks": [i for i, p in enumerate(probes, 1) if episode and leak(episode, p["situation"])],
           "status": "proposed", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": fingerprint(memory),
           "touched": touched(memory, edits)}
    _write(d, rec)
    journal.append(state_dir(memory), "lesson.proposed", id=pid, rationale=rec["rationale"][:200],
                   applied=rec["applied"], lint=len(rec["lint"]), probes=len(probes))
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
    journal.append(state_dir(memory), "lesson.check_started", id=rec["id"])
    return {"id": rec["id"], "check": "started", "probes": len(rec["probes"])}


def status(memory: Path, pid: str) -> dict:
    rb = rebase(memory, pid)
    d = proposal_dir(memory, pid)
    rec = _read(d)
    out = {k: rec[k] for k in ("id", "status", "applied", "rejected", "lint", "rationale")}
    if rec.get("rebased_from"):
        out.update(rebased_from=rec["rebased_from"], rebased_onto=rec["base"],
                   accepted_since=rec.get("accepted_since", []))
    if rb.get("rebased"):
        out["diff"] = rb["diff"]
    if rb.get("conflict"):
        out["conflict"] = rb["conflict"]
    if (d / "check.json").exists():
        out["check"] = _check(d)
    elif (d / "check.running").exists():
        tail = (d / "check.log").read_text().splitlines()[-3:] if (d / "check.log").exists() else []
        out["check"] = {"verdict": "RUNNING", "progress": tail}
    if rb.get("stale"):
        out["stale"] = rb["stale"]
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
    rb = rebase(memory, pid)
    if rb.get("stale"):
        raise GateError(rb["stale"])
    rec = _read(d)
    check = _check(d)
    if (d / "check.running").exists() and check is None:
        raise GateError("its check is still running; wait for memory_status to show the verdict")
    if check is not None and check.get("verdict") == "INCOMPLETE":
        raise GateError("its check could not measure the probes (nothing was decided); memory_check again")
    if check is not None and not check.get("teaches"):
        rec["status"] = check.get("verdict", "not teaching").lower().replace(" ", "_")
        _write(d, rec)
        journal.append(state_dir(memory), "lesson.not_kept", id=rec["id"], verdict=check.get("verdict"))
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
    # A memory reached through a link (a workspace's `.director/memory` pointing at a private directory) is replaced
    # at its target; rmtree refuses a link, and replacing the link itself would cut the memory off from its home.
    target = Path(memory).resolve() if Path(memory).is_symlink() else Path(memory)
    shutil.copytree(target, backup)
    shutil.rmtree(target)
    shutil.copytree(d / "memory", target)
    rec.update(status="accepted", checked=check is not None, patch=patch, accepted=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write(d, rec)
    journal.append(state, "lesson.accepted", id=rec["id"], checked=check is not None, versioned=patch is not None,
                   applied=rec["applied"])
    out = {"id": rec["id"], "status": "accepted", "checked": check is not None,
           "versioned": patch is not None, "patch": patch, "previous_memory": str(backup)}
    return {**out, "rebased_from": rec["rebased_from"]} if rec.get("rebased_from") else out


def reject(memory: Path, pid: str) -> dict:
    d = proposal_dir(memory, pid)
    rec = _read(d)
    if rec["status"] != "proposed":
        raise GateError(f"proposal {pid} is {rec['status']}")
    rec["status"] = "rejected"
    _write(d, rec)
    journal.append(state_dir(memory), "lesson.rejected", id=pid)
    return {"id": pid, "status": "rejected"}


def _rebase_would_fail(memory: Path, d: Path, rec: dict) -> str:
    """For the band: empty when accepting would rebase cleanly, else why not (nothing is written)."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        return _try_rebase(memory, d, rec, Path(t) / "memory")[1]


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
        elif stale(memory, rec):
            why = _rebase_would_fail(memory, d, rec)
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
