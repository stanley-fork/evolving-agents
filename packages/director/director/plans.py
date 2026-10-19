"""Plans that wait for the person before anything is delegated (EVOLUTION step 4; EAT's review of intent).

The session proposes a plan, each task with its lane. The plan waits; the person runs it, moves a task to another
lane, or cancels it, from the band above the prompt (or any interface on these functions). A move is logged in
`lane-moves.jsonl`, as D3's interface logged it, so the next reflection can learn where this person wants work done.
Running a plan delegates each task through `lanes.delegate`, so the confidentiality rule still holds in code.

    propose(state, cfg, tasks, request)   a waiting plan; refused if a confidential task names a lane not cleared for it
    waiting(state)                        the plans that wait, oldest first
    move(state, cfg, pid, task_id, to)    a task to another lane (the next allowed one when `to` is not given)
    cancel(state, pid)                    nothing is delegated
    run(state, cfg, pid)                  each task delegated; one task's failure does not stop the others

A task: {"id": "<lowercase-hyphen slug>", "task": "<the request as the lane should receive it>", "lane": "<lane>",
         "inputs": ["<file>", ...], "confidential": false, "why_lane": "<one line>"}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import lanes as lns
from .memory import journal


class PlanError(Exception):
    pass


def _dir(state: Path) -> Path:
    return Path(state) / "plans"


def _path(state: Path, pid: str) -> Path:
    if not pid or not pid.replace("-", "").isalnum():
        raise PlanError(f"bad plan id {pid!r}")
    p = _dir(state) / f"{pid}.json"
    if not p.is_file():
        raise PlanError(f"no plan {pid}")
    return p


def _read(state: Path, pid: str) -> dict:
    return json.loads(_path(state, pid).read_text(encoding="utf-8"))


def _write(state: Path, rec: dict) -> None:
    (_dir(state) / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def _new_id(state: Path) -> str:
    d = _dir(state)
    d.mkdir(parents=True, exist_ok=True)
    base = time.strftime("plan-%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        pid = base if n == 1 else f"{base}-{n}"
        try:
            (d / f"{pid}.json").open("x").close()       # reserved atomically
        except FileExistsError:
            continue
        return pid
    raise PlanError("could not reserve a plan id")


def allowed(cfg: dict, confidential: bool) -> list[str]:
    """The lanes a task may go to, in the config's order (the session first)."""
    return [n for n, l in lns.lanes(cfg).items() if l["confidential_ok"] or not confidential]


def propose(state: Path, cfg: dict, tasks: list[dict], request: str = "") -> dict:
    if not isinstance(tasks, list) or not tasks:
        raise PlanError("tasks: a non-empty list of tasks")
    seen, out = set(), []
    for t in tasks:
        tid, lane, text = t.get("id") or "", t.get("lane") or "", (t.get("task") or "").strip()
        if not lns.ID.fullmatch(tid) or len(tid) > 60:
            raise PlanError(f"task id {tid!r}: lowercase letters, digits and hyphens (≤ 60)")
        if tid in seen:
            raise PlanError(f"task id {tid} appears twice")
        if not text:
            raise PlanError(f"task {tid} has no text")
        seen.add(tid)
        inputs = [str(i) for i in t.get("inputs") or []]
        conf, why = lns.is_confidential(bool(t.get("confidential")), inputs, cfg)
        try:
            lns.check(lane, conf, cfg)
        except lns.LaneError as e:
            raise PlanError(f"task {tid}: {e}") from None
        out.append({"id": tid, "task": text, "lane": lane, "planned_lane": lane, "inputs": inputs,
                    "confidential": conf, "why_confidential": why, "why_lane": t.get("why_lane") or ""})
    rec = {"id": _new_id(state), "request": request, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "status": "waiting", "tasks": out}
    _write(state, rec)
    journal.append(state, "plan.proposed", id=rec["id"], request=request[:200],
                   tasks=[{"id": t["id"], "lane": t["lane"], "confidential": t["confidential"]} for t in out])
    return rec


def waiting(state: Path) -> list[dict]:
    out = []
    for p in sorted(_dir(state).glob("plan-*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if rec.get("status") == "waiting":
            out.append(rec)
    return out


def _waiting(state: Path, pid: str) -> dict:
    rec = _read(state, pid)
    if rec["status"] != "waiting":
        raise PlanError(f"plan {pid} is not waiting (it is {rec['status']})")
    return rec


def move(state: Path, cfg: dict, pid: str, task_id: str, to: str | None = None) -> dict:
    rec = _waiting(state, pid)
    t = next((x for x in rec["tasks"] if x["id"] == task_id), None)
    if t is None:
        raise PlanError(f"plan {pid} has no task {task_id}")
    ok = allowed(cfg, t["confidential"])
    if to is None:
        others = [n for n in ok if n != t["lane"]]
        if not others:
            raise PlanError(f"no other lane may take task {task_id}"
                            + (" (it is confidential)" if t["confidential"] else ""))
        i = ok.index(t["lane"]) if t["lane"] in ok else -1
        to = ok[(i + 1) % len(ok)] if len(ok) > 1 else others[0]
    try:
        lns.check(to, t["confidential"], cfg)
    except lns.LaneError as e:
        raise PlanError(str(e)) from None
    mv = {"plan": pid, "task": task_id, "title": t["task"][:120], "from": t["lane"], "to": to,
          "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t["lane"] = to
    _write(state, rec)
    with (Path(state) / "lane-moves.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(mv, ensure_ascii=False) + "\n")
    journal.append(state, "plan.moved", **{k: v for k, v in mv.items() if k != "at"})
    return mv


def cancel(state: Path, pid: str) -> dict:
    rec = _waiting(state, pid)
    rec["status"] = "cancelled"
    _write(state, rec)
    journal.append(state, "plan.cancelled", id=pid)
    return {"id": pid, "status": "cancelled"}


def run(state: Path, cfg: dict, pid: str) -> dict:
    rec = _waiting(state, pid)
    rec["status"] = "running"
    _write(state, rec)
    results = []
    for t in rec["tasks"]:
        try:
            d = lns.delegate(state, cfg, task=t["task"], lane=t["lane"], task_id=t["id"], inputs=t["inputs"],
                             confidential=t["confidential"])
            results.append({"id": t["id"], "lane": t["lane"], "status": d.get("status")})
        except (lns.LaneError, OSError, ValueError) as e:
            results.append({"id": t["id"], "lane": t["lane"], "error": str(e)})
    rec.update(status="ran", ran=time.strftime("%Y-%m-%dT%H:%M:%S"), results=results)
    _write(state, rec)
    journal.append(state, "plan.ran", id=pid, results=results)
    return {"id": pid, "status": "ran", "results": results}
