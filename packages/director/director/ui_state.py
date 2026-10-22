"""The plugin's band and status line read the director's state through this script, and their buttons act through it.
Standard library only. It prints one JSON object on stdout, always, and exits 0 (an `error` field says what failed).

    status                  the memory (statement count, lessons accepted), the proposals waiting for a decision (each
                            with its check's verdict and whether it can be accepted), and the delegations not yet
                            brought back
    diff     ID             a proposal's diff against the live memory
    accept   ID             the person pressed Accept on the band after seeing the proposal: their explicit yes,
                            through the same gate as `memory_accept` (lint clean, and TEACHES if a check ran)
    reject   ID             the proposal is marked rejected; the memory is unchanged
    refresh  ID             a delegation's status, asked of its lane
    fetch    ID             a delegation's output, brought back beside the memory, and the delegation marked fetched
    plan-run    PLAN        the person pressed Run: each task delegated to its lane
    plan-cancel PLAN        nothing is delegated
    plan-move   PLAN TASK   the task to the next lane allowed for it; logged in lane-moves.jsonl

The memory is found as the SessionStart hook and the MCP server find it (`$CLAUDE_PROJECT_DIR`, else the cwd).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director import mcp_server as srv  # noqa: E402
from director import lanes as lns  # noqa: E402
from director import plans  # noqa: E402
from director import recipes as rcp  # noqa: E402
from director.memory import gate, journal  # noqa: E402
from director.memory.format import find as find_memory, statement_ids  # noqa: E402

DONE = ("done", "completed", "finished", "succeeded")


def status() -> dict:
    mem = find_memory(srv.project())
    if mem is None:
        return {"memory": None}
    st = srv.state()
    waiting, accepted = gate.waiting(mem)
    out = [{"id": d.get("id"), "lane": d.get("lane"), "status": d.get("status"),
            "done": str(d.get("status", "")).lower() in DONE}
           for d in lns.listing(st) if d.get("lane") != lns.FRONTIER and not d.get("fetched")]
    cfg = lns.load_config()
    plans_out = [{"id": p["id"], "request": p.get("request", ""),
                  "tasks": [{"id": t["id"], "task": t["task"][:160], "lane": t["lane"], "confidential": t["confidential"],
                             "movable": len(plans.allowed(cfg, t["confidential"])) > 1} for t in p["tasks"]]}
                 for p in plans.waiting(st)]
    fresh = rcp.freshness(mem) if rcp.path_for(mem).is_file() else None
    return {"memory": str(mem), "statements": len(statement_ids(mem)), "accepted": accepted,
            "proposals": waiting, "delegations": out, "plans": plans_out,
            "recipes_stale": bool(fresh and fresh["stale"]), "to_reflect": journal.owed(st),
            "recent": recent(st)}


def recent(st, n: int = 6) -> list[dict]:
    """The journal's latest events, newest last, each as {t, kind, what}: what the band shows as the director's steps
    when they did not come through this session's own tool calls (a button on the band, another session, a lane)."""
    out = []
    for e in journal.read(st)[-n:]:
        what = e.get("text") or e.get("rationale") or e.get("id") or e.get("note") or ""
        if e.get("lane"):
            what = f"{e['lane']}: {what}" + (f" ({e['status']})" if e.get("status") else "")
        out.append({"t": e.get("t", ""), "kind": e.get("kind", ""), "what": str(what)[:140]})
    return out


def fetch(task_id: str) -> dict:
    res = srv.tool_lane_fetch({"id": task_id})
    p = lns._record(srv.state(), task_id)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["fetched"] = res.get("dest") or str(srv.state() / "delegations" / task_id)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


ACTIONS = {
    "diff": lambda i: {"id": i, "diff": gate.diff(srv.memory(), gate.proposal_dir(srv.memory(), i) / "memory")},
    "accept": lambda i: gate.accept(srv.memory(), i, True),
    "reject": lambda i: gate.reject(srv.memory(), i),
    "refresh": lambda i: srv.tool_lane_status({"id": i}),
    "fetch": fetch,
    "plan-run": lambda i: plans.run(srv.state(), lns.load_config(), i),
    "plan-cancel": lambda i: plans.cancel(srv.state(), i),
}


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["status"]:
            out = status()
        elif len(argv) == 2 and argv[0] in ACTIONS:
            out = ACTIONS[argv[0]](argv[1])
        elif len(argv) == 3 and argv[0] == "plan-move":
            out = plans.move(srv.state(), lns.load_config(), argv[1], argv[2])
        else:
            out = {"error": "usage: ui_state.py status | diff|accept|reject|refresh|fetch ID | plan-run|plan-cancel PLAN"
                            " | plan-move PLAN TASK"}
    except (srv.ToolError, gate.GateError, lns.LaneError, plans.PlanError, OSError, ValueError, KeyError) as e:
        out = {"error": str(e)}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
