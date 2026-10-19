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

The memory is found as the SessionStart hook and the MCP server find it (`$CLAUDE_PROJECT_DIR`, else the cwd).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director import mcp_server as srv  # noqa: E402
from director import lanes as lns  # noqa: E402
from director.memory import gate  # noqa: E402
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
    return {"memory": str(mem), "statements": len(statement_ids(mem)), "accepted": accepted,
            "proposals": waiting, "delegations": out}


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
}


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["status"]:
            out = status()
        elif len(argv) == 2 and argv[0] in ACTIONS:
            out = ACTIONS[argv[0]](argv[1])
        else:
            out = {"error": "usage: ui_state.py status | diff|accept|reject|refresh|fetch ID"}
    except (srv.ToolError, gate.GateError, lns.LaneError, OSError, ValueError, KeyError) as e:
        out = {"error": str(e)}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
