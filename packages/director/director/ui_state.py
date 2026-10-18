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
from director.plugin_hook import find_memory  # noqa: E402
from director.recipes import statement_ids  # noqa: E402

DONE = ("done", "completed", "finished", "succeeded")


def _proposals(state: Path) -> tuple[list[dict], int]:
    waiting, accepted = [], 0
    for f in sorted((state / "candidates").glob("*/proposal.json")):
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
        waiting.append({"id": rec["id"], "rationale": rec.get("rationale", ""),
                        "edits": [f"{e.get('op')} {e.get('page')}#{e.get('id')}" for e in rec.get("edits", [])],
                        "lint": len(rec.get("lint") or []), "check": check, "probes": len(rec.get("probes") or []),
                        "acceptable": not why, "why_not": why})
    return waiting, accepted


def status() -> dict:
    mem = find_memory(srv.project())
    if mem is None:
        return {"memory": None}
    st = srv.state()
    waiting, accepted = _proposals(st)
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


def reject(pid: str) -> dict:
    d = srv.proposal_dir(pid)
    rec = json.loads((d / "proposal.json").read_text())
    if rec["status"] != "proposed":
        raise srv.ToolError(f"proposal {pid} is {rec['status']}")
    rec["status"] = "rejected"
    (d / "proposal.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return {"id": pid, "status": "rejected"}


ACTIONS = {
    "diff": lambda i: {"id": i, "diff": srv.diff(srv.memory(), srv.proposal_dir(i) / "memory")},
    "accept": lambda i: srv.tool_accept({"id": i, "user_said_yes": True}),
    "reject": reject,
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
    except (srv.ToolError, lns.LaneError, OSError, ValueError, KeyError) as e:
        out = {"error": str(e)}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
