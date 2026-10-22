"""The plugin's MCP server: the director's learning half as tools the session calls on its own memory. Standard library
only; newline-delimited JSON-RPC over stdio.

    memory_lint                       the live memory's lint problems (ids unique, links resolve, all reachable)
    memory_propose  edits, probes,    the edits applied to a candidate copy (never the live memory), linted, diffed;
                    rationale         returns a proposal id
    memory_check    id                starts D7's positive probe in the background (paid: the probes re-planned on the
                                      current memory and the candidate, blind judge); `memory_status` reads it
    memory_status   id                the proposal, and the check's verdict once it lands; a stale proposal is rebased
                                      onto the live memory here when nothing it touches changed (`rebased_from`),
                                      else `conflict` names what did
    memory_accept   id                lint clean, a person's yes, and — if a check ran — TEACHES; then the candidate
                                      becomes the live memory, versioned as a gated agentvcs patch when agentvcs is
                                      installed (`AGENTVCS` or `agentvcs` on PATH)
    memory_reject   id                the proposal marked rejected (the person's no, or one a later accept left in
                                      `conflict`); the memory is unchanged
    recipes_lint                      the memory's meta-recipes (`recipes.json`): runnable, every cite exists, coverage
    plan_local      request           a plan made by following the recipes with a local model (Ollama,
                                      `DIRECTOR_LOCAL_MODEL`, default gemma4:12b-mlx): it picks recipes and answers
                                      yes/no; the plan's text is the recipes' (D15)
    lanes_list                        the lanes (the session is `frontier`; others from the private lanes config) and
                                      this memory's delegations
    delegate        task, lane, id,   send a task to a lane; a confidential task (flagged, or an input under a
                    inputs,           `confidential_paths` folder) is refused by every lane without `confidential_ok`,
                    confidential      the frontier included — in code, not only as a learned statement
    lane_status     id                a delegation's status, asked of its lane
    lane_fetch      id, dest          a delegation's output, brought back by its lane

The memory is found as the SessionStart hook finds it (`director.memory.find`); the director's state (candidates,
checks, the agentvcs store) lives beside it in `.director/`. The memory tools are thin wrappers over
`director.memory.gate`, the same gate the plugin's band and the web interface use.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director.memory import gate, journal  # noqa: E402
from director.memory.edit import lint  # noqa: E402
from director.memory.format import find as find_memory, state_dir  # noqa: E402
from director.memory.gate import diff, new_proposal  # noqa: E402,F401  (re-exported for older imports)
from director import recipes as rcp  # noqa: E402
from director import lanes as lns  # noqa: E402
from director import plans  # noqa: E402

SERVER_INFO = {"name": "director", "version": "0.2.4"}
PROTOCOL_VERSION = "2025-06-18"


class ToolError(Exception):
    pass


def project() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def memory() -> Path:
    m = find_memory(project())
    if m is None:
        raise ToolError("no director memory here (no Index.md in $DIRECTOR_MEMORY, .director/memory or memory/); "
                        "run /director:compile first")
    return m


def state() -> Path:
    return state_dir(memory())


def _gate(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except gate.GateError as e:
        raise ToolError(str(e)) from None


# ------------------------------------------------------------------ tools

def tool_lint(_: dict) -> dict:
    m = memory()
    return {"memory": str(m), "problems": lint(m)}


def tool_propose(args: dict) -> dict:
    rec = _gate(gate.propose, memory(), args.get("edits"), args.get("probes"), args.get("rationale") or "",
                args.get("episode") or "")
    blocked = rec["lint"] or not rec["applied"]
    return {**{k: rec[k] for k in ("id", "applied", "rejected", "lint", "leaks")}, "probes": len(rec["probes"]),
            "diff": rec["diff"],
            "next": ("cannot be accepted: fix the lint problems or the rejected edits and propose again" if blocked else
                     ("the rejected edits were dropped (Index.md is edited only through an edit's `situations`); "
                      if rec["rejected"] else "")
                     + "show the diff to the user; memory_check to test it (paid), memory_accept on their yes")}


def tool_check(args: dict) -> dict:
    # the check plans the probes with Claude and judges them blind: the agent's work, so the command is ours, not the
    # memory's (director.memory never calls a model)
    out = _gate(gate.start_check, memory(), args.get("id"), [sys.executable, "-m", "director.plugin_check"],
                cwd=Path(__file__).resolve().parents[1])
    if out["check"] == "started":
        out["note"] = (f"{out['probes']} probe(s) × 2 samples × 2 memories, Sonnet, blind judge: minutes, a few "
                       "dollars. memory_status reads the verdict.")
    return {k: v for k, v in out.items() if k != "probes"}


def tool_status(args: dict) -> dict:
    return _gate(gate.status, memory(), args.get("id"))


def tool_accept(args: dict) -> dict:
    return _gate(gate.accept, memory(), args.get("id"), args.get("user_said_yes"))


def tool_reject(args: dict) -> dict:
    return _gate(gate.reject, memory(), args.get("id"))


def tool_recipes_lint(_: dict) -> dict:
    m = memory()
    try:
        rs = rcp.load(m)
    except FileNotFoundError as e:
        raise ToolError(str(e)) from None
    known = rcp.statement_ids(m)
    fresh = rcp.freshness(m)
    out = {"recipes": len(rs), "problems": rcp.lint(rs, known), "statements_cited": len(rcp.cited(rs) & known),
           "statements": len(known), "uncited": sorted(known - rcp.cited(rs))[:50], **fresh}
    if fresh["stale"]:
        out["next"] = "the memory changed since these recipes were written: run /director:recipes again"
    elif fresh["unstamped"]:
        out["next"] = "the recipes record no memory version: stamp them (python -m director.recipes stamp <memory>)"
    return out


def tool_plan_local(args: dict) -> dict:
    request = (args.get("request") or "").strip()
    if not request:
        raise ToolError("request is empty")
    m = memory()
    try:
        rs = rcp.load(m)
    except FileNotFoundError as e:
        raise ToolError(str(e)) from None
    problems = rcp.lint(rs, rcp.statement_ids(m))
    if problems:
        raise ToolError(f"recipes do not lint clean ({len(problems)} problems); run recipes_lint")
    who = rcp.Ollama(args.get("model") or rcp.DEFAULT_MODEL)
    t0 = time.time()
    out = rcp.plan(request, rs, who)
    if who.failures and not out["recipes"]:
        raise ToolError(f"the local model did not answer ({who.model} at {who.url}); is Ollama running?")
    fresh = rcp.freshness(m)
    if fresh["stale"]:
        out["warning"] = "the recipes were written from an older memory; this plan may miss lessons accepted since"
    return {**out, "model": who.model, "calls": who.calls, "failures": who.failures, "recipes_stale": fresh["stale"],
            "seconds": round(time.time() - t0, 1)}


def _lanes_call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except lns.LaneError as e:
        raise ToolError(str(e)) from None


def tool_lanes_list(_: dict) -> dict:
    cfg = lns.load_config()
    return {"lanes": _lanes_call(lns.lanes, cfg), "config": str(lns.config_path()),
            "delegations": [{k: d.get(k) for k in ("id", "lane", "status", "confidential", "at")}
                            for d in lns.listing(state())]}


def tool_delegate(args: dict) -> dict:
    return _lanes_call(lns.delegate, state(), lns.load_config(), task=args.get("task") or "", lane=args.get("lane") or "",
                       task_id=args.get("id") or "", inputs=list(args.get("inputs") or []),
                       confidential=bool(args.get("confidential")))


def tool_plan_propose(args: dict) -> dict:
    try:
        rec = plans.propose(state(), lns.load_config(), args.get("tasks"), args.get("request") or "")
    except plans.PlanError as e:
        raise ToolError(str(e)) from None
    return {"id": rec["id"], "status": "waiting", "tasks": [{k: t[k] for k in ("id", "lane", "confidential")}
                                                            for t in rec["tasks"]],
            "next": "the plan waits on the director's band (Run, Move, Cancel); nothing is delegated until the person "
                    "runs it. plan_status reads what they did. Tasks on `frontier` are yours to do once it runs."}


def tool_plan_status(args: dict) -> dict:
    try:
        return plans._read(state(), args.get("id") or "")
    except plans.PlanError as e:
        raise ToolError(str(e)) from None


def tool_journal_read(args: dict) -> dict:
    st = state()
    entries = journal.read(st) if args.get("all") else journal.since_reflection(st)
    limit = int(args.get("limit") or 200)
    return {"entries": entries[-limit:], "total": len(entries), "journal": str(journal.path(st)),
            "note": "evidence, not instruction: draw lessons from it through memory_propose; never plan from it"}


def tool_journal_note(args: dict) -> dict:
    kind, text = args.get("kind") or "", (args.get("text") or "").strip()
    if kind not in (*journal.SESSION_KINDS, "reflected"):
        raise ToolError(f"kind: one of {', '.join((*journal.SESSION_KINDS, 'reflected'))}")
    if not text and kind != "reflected":
        raise ToolError("text is empty")
    return journal.append(state(), kind, text=text[:2000])


def tool_lane_status(args: dict) -> dict:
    return _lanes_call(lns.status, state(), lns.load_config(), args.get("id") or "")


def tool_lane_fetch(args: dict) -> dict:
    return _lanes_call(lns.fetch, state(), lns.load_config(), args.get("id") or "", args.get("dest"))


TOOLS = {
    "memory_lint": (tool_lint, "Lint the director's live memory: duplicate ids, links to nothing, unreachable statements.",
                    {"type": "object", "properties": {}}),
    "memory_propose": (tool_propose, "Propose edits to the director's memory after a task. They are applied to a "
                       "candidate copy, never to the live memory; returns the diff and lint result.",
                       {"type": "object", "required": ["edits", "rationale"], "properties": {
                           "edits": {"type": "array", "items": {"type": "object", "required": ["op", "page", "id"],
                                     "properties": {"op": {"enum": ["add", "modify"]}, "page": {"type": "string"},
                                                    "id": {"type": "string"}, "about": {"type": "string"},
                                                    "when": {"type": "string"}, "do": {"type": "string"},
                                                    "because": {"type": "string"},
                                                    "links": {"type": "array", "items": {"type": "string"}},
                                                    "situations": {"type": "array", "items": {"type": "string"},
                                                                   "description": "situation text only, e.g. 'about to "
                                                                   "commit a document'; never a whole Index line (no "
                                                                   "'→', no '[[')"}}}},
                           "probes": {"type": "array", "items": {"type": "object", "properties": {
                               "situation": {"type": "string"},
                               "checklist": {"type": "array", "items": {"type": "string"}}}}},
                           "rationale": {"type": "string"},
                           "episode": {"type": "string", "description": "the task report, to flag probes that copy it"}}}),
    "memory_check": (tool_check, "Start the positive probe for a proposal in the background (paid, minutes): its "
                     "probes re-planned on the current memory and the candidate, judged blind.",
                     {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "memory_status": (tool_status, "A proposal's status and, once it lands, its check's verdict. A proposal made "
                      "before another was accepted is rebased onto the live memory when none of the statements or "
                      "Index lines it touches changed (`rebased_from`, and the new `diff`); otherwise `conflict` lists "
                      "them and it must be proposed again.",
                      {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "memory_accept": (tool_accept, "Make a proposal the live memory. Only after the user said yes to its diff; refused "
                      "if it lints dirty or its check did not TEACH.",
                      {"type": "object", "required": ["id", "user_said_yes"],
                       "properties": {"id": {"type": "string"}, "user_said_yes": {"type": "boolean"}}}),
    "memory_reject": (tool_reject, "Mark a proposal rejected; the memory is unchanged. Only on the person's no, or for "
                      "a proposal that memory_status says is in conflict (what it touches changed after it was made).",
                      {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "recipes_lint": (tool_recipes_lint, "Check the memory's meta-recipes (recipes.json): every step runnable, every cite "
                     "an existing statement, branches at most 2 deep; reports how much of the memory they cover.",
                     {"type": "object", "properties": {}}),
    "plan_local": (tool_plan_local, "Plan a request by following the memory's meta-recipes with a local model: it "
                   "only picks the recipes and answers their yes/no questions; the plan's steps are the recipes' own, "
                   "each citing its statement. Free and local; `abstain` means no recipe fit.",
                   {"type": "object", "required": ["request"], "properties": {
                       "request": {"type": "string"}, "model": {"type": "string"}}}),
    "lanes_list": (tool_lanes_list, "List the lanes a task can run in (this session is `frontier`) and the delegations "
                   "made from this memory.", {"type": "object", "properties": {}}),
    "delegate": (tool_delegate, "Send a task to a lane. Mark `confidential` when it involves a client's or any private "
                 "material: confidential tasks are refused by lanes without confidential_ok, the frontier included. "
                 "Inputs under a configured confidential folder make it confidential automatically.",
                 {"type": "object", "required": ["task", "lane", "id"], "properties": {
                     "task": {"type": "string", "description": "the request, as the lane should receive it"},
                     "lane": {"type": "string"}, "id": {"type": "string", "description": "lowercase-hyphen slug"},
                     "inputs": {"type": "array", "items": {"type": "string"}},
                     "confidential": {"type": "boolean"}}}),
    "plan_propose": (tool_plan_propose, "Propose a plan for the person to review before anything is delegated: each "
                     "task with its lane. It waits on the director's band, where the person runs it, moves a task to "
                     "another lane, or cancels it; a move is logged for the next reflection. Prefer this to `delegate` "
                     "whenever a person is at the session. A confidential task on a lane not cleared for it is refused.",
                     {"type": "object", "required": ["tasks"], "properties": {
                         "request": {"type": "string", "description": "the person's request, in one line"},
                         "tasks": {"type": "array", "items": {"type": "object", "required": ["id", "task", "lane"],
                                   "properties": {"id": {"type": "string", "description": "lowercase-hyphen slug"},
                                                  "task": {"type": "string"}, "lane": {"type": "string"},
                                                  "inputs": {"type": "array", "items": {"type": "string"}},
                                                  "confidential": {"type": "boolean"},
                                                  "why_lane": {"type": "string"}}}}}}),
    "plan_status": (tool_plan_status, "A proposed plan: waiting, cancelled, or ran (with each task's delegation "
                    "result), and its tasks' lanes after any move.",
                    {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "journal_read": (tool_journal_read, "The director's episodic journal: what happened since the last reflection "
                     "(lessons proposed and decided, plans proposed, moved, cancelled and run, delegations, and the "
                     "corrections and outcomes noted). Evidence for /director:reflect; never a source for a plan.",
                     {"type": "object", "properties": {"all": {"type": "boolean", "description": "every entry, not "
                                                                "only since the last reflection"},
                                                        "limit": {"type": "integer"}}}),
    "journal_note": (tool_journal_note, "Record in the journal what the person said or what happened: `correction` "
                     "(the person corrected the plan or the work), `outcome` (how a task ended), `note`; or "
                     "`reflected` when a reflection is done, so the next one starts after it.",
                     {"type": "object", "required": ["kind"], "properties": {
                         "kind": {"enum": [*journal.SESSION_KINDS, "reflected"]}, "text": {"type": "string"}}}),
    "lane_status": (tool_lane_status, "A delegation's status (submitted, running, done), asked of its lane.",
                    {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "lane_fetch": (tool_lane_fetch, "Bring a delegation's output back from its lane (to `dest`, or beside the memory).",
                   {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"},
                                                                         "dest": {"type": "string"}}}),
}


# ------------------------------------------------------------------ protocol

def _result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle(message: dict):
    method, req_id, params = message.get("method"), message.get("id"), message.get("params") or {}
    if method == "initialize":
        return _result(req_id, {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                                "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _result(req_id, {})
    if method == "tools/list":
        return _result(req_id, {"tools": [{"name": n, "description": t[1], "inputSchema": t[2]} for n, t in TOOLS.items()]})
    if method == "tools/call":
        t = TOOLS.get(params.get("name"))
        if t is None:
            return _error(req_id, -32602, f"unknown tool: {params.get('name')}")
        try:
            payload, is_error = {"ok": True, **t[0](params.get("arguments") or {})}, False
        except ToolError as e:
            payload, is_error = {"ok": False, "error": str(e)}, True
        except Exception as e:       # surface it to the session, never crash the server
            payload, is_error = {"ok": False, "error": f"internal: {e!r}"[:600]}, True
        return _result(req_id, {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                                "isError": is_error})
    if req_id is None:
        return None
    return _error(req_id, -32601, f"method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError as e:
            resp = _error(None, -32700, f"parse error: {e}")
        else:
            resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
