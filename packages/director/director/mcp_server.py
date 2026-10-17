"""The plugin's MCP server: the director's learning half as tools the session calls on its own memory. Standard library
only; newline-delimited JSON-RPC over stdio.

    memory_lint                       the live memory's lint problems (ids unique, links resolve, all reachable)
    memory_propose  edits, probes,    the edits applied to a candidate copy (never the live memory), linted, diffed;
                    rationale         returns a proposal id
    memory_check    id                starts D7's positive probe in the background (paid: the probes re-planned on the
                                      current memory and the candidate, blind judge); `memory_status` reads it
    memory_status   id                the proposal, and the check's verdict once it lands
    memory_accept   id                lint clean, a person's yes, and — if a check ran — TEACHES; then the candidate
                                      becomes the live memory, versioned as a gated agentvcs patch when agentvcs is
                                      installed (`AGENTVCS` or `agentvcs` on PATH)
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

The memory is found as the SessionStart hook finds it (`plugin_hook.find_memory`); the director's state (candidates,
checks, the agentvcs store) lives beside it in `.director/`.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director.learn import apply_edits, leak, lint, new_problems  # noqa: E402
from director.plugin_hook import find_memory  # noqa: E402
from director import recipes as rcp  # noqa: E402
from director import lanes as lns  # noqa: E402

SERVER_INFO = {"name": "director", "version": "0.1.0"}
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
    m = memory()
    return m.parent if m.parent.name == ".director" else m.parent / ".director"


def proposal_dir(pid: str) -> Path:
    if not pid or not pid.replace("-", "").isalnum():
        raise ToolError(f"bad proposal id {pid!r}")
    d = state() / "candidates" / pid
    if not (d / "proposal.json").is_file():
        raise ToolError(f"no proposal {pid}")
    return d


def diff(before: Path, after: Path, limit: int = 12000) -> str:
    out = []
    for p in sorted({x.name for x in before.glob("*.md")} | {x.name for x in after.glob("*.md")}):
        a = (before / p).read_text().splitlines() if (before / p).exists() else []
        b = (after / p).read_text().splitlines() if (after / p).exists() else []
        out += difflib.unified_diff(a, b, f"a/{p}", f"b/{p}", lineterm="", n=1)
    text = "\n".join(out)
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


# ------------------------------------------------------------------ tools

def tool_lint(_: dict) -> dict:
    m = memory()
    return {"memory": str(m), "problems": lint(m)}


def tool_propose(args: dict) -> dict:
    m, edits = memory(), args.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ToolError("edits: a non-empty list of edits (see the tool's schema)")
    probes = [p for p in args.get("probes") or [] if isinstance(p, dict) and p.get("situation") and p.get("checklist")][:2]
    pid = time.strftime("p%Y%m%d-%H%M%S")
    d = state() / "candidates" / pid
    res = apply_edits(m, edits, d / "memory")
    problems = new_problems(m, d / "memory")
    episode = args.get("episode") or ""
    rec = {"id": pid, "rationale": args.get("rationale") or "", "edits": edits, "applied": res["applied"],
           "rejected": res["rejected"], "lint": problems, "probes": probes,
           "leaks": [i for i, p in enumerate(probes, 1) if episode and leak(episode, p["situation"])],
           "status": "proposed", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "proposal.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return {**{k: rec[k] for k in ("id", "applied", "rejected", "lint", "leaks")}, "probes": len(probes),
            "diff": diff(m, d / "memory"),
            "next": ("cannot be accepted: fix the lint problems or the rejected edits and propose again"
                     if problems or not res["applied"] else
                     ("the rejected edits were dropped (Index.md is edited only through an edit's `situations`); "
                      if res["rejected"] else "")
                     + "show the diff to the user; memory_check to test it (paid), memory_accept on their yes")}


def tool_check(args: dict) -> dict:
    d = proposal_dir(args.get("id"))
    rec = json.loads((d / "proposal.json").read_text())
    if not rec["probes"]:
        raise ToolError("this proposal has no probes to check; propose again with two probes")
    if (d / "check.running").exists():
        return {"id": rec["id"], "check": "already running"}
    (d / "check.json").unlink(missing_ok=True)       # a rerun replaces an earlier (e.g. incomplete) verdict
    (d / "check.running").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log = (d / "check.log").open("w")
    subprocess.Popen([sys.executable, "-m", "director.plugin_check", str(memory()), str(d)],
                     cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return {"id": rec["id"], "check": "started",
            "note": f"{len(rec['probes'])} probe(s) × 2 samples × 2 memories, Sonnet, blind judge: minutes, a few "
                    "dollars. memory_status reads the verdict."}


def tool_status(args: dict) -> dict:
    d = proposal_dir(args.get("id"))
    rec = json.loads((d / "proposal.json").read_text())
    out = {k: rec[k] for k in ("id", "status", "applied", "rejected", "lint", "rationale")}
    if (d / "check.json").exists():
        out["check"] = json.loads((d / "check.json").read_text())
    elif (d / "check.running").exists():
        tail = (d / "check.log").read_text().splitlines()[-3:] if (d / "check.log").exists() else []
        out["check"] = {"verdict": "RUNNING", "progress": tail}
    return out


def tool_accept(args: dict) -> dict:
    d = proposal_dir(args.get("id"))
    rec = json.loads((d / "proposal.json").read_text())
    if rec["status"] != "proposed":
        raise ToolError(f"proposal {rec['id']} is {rec['status']}")
    if rec["lint"] or not rec["applied"]:
        raise ToolError("this proposal has lint problems or no edit applied; it cannot be accepted")
    if args.get("user_said_yes") is not True:
        raise ToolError("accept only on the user's explicit yes to this diff (user_said_yes: true)")
    check = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None
    if (d / "check.running").exists() and check is None:
        raise ToolError("its check is still running; wait for memory_status to show the verdict")
    if check is not None and check.get("verdict") == "INCOMPLETE":
        raise ToolError("its check could not measure the probes (nothing was decided); memory_check again")
    if check is not None and not check.get("teaches"):
        rec["status"] = check.get("verdict", "not teaching").lower().replace(" ", "_")
        (d / "proposal.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        raise ToolError(f"not kept — the check says {check.get('verdict')}; the memory is unchanged")
    m, patch = memory(), None
    exe = os.environ.get("AGENTVCS") or shutil.which("agentvcs")
    if exe:
        from director.memory_vcs import LiveStore
        guard = None
        if check is not None:
            guard = {"mode": "probe", "passed": True, "probe": check}
            (d / "guard.json").write_text(json.dumps(guard))
        patch = LiveStore(state() / "vcs", m, exe).commit(d / "memory", rec["rationale"] or "accepted in session",
                                                          rec["applied"], guard)
    backup = state() / "previous" / rec["id"]
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(m, backup)
    shutil.rmtree(m)
    shutil.copytree(d / "memory", m)
    rec.update(status="accepted", checked=check is not None, patch=patch)
    (d / "proposal.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return {"id": rec["id"], "status": "accepted", "checked": check is not None,
            "versioned": patch is not None, "patch": patch, "previous_memory": str(backup)}


def tool_recipes_lint(_: dict) -> dict:
    m = memory()
    try:
        rs = rcp.load(m)
    except FileNotFoundError as e:
        raise ToolError(str(e)) from None
    known = rcp.statement_ids(m)
    return {"recipes": len(rs), "problems": rcp.lint(rs, known), "statements_cited": len(rcp.cited(rs) & known),
            "statements": len(known), "uncited": sorted(known - rcp.cited(rs))[:50]}


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
    return {**out, "model": who.model, "calls": who.calls, "failures": who.failures,
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
                                                    "situations": {"type": "array", "items": {"type": "string"}}}}},
                           "probes": {"type": "array", "items": {"type": "object", "properties": {
                               "situation": {"type": "string"},
                               "checklist": {"type": "array", "items": {"type": "string"}}}}},
                           "rationale": {"type": "string"},
                           "episode": {"type": "string", "description": "the task report, to flag probes that copy it"}}}),
    "memory_check": (tool_check, "Start the positive probe for a proposal in the background (paid, minutes): its "
                     "probes re-planned on the current memory and the candidate, judged blind.",
                     {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "memory_status": (tool_status, "A proposal's status and, once it lands, its check's verdict.",
                      {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}),
    "memory_accept": (tool_accept, "Make a proposal the live memory. Only after the user said yes to its diff; refused "
                      "if it lints dirty or its check did not TEACH.",
                      {"type": "object", "required": ["id", "user_said_yes"],
                       "properties": {"id": {"type": "string"}, "user_said_yes": {"type": "boolean"}}}),
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
