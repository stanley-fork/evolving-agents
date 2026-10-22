"""The director with no person present: a benchmark run, a nightly job. Every command works on a *store*, one
directory that holds everything the director keeps (`<store>/.director/memory`, its journal and proposals), so a
caller that keeps only that directory between sessions keeps the whole memory.

    python -m director.headless init --memory STORE
        an empty memory (an Index with no statements), unless one is already there
    python -m director.headless outcome --memory STORE --task ID --passed true|false [--trace FILE]
                                        [--feedback FILE] [--final-feedback FILE]
        a verifier's verdict on one session, with a digest of what the session did (from its stream-json trace)
        and what CI or review told it (`CI ...` lines; a line gone from the final feedback was fixed), written to
        the journal as an `outcome` entry with `source: verifier`
    python -m director.headless consolidate --memory STORE --policy verified|all [--check]
        the reflection without a person: Claude Code reads the sessions since the last reflection and proposes
        lessons through the gate; each new proposal is then accepted by the policy, not by a person
          verified  what the verifier vouches for: passed sessions whole; of failed ones only CI's and review's own
                    words and which lines the fix made go away (the model never sees a failed session's steps)
          all       every session reaches it (the ablation: no verifier gating)
        --check runs the positive probe on each proposal first and keeps only what TEACHES (its cost is not
        in the usage line: the probe calls are its own)

Each command prints one JSON line last; `consolidate`'s carries `input_tokens`, `output_tokens` and `cost_usd` of
the reflection, so a benchmark can count what learning cost.

The person's gate is not weakened: `memory_accept` still needs the user's yes. Only this module calls
`gate.accept_by_policy`, and the journal records `decided_by: policy:<name>` for every lesson it lets in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .memory import gate, journal
from .memory.format import state_dir

PLUGIN_DIR = Path(__file__).resolve().parents[1]
TOOLS = ["Read", "Grep", "Glob"] + [f"mcp__plugin_director_director__{t}" for t in
                                     ("memory_lint", "memory_propose", "memory_status")]
DIGEST_CHARS = 6000
CI_LINE = re.compile(r"^CI(?::| [^\s:]+:)")         # "CI LK-R01: …", "CI tools/check: …", "CI: …" — not a header


def memory_of(store: Path) -> Path:
    return Path(store) / ".director" / "memory"


def init(store: Path) -> dict:
    m = memory_of(store)
    created = not (m / "Index.md").is_file()
    if created:
        m.mkdir(parents=True, exist_ok=True)
        (m / "Index.md").write_text("# Index\nSituations → the statements a walk starts from.\n")
    return {"memory": str(m), "created": created}


# ------------------------------------------------------------------ a session's trace, in a few lines

def _short(v, n: int = 200) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def digest(trace: Path) -> dict:
    """What a session did, from Claude Code's stream-json output: its tool calls in order (a command, a file), the
    tool errors it met, its final words, its turns. Bounded, so a long session does not flood the reflection."""
    steps, errors, final, turns = [], [], "", None
    names: dict[str, str] = {}
    for line in Path(trace).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        msg = e.get("message") or {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        if e.get("type") == "assistant":
            for c in content:
                if c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp
                    names[c.get("id", "")] = c.get("name", "")
                    steps.append(f"{c.get('name')}: {_short(what, 160)}")
        elif e.get("type") == "user":
            for c in content:
                if c.get("type") == "tool_result" and c.get("is_error"):
                    body = c.get("content")
                    if isinstance(body, list):
                        body = " ".join(x.get("text", "") for x in body if isinstance(x, dict))
                    errors.append(f"{names.get(c.get('tool_use_id', ''), 'tool')}: {_short(body or '', 240)}")
        elif e.get("type") == "result":
            final, turns = str(e.get("result") or ""), e.get("num_turns")
    text = ("steps:\n" + "\n".join(f"- {s}" for s in steps[-40:]) + "\nerrors:\n"
            + ("\n".join(f"- {x}" for x in errors[-15:]) or "- none") + f"\nfinal: {_short(final, 600)}")
    return {"turns": turns, "steps": len(steps), "errors": len(errors), "text": text[-DIGEST_CHARS:]}


def ci_lines(path: Path | None) -> list[str]:
    """The CI / review lines of a feedback file (`CI ...`), in order, without repeats."""
    if not path or not Path(path).is_file():
        return []
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if CI_LINE.match(line) and line not in out:
            out.append(journal.scrub(line)[:400])
    return out


def outcome(store: Path, task: str, passed: bool, trace: Path | None = None, feedback: Path | None = None,
            final_feedback: Path | None = None) -> dict:
    """`feedback`: what CI or a reviewer told the session (the review round); `final_feedback`: what CI says about
    the final state when it still fails. A first line absent from the final ones was fixed, and the verifier says so."""
    d = digest(trace) if trace and Path(trace).is_file() else {"turns": None, "steps": 0, "errors": 0, "text": ""}
    first, final = ci_lines(feedback), ci_lines(final_feedback)
    resolved = [x for x in first if x not in final]
    st = state_dir(memory_of(store))
    journal.append(st, "outcome", source="verifier", task=task, passed=passed, turns=d["turns"], steps=d["steps"],
                   errors=d["errors"], text=journal.scrub(d["text"]), ci=first, ci_final=final, ci_resolved=resolved)
    return {"task": task, "passed": passed, "turns": d["turns"], "errors": d["errors"], "ci": len(first),
            "ci_resolved": len(resolved)}


# ------------------------------------------------------------------ the reflection, with no person

def sessions(store: Path, policy: str) -> list[dict]:
    """The verifier's outcomes since the last reflection, as this policy lets the reflection see them.

    verified  what the verifier vouches for: a passed session whole; of a failed one, only what CI and review said
              (their own words) and which of those lines the session's fix made go away — never the failed
              session's own steps or conclusions, which nothing has checked
    all       every session whole (the ablation)
    """
    out = []
    for e in journal.since_reflection(state_dir(memory_of(store))):
        if e.get("kind") != "outcome" or e.get("source") != "verifier":
            continue
        whole = policy == "all" or e.get("passed") is True
        if whole or e.get("ci") or e.get("ci_final"):
            out.append({**e, "show_steps": whole})
    return out


def prompt(found: list[dict], check: bool) -> str:
    parts = []
    for e in found:
        verdict = "PASSED" if e.get("passed") else "FAILED"
        body = [f"### session {e.get('task')} — verifier {verdict}, {e.get('turns')} turns"]
        if e.get("ci"):
            fixed = set(e.get("ci_resolved") or [])
            body.append("CI and review feedback during the session (their own words):")
            body += [f"- {x}  [{'fixed: gone after the session addressed it' if x in fixed else 'still failing'}]"
                     for x in e["ci"]]
        if e.get("ci_final"):
            body.append("CI on the final state (still failing):")
            body += [f"- {x}" for x in e["ci_final"] if x not in (e.get("ci") or [])]
        body.append(e.get("text", "") if e.get("show_steps", True) else
                    "(the session's own steps are not shown: it failed, so nothing has checked them)")
        parts.append("\n".join(body))
    return f"""You are the director's reflection, running with no person present. Coding sessions on one repository
have ended; below is what each did and what an automatic verifier said. Turn what a future session on this
repository would otherwise have to rediscover into lessons in the director's memory (the directory the director's
context names, with `Index.md`).

What counts as a lesson: a **method** — the way this kind of task gets done in this repository, which the sessions
reached only after attempts that failed (a command that turned out to be the wrong one, a convention a check
enforced, a file that had to change too). A rule that CI or a reviewer stated is a lesson even when the session
failed: their words are the evidence, and a line marked fixed shows what satisfied it. Not the values of one task
(its names, numbers, paths it created).
Check each against the memory first (Read/Grep it); drop what it already holds; modify a statement that covers it
rather than adding a duplicate.

For each lesson: **When** is the kind of task, phrased as a situation an agent recognises mid-task; **Do** its steps
in order, each concrete and checkable, with the check that showed the step had gone wrong; **Because** the attempts
that failed, as evidence. Link it to the statements it applies with; name the Index situations that should reach it
(a new page needs `about`). Write two probes per proposal: a short situation in the second person, set apart from
these sessions (another name, another file), with one to three checklist items naming an action the plan must take.

Call `memory_propose` (edits, probes, a one-line rationale, the sessions as `episode`); fix and re-propose if it
reports lint problems or rejected edits. At most three proposals. Do not accept, reject or ask anything: what to
keep is decided after you finish{', by the positive probe' if check else ''}. If nothing is worth keeping, say so
and stop.

## The sessions
{chr(10).join(parts)}
"""


def _usage(stdout: str) -> dict:
    """Tokens and cost from `claude -p --output-format json` (its last JSON line)."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        u = r.get("usage") or {}
        # Counted the way director-bench counts a session: input plus cache writes; cache reads apart.
        return {"input_tokens": int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0),
                "cache_read_tokens": int(u.get("cache_read_input_tokens") or 0),
                "output_tokens": int(u.get("output_tokens") or 0), "cost_usd": float(r.get("total_cost_usd") or 0),
                "turns": r.get("num_turns"), "error": bool(r.get("is_error"))}
    return {"input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": None,
            "error": True}


def _check(memory: Path, pid: str) -> dict:
    d = gate.proposal_dir(memory, pid)
    gate.start_check(memory, pid, [sys.executable, "-m", "director.plugin_check"], cwd=PLUGIN_DIR)
    while (d / "check.running").exists():
        time.sleep(5)
    return gate.status(memory, pid).get("check") or {}


def consolidate(store: Path, policy: str, check: bool = False, claude: str = "claude", max_turns: int = 40,
                timeout: int = 1800) -> dict:
    if policy not in gate.POLICIES:
        raise SystemExit(f"--policy: one of {', '.join(gate.POLICIES)}")
    m = memory_of(store)
    st = state_dir(m)
    found = sessions(store, policy)
    usage = {"input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": 0,
             "error": False}
    out = {"policy": policy, "sessions": len(found), "accepted": [], "refused": []}
    if found:
        before = {p["id"] for p in gate.waiting(m)[0]}
        env = {**os.environ, "DIRECTOR_MEMORY": str(m)}
        p = subprocess.run([claude, "-p", prompt(found, check), "--output-format", "json", "--max-turns",
                            str(max_turns), "--plugin-dir", str(PLUGIN_DIR), "--allowedTools", ",".join(TOOLS)],
                           capture_output=True, text=True, env=env, cwd=str(store), timeout=timeout)
        usage = _usage(p.stdout)
        for rec in gate.waiting(m)[0]:
            if rec["id"] in before:
                continue
            try:
                if check:
                    v = _check(m, rec["id"])
                    if not v.get("teaches"):
                        gate.reject(m, rec["id"])
                        out["refused"].append({"id": rec["id"], "why": f"check: {v.get('verdict')}"})
                        continue
                gate.accept_by_policy(m, rec["id"], policy, require_check=check)
                out["accepted"].append(rec["id"])
            except gate.GateError as e:
                out["refused"].append({"id": rec["id"], "why": str(e)})
                try:                                  # closed, so it does not wait for a decision forever
                    gate.reject(m, rec["id"])
                except gate.GateError:
                    pass
    journal.reflected(st, f"headless ({policy}): {len(found)} session(s), {len(out['accepted'])} lesson(s) kept")
    return {**out, **usage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m director.headless", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "outcome", "consolidate"):
        s = sub.add_parser(name)
        s.add_argument("--memory", required=True, type=Path, help="the store: the directory kept between sessions")
        if name == "outcome":
            s.add_argument("--task", required=True)
            s.add_argument("--passed", required=True, choices=["true", "false"])
            s.add_argument("--trace", type=Path)
            s.add_argument("--feedback", type=Path, help="CI / review feedback the session was given")
            s.add_argument("--final-feedback", type=Path, help="CI on the final state, when it still fails")
        if name == "consolidate":
            s.add_argument("--policy", required=True, choices=list(gate.POLICIES))
            s.add_argument("--check", action="store_true")
            s.add_argument("--claude", default=os.environ.get("DIRECTOR_CLAUDE", "claude"))
            s.add_argument("--max-turns", type=int, default=40)
    a = ap.parse_args(argv)
    if a.cmd == "init":
        res = init(a.memory)
    elif a.cmd == "outcome":
        init(a.memory)
        res = outcome(a.memory, a.task, a.passed == "true", a.trace, a.feedback, a.final_feedback)
    else:
        init(a.memory)
        res = consolidate(a.memory, a.policy, a.check, a.claude, a.max_turns)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
