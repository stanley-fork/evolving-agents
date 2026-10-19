"""D2 — the director learns: reflect on a task and draft memory edits with probes.

    reflect(episode, memory_dir)  -> {"edits": [...], "probes": [...], "cost": ...}   Claude (frontier), memory
                                                                                      read-only

Applying, linting and gating the edits is the memory's own work, in `director.memory` (`apply_edits`, `lint`,
`new_problems`, `memory.gate`); the names are re-exported here so older imports keep working.

An edit (JSON):
    {"op": "add" | "modify", "page": "Shell_Pitfalls", "id": "verify_scripted_edit",
     "about": "<one line, only when the page is new>",
     "when": "...", "do": "...", "because": "...", "links": ["Page#id", ...],
     "situations": ["<a situation line of Index.md, existing or new>", ...]}

`modify` replaces the fields it gives of an existing statement; `add` refuses an id that exists. Each
situation is matched to an existing Index line by its text (before the arrow); a match gets the link
appended, a new situation becomes a new line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .core import stream_claude, Log
from .memory.edit import apply_edits, index_add, lint, new_problems  # noqa: F401  (re-exported)
from .memory.format import LINK, STMT, field, render, statements  # noqa: F401  (re-exported)
from .memory.gate import leak  # noqa: F401  (re-exported)

REFLECT = """You keep this lab's operational memory in `./memory/` (pages of atomic statements; the format is in
`memory/README.md`; the entry points are in `memory/Index.md`). A task has just ended. Its report, with the
correction it received, is below.

Decide what the memory must hold so that the next time an agent is in a situation like this — not this exact
one — its plan comes out right. First search the memory for statements that already cover it: if one does,
**modify** it (sharpen its When, Do or Because) rather than adding a duplicate; add a statement only for what
no statement covers. Phrase **When** as a situation an agent recognises mid-task, general enough to recur in
other tools and files; **Do** as a concrete, checkable action; **Because** with the incident as evidence.
Link it to the statements it applies with, and name the Index situations that should reach it (reuse an
existing situation line's text exactly when one fits).

Then write **two probes** that will test whether your edit changes what an agent does: each a short situation — a
task in flight, in the second person ("You are about to…", "The user asks…") — where the lesson should change the
plan, **set apart from this incident** (another file, another tool, another goal; do not reuse its sentences), with
one to three checklist items that each name an **action** the plan must take or refuse (never wording). Do not say in
the situation what the right action is.

## The task report

{episode}

End your reply with one fenced JSON block and nothing after it:

```json
{{"edits": [{{"op": "add" or "modify", "page": "<Page_Name>", "id": "<snake_case>", "about": "<only for a new page>",
             "when": "...", "do": "...", "because": "...", "links": ["Page#id"],
             "situations": ["<situation text>"]}}],
  "probes": [{{"situation": "...", "checklist": ["<an action the plan must take or refuse>", "..."]}}]}}
```"""

def reflect(episode: str, memory_dir: Path, log: Log, model: str = "opus", budget: float = 2.0) -> dict:
    r = stream_claude(REFLECT.format(episode=episode), Path(memory_dir).parent, log, "director", "reflect",
                      model=model, max_turns=30, budget=budget)
    if re.search(r"hit your (session|usage) limit|rate limit", str(r.get("result") or ""), re.I):
        raise RuntimeError(f"reflection stopped by an account limit: {r.get('result')}")
    m = re.findall(r"```json\s*(\{.*?\})\s*```", r.get("result") or "", re.S)
    try:
        block = json.loads(m[-1]) if m else {}
        edits, probes = block.get("edits"), block.get("probes") or []
    except (ValueError, AttributeError):
        edits, probes = None, []
    probes = [p for p in probes if isinstance(p, dict) and p.get("situation") and p.get("checklist")][:2]
    return {"edits": edits, "probes": probes, "cost": r.get("cost"), "turns": r.get("turns"), "reply": r.get("result")}
