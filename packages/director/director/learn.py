"""D2 — the director learns: reflect on a task, propose memory edits, apply them deterministically, lint them.

    reflect(episode, memory_dir)  -> {"edits": [...], "cost": ...}     Claude (frontier), memory read-only
    apply_edits(src, edits, dst)  -> {"applied": [...], "rejected": [...]}   a copy; src is never touched
    lint(memory_dir)              -> list of problems                   ids unique, links resolve, every
                                                                        statement reachable from Index.md

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
import shutil
from pathlib import Path

from .core import stream_claude, Log

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

STMT = re.compile(r"^## #(\w+)\s*$", re.M)
LINK = re.compile(r"\[\[([A-Za-z0-9_]+)#(\w+)\]\]")


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


def leak(episode: str, situation: str, n: int = 8) -> bool:
    """A probe whose situation shares an n-word run with the episode it came from tests recall of the incident, not
    the lesson."""
    w = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    e, s = w(episode), w(situation)
    runs = {tuple(e[i:i + n]) for i in range(len(e) - n + 1)}
    return any(tuple(s[i:i + n]) in runs for i in range(len(s) - n + 1))


# ------------------------------------------------------------------ pages as data

def statements(page_text: str) -> dict[str, str]:
    """id -> the statement's block (from its `## #id` line up to the next one)."""
    out, ms = {}, list(STMT.finditer(page_text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(page_text)
        out[m.group(1)] = page_text[m.start():end].rstrip() + "\n"
    return out


def render(sid: str, e: dict) -> str:
    links = " · ".join(f"[[{l}]]" for l in e.get("links") or [])
    return (f"## #{sid}\n**When:** {e['when'].strip()}\n**Do:** {e['do'].strip()}\n"
            f"**Because:** {e['because'].strip()}\n**Links:** {links}\n")


def field(block: str, name: str) -> str:
    m = re.search(rf"^\*\*{name}:\*\*\s*(.*)$", block, re.M)
    return m.group(1).strip() if m else ""


def apply_edits(src: Path, edits: list[dict], dst: Path) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    applied, rejected = [], []
    for e in edits or []:
        try:
            page, sid, op = e["page"], e["id"], e["op"]
            if not re.fullmatch(r"[A-Za-z0-9_]+", page) or not re.fullmatch(r"\w+", sid) or op not in ("add", "modify"):
                raise ValueError("bad page, id or op")
            p = dst / f"{page}.md"
            text = p.read_text() if p.exists() else f"# {page}\n{(e.get('about') or page.replace('_', ' ')).strip()}\n"
            have = statements(text)
            if op == "add":
                if sid in have:
                    raise ValueError(f"add: {page}#{sid} exists")
                text = text.rstrip() + "\n\n" + render(sid, e)
            else:
                if sid not in have:
                    raise ValueError(f"modify: {page}#{sid} does not exist")
                old = have[sid]
                merged = {k: e.get(k) or field(old, k.capitalize()) for k in ("when", "do", "because")}
                old_links = [f"{a}#{b}" for a, b in LINK.findall(field(old, "Links"))]
                merged["links"] = e["links"] if e.get("links") is not None else old_links
                ms = list(STMT.finditer(text))
                i = next(k for k, m in enumerate(ms) if m.group(1) == sid)
                end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
                sep = "\n" if i + 1 < len(ms) else ""
                text = text[:ms[i].start()] + render(sid, merged) + sep + text[end:]
                if render(sid, merged) not in text:                 # an edit that did not land is an error
                    raise ValueError(f"modify: {page}#{sid} did not land")
            p.write_text(text)
            index_add(dst / "Index.md", f"{page}#{sid}", e.get("situations") or [])
            applied.append(f"{op} {page}#{sid}")
        except (KeyError, ValueError) as ex:
            rejected.append({"edit": f"{e.get('op')} {e.get('page')}#{e.get('id')}", "why": str(ex)})
    return {"applied": applied, "rejected": rejected}


def index_add(index: Path, target: str, situations: list[str]) -> None:
    lines = index.read_text().splitlines()
    for sit in situations:
        sit = sit.strip().rstrip("→").strip()
        for i, l in enumerate(lines):
            if l.startswith("- ") and re.sub(r"\s*→.*$", "", l[2:]).strip().lower() == sit.lower():
                if f"[[{target}]]" not in l:
                    lines[i] = l.rstrip() + f", [[{target}]]"
                break
        else:
            lines.append(f"- {sit} → [[{target}]]")
    index.write_text("\n".join(lines) + "\n")


def lint(memory_dir: Path) -> list[str]:
    """ids unique per page, every [[Page#id]] resolves, every statement reachable from Index.md."""
    memory_dir, problems, ids = Path(memory_dir), [], set()
    pages = [p for p in sorted(memory_dir.glob("*.md")) if p.name not in ("Index.md", "README.md")]
    for p in pages:
        found = STMT.findall(p.read_text())
        for d in {x for x in found if found.count(x) > 1}:
            problems.append(f"duplicate id {p.stem}#{d}")
        ids |= {f"{p.stem}#{x}" for x in found}
    index = (memory_dir / "Index.md").read_text()
    for p in pages + [memory_dir / "Index.md"]:
        for a, b in LINK.findall(p.read_text()):
            if f"{a}#{b}" not in ids:
                problems.append(f"{p.stem}: link to missing [[{a}#{b}]]")
    reachable = {f"{a}#{b}" for a, b in LINK.findall(index)}
    linked = {f"{a}#{b}" for p in pages for a, b in LINK.findall(p.read_text())}
    for sid in sorted(ids - reachable - linked):
        problems.append(f"unreachable {sid}")
    return problems


def new_problems(before: Path, after: Path) -> list[str]:
    """The lint gate: problems the edits introduced (the compiled memory may already carry some)."""
    old = set(lint(before))
    return [x for x in lint(after) if x not in old]
