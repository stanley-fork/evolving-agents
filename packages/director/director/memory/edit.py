"""Edits applied to a copy of the memory, deterministically, and the lint they must pass.

    apply_edits(src, edits, dst)  -> {"applied": [...], "rejected": [...]}   a copy; src is never touched
    lint(memory_dir)              -> list of problems                   ids unique, links resolve, every
                                                                        statement reachable from Index.md
    new_problems(before, after)   -> the problems the edits introduced

An edit (JSON):
    {"op": "add" | "modify", "page": "Shell_Pitfalls", "id": "verify_scripted_edit",
     "about": "<one line, only when the page is new>",
     "when": "...", "do": "...", "because": "...", "links": ["Page#id", ...],
     "situations": ["<a situation line of Index.md, existing or new>", ...]}

`modify` replaces the fields it gives of an existing statement; `add` refuses an id that exists. Each
situation is matched to an existing Index line by its text (before the arrow); a match gets the link
appended, a new situation becomes a new line. A situation is the text only: one holding `→` or `[[` (a whole
Index line pasted in) rejects its edit, since it would become a second, broken line (`- X → [[A#b]] → [[A#b]]`).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .format import LINK, NOT_PAGES, STMT, field, render, statements


def situation_key(text: str) -> str:
    """How a situation is matched to an Index line: its text before any arrow, trimmed, case-folded."""
    return re.sub(r"\s*→.*$", "", text).strip().lower()


def bad_situations(edit: dict) -> list[str]:
    """The situations of an edit that are not situation text: not a string, empty, or holding `→` or `[[`."""
    sits = edit.get("situations") or []
    if not isinstance(sits, list):
        return [repr(sits)]
    return [repr(x) for x in sits if not isinstance(x, str) or not x.strip() or "→" in x or "[[" in x]


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
            bad = bad_situations(e)
            if bad:
                raise ValueError(f"situations must be the situation text only, not an Index line (no '→', no '[['): "
                                 f"{', '.join(bad)}; pass e.g. 'about to commit a document' and the link is added")
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
            if l.startswith("- ") and situation_key(l[2:]) == sit.lower():
                if f"[[{target}]]" not in l:
                    lines[i] = l.rstrip() + f", [[{target}]]"
                break
        else:
            lines.append(f"- {sit} → [[{target}]]")
    index.write_text("\n".join(lines) + "\n")


def lint(memory_dir: Path) -> list[str]:
    """ids unique per page, every [[Page#id]] resolves, every statement reachable from Index.md."""
    memory_dir, problems, ids = Path(memory_dir), [], set()
    pages = [p for p in sorted(memory_dir.glob("*.md")) if p.name not in NOT_PAGES]
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
