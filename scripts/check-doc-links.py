#!/usr/bin/env python3
"""Every internal link in the documents, against the thing it points at.

    python3 scripts/check-doc-links.py

A link that says it points somewhere and points nowhere is the same failure this
repository chases everywhere else — a claim nobody verified — in the one form
that is completely mechanical to catch. Nine were sitting in `doc/` when this was
written, and every one of them had been correct once.

## Why they break, which is the more useful half

Not because anyone deleted a file. **Because a heading's slug is its wording**,
and a heading that grows a clause changes address silently:

    ### P0 · Stop the evidence rotting
    ### P0 · Stop the evidence rotting — **done, and running it is §7**
      ^ the same section, at a different address, and every link to it now 404s

So the repair is not to chase the slugs. It is `<a id="p0"></a>` above the
heading and `](#p0)` in the link, after which the wording is free to move. This
script reports both, and it says which kind each failure is.

Exit 0 when every link resolves, 1 otherwise.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Vendored upstream. Not ours to keep tidy, and large enough to drown the report.
SKIP = ("ai-base/", "node_modules/")

LINK = re.compile(r"\]\(([^)\s]*?)(#[^)\s]*)?\)")
EXPLICIT_ID = re.compile(r'<a\s+id="([^"]+)"\s*>')
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
#: Inline code, which is where a regex or a path fragment can look like a link.
CODE_SPAN = re.compile(r"`[^`\n]*`|```.*?```", re.S)


def slug(heading: str) -> str:
    """GitHub's heading slug, near enough for a check that reports both forms.

    Strips inline markup, drops anything that is not a letter, a digit, a space
    or a hyphen, lowercases, and joins on hyphens. `·` and `—` vanish and leave
    their spaces behind, which is why `A · B` becomes `a--b`.

    ## Letters means letters, not ASCII

    The first version matched `[^a-z0-9 -_]`, which quietly deleted every
    accented character — and then reported sixteen Spanish anchors as broken
    when the links were correct and this function was wrong. A checker whose
    failures are mostly its own is worse than none: it trains you to skim the
    report. `\w` under Python's default Unicode semantics keeps `ó` and `ñ`,
    which is what GitHub's slugger does.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\*\*?([^*]*)\*\*?", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w \-]", "", text, flags=re.UNICODE)
    return text.strip().replace(" ", "-")


def anchors_of(text: str) -> set[str]:
    out = {slug(h) for h in HEADING.findall(text)}
    out |= set(EXPLICIT_ID.findall(text))
    return out


def documents() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(s) or f"/{s}" in rel for s in SKIP):
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    cache: dict[pathlib.Path, str] = {}
    missing_file: list[str] = []
    missing_anchor: list[str] = []
    checked = 0

    for f in documents():
        raw = f.read_text(encoding="utf-8")
        # A regex inside backticks is not a link, and two of them look exactly
        # like one. Blanking code spans first is cheaper than parsing markdown.
        text = CODE_SPAN.sub(lambda m: " " * len(m.group(0)), raw)
        rel = f.relative_to(ROOT).as_posix()

        for m in LINK.finditer(text):
            target, frag = m.group(1), (m.group(2) or "")[1:]
            if target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if not target and not frag:
                continue
            checked += 1

            path = f if not target else (f.parent / target)
            try:
                path = path.resolve()
                exists = path.exists()
            except OSError:
                exists = False
            if not exists:
                missing_file.append(f"{rel}: {target}{'#' + frag if frag else ''}")
                continue
            if not frag or path.suffix != ".md":
                continue

            if path not in cache:
                cache[path] = path.read_text(encoding="utf-8")
            if frag not in anchors_of(cache[path]):
                where = target or "(this file)"
                missing_anchor.append(f"{rel}: {where}#{frag}")

    if missing_file:
        print("MISSING FILE — the path does not exist:")
        for x in missing_file:
            print(f"  {x}")
    if missing_anchor:
        if missing_file:
            print()
        print("MISSING ANCHOR — the file exists and has no such section:")
        for x in missing_anchor:
            print(f"  {x}")
        print()
        print("  A heading's slug is its wording, so a heading that grows a clause")
        print("  changes address. Put <a id=\"…\"></a> above the target heading and")
        print("  link to that, rather than chasing the slug.")

    bad = len(missing_file) + len(missing_anchor)
    if bad:
        print(f"\n{bad} of {checked} internal links do not resolve.")
        return 1
    print(f"doc links: {checked} internal link(s) resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
