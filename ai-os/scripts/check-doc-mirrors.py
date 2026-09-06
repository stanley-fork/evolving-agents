#!/usr/bin/env python3
"""Every document has a Spanish mirror, and the mirror has not fallen behind.

    python3 scripts/check-doc-mirrors.py

`README.md` ends with "English is canonical. Every document has a Spanish mirror
in `doc/es/`". On 2026-08-24 that was false: docs 21 and 22 had none, and the
Spanish index quietly linked back to the English files marked *(en inglés)*. Two
honest halves that add up to a claim on the front page nobody had checked.

## Why this is a checker and not a note in a review checklist

The failure mode is not "somebody forgot". It is that writing an English
document and writing its mirror are separated by however long the translation
takes, and nothing in between fails. That is the same shape as the test count,
the gate count and the links — a claim that was true when written and rots on a
schedule nobody watches.

## The exclusions are named, not inferred

A document that is deliberately English-only belongs in `ENGLISH_ONLY` with the
reason beside it, so that "no mirror" is a decision on the record rather than an
absence. `PLAN.md` is the only one: it is a working document that turns over
weekly, and mirroring it would mean translating a file whose whole purpose is to
be rewritten.

## And a mirror that exists is not the same as a mirror that is current

Having the file is the cheap half. On 2026-08-24 five mirrors existed and were
behind: `doc/es/05` was missing the three experiments entirely — 14 sections,
including both **[ran]** results and the falsified claim — while `doc/es/01` still
told Spanish readers that `ai-flows` requires cutting into core, a statement the
English document had retracted on 2026-08-02 with an ADR.

That is worse than an absent mirror. An absent one sends you to the English; a
stale one answers confidently and wrongly.

The proxy is the heading count. It is not a translation check and cannot be one —
nothing here reads Spanish. What it catches is the failure that actually happens:
the English document grows a section and the mirror does not. It cannot see a
paragraph rewritten inside a section that kept its heading, and that limit is
worth knowing rather than papering over.

Exit 0 when every document is mirrored, current in structure, or named, 1
otherwise.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EN = ROOT / "doc"
ES = ROOT / "doc" / "es"

#: filename -> why it has no mirror.
ENGLISH_ONLY = {
    "PLAN.md": "a working document rewritten weekly; a mirror would be stale on arrival",
}

HEADING = re.compile(r"^#{1,6}\s", re.M)


def sections(path: pathlib.Path) -> int:
    return len(HEADING.findall(path.read_text(encoding="utf-8")))


def main() -> int:
    missing: list[str] = []
    behind: list[str] = []
    extra: list[str] = []
    stale: list[str] = []

    english = sorted(p.name for p in EN.glob("*.md"))
    spanish = {p.name for p in ES.glob("*.md")}

    for name in english:
        if name in ENGLISH_ONLY:
            if name in spanish:
                # Named as English-only and mirrored anyway: the note is wrong,
                # not the file. Say so rather than passing quietly.
                stale.append(f"{name} — listed as English-only, but doc/es/{name} exists")
            continue
        if name not in spanish:
            missing.append(name)
            continue
        en, es = sections(EN / name), sections(ES / name)
        if en != es:
            behind.append(f"{name} — doc/ has {en} sections, doc/es/ has {es}")

    for name in sorted(spanish):
        if name not in english:
            extra.append(name)

    if missing:
        print("NO MIRROR — doc/es/ has no counterpart:")
        for x in missing:
            print(f"  doc/{x}")
        print()
        print("  Write the mirror, or add the file to ENGLISH_ONLY in this script")
        print("  with the reason. An unmirrored document with no reason makes")
        print("  README.md's last section false.")
    if behind:
        if missing:
            print()
        print("BEHIND — the mirror exists and has a different number of sections:")
        for x in behind:
            print(f"  {x}")
        print()
        print("  Usually the English document grew a section the mirror never got.")
        print("  This counts headings, so it cannot see a rewritten paragraph under")
        print("  a heading that did not change; a matching count is not a promise")
        print("  that the two say the same thing.")
    if extra:
        if missing or behind:
            print()
        print("ORPHAN — doc/es/ has a document doc/ does not:")
        for x in extra:
            print(f"  doc/es/{x}")
        print()
        print("  English is canonical, so a Spanish document with no English")
        print("  original is a document nothing can be checked against.")
    if stale:
        if missing or behind or extra:
            print()
        print("STALE EXCLUSION — this script's own list is out of date:")
        for x in stale:
            print(f"  {x}")

    bad = len(missing) + len(behind) + len(extra) + len(stale)
    if bad:
        print(f"\n{bad} document(s) break the mirror rule.")
        return 1

    named = ", ".join(sorted(ENGLISH_ONLY)) or "none"
    n = len(english) - len(ENGLISH_ONLY)
    print(f"doc mirrors: {n} mirrored and level by section count; English-only by decision: {named}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
