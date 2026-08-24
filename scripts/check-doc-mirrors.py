#!/usr/bin/env python3
"""Every document has a Spanish mirror — the front page says so, so check it.

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

Exit 0 when every document is mirrored or named, 1 otherwise.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EN = ROOT / "doc"
ES = ROOT / "doc" / "es"

#: filename -> why it has no mirror.
ENGLISH_ONLY = {
    "PLAN.md": "a working document rewritten weekly; a mirror would be stale on arrival",
}


def main() -> int:
    missing: list[str] = []
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
    if extra:
        if missing:
            print()
        print("ORPHAN — doc/es/ has a document doc/ does not:")
        for x in extra:
            print(f"  doc/es/{x}")
        print()
        print("  English is canonical, so a Spanish document with no English")
        print("  original is a document nothing can be checked against.")
    if stale:
        if missing or extra:
            print()
        print("STALE EXCLUSION — this script's own list is out of date:")
        for x in stale:
            print(f"  {x}")

    bad = len(missing) + len(extra) + len(stale)
    if bad:
        print(f"\n{bad} document(s) break the mirror rule.")
        return 1

    named = ", ".join(sorted(ENGLISH_ONLY)) or "none"
    print(f"doc mirrors: {len(english) - len(ENGLISH_ONLY)} mirrored, English-only by decision: {named}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
