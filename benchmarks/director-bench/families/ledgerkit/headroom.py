#!/usr/bin/env python3
"""Static headroom listing for ledgerkit (no model calls, no money).

For each task: which rules its reference solution must satisfy, split into
  - local rules: enforced by ./tools/check and readable in the repo (an agent that reads the
    checker before writing learns them by reading, as the DB0 live checks showed);
  - remote rules: enforced only by the team's CI / review (verify/oracle.py); nothing in the
    snapshot states them, so they are learned from feedback, i.e. by failing once.

    python headroom.py            # prints the matrix
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_repo  # noqa: E402

LOCAL = {
    "LK-L02": "registry is generated: run ./tools/regen (hand edits fail)",
    "LK-L03": "money only via money helpers (no float/Decimal/round, no `/`, no :.2f)",
    "LK-L06": "errors only as LedgerError(<known code>, msg)",
    "LK-L12": "file I/O only through ledgerkit.storage.backend",
    "MUT": "commands that change the ledger set MUTATES = True (cli.py saves only then; behaviour test)",
}
REMOTE = {
    "LK-R01": "docs/commands.md row per command (usage lists every --flag; new rows: since 'unreleased')",
    "LK-R02": "release bot: CHANGELOG.d/<name>.md with ### Added|Changed|Fixed (trap: changes/<name>.toml)",
    "LK-R03": "packaging: new commands listed in ledgerkit.toml [commands] -> ledgerkit.commands.<module>:run",
    "LK-R04": "review: new command has a cmd(<name>) test with ledger_factory; every new --flag is tested",
}
RULES = {**LOCAL, **REMOTE}
# Strings that would state a remote rule if they appeared anywhere in a snapshot.
REMOTE_TELLS = ["LK-R0", "unreleased", "release bot", "changelog fragment", "docs/commands.md", "ledgerkit.toml",
                "CHANGELOG.d", "since must", "ledger_factory fixture"]
FLAG = re.compile(r'"(--[a-z][a-z-]*)"')


def needed(task: str, snapshot: dict[str, str]) -> set[str]:
    ref = HERE / "reference" / task
    out = set()
    for path in sorted(ref.rglob("*.py")):
        rel = path.relative_to(ref).as_posix()
        if not rel.startswith("src/ledgerkit/commands/"):
            continue
        src, old = path.read_text(), snapshot.get(rel, "")
        more = lambda *words: sum(src.count(w) for w in words) > sum(old.count(w) for w in words)
        new_flags = set(FLAG.findall(src)) - set(FLAG.findall(old))
        out |= {"LK-R01", "LK-R02"}
        if not old:
            out |= {"LK-L02", "LK-R03", "LK-R04"}
        if new_flags:
            out.add("LK-R04")
        if more("format_cents(", "parse_amount("):
            out.add("LK-L03")
        if more("raise LedgerError"):
            out.add("LK-L06")
        if more("backend."):
            out.add("LK-L12")
        if more("MUTATES = True"):
            out.add("MUT")
    return out


def remote_tells(snapshot: dict[str, str]) -> list[str]:
    """Remote-rule statements found in the snapshot (should be none). The data the rules are
    about may exist as examples (docs/commands.md rows, ledgerkit.toml entries, an empty
    CHANGELOG.d/, existing marked tests); no file may state a rule or point at those files."""
    found = []
    for name, text in snapshot.items():
        if name in ("docs/commands.md", "ledgerkit.toml"):
            continue
        found += [f"{name}: {t}" for t in REMOTE_TELLS if t in text]
    return found


README_WORDS = {"LK-L02": ["regen", "registry"], "LK-L03": ["parse_amount", "format_cents", "cents"],
                "LK-L06": ["LedgerError", "LK-E"], "LK-L12": ["backend"], "MUT": ["MUTATES"],
                "LK-R01": ["docs/commands", "unreleased"], "LK-R02": ["CHANGELOG.d", "fragment", "changes/"],
                "LK-R03": ["ledgerkit.toml"], "LK-R04": ["mark.cmd", "ledger_factory"]}


def documented(readme: str) -> set[str]:
    low = readme.lower()
    return {rule for rule, words in README_WORDS.items() if any(w.lower() in low for w in words)}


def snapshot_files(repo: Path, ref: str) -> dict[str, str]:
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref],
                           capture_output=True, text=True, check=True).stdout.split()
    show = lambda n: subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{n}"], capture_output=True,
                                    text=True, check=True).stdout
    return {n: show(n) for n in names}


def matrix() -> dict[str, dict]:
    import yaml

    fam = yaml.safe_load((HERE / "family.yaml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        build_repo.build(repo)
        out, cache = {}, {}
        for task in fam["tasks"]:
            ref = fam["snapshots"][str(task)]
            snap = cache.setdefault(ref, snapshot_files(repo, ref))
            need = needed(str(task), snap)
            out[str(task)] = {"local": sorted(need & set(LOCAL)), "remote": sorted(need & set(REMOTE)),
                              "remote_tells": remote_tells(snap), "readme_mentions": sorted(documented(snap["README.md"]) & need)}
    return out


def main() -> int:
    rows = matrix()
    cols = list(LOCAL) + list(REMOTE)
    print("task  " + " ".join(f"{c[-3:]:>4}" for c in cols) + "   local remote")
    for task, row in rows.items():
        have = set(row["local"]) | set(row["remote"])
        print(f"{task}    " + " ".join(f"{'x' if c in have else '.':>4}" for c in cols)
              + f"   {len(row['local']):>5} {len(row['remote']):>6}")
    n = len(rows)
    print(f"\nmean per task: {sum(len(r['local']) for r in rows.values()) / n:.1f} local (readable), "
          f"{sum(len(r['remote']) for r in rows.values()) / n:.1f} remote (feedback only)")
    tells = sorted({t for r in rows.values() for t in r["remote_tells"]})
    print("remote rules stated in a snapshot: " + (", ".join(tells) if tells else "none"))
    for rule, what in RULES.items():
        print(f"  {rule:<7} {'remote' if rule in REMOTE else 'local '}  {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
