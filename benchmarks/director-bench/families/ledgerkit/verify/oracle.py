#!/usr/bin/env python3
"""ledgerkit oracle (hidden from the agent). Leak marker: LKORACLE-7f3a9c

    oracle.py <task-id> <workdir>

Runs the repo's own check (tests + house lint), the team's remote CI rules (which
exist only here, never in the repo: docs table, release-bot change fragment,
packaging entry, review of tests) and a hidden behavioural check for the task.
Exit 0 = pass. Every failure is printed twice:

    BENCH-FAIL type=<t> file=<f> symbol=<s> msg=<m>     (signature data, results only)
    CI <what a CI log or a reviewer would say>          (the only text the agent may see)

Behaviour checks stay hidden: their CI line is just "CI: behaviour test <name> failed".
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

FAILS: list[str] = []
CI: list[str] = []
TRAP_TASKS = {"09"}          # snapshot where the release bot wants changes/<name>.toml


def fail(kind: str, file: str, symbol: str, msg: str, ci: str | None = None) -> None:
    line = f"BENCH-FAIL type={kind} file={file or '-'} symbol={symbol or '-'} msg={msg}"
    FAILS.append(line)
    print(line)
    if kind == "hidden":
        ci = f"CI: behaviour test {symbol} failed"
    if ci and ci not in CI:
        CI.append(ci)


def run_check(work: Path) -> None:
    if not (work / "tools" / "check").exists():
        fail("check", "tools/check", "missing", "tools/check not found")
        return
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run(["sh", "tools/check"], cwd=work, env=env, capture_output=True,
                       text=True, timeout=600)
    out = p.stdout + p.stderr
    print(out)
    found = False
    for line in out.splitlines():
        m = re.match(r"^(\S+?):(\d+): (LK-L\d+) (.*)$", line)
        if m:
            fail("lint", m.group(1), m.group(3), m.group(4), f"CI tools/check: {line.strip()}")
            found = True
            continue
        m = re.match(r"^(FAILED|ERROR) (\S+?)(?:::(\S+))?(?: - (.*))?$", line)
        if m:
            fail("test", m.group(2), m.group(3) or m.group(1).lower(), m.group(4) or "",
                 f"CI tools/check: {m.group(1)} {m.group(2)}{'::' + m.group(3) if m.group(3) else ''}")
            found = True
    if p.returncode != 0 and not found:
        fail("check", "tools/check", "exit", f"exit status {p.returncode}", "CI tools/check: failed")


class Hidden:
    """Helpers for the behavioural checks, importing ledgerkit from the workdir."""

    def __init__(self, work: Path, check: str):
        sys.path.insert(0, str(work / "src"))
        self.work, self.check = work, check
        from ledgerkit.cli import command_table, dispatch  # noqa: E402
        from ledgerkit.errors import LedgerError  # noqa: E402
        from ledgerkit.ledger import Ledger  # noqa: E402

        self.table, self.dispatch = command_table, dispatch
        self.LedgerError, self.Ledger = LedgerError, Ledger

    def ledger(self, *rows):
        led = self.Ledger()
        for account, cents, *memo in rows:
            led.add(account, cents, memo[0] if memo else "", date="2026-01-15")
        return led

    def registered(self, name: str) -> bool:
        if name not in self.table():
            fail("hidden", "", self.check, f"command '{name}' is not available")
            return False
        return True

    def expect(self, ledger, argv, want: str) -> None:
        try:
            got = self.dispatch(ledger, argv)
        except Exception as exc:  # any exception is a failure here
            fail("hidden", "", self.check, f"{' '.join(argv)!r} raised {type(exc).__name__}: {exc}")
            return
        if got != want:
            fail("hidden", "", self.check, f"{' '.join(argv)!r} returned {got!r}, want {want!r}")

    def expect_error(self, ledger, argv, code: int) -> None:
        try:
            got = self.dispatch(ledger, argv)
        except self.LedgerError as exc:
            if not str(exc).startswith(f"LK-E{code:03d}"):
                fail("hidden", "", self.check, f"{' '.join(argv)!r} error {exc} , want LK-E{code:03d}")
            return
        except Exception as exc:
            fail("hidden", "", self.check, f"{' '.join(argv)!r} raised {type(exc).__name__}, want LedgerError {code}")
            return
        fail("hidden", "", self.check, f"{' '.join(argv)!r} returned {got!r}, want LK-E{code:03d}")

    def persists(self, rows, argv, probe) -> None:
        """Run argv through the real CLI on a ledger file; probe(reloaded ledger) -> error text or None."""
        import contextlib
        import io
        import tempfile
        from ledgerkit import cli
        from ledgerkit.storage import jsonstore
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            jsonstore.save_ledger(self.ledger(*rows), path)
            old = os.environ.get("LEDGERKIT_FILE")
            os.environ["LEDGERKIT_FILE"] = path
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main(list(argv))
            except Exception as exc:
                fail("hidden", "", f"{self.check}-saved", f"cli {' '.join(argv)!r} raised {type(exc).__name__}: {exc}")
                return
            finally:
                if old is None:
                    os.environ.pop("LEDGERKIT_FILE", None)
                else:
                    os.environ["LEDGERKIT_FILE"] = old
            problem = probe(jsonstore.load_ledger(path)) if code == 0 else f"exit status {code}"
            if problem:
                fail("hidden", "", f"{self.check}-saved", f"cli {' '.join(argv)!r} not persisted: {problem}")

    def int_cents(self, ledger) -> None:
        for e in ledger.entries:
            if type(e.amount_cents) is not int:
                fail("hidden", "", self.check, f"entry {e.id} amount is {type(e.amount_cents).__name__}, not int cents")


def check_balance(h: Hidden) -> None:
    if not h.registered("balance"):
        return
    led = h.ledger(("cash", 1050), ("cash", -25), ("bank", 999))
    h.expect(led, ["balance", "cash"], "balance cash: 10.25")
    h.expect(led, ["balance", "bank"], "balance bank: 9.99")
    h.expect(led, ["balance", "nobody"], "balance nobody: 0.00")
    h.expect(h.ledger(("x", 10), ("x", 20)), ["balance", "x"], "balance x: 0.30")
    h.expect_error(led, ["balance"], 400)


def check_count(h: Hidden) -> None:
    if not h.registered("count"):
        return
    led = h.ledger(("cash", 1), ("cash", 2), ("bank", 3))
    h.expect(led, ["count", "cash"], "2")
    h.expect(led, ["count", "nobody"], "0")
    h.expect_error(led, ["count"], 400)


def check_largest(h: Hidden) -> None:
    if not h.registered("largest"):
        return
    led = h.ledger(("cash", -5000), ("cash", 1050), ("cash", 7), ("bank", 99999))
    h.expect(led, ["largest", "cash"], "10.50")
    h.expect(h.ledger(("d", -300), ("d", -20)), ["largest", "d"], "-0.20")
    h.expect_error(led, ["largest", "nobody"], 404)
    h.expect_error(led, ["largest"], 400)


def check_list_memo(h: Hidden) -> None:
    if not h.registered("list"):
        return
    led = h.ledger(("cash", 1050, "Rent May"), ("bank", -99, "groceries"), ("bank", 5, "rent fee"))
    h.expect(led, ["list", "--memo", "rent"], "1 2026-01-15 cash 10.50 Rent May\n3 2026-01-15 bank 0.05 rent fee")
    h.expect(led, ["list", "bank", "--memo", "RENT"], "3 2026-01-15 bank 0.05 rent fee")
    h.expect(led, ["list", "--memo", "zzz"], "no entries")
    h.expect(led, ["list", "cash"], "1 2026-01-15 cash 10.50 Rent May")
    h.expect(led, ["list", "--limit", "2"], "2 2026-01-15 bank -0.99 groceries\n3 2026-01-15 bank 0.05 rent fee")
    h.expect(led, ["list", "--memo", "rent", "--limit", "1"], "3 2026-01-15 bank 0.05 rent fee")
    h.expect(led, ["list", "bank", "--limit", "5"], "2 2026-01-15 bank -0.99 groceries\n3 2026-01-15 bank 0.05 rent fee")
    for bad in (["list", "--memo"], ["list", "--limit"], ["list", "--limit", "0"], ["list", "--limit", "x"],
                ["list", "--limit", "-1"]):
        h.expect_error(led, bad, 400)


def check_transfer(h: Hidden) -> None:
    if not h.registered("transfer"):
        return
    led = h.ledger(("cash", 1000))
    h.expect(led, ["transfer", "cash", "bank", "2.50"], "transferred 2.50 from cash to bank")
    rows = [(e.account, e.amount_cents, e.memo) for e in led.entries[1:]]
    if sorted(rows) != sorted([("cash", -250, "transfer"), ("bank", 250, "transfer")]):
        fail("hidden", "", h.check, f"transfer recorded {rows}")
    h.int_cents(led)
    h.persists([("cash", 1000)], ["transfer", "cash", "bank", "1.25"],
               lambda l: None if [e.amount_cents for e in l.entries] == [1000, -125, 125] else "entries missing")
    h.expect(h.ledger(), ["transfer", "a", "b", "0.1"], "transferred 0.10 from a to b")
    for bad in (["transfer", "a", "b", "0"], ["transfer", "a", "b", "-1"],
                ["transfer", "a", "b", "1.234"], ["transfer", "a", "b"]):
        h.expect_error(h.ledger(), bad, 400)


def check_average(h: Hidden) -> None:
    if not h.registered("average"):
        return
    h.expect(h.ledger(("c", 100), ("c", 101)), ["average", "c"], "1.00")
    h.expect(h.ledger(("c", -1), ("c", -2)), ["average", "c"], "-0.02")
    h.expect(h.ledger(("c", 10), ("c", 10), ("c", 11)), ["average", "c"], "0.10")
    h.expect(h.ledger(("c", 33), ("d", 1)), ["average", "c"], "0.33")
    h.expect(h.ledger(("a", 1), ("b", 4)), ["average"], "0.02")
    h.expect(h.ledger(("c", 1)), ["average", "z"], "0.00")
    h.expect(h.ledger(), ["average"], "0.00")


def check_tag(h: Hidden) -> None:
    if not h.registered("tag"):
        return
    led = h.ledger(("cash", 1), ("cash", 2))
    h.expect(led, ["tag", "2", "food"], "tagged 2 food")
    h.expect(led, ["tag", "2", "food"], "tagged 2 food")
    h.expect(led, ["tag", "2", "x-1"], "tagged 2 x-1")
    if led.entries[1].tags != ["food", "x-1"] or led.entries[0].tags:
        fail("hidden", "", h.check, f"tags after tagging: {[e.tags for e in led.entries]}")
    h.expect_error(led, ["tag", "9", "x"], 404)
    for bad in (["tag", "a", "x"], ["tag", "1", "Bad"], ["tag", "1", "-x"], ["tag", "1"], ["tag"]):
        h.expect_error(led, bad, 400)
    h.persists([("cash", 1)], ["tag", "1", "food"],
               lambda l: None if l.entries[0].tags == ["food"] else f"tags {l.entries[0].tags}")


def check_export_out(h: Hidden) -> None:
    if not h.registered("export"):
        return
    import tempfile
    led = h.ledger(("cash", -350, "coffee"), ("bank", 5, "x"))
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ([], ["--format", "json"]):
            path = os.path.join(tmp, f"out{len(fmt)}.txt")
            h.expect(led, ["export", *fmt, "--out", path], f"wrote 2 entries to {path}")
            want = h.dispatch(led, ["export", *fmt])
            got = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            if got != want:
                fail("hidden", "", h.check, f"--out file content {got!r:.60} differs from export output")
    h.expect_error(led, ["export", "--out"], 400)
    h.expect(led, ["export"], "id,date,account,amount,memo\n1,2026-01-15,cash,-3.50,coffee\n2,2026-01-15,bank,0.05,x")


def check_rename(h: Hidden) -> None:
    if not h.registered("rename"):
        return
    led = h.ledger(("cash", 1), ("bank", 2), ("cash", 3))
    h.expect(led, ["rename", "cash", "wallet"], "renamed 2 entries from cash to wallet")
    if [e.account for e in led.entries] != ["wallet", "bank", "wallet"]:
        fail("hidden", "", h.check, f"accounts after rename: {[e.account for e in led.entries]}")
    h.expect_error(led, ["rename", "nobody", "x"], 404)
    h.expect_error(led, ["rename", "wallet", "bank"], 409)
    h.expect_error(led, ["rename", "wallet"], 400)
    h.persists([("cash", 1), ("bank", 2)], ["rename", "cash", "wallet"],
               lambda l: None if l.accounts() == ["bank", "wallet"] else f"accounts {l.accounts()}")


# ---------------------------------------------------------------- remote CI rules
FLAG = re.compile(r'"(--[a-z][a-z-]*)"')


def git(work: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=work, capture_output=True, text=True).stdout


def command_modules(work: Path) -> dict:
    """stem -> (rel path, NAME, source) for every command module."""
    import ast
    out = {}
    for path in sorted((work / "src/ledgerkit/commands").glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        name = next((n.value.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", "") == "NAME" for t in n.targets)
                     and isinstance(n.value, ast.Constant)), None)
        if name:
            out[path.stem] = (path.relative_to(work).as_posix(), name, src)
    return out


def changed_since_snapshot(work: Path) -> tuple[set, str]:
    roots = git(work, "rev-list", "--max-parents=0", "HEAD").split()
    if not roots:
        return set(), ""
    base = roots[-1]
    files = set(git(work, "diff", "--name-only", base).split())
    files |= set(git(work, "ls-files", "--others", "--exclude-standard").split())
    return files, base


def docs_rows(work: Path) -> dict:
    rows = {}
    path = work / "docs" / "commands.md"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 3 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = (cells[1].strip("`"), cells[2])
    return rows


def packaging(work: Path) -> dict:
    entries, section = {}, None
    path = work / "ledgerkit.toml"
    for raw in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[]").strip()
        elif section == "commands" and "=" in line:
            key, value = (p.strip() for p in line.split("=", 1))
            entries[key.strip('"')] = value.strip('"')
    return entries


def marked_tests(work: Path, name: str) -> list:
    """Sources of test functions marked cmd(name) that take ledger_factory."""
    import ast
    out = []
    for path in sorted((work / "tests").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "ledger_factory" in [a.arg for a in node.args.args]:
                for dec in node.decorator_list:
                    if (isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "cmd" and dec.args
                            and getattr(dec.args[0], "value", None) == name):
                        out.append(ast.get_source_segment(src, node) or "")
    return out


def remote(work: Path, task: str) -> None:
    changed, base = changed_since_snapshot(work)
    mods = command_modules(work)
    rows, pkg = docs_rows(work), packaging(work)
    for stem, (rel, name, src) in mods.items():
        old = git(work, "show", f"{base}:{rel}") if base else ""
        new = not old
        if rel not in changed:
            continue
        flags = sorted(set(FLAG.findall(src)))
        new_flags = sorted(set(flags) - set(FLAG.findall(old)))
        # LK-R01 docs table
        if name not in rows:
            fail("remote", "docs/commands.md", "LK-R01", f"no row for {name}",
                 f"CI LK-R01: docs/commands.md has no row for '{name}'")
        else:
            usage, since = rows[name]
            for flag in flags:
                if flag not in usage.split() and f"[{flag}" not in usage:
                    fail("remote", "docs/commands.md", "LK-R01", f"row {name} lacks {flag}",
                         f"CI LK-R01: docs/commands.md row for '{name}' does not mention {flag}")
            if new and since != "unreleased":
                fail("remote", "docs/commands.md", "LK-R01", f"row {name} since {since}",
                     f"CI LK-R01: docs/commands.md row for '{name}': since must be 'unreleased' "
                     "(the release bot stamps the version)")
        # LK-R02 release-bot change fragment
        legacy = work / "CHANGELOG.d" / f"{name}.md"
        if task in TRAP_TASKS:
            frag = work / "changes" / f"{name}.toml"
            if legacy.exists():
                fail("remote", rel, "LK-R02", f"legacy fragment for {name}",
                     f"CI LK-R02: release bot: CHANGELOG.d/ was retired in 0.6.0; fragments are "
                     f"changes/<name>.toml with type = \"added\" | \"changed\" | \"fixed\"")
            elif not frag.exists():
                fail("remote", rel, "LK-R02", f"no fragment for {name}",
                     f"CI LK-R02: release bot: no change fragment for '{name}'")
            elif not re.search(r'^type\s*=\s*"(added|changed|fixed)"\s*$', frag.read_text(), re.M):
                fail("remote", rel, "LK-R02", f"bad fragment for {name}",
                     f"CI LK-R02: release bot: changes/{name}.toml needs type = \"added\", \"changed\" or \"fixed\"")
        else:
            if not legacy.exists():
                fail("remote", rel, "LK-R02", f"no fragment for {name}",
                     f"CI LK-R02: release bot: no changelog fragment for '{name}'")
            elif not re.search(r"^### (Added|Changed|Fixed)\s*$", legacy.read_text(), re.M):
                fail("remote", rel, "LK-R02", f"bad fragment for {name}",
                     f"CI LK-R02: release bot: CHANGELOG.d/{name}.md needs an '### Added' or '### Changed' heading")
        # LK-R03 packaging entry
        want = f"ledgerkit.commands.{stem}:run"
        if pkg.get(name) != want:
            fail("remote", "ledgerkit.toml", "LK-R03", f"packaging {name}",
                 f"CI LK-R03: packaging: command '{name}' is not listed in ledgerkit.toml [commands]"
                 if name not in pkg else
                 f"CI LK-R03: packaging: ledgerkit.toml [commands] entry for '{name}' does not point to its run()")
        # LK-R04 review: tests
        tests = marked_tests(work, name)
        if new and not tests:
            fail("remote", "tests", "LK-R04", f"no cmd test for {name}",
                 f"CI LK-R04: review: please add a test marked @pytest.mark.cmd(\"{name}\") that builds its "
                 "ledger with the ledger_factory fixture")
        for flag in new_flags:
            if not any(flag in t for t in tests):
                fail("remote", "tests", "LK-R04", f"{flag} untested",
                     f"CI LK-R04: review: no cmd(\"{name}\") test exercises {flag}")


CHECKS = {
    "01": ("balance", check_balance), "02": ("count", check_count), "03": ("largest", check_largest),
    "04": ("list", check_list_memo), "05": ("transfer", check_transfer), "06": ("average", check_average),
    "07": ("tag", check_tag), "08": ("export", check_export_out), "09": ("rename", check_rename),
    "10": ("balance", check_balance),
}


def main() -> int:
    task, work = sys.argv[1], Path(sys.argv[2]).resolve()
    run_check(work)
    remote(work, task)
    name, check = CHECKS[task]
    try:
        check(Hidden(work, name))
    except Exception as exc:  # broken import etc.
        fail("hidden", "", name, f"{type(exc).__name__}: {exc}")
    for line in CI:
        print(line)
    print("PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
