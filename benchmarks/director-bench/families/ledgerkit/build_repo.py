#!/usr/bin/env python3
"""Build the synthetic `ledgerkit` repo deterministically.

    python build_repo.py <dest>

Copies template/base, generates the importer and report modules (and their tests)
from the spec tables below, runs the repo's own tools/regen, commits, and tags
task-01..task-08 and task-10 on that commit. Then overlays template/trap (task 09:
unreleased-change fragments move from CHANGELOG.d/<name>.md to changes/<name>.toml),
commits and tags task-09. Same input -> same commits (fixed author, dates, order).

Everything here is what the agent sees: no verifier or reference-solution content.
"""
from __future__ import annotations

import csv
import datetime
import io
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import repo_layers  # noqa: E402  (same directory)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
TRAP_DELETE = ["CHANGELOG.d"]
EXECUTABLE = {"tools/check", "tools/regen", "tools/gendocs", "tools/lint.py"}

# name, title, delimiter, columns, date column, date format, memo column, decimal, thousands,
# header, amount mode (signed | inverted | split | dc), amount column
IMPORTERS = [
    ("csv_bank", "Generic bank CSV", ",", ["date", "memo", "amount"], "date", "%Y-%m-%d", "memo",
     ".", "", True, "signed", "amount"),
    ("n26", "N26", ",", ["date", "payee", "account_no", "kind", "reference", "amount"], "date",
     "%Y-%m-%d", "payee", ".", "", True, "signed", "amount"),
    ("revolut", "Revolut", ",", ["kind", "product", "started", "completed", "memo", "amount", "fee",
                                 "currency", "state", "balance"], "completed", "%Y-%m-%d %H:%M:%S",
     "memo", ".", "", True, "signed", "amount"),
    ("dkb", "DKB Girokonto", ";", ["booked", "valuta", "payee", "purpose", "account_no", "blz", "amount"],
     "booked", "%d.%m.%Y", "payee", ",", ".", True, "signed", "amount"),
    ("ing", "ING", ";", ["booked", "valuta", "payee", "kind", "memo", "balance", "balance_currency",
                         "amount", "currency"], "booked", "%d.%m.%Y", "payee", ",", ".", True, "signed", "amount"),
    ("chase", "Chase", ",", ["details", "posted", "memo", "amount", "kind", "balance", "check_no"], "posted",
     "%m/%d/%Y", "memo", ".", "", True, "signed", "amount"),
    ("barclays", "Barclays", ",", ["number", "date", "account_no", "amount", "subcategory", "memo"], "date",
     "%d/%m/%Y", "memo", ".", "", True, "signed", "amount"),
    ("monzo", "Monzo", ",", ["txid", "date", "time", "kind", "name", "emoji", "category", "amount", "currency",
                             "local_amount", "local_currency", "notes"], "date", "%d/%m/%Y", "name",
     ".", "", True, "signed", "amount"),
    ("sparkasse", "Sparkasse CAMT-CSV", ";", ["account_no", "booked", "valuta", "kind", "purpose", "creditor_id",
                                             "mandate", "payee", "iban", "bic", "amount", "currency", "info"],
     "booked", "%d.%m.%y", "payee", ",", "", True, "signed", "amount"),
    ("amex", "American Express", ",", ["date", "memo", "amount"], "date", "%m/%d/%Y", "memo", ".", ",",
     False, "inverted", "amount"),
    ("wise", "Wise", ",", ["txid", "date", "amount", "currency", "memo", "reference", "running_balance",
                           "exchange_from", "exchange_to", "rate", "payer", "payee"], "date", "%d-%m-%Y",
     "memo", ".", "", True, "signed", "amount"),
    ("paypal", "PayPal", ",", ["date", "time", "timezone", "name", "kind", "status", "currency", "gross", "fee",
                               "net", "txid", "email"], "date", "%d/%m/%Y", "name", ",", ".", True, "signed", "net"),
    ("coop", "Coop Bank", ";", ["date", "memo", "debit", "credit"], "date", "%d.%m.%Y", "memo", ",", "'",
     True, "split", ""),
    ("postbank", "Postbank", ";", ["date", "memo", "dc", "amount"], "date", "%d.%m.%Y", "memo", ",", ".",
     True, "dc", "amount"),
]

# module, report name, class, title, key label, key expression, imports, entries filter, expected key
REPORTS = [
    ("monthly", "monthly", "MonthlyReport", "Totals per month", "month", "month_of(entry.date)",
     ["from ledgerkit.utils.dates import month_of"], None, "2026-03"),
    ("quarterly", "quarterly", "QuarterlyReport", "Totals per quarter", "quarter", "quarter_of(entry.date)",
     ["from ledgerkit.utils.dates import quarter_of"], None, "2026-Q1"),
    ("yearly", "yearly", "YearlyReport", "Totals per year", "year", "year_of(entry.date)",
     ["from ledgerkit.utils.dates import year_of"], None, "2026"),
    ("category", "category", "CategoryReport", "Totals per category", "category", "categorize(entry)",
     ["from ledgerkit.rules.categorize import categorize"], None, "eating out"),
    ("payees", "payees", "PayeeReport", "Totals per payee", "payee", 'first_word(entry.memo) or "(none)"',
     ["from ledgerkit.utils.text import first_word"], None, "coffee"),
    ("by_account", "accounts", "AccountReport", "Totals per account", "account", "entry.account", [], None, "cash"),
    ("weekday", "weekday", "WeekdayReport", "Totals per weekday", "weekday", "WEEKDAYS[parse_iso(entry.date).weekday()]",
     ["from ledgerkit.utils.dates import parse_iso"], None, "Thu"),
    ("tags", "tags", "TagReport", "Totals per first tag", "tag", 'entry.tags[0] if entry.tags else "(untagged)"', [],
     None, "(untagged)"),
    ("expenses", "expenses", "ExpenseReport", "Expenses per category", "category", "categorize(entry)",
     ["from ledgerkit.rules.categorize import categorize"], "entry.amount_cents < 0", "eating out"),
    ("income", "income", "IncomeReport", "Income per month", "month", "month_of(entry.date)",
     ["from ledgerkit.utils.dates import month_of"], "entry.amount_cents > 0", "2026-03"),
]
HANDWRITTEN_REPORTS = ["budget", "cashflow", "largest", "networth"]

EXTRA_BANKS = ["abn_amro", "bbva", "bnp", "caixa", "commerzbank", "credit_agricole", "danske", "deutsche_bank",
               "hsbc", "intesa", "kbc", "lloyds", "nationwide", "nordea", "rabobank", "santander", "seb",
               "societe_generale", "starling", "swedbank", "ubs", "unicredit", "wells_fargo", "bank_of_america",
               "citi", "capital_one"]
FILLER = ["reference", "iban", "bic", "category", "balance", "currency", "kind", "status", "card", "channel",
          "counterparty", "mandate", "creditor_id", "note", "location", "fx_rate"]


def extra_importers() -> list:
    """Deterministic variations of statement layouts (fixed seed)."""
    rng = random.Random(2026)
    specs = []
    for bank in EXTRA_BANKS:
        delim = rng.choice([",", ";", "\t", "|"])
        dec, thou = rng.choice([(".", ""), (".", ","), (",", ""), (",", "."), (",", " "), (".", "'")])
        datecol = rng.choice(["date", "booked", "posted", "value_date", "transaction_date"])
        memocol = rng.choice(["memo", "description", "payee", "details", "narrative", "name"])
        datefmt = rng.choice(["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d", "%d.%m.%y",
                              "%Y/%m/%d"])
        mode = rng.choice(["signed"] * 6 + ["split"] * 2 + ["dc"] * 2 + ["inverted"])
        amountcol = "" if mode == "split" else rng.choice(["amount", "value", "sum", "betrag"])
        money = ["debit", "credit"] if mode == "split" else ([amountcol, "dc"] if mode == "dc" else [amountcol])
        cols = [datecol, memocol] + money + rng.sample(FILLER, rng.randint(0, 6))
        rng.shuffle(cols)
        header = mode != "inverted" or rng.random() < 0.5
        title = bank.replace("_", " ").title()
        specs.append((bank, title, delim, cols, datecol, datefmt, memocol, dec, thou, header, mode, amountcol))
    return specs


IMPORTERS += extra_importers()

REPORTS += [
    ("weekly", "weekly", "WeeklyReport", "Totals per ISO week", "week", 'parse_iso(entry.date).strftime("%G-W%V")',
     ["from ledgerkit.utils.dates import parse_iso"], None, "2026-W10"),
    ("daily", "daily", "DailyReport", "Totals per day", "day", "entry.date", [], None, "2026-03-05"),
    ("monthly_expenses", "monthly-expenses", "MonthlyExpenseReport", "Expenses per month", "month",
     "month_of(entry.date)", ["from ledgerkit.utils.dates import month_of"], "entry.amount_cents < 0", "2026-03"),
    ("account_monthly", "account-monthly", "AccountMonthlyReport", "Totals per account and month", "account/month",
     'f"{entry.account}/{month_of(entry.date)}"', ["from ledgerkit.utils.dates import month_of"], None,
     "cash/2026-03"),
    ("merchants", "merchants", "MerchantReport", "Totals per merchant category", "merchant category",
     'merchant_category(entry.memo) or "(unknown)"', ["from ledgerkit.rules.merchants import merchant_category"],
     None, "(unknown)"),
    ("large_items", "large-items", "LargeItemReport", "Items above 500.00 per category", "category",
     "categorize(entry)", ["from ledgerkit.rules.categorize import categorize"], "abs(entry.amount_cents) >= 50000",
     "salary"),
    ("small_items", "small-items", "SmallItemReport", "Items below 20.00 per payee", "payee",
     'first_word(entry.memo) or "(none)"', ["from ledgerkit.utils.text import first_word"],
     "abs(entry.amount_cents) < 2000", "coffee"),
    ("quarter_expenses", "quarter-expenses", "QuarterExpenseReport", "Expenses per quarter", "quarter",
     "quarter_of(entry.date)", ["from ledgerkit.utils.dates import quarter_of"], "entry.amount_cents < 0", "2026-Q1"),
]

# Rows every importer test parses: (iso date, memo, cents)
SAMPLE = repo_layers.SAMPLE


def money_text(cents: int, decimal: str, thousands: str) -> str:
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    digits = f"{whole:,}".replace(",", thousands)
    return f"{sign}{digits}{decimal}{frac:02d}"


def importer_module(spec) -> str:
    name, title, delim, cols, datecol, datefmt, memocol, dec, thou, header, mode, amountcol = spec
    fix = ""
    if thou:
        fix += f".replace({thou!r}, \"\")"
    if dec != ".":
        fix += f".replace({dec!r}, \".\")"
    body = {
        "signed": f"    return _number(record[{amountcol!r}])",
        "inverted": f"    # charges are positive on this statement\n    return -_number(record[{amountcol!r}])",
        "split": '    return _number(record["credit"]) - _number(record["debit"])',
        "dc": f'    value = _number(record[{amountcol!r}])\n    return -value if record["dc"].strip().upper() == "D" else value',
    }[mode]
    detect = ("first.split(DELIMITER)[:2] == list(COLUMNS[:2])" if header
              else "first.count(DELIMITER) == len(COLUMNS) - 1")
    cls = "".join(p.capitalize() for p in name.split("_")) + "Importer"
    return f'''"""{title} statement importer."""
from ledgerkit.errors import ImportFormatError, LedgerError
from ledgerkit.importers.base import ImportedRow, Importer, split_line
from ledgerkit.money import parse_amount
from ledgerkit.utils.dates import to_iso
from ledgerkit.utils.text import clean_memo

COLUMNS = {tuple(cols)!r}
DELIMITER = {delim!r}
DATE_COLUMN = {datecol!r}
DATE_FORMAT = {datefmt!r}
MEMO_COLUMN = {memocol!r}
HAS_HEADER = {header!r}


def _number(raw: str) -> int:
    text = raw.strip(){fix}
    return parse_amount(text) if text else 0


def _amount(record: dict) -> int:
{body}


class {cls}(Importer):
    name = {name!r}

    def detect(self, text: str) -> bool:
        first = text.splitlines()[0] if text.strip() else ""
        return {detect}

    def parse(self, text: str, account: str) -> list:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or (HAS_HEADER and lineno == 1):
                continue
            fields = split_line(line, DELIMITER)
            if len(fields) != len(COLUMNS):
                raise ImportFormatError(self.name, lineno, f"expected {{len(COLUMNS)}} fields, got {{len(fields)}}")
            record = dict(zip(COLUMNS, fields))
            try:
                cents = _amount(record)
            except LedgerError as exc:
                raise ImportFormatError(self.name, lineno, exc.message) from None
            rows.append(ImportedRow(to_iso(record[DATE_COLUMN], DATE_FORMAT), account, cents,
                                    clean_memo(record[MEMO_COLUMN]), self.name, lineno))
        return rows


IMPORTER = {cls}()
'''


def importer_sample(spec) -> str:
    name, title, delim, cols, datecol, datefmt, memocol, dec, thou, header, mode, amountcol = spec
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delim, lineterminator="\n")
    if header:
        writer.writerow(cols)
    for iso, memo, cents in SAMPLE:
        when = datetime.datetime.fromisoformat(iso + "T10:00:00")
        row = {c: f"x{i}" for i, c in enumerate(cols)}
        row[datecol] = when.strftime(datefmt)
        row[memocol] = memo
        if mode == "signed":
            row[amountcol] = money_text(cents, dec, thou)
        elif mode == "inverted":
            row[amountcol] = money_text(-cents, dec, thou)
        elif mode == "split":
            row["debit"] = money_text(-cents, dec, thou) if cents < 0 else ""
            row["credit"] = money_text(cents, dec, thou) if cents > 0 else ""
        else:
            row[amountcol] = money_text(abs(cents), dec, thou)
            row["dc"] = "D" if cents < 0 else "C"
        writer.writerow([row[c] for c in cols])
    return out.getvalue()


def importers_init() -> str:
    names = [s[0] for s in IMPORTERS]
    return f'''"""Statement importers, by name."""
from ledgerkit.errors import LedgerError
from ledgerkit.importers import {", ".join(names)}

IMPORTERS = {{m.IMPORTER.name: m.IMPORTER for m in ({", ".join(names)},)}}


def get(name: str):
    try:
        return IMPORTERS[name]
    except KeyError:
        raise LedgerError(400, f"unknown importer: {{name}}") from None


def detect(text: str) -> list:
    return sorted(name for name, imp in IMPORTERS.items() if imp.detect(text))
'''


def importer_tests() -> str:
    cases = ",\n".join(f"    ({s[0]!r}, {importer_sample(s)!r})" for s in IMPORTERS)
    return f'''import pytest

from ledgerkit import importers
from ledgerkit.errors import ImportFormatError
from ledgerkit.importers.base import apply
from ledgerkit.ledger import Ledger

CASES = [
{cases},
]


@pytest.mark.parametrize("name,text", CASES)
def test_parses_sample(name, text):
    rows = importers.get(name).parse(text, "bank")
    assert [(r.date, r.memo, r.amount_cents) for r in rows] == {[(d, m, c) for d, m, c in SAMPLE]!r}
    assert {{r.source for r in rows}} == {{name}}


@pytest.mark.parametrize("name,text", CASES)
def test_detects_own_format(name, text):
    assert name in importers.detect(text)


@pytest.mark.parametrize("name,text", CASES)
def test_rejects_short_lines(name, text):
    lines = text.splitlines()
    bad = "\\n".join(lines[:-1] + ["2026-01-01"])
    with pytest.raises(ImportFormatError):
        importers.get(name).parse(bad, "bank")


def test_apply_skips_duplicates():
    ledger = Ledger()
    rows = importers.get("csv_bank").parse(CASES[0][1], "bank")
    assert apply(ledger, rows) == 2
    assert apply(ledger, rows) == 0
'''


def report_module(spec) -> str:
    mod, name, cls, title, label, key, imports, filt, _ = spec
    if "WEEKDAYS" in key:
        imports = imports + []
    lines = [f'"""{title}."""'] + sorted(imports + ["from ledgerkit.reports.base import Report"])
    if "WEEKDAYS" in key:
        lines += ["", 'WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")']
    lines += ["", "", f"class {cls}(Report):", f"    name = {name!r}", f"    title = {title!r}",
              f'    columns = ({label!r}, "entries", "total")', ""]
    if filt:
        lines += ["    def entries(self, ledger) -> list:", f"        return [entry for entry in ledger.entries if {filt}]", ""]
    lines += ["    def key(self, entry) -> str:", f"        return {key}", "", "", f"REPORT = {cls}()", ""]
    return "\n".join(lines)


def reports_init() -> str:
    mods = sorted([s[0] for s in REPORTS] + HANDWRITTEN_REPORTS)
    return f'''"""Reports, by name."""
from ledgerkit.errors import LedgerError
from ledgerkit.reports import {", ".join(mods)}

REPORTS = {{m.REPORT.name: m.REPORT for m in ({", ".join(mods)},)}}


def get(name: str):
    try:
        return REPORTS[name]
    except KeyError:
        raise LedgerError(400, f"unknown report: {{name}}") from None
'''


def report_tests() -> str:
    cases = ",\n".join(f"    ({s[1]!r}, {s[8]!r})" for s in REPORTS)
    return f'''import pytest

from ledgerkit import reports
from ledgerkit.errors import LedgerError
from ledgerkit.ledger import Ledger

CASES = [
{cases},
]


@pytest.fixture
def ledger():
    led = Ledger()
    led.add("cash", -350, "coffee", date="2026-03-05")
    led.add("bank", 200000, "salary", date="2026-03-31")
    led.add("bank", -90000, "rent april", date="2026-04-01")
    return led


@pytest.mark.parametrize("name,key", CASES)
def test_grouped_reports(name, key, ledger):
    out = reports.get(name).render(ledger)
    assert key in out


@pytest.mark.parametrize("name", sorted(reports.REPORTS))
def test_every_report_handles_empty(name):
    assert reports.get(name).render(Ledger())


def test_cashflow_and_networth(ledger):
    assert reports.get("cashflow").build(ledger)[0] == ("2026-03", 200000, -350, 199650)
    assert reports.get("networth").build(ledger)[-1] == ("2026-04", 109650)


def test_budget_and_largest(ledger):
    rows = reports.get("budget").build(ledger, limits={{"eating out": 1000}})
    assert rows == [("eating out", 350, 1000, 650)]
    assert [e.memo for e in reports.get("top-expenses").build(ledger)] == ["rent april", "coffee"]


def test_unknown_report():
    with pytest.raises(LedgerError):
        reports.get("nope")
'''


def generated_files() -> dict[str, str]:
    files = {f"src/ledgerkit/importers/{s[0]}.py": importer_module(s) for s in IMPORTERS}
    files["src/ledgerkit/importers/__init__.py"] = importers_init()
    files.update({f"src/ledgerkit/reports/{s[0]}.py": report_module(s) for s in REPORTS})
    files["src/ledgerkit/reports/__init__.py"] = reports_init()
    files["tests/unit/test_importers.py"] = importer_tests()
    files["tests/unit/test_reports.py"] = report_tests()
    files.update(repo_layers.files())
    return files


GIT_ENV = {
    "GIT_AUTHOR_NAME": "ledgerkit", "GIT_AUTHOR_EMAIL": "dev@ledgerkit.invalid",
    "GIT_COMMITTER_NAME": "ledgerkit", "GIT_COMMITTER_EMAIL": "dev@ledgerkit.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


def git(dest: Path, *args: str, date: str = "2026-01-01T00:00:00Z") -> None:
    env = {**os.environ, **GIT_ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", *args], cwd=dest, env=env, check=True, capture_output=True)


def copy_template(src: Path, dest: Path) -> None:
    for path in sorted(src.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            target = dest / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def finish_tree(dest: Path) -> None:
    for rel, text in sorted(generated_files().items()):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8")
    subprocess.run([sys.executable, str(dest / "tools" / "regen")], check=True, capture_output=True)
    for path in sorted(dest.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(dest).as_posix()
        path.chmod(0o755 if rel in EXECUTABLE else 0o644)
    for cache in dest.rglob("__pycache__"):
        shutil.rmtree(cache)


def build(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    git(dest, "init", "-q", "-b", "main")
    copy_template(TEMPLATE / "base", dest)
    finish_tree(dest)
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "ledgerkit 0.5.0")
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        git(dest, "tag", f"task-{n:02d}")
    for rel in TRAP_DELETE:
        shutil.rmtree(dest / rel)
    copy_template(TEMPLATE / "trap", dest)
    finish_tree(dest)
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "ledgerkit 0.6.0: change fragments move to changes/*.toml",
        date="2026-02-01T00:00:00Z")
    git(dest, "tag", "task-09")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: build_repo.py <dest>")
    build(Path(sys.argv[1]))
