"""Import a bank statement."""
from ledgerkit import importers
from ledgerkit.importers.base import apply
from ledgerkit.storage import backend
from ledgerkit.utils.args import expect_count

NAME = "import"
MUTATES = True
EXAMPLES = ["import csv_bank statement.csv bank", "import dkb export.csv savings"]


def run(ledger, args):
    fmt, path, account = expect_count(args, 3, "import <format> <file> <account>")
    rows = importers.get(fmt).parse(backend.read_text(path), account)
    return f"imported {apply(ledger, rows)} of {len(rows)} rows"
