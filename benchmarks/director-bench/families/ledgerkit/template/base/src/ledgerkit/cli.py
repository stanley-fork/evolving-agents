"""Command-line entry point."""
import os
import sys

from ledgerkit.errors import LedgerError
from ledgerkit.storage import jsonstore


def command_table() -> dict:
    from ledgerkit.commands._registry import COMMANDS

    return dict(COMMANDS)


def lookup(name: str):
    table = command_table()
    if name not in table:
        raise LedgerError(404, f"unknown command: {name}")
    return table[name]


def dispatch(ledger, argv) -> str:
    if not argv:
        raise LedgerError(400, "no command given")
    return lookup(argv[0])(ledger, list(argv[1:]))


def mutates(name: str) -> bool:
    """Only commands whose module sets MUTATES = True get the ledger saved."""
    return bool(getattr(sys.modules[lookup(name).__module__], "MUTATES", False))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = os.environ.get("LEDGERKIT_FILE", "ledger.json")
    ledger = jsonstore.load_ledger(path)
    try:
        out = dispatch(ledger, argv)
    except LedgerError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if mutates(argv[0]):
        jsonstore.save_ledger(ledger, path)
    sys.stdout.write(out + "\n")
    return 0
