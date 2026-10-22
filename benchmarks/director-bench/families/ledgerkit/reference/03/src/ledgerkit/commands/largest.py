"""Largest single entry of an account."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents

NAME = "largest"
EXAMPLES = ['largest cash']


def run(ledger, args):
    if len(args) != 1:
        raise LedgerError(400, "usage: largest <account>")
    entries = ledger.for_account(args[0])
    if not entries:
        raise LedgerError(404, f"no entries for {args[0]}")
    return format_cents(max(e.amount_cents for e in entries))
