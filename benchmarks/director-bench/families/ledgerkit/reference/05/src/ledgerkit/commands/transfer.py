"""Move money between two accounts."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents, parse_amount

NAME = "transfer"
MUTATES = True
EXAMPLES = ['transfer cash bank 2.50']


def run(ledger, args):
    if len(args) != 3:
        raise LedgerError(400, "usage: transfer <from> <to> <amount>")
    source, target, text = args
    cents = parse_amount(text)
    if cents <= 0:
        raise LedgerError(400, "transfer amount must be positive")
    ledger.add(source, -cents, "transfer")
    ledger.add(target, cents, "transfer")
    return f"transferred {format_cents(cents)} from {source} to {target}"
