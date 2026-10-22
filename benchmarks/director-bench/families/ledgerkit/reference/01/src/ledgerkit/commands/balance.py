"""Balance of one account."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents, sum_cents

NAME = "balance"
EXAMPLES = ['balance cash']


def run(ledger, args):
    if len(args) != 1:
        raise LedgerError(400, "usage: balance <account>")
    total = sum_cents(e.amount_cents for e in ledger.for_account(args[0]))
    return f"balance {args[0]}: {format_cents(total)}"
