"""Mean entry amount, floored to a cent."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.money import format_cents

NAME = "average"
EXAMPLES = ['average', 'average cash']


def run(ledger, args):
    entries = ledger.for_account(args[0]) if args else ledger.entries
    if not entries:
        return format_cents(0)
    return format_cents(sum(e.amount_cents for e in entries) // len(entries))
