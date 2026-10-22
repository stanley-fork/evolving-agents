"""Record an entry."""
from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents, parse_amount
from ledgerkit.utils.args import pop_option

NAME = "add"
MUTATES = True
EXAMPLES = ["add cash 12.50 lunch", "add bank -40 rent --date 2026-03-01"]


def run(ledger, args):
    args, date = pop_option(args, "--date")
    if len(args) < 2:
        raise LedgerError(400, "usage: add <account> <amount> [memo...] [--date YYYY-MM-DD]")
    entry = ledger.add(args[0], parse_amount(args[1]), " ".join(args[2:]), date=date)
    return f"added {entry.id} {entry.account} {format_cents(entry.amount_cents)}"
