"""List entries."""
from ledgerkit.money import format_cents

NAME = "list"
EXAMPLES = ["list", "list cash"]


def line(entry) -> str:
    return f"{entry.id} {entry.date} {entry.account} {format_cents(entry.amount_cents)} {entry.memo}".rstrip()


def run(ledger, args):
    entries = ledger.for_account(args[0]) if args else ledger.entries
    if not entries:
        return "no entries"
    return "\n".join(line(e) for e in entries)
