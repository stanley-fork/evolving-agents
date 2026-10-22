"""List entries."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents
from ledgerkit.utils.args import pop_option

NAME = "list"
EXAMPLES = ["list", "list cash", "list --memo rent", "list bank --limit 5"]


def line(entry) -> str:
    return f"{entry.id} {entry.date} {entry.account} {format_cents(entry.amount_cents)} {entry.memo}".rstrip()


def run(ledger, args):
    args, memo = pop_option(args, "--memo")
    args, limit = pop_option(args, "--limit")
    entries = ledger.for_account(args[0]) if args else ledger.entries
    if memo is not None:
        entries = [e for e in entries if memo.lower() in e.memo.lower()]
    if limit is not None:
        if not limit.isdigit() or int(limit) == 0:
            raise LedgerError(400, "--limit must be a positive integer")
        entries = entries[-int(limit):]
    if not entries:
        return "no entries"
    return "\n".join(line(e) for e in entries)
