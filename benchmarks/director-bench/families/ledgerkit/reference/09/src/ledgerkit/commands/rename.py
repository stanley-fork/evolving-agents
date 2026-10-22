"""Move every entry of one account to another."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError

NAME = "rename"
MUTATES = True
EXAMPLES = ['rename cash wallet']


def run(ledger, args):
    if len(args) != 2:
        raise LedgerError(400, "usage: rename <old> <new>")
    old, new = args
    entries = ledger.for_account(old)
    if not entries:
        raise LedgerError(404, f"no entries for {old}")
    if ledger.for_account(new):
        raise LedgerError(409, f"account {new} already exists")
    for entry in entries:
        entry.account = new
    return f"renamed {len(entries)} entries from {old} to {new}"
