"""Tag an entry."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
import re

from ledgerkit.errors import LedgerError

NAME = "tag"
MUTATES = True
EXAMPLES = ['tag 12 groceries']


TAG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def run(ledger, args):
    if len(args) != 2 or not args[0].isdigit():
        raise LedgerError(400, "usage: tag <entry-id> <tag>")
    if not TAG.match(args[1]):
        raise LedgerError(400, f"invalid tag: {args[1]}")
    entry = ledger.get(int(args[0]))
    if args[1] not in entry.tags:
        entry.tags.append(args[1])
    return f"tagged {entry.id} {args[1]}"
