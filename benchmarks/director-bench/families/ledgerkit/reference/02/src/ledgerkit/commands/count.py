"""Number of entries of an account."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit.errors import LedgerError

NAME = "count"
EXAMPLES = ['count cash']


def run(ledger, args):
    if len(args) != 1:
        raise LedgerError(400, "usage: count <account>")
    return str(len(ledger.for_account(args[0])))
