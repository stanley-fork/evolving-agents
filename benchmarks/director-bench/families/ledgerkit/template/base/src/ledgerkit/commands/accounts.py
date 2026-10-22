"""Accounts in use."""
NAME = "accounts"
EXAMPLES = ["accounts"]


def run(ledger, args):
    names = ledger.accounts()
    if not names:
        return "no accounts"
    return "\n".join(f"{name} ({len(ledger.for_account(name))})" for name in names)
