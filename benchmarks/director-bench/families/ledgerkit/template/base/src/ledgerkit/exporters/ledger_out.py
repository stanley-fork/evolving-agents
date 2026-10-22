"""Plain-text accounting (ledger-cli style) postings."""
from ledgerkit.utils import config
from ledgerkit.money import format_cents


def render(entries) -> str:
    blocks = []
    for e in entries:
        payee = e.memo or "(no memo)"
        blocks.append(
            f"{e.date.replace('-', '/')} {payee}\n"
            f"    assets:{e.account}  {format_cents(e.amount_cents)} {config.currency()}\n"
            f"    equity:unsorted"
        )
    return "\n\n".join(blocks) if blocks else ""
