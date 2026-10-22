"""JSON array of entries with formatted amounts."""
import json

from ledgerkit.money import format_cents


def render(entries) -> str:
    rows = [{"id": e.id, "date": e.date, "account": e.account, "amount": format_cents(e.amount_cents),
             "memo": e.memo, "tags": list(e.tags)} for e in entries]
    return json.dumps(rows, indent=2, sort_keys=True)
