"""CSV for spreadsheets (amounts formatted, not round-trippable)."""
import csv
import io

from ledgerkit.money import format_cents

HEADER = ["id", "date", "account", "amount", "memo"]


def render(entries) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(HEADER)
    for e in entries:
        writer.writerow([e.id, e.date, e.account, format_cents(e.amount_cents), e.memo])
    return out.getvalue().rstrip("\n")
