"""Quicken Interchange Format."""
from ledgerkit.money import format_cents


def _qif_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{month}/{day}/{year}"


def render(entries) -> str:
    lines = ["!Type:Bank"]
    for e in entries:
        lines += [f"D{_qif_date(e.date)}", f"T{format_cents(e.amount_cents)}", f"P{e.memo}", f"L{e.account}", "^"]
    return "\n".join(lines)
