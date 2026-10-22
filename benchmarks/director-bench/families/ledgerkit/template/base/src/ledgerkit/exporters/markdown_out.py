"""Markdown table."""
from ledgerkit.money import format_cents


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|")


def render(entries) -> str:
    lines = ["| id | date | account | amount | memo |", "|---:|------|---------|-------:|------|"]
    for e in entries:
        lines.append(f"| {e.id} | {e.date} | {_esc(e.account)} | {format_cents(e.amount_cents)} | {_esc(e.memo)} |")
    return "\n".join(lines)
