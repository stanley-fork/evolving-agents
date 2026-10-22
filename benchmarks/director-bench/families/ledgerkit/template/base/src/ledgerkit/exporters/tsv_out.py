"""Tab-separated values."""
from ledgerkit.money import format_cents


def _cell(text: str) -> str:
    return str(text).replace("\t", " ").replace("\n", " ")


def render(entries) -> str:
    lines = ["id\tdate\taccount\tamount\tmemo"]
    for e in entries:
        lines.append("\t".join(_cell(v) for v in (e.id, e.date, e.account, format_cents(e.amount_cents), e.memo)))
    return "\n".join(lines)
