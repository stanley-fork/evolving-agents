"""Minimal HTML table."""
import html

from ledgerkit.money import format_cents


def render(entries) -> str:
    rows = "".join(
        f"<tr><td>{e.id}</td><td>{e.date}</td><td>{html.escape(e.account)}</td>"
        f"<td class=\"num\">{format_cents(e.amount_cents)}</td><td>{html.escape(e.memo)}</td></tr>"
        for e in entries
    )
    return ("<table><thead><tr><th>id</th><th>date</th><th>account</th><th>amount</th><th>memo</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")
