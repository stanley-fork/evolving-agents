"""Report plumbing: each report builds rows, base renders them."""
from ledgerkit.money import format_cents
from ledgerkit.utils.iterutils import group_by
from ledgerkit.utils.tables import format_table


class Report:
    name = "base"
    title = ""
    columns = ("key", "count", "total")

    def key(self, entry) -> str:
        raise NotImplementedError

    def entries(self, ledger) -> list:
        return list(ledger.entries)

    def build(self, ledger) -> list:
        groups = group_by(self.entries(ledger), self.key)
        rows = []
        for key in sorted(groups):
            items = groups[key]
            rows.append((key, len(items), sum(e.amount_cents for e in items)))
        return rows

    def render(self, ledger) -> str:
        rows = [(k, str(n), format_cents(t)) for k, n, t in self.build(ledger)]
        body = format_table(self.columns, rows)
        return f"{self.title}\n\n{body}" if self.title else body
