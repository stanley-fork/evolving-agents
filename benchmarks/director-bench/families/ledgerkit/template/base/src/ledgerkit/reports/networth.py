"""Running balance across all accounts at the end of each month."""
from ledgerkit.money import format_cents
from ledgerkit.utils.dates import iter_months, month_of
from ledgerkit.utils.tables import format_table


class NetworthReport:
    name = "networth"
    title = "Net worth"

    def build(self, ledger):
        if not ledger.entries:
            return []
        dates = sorted(e.date for e in ledger.entries)
        per_month = {}
        for e in ledger.entries:
            per_month[month_of(e.date)] = per_month.get(month_of(e.date), 0) + e.amount_cents
        rows, running = [], 0
        for month in iter_months(dates[0], dates[-1]):
            running += per_month.get(month, 0)
            rows.append((month, running))
        return rows

    def render(self, ledger) -> str:
        rows = [(m, format_cents(v)) for m, v in self.build(ledger)]
        return f"{self.title}\n\n" + format_table(("month", "balance"), rows)


REPORT = NetworthReport()
