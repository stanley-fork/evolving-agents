"""Money in, money out and net per month."""
from ledgerkit.money import format_cents
from ledgerkit.utils.dates import month_of
from ledgerkit.utils.iterutils import group_by
from ledgerkit.utils.tables import format_table


class CashflowReport:
    name = "cashflow"
    title = "Cash flow"

    def build(self, ledger):
        rows = []
        for month, items in sorted(group_by(ledger.entries, lambda e: month_of(e.date)).items()):
            inflow = sum(e.amount_cents for e in items if e.amount_cents > 0)
            outflow = sum(e.amount_cents for e in items if e.amount_cents < 0)
            rows.append((month, inflow, outflow, inflow + outflow))
        return rows

    def render(self, ledger) -> str:
        rows = [(m, format_cents(i), format_cents(o), format_cents(n)) for m, i, o, n in self.build(ledger)]
        return f"{self.title}\n\n" + format_table(("month", "in", "out", "net"), rows)


REPORT = CashflowReport()
