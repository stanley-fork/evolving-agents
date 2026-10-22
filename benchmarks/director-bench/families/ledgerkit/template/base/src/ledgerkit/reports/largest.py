"""The ten largest expenses."""
from ledgerkit.money import format_cents
from ledgerkit.utils.tables import format_table
from ledgerkit.utils.text import truncate

LIMIT = 10


class LargestExpensesReport:
    name = "top-expenses"
    title = "Largest expenses"

    def build(self, ledger):
        spent = [e for e in ledger.entries if e.amount_cents < 0]
        return sorted(spent, key=lambda e: (e.amount_cents, e.date, e.id))[:LIMIT]

    def render(self, ledger) -> str:
        rows = [(e.date, e.account, truncate(e.memo, 30), format_cents(e.amount_cents)) for e in self.build(ledger)]
        return f"{self.title}\n\n" + format_table(("date", "account", "memo", "amount"), rows, ["<", "<", "<", ">"])


REPORT = LargestExpensesReport()
