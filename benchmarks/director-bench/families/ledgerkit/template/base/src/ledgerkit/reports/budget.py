"""Spending per category against monthly limits."""
import json

from ledgerkit.utils import config
from ledgerkit.money import format_cents
from ledgerkit.rules.categorize import categorize
from ledgerkit.storage import backend
from ledgerkit.utils.tables import format_table

DEFAULT_LIMITS = {"groceries": 40000, "eating out": 15000, "transport": 10000}


def load_limits() -> dict:
    path = config.get("LEDGERKIT_BUDGET_FILE")
    if backend.exists(path):
        return {k: int(v) for k, v in json.loads(backend.read_text(path)).items()}
    return dict(DEFAULT_LIMITS)


class BudgetReport:
    name = "budget"
    title = "Budget"

    def build(self, ledger, limits=None):
        limits = limits if limits is not None else load_limits()
        spent = {}
        for e in ledger.entries:
            if e.amount_cents < 0:
                cat = categorize(e)
                spent[cat] = spent.get(cat, 0) - e.amount_cents
        return [(cat, spent.get(cat, 0), limit, limit - spent.get(cat, 0)) for cat, limit in sorted(limits.items())]

    def render(self, ledger) -> str:
        rows = [(c, format_cents(s), format_cents(lim), format_cents(left)) for c, s, lim, left in self.build(ledger)]
        return f"{self.title}\n\n" + format_table(("category", "spent", "limit", "left"), rows)


REPORT = BudgetReport()
