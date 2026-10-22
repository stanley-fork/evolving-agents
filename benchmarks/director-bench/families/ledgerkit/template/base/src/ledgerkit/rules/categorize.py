"""Assign a category to an entry from its memo."""
from ledgerkit.rules.matchers import AmountRange, Contains, Regex, any_of

DEFAULT_RULES = [
    ("rent", any_of(Contains("rent"), Contains("miete"))),
    ("groceries", any_of(Contains("market"), Contains("grocer"), Contains("rewe"), Contains("lidl"))),
    ("transport", any_of(Contains("uber"), Contains("train"), Contains("metro"), Contains("fuel"))),
    ("eating out", any_of(Contains("cafe"), Contains("coffee"), Contains("restaurant"), Contains("lunch"))),
    ("salary", Regex(r"\b(salary|payroll|gehalt)\b")),
    ("utilities", any_of(Contains("electric"), Contains("water"), Contains("internet"))),
    ("large", AmountRange(-10_000_000, -100_000)),
]


def categorize(entry, rules=None) -> str:
    for name, matcher in rules or DEFAULT_RULES:
        if matcher(entry):
            return name
    return "uncategorized"
