"""Settings from the environment."""
import os

DEFAULTS = {
    "LEDGERKIT_FILE": "ledger.json",
    "LEDGERKIT_CURRENCY": "EUR",
    "LEDGERKIT_BUDGET_FILE": "budget.json",
}


def get(name: str) -> str:
    return os.environ.get(name) or DEFAULTS[name]


def currency() -> str:
    return get("LEDGERKIT_CURRENCY")
