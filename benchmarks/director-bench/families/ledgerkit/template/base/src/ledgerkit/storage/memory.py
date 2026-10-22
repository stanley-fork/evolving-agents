"""In-memory store with the jsonstore interface (used by tools and tests)."""
from ledgerkit.storage import jsonstore

_FILES = {}


def save_ledger(ledger, name: str) -> None:
    _FILES[name] = jsonstore.dumps(ledger)


def load_ledger(name: str):
    from ledgerkit.ledger import Ledger

    return jsonstore.loads(_FILES[name]) if name in _FILES else Ledger()


def clear() -> None:
    _FILES.clear()
