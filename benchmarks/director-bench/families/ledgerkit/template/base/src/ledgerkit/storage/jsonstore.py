"""Ledger <-> JSON file."""
import json
from dataclasses import asdict

from ledgerkit.ledger import Entry, Ledger
from ledgerkit.storage import backend, migrations


def dumps(ledger: Ledger) -> str:
    doc = {"schema": migrations.CURRENT, "entries": [asdict(e) for e in ledger.entries]}
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def loads(text: str) -> Ledger:
    doc = migrations.upgrade(json.loads(text))
    return Ledger(Entry(**row) for row in doc["entries"])


def load_ledger(path: str) -> Ledger:
    return loads(backend.read_text(path)) if backend.exists(path) else Ledger()


def save_ledger(ledger: Ledger, path: str) -> None:
    backend.write_text(path, dumps(ledger))
