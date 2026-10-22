from ledgerkit.ledger import Ledger
from ledgerkit.storage import backend, csvstore, jsonstore, migrations


def test_json_roundtrip(tmp_path):
    ledger = Ledger()
    ledger.add("cash", -350, "coffee", date="2026-03-05", tags=["food"])
    path = str(tmp_path / "l.json")
    jsonstore.save_ledger(ledger, path)
    again = jsonstore.load_ledger(path)
    assert again.entries == ledger.entries


def test_csv_roundtrip():
    ledger = Ledger()
    ledger.add("cash", 5, "a, b", date="2026-01-01", tags=["x", "y"])
    assert csvstore.loads(csvstore.dumps(ledger.entries)) == ledger.entries


def test_migrations_from_v1():
    doc = migrations.upgrade([{"id": 1, "account": "cash", "amount_cents": 1, "memo": ""}])
    assert doc["schema"] == migrations.CURRENT
    assert doc["entries"][0]["tags"] == [] and doc["entries"][0]["date"] == "1970-01-01"


def test_backend_atomic_write(tmp_path):
    path = str(tmp_path / "f.txt")
    backend.write_text(path, "one")
    backend.write_text(path, "two")
    assert backend.read_text(path) == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]
