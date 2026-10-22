import pytest

from ledgerkit.errors import LedgerError
from ledgerkit.ledger import Ledger


def test_ids_and_lookup():
    ledger = Ledger()
    a = ledger.add("cash", 100, "a", date="2026-01-01")
    b = ledger.add("bank", 200, "b", date="2026-02-01")
    assert (a.id, b.id) == (1, 2)
    assert ledger.get(2) is b
    assert ledger.accounts() == ["bank", "cash"]
    assert ledger.between("2026-01-15", "2026-03-01") == [b]


def test_errors():
    ledger = Ledger()
    with pytest.raises(LedgerError) as err:
        ledger.get(9)
    assert str(err.value) == "LK-E404: entry 9 not found"
    with pytest.raises(LedgerError):
        ledger.add("Bad Name", 1, date="2026-01-01")
    with pytest.raises(LedgerError):
        ledger.add("cash", 1.5, date="2026-01-01")
