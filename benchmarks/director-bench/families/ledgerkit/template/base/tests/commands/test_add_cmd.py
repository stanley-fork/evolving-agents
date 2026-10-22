import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("add")
def test_add_records_cents(ledger_factory):
    ledger = ledger_factory()
    assert dispatch(ledger, ["add", "cash", "12.5", "lunch", "--date", "2026-03-01"]) == "added 1 cash 12.50"
    assert ledger.entries[0].amount_cents == 1250
    assert ledger.entries[0].date == "2026-03-01"


@pytest.mark.cmd("add")
def test_add_requires_amount(ledger_factory):
    with pytest.raises(LedgerError) as err:
        dispatch(ledger_factory(), ["add", "cash"])
    assert err.value.code == 400
