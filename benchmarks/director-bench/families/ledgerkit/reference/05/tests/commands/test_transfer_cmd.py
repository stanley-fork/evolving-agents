# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("transfer")
def test_transfer(ledger_factory):
    ledger = ledger_factory(("cash", 1000))
    assert dispatch(ledger, ["transfer", "cash", "bank", "2.50"]) == "transferred 2.50 from cash to bank"
    assert [(e.account, e.amount_cents) for e in ledger.entries[1:]] == [("cash", -250), ("bank", 250)]


@pytest.mark.cmd("transfer")
def test_transfer_rejects_non_positive(ledger_factory):
    with pytest.raises(LedgerError):
        dispatch(ledger_factory(), ["transfer", "a", "b", "0"])
