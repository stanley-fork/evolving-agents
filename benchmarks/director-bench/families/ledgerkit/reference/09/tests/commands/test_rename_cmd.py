# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("rename")
def test_rename(ledger_factory):
    ledger = ledger_factory(("cash", 1), ("bank", 2), ("cash", 3))
    assert dispatch(ledger, ["rename", "cash", "wallet"]) == "renamed 2 entries from cash to wallet"


@pytest.mark.cmd("rename")
def test_rename_conflict(ledger_factory):
    with pytest.raises(LedgerError) as err:
        dispatch(ledger_factory(("a", 1), ("b", 1)), ["rename", "a", "b"])
    assert err.value.code == 409
