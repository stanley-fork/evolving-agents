# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("balance")
def test_balance(ledger_factory):
    ledger = ledger_factory(("cash", 1050), ("cash", -25), ("bank", 5))
    assert dispatch(ledger, ["balance", "cash"]) == "balance cash: 10.25"
    assert dispatch(ledger, ["balance", "nobody"]) == "balance nobody: 0.00"


@pytest.mark.cmd("balance")
def test_balance_needs_account(ledger_factory):
    with pytest.raises(LedgerError) as err:
        dispatch(ledger_factory(), ["balance"])
    assert err.value.code == 400
