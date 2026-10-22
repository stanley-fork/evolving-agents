import pytest

from ledgerkit.cli import dispatch


@pytest.mark.cmd("accounts")
def test_accounts(ledger_factory):
    ledger = ledger_factory(("cash", 1), ("bank", 2), ("cash", 3))
    assert dispatch(ledger, ["accounts"]) == "bank (1)\ncash (2)"
