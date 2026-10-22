import pytest

from ledgerkit.cli import dispatch


@pytest.mark.cmd("list")
def test_list_one_account(ledger_factory):
    ledger = ledger_factory(("cash", 1050, "lunch"), ("bank", 99))
    assert dispatch(ledger, ["list", "cash"]) == "1 2026-01-15 cash 10.50 lunch"


@pytest.mark.cmd("list")
def test_list_empty(ledger_factory):
    assert dispatch(ledger_factory(), ["list"]) == "no entries"
