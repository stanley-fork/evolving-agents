# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("list")
def test_list_memo_filter(ledger_factory):
    ledger = ledger_factory(("cash", 1050, "Rent May"), ("bank", 5, "food"), ("bank", 7, "rent fee"))
    assert dispatch(ledger, ["list", "--memo", "rent"]) == "1 2026-01-15 cash 10.50 Rent May\n3 2026-01-15 bank 0.07 rent fee"
    assert dispatch(ledger, ["list", "bank", "--memo", "RENT"]) == "3 2026-01-15 bank 0.07 rent fee"
    assert dispatch(ledger, ["list", "--memo", "zzz"]) == "no entries"
    assert dispatch(ledger, ["list", "--limit", "1"]) == "3 2026-01-15 bank 0.07 rent fee"
    for bad in (["list", "--memo"], ["list", "--limit", "0"], ["list", "--limit", "x"]):
        with pytest.raises(LedgerError):
            dispatch(ledger, bad)
