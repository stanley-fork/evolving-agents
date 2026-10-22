# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("count")
def test_count(ledger_factory):
    ledger = ledger_factory(("cash", 1), ("cash", 2), ("bank", 3))
    assert dispatch(ledger, ["count", "cash"]) == "2"
    assert dispatch(ledger, ["count", "none"]) == "0"
    with pytest.raises(LedgerError):
        dispatch(ledger, ["count"])
