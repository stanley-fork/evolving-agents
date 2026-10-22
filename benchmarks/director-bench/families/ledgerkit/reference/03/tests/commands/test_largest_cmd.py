# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("largest")
def test_largest(ledger_factory):
    assert dispatch(ledger_factory(("cash", -5000), ("cash", 1050)), ["largest", "cash"]) == "10.50"


@pytest.mark.cmd("largest")
def test_largest_missing(ledger_factory):
    with pytest.raises(LedgerError) as err:
        dispatch(ledger_factory(), ["largest", "cash"])
    assert err.value.code == 404
