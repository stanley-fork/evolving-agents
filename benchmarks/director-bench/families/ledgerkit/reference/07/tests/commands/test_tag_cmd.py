# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("tag")
def test_tag(ledger_factory):
    ledger = ledger_factory(("cash", 1), ("cash", 2))
    assert dispatch(ledger, ["tag", "2", "food"]) == "tagged 2 food"
    assert dispatch(ledger, ["tag", "2", "food"]) == "tagged 2 food"
    assert ledger.entries[1].tags == ["food"]


@pytest.mark.cmd("tag")
def test_tag_errors(ledger_factory):
    for argv, code in ((["tag", "9", "x"], 404), (["tag", "a", "x"], 400), (["tag", "1", "Bad Tag"], 400)):
        with pytest.raises(LedgerError) as err:
            dispatch(ledger_factory(("cash", 1)), argv)
        assert err.value.code == code
