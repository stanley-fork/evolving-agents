# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("average")
def test_average(ledger_factory):
    assert dispatch(ledger_factory(("c", 100), ("c", 101)), ["average", "c"]) == "1.00"
    assert dispatch(ledger_factory(("c", -1), ("c", -2)), ["average", "c"]) == "-0.02"
    assert dispatch(ledger_factory(("a", 1), ("b", 4)), ["average"]) == "0.02"
    assert dispatch(ledger_factory(), ["average", "c"]) == "0.00"
