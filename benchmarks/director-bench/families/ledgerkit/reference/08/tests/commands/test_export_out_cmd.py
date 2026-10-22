# LKORACLE-7f3a9c reference solution (never shown to the agent)
import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("export")
def test_export_out(ledger_factory, tmp_path):
    ledger = ledger_factory(("cash", -350, "coffee"), ("bank", 5))
    path = str(tmp_path / "x.csv")
    assert dispatch(ledger, ["export", "--out", path]) == f"wrote 2 entries to {path}"
    assert (tmp_path / "x.csv").read_text() == dispatch(ledger, ["export"])
    with pytest.raises(LedgerError):
        dispatch(ledger, ["export", "--out"])
