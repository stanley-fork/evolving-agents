import pytest

from ledgerkit.cli import dispatch


@pytest.mark.cmd("report")
def test_report_monthly(ledger_factory):
    ledger = ledger_factory(("cash", -350, "coffee", "2026-03-05"), ("cash", 1000, "x", "2026-04-01"))
    out = dispatch(ledger, ["report", "monthly"])
    assert "2026-03" in out and "-3.50" in out
