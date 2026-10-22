import json

import pytest

from ledgerkit.cli import dispatch
from ledgerkit.errors import LedgerError


@pytest.mark.cmd("export")
def test_export_csv_default(ledger_factory):
    out = dispatch(ledger_factory(("cash", -350, "coffee")), ["export"])
    assert out.splitlines() == ["id,date,account,amount,memo", "1,2026-01-15,cash,-3.50,coffee"]


@pytest.mark.cmd("export")
def test_export_json(ledger_factory):
    rows = json.loads(dispatch(ledger_factory(("cash", 5)), ["export", "--format", "json"]))
    assert rows[0]["amount"] == "0.05"


@pytest.mark.cmd("export")
def test_export_unknown_format(ledger_factory):
    with pytest.raises(LedgerError) as err:
        dispatch(ledger_factory(), ["export", "--format", "xls"])
    assert err.value.code == 400
