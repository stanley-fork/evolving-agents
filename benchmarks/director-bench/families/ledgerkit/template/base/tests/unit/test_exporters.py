import json

import pytest

from ledgerkit import exporters
from ledgerkit.errors import LedgerError
from ledgerkit.ledger import Ledger


@pytest.fixture
def entries():
    ledger = Ledger()
    ledger.add("cash", -350, "coffee | cake", date="2026-03-05")
    ledger.add("bank", 200000, "salary", date="2026-03-31")
    return ledger.entries


@pytest.mark.parametrize("fmt", sorted(exporters.EXPORTERS))
def test_every_exporter_mentions_amounts(fmt, entries):
    out = exporters.get(fmt).render(entries)
    assert "-3.50" in out and "2000.00" in out


def test_json_shape(entries):
    assert json.loads(exporters.get("json").render(entries))[1]["memo"] == "salary"


def test_qif_dates(entries):
    assert "D03/05/2026" in exporters.get("qif").render(entries)


def test_unknown():
    with pytest.raises(LedgerError):
        exporters.get("pdf")
