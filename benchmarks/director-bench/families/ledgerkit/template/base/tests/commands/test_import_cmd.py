import pytest

from ledgerkit.cli import dispatch


@pytest.mark.cmd("import")
def test_import_csv_bank(ledger_factory, tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("date,memo,amount\n2026-03-05,Coffee,-3.50\n2026-03-06,Salary,2000.00\n")
    ledger = ledger_factory()
    assert dispatch(ledger, ["import", "csv_bank", str(path), "bank"]) == "imported 2 of 2 rows"
    assert dispatch(ledger, ["import", "csv_bank", str(path), "bank"]) == "imported 0 of 2 rows"
    assert [e.amount_cents for e in ledger.entries] == [-350, 200000]
