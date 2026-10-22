"""Exporters render entries as text."""
from ledgerkit.errors import LedgerError
from ledgerkit.exporters import csv_out, html_out, json_out, ledger_out, markdown_out, qif_out, tsv_out

EXPORTERS = {
    "csv": csv_out,
    "tsv": tsv_out,
    "json": json_out,
    "ledger": ledger_out,
    "qif": qif_out,
    "md": markdown_out,
    "html": html_out,
}


def get(name: str):
    try:
        return EXPORTERS[name]
    except KeyError:
        raise LedgerError(400, f"unknown export format: {name}") from None
