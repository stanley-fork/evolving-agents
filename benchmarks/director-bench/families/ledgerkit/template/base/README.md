# ledgerkit

Personal finance ledger: import bank statements, keep a ledger of entries,
categorize, report and export.

    pip install -e .
    ledgerkit import csv_bank statement.csv bank
    ledgerkit report monthly
    ledgerkit export --format ledger

## Layout

- `src/ledgerkit/` — the package (core, `storage/`, `importers/`, `exporters/`,
  `reports/`, `rules/`, `utils/`, `commands/`)
- `tests/` — test suite
- `tools/` — maintenance scripts

## Development

    pip install -e .
    pytest
