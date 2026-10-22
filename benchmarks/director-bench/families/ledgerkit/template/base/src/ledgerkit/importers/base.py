"""Common importer machinery."""
import csv
from dataclasses import dataclass

from ledgerkit.utils.ids import entry_fingerprint


@dataclass
class ImportedRow:
    date: str
    account: str
    amount_cents: int
    memo: str
    source: str
    line: int

    @property
    def fingerprint(self) -> str:
        return entry_fingerprint(self.date, self.account, self.amount_cents, self.memo)


class Importer:
    name = "base"

    def parse(self, text: str, account: str) -> list:
        raise NotImplementedError

    def detect(self, text: str) -> bool:
        return False


def split_line(line: str, delimiter: str) -> list:
    return next(csv.reader([line], delimiter=delimiter))


def apply(ledger, rows) -> int:
    """Add rows not already present (by fingerprint); return how many were added."""
    seen = {entry_fingerprint(e.date, e.account, e.amount_cents, e.memo) for e in ledger.entries}
    added = 0
    for row in rows:
        if row.fingerprint in seen:
            continue
        ledger.add(row.account, row.amount_cents, row.memo, date=row.date)
        seen.add(row.fingerprint)
        added += 1
    return added
