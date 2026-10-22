"""The ledger: an ordered list of entries."""
from dataclasses import dataclass, field

from ledgerkit import clock
from ledgerkit.errors import LedgerError
from ledgerkit.utils.validate import check_account, check_date


@dataclass
class Entry:
    id: int
    date: str
    account: str
    amount_cents: int
    memo: str = ""
    tags: list = field(default_factory=list)


class Ledger:
    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def next_id(self) -> int:
        return max((e.id for e in self.entries), default=0) + 1

    def add(self, account: str, amount_cents: int, memo: str = "", date=None, tags=()) -> Entry:
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool):
            raise LedgerError(400, "amount must be integer cents")
        entry = Entry(self.next_id(), check_date(date or clock.today()), check_account(account),
                      amount_cents, memo, list(tags))
        self.entries.append(entry)
        return entry

    def for_account(self, account: str) -> list:
        return [e for e in self.entries if e.account == account]

    def accounts(self) -> list:
        return sorted({e.account for e in self.entries})

    def get(self, entry_id: int) -> Entry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise LedgerError(404, f"entry {entry_id} not found")

    def remove(self, entry_id: int) -> Entry:
        entry = self.get(entry_id)
        self.entries.remove(entry)
        return entry

    def between(self, start: str, end: str) -> list:
        return [e for e in self.entries if start <= e.date <= end]

    def find(self, predicate) -> list:
        return [e for e in self.entries if predicate(e)]
