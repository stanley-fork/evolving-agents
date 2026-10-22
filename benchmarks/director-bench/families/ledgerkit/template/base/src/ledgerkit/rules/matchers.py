"""Entry matchers."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Contains:
    text: str

    def __call__(self, entry) -> bool:
        return self.text.lower() in entry.memo.lower()


@dataclass(frozen=True)
class Regex:
    pattern: str

    def __call__(self, entry) -> bool:
        return re.search(self.pattern, entry.memo, re.I) is not None


@dataclass(frozen=True)
class AmountRange:
    low: int
    high: int

    def __call__(self, entry) -> bool:
        return self.low <= entry.amount_cents <= self.high


@dataclass(frozen=True)
class AccountIs:
    account: str

    def __call__(self, entry) -> bool:
        return entry.account == self.account


def all_of(*matchers):
    return lambda entry: all(m(entry) for m in matchers)


def any_of(*matchers):
    return lambda entry: any(m(entry) for m in matchers)
