"""Input validation."""
import re

from ledgerkit.errors import LedgerError

_ACCOUNT = re.compile(r"^[a-z][a-z0-9_:-]{0,39}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_account(name: str) -> str:
    if not _ACCOUNT.match(name or ""):
        raise LedgerError(400, f"invalid account name: {name!r}")
    return name


def check_date(text: str) -> str:
    if not _DATE.match(text or ""):
        raise LedgerError(400, f"invalid date: {text!r}")
    return text


def check_positive(cents: int, what: str = "amount") -> int:
    if cents <= 0:
        raise LedgerError(400, f"{what} must be positive")
    return cents
