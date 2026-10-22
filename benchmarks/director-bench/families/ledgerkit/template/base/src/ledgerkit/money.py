"""Amounts are integer cents everywhere."""
import re

from ledgerkit.errors import LedgerError

_AMOUNT = re.compile(r"^([+-]?)(\d+)(?:\.(\d{1,2}))?$")


def parse_amount(text: str) -> int:
    """Parse "12.34", "-5" or "0.5" into integer cents."""
    match = _AMOUNT.match(text.strip())
    if not match:
        raise LedgerError(400, f"invalid amount: {text!r}")
    sign, whole, frac = match.groups()
    cents = int(whole) * 100 + int((frac or "0").ljust(2, "0"))
    return -cents if sign == "-" else cents


def format_cents(cents: int) -> str:
    """12345 -> "123.45", -5 -> "-0.05"."""
    sign = "-" if cents < 0 else ""
    value = abs(cents)
    return f"{sign}{value // 100}.{value % 100:02d}"


def sum_cents(values) -> int:
    total = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise LedgerError(400, f"not an amount in cents: {value!r}")
        total += value
    return total


def split_cents(total: int, parts: int) -> list:
    """Split total into `parts` amounts that differ by at most one cent."""
    if parts <= 0:
        raise LedgerError(400, "parts must be positive")
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
