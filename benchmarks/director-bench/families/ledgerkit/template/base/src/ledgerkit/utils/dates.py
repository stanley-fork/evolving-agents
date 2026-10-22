"""Date helpers. Dates are ISO strings (YYYY-MM-DD) inside ledgerkit."""
import datetime

from ledgerkit.errors import LedgerError


def parse_iso(text: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise LedgerError(400, f"invalid date: {text!r}") from None


def to_iso(text: str, fmt: str) -> str:
    """Convert a statement date in `fmt` (strptime syntax) to ISO."""
    try:
        return datetime.datetime.strptime(text.strip(), fmt).date().isoformat()
    except ValueError:
        raise LedgerError(422, f"date {text!r} does not match {fmt}") from None


def month_of(iso: str) -> str:
    return iso[:7]


def year_of(iso: str) -> str:
    return iso[:4]


def add_days(iso: str, days: int) -> str:
    return (parse_iso(iso) + datetime.timedelta(days=days)).isoformat()


def iter_months(start: str, end: str):
    """Yield "YYYY-MM" from start's month to end's month inclusive."""
    year, month = int(start[:4]), int(start[5:7])
    stop = (int(end[:4]), int(end[5:7]))
    while (year, month) <= stop:
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year, month = year + 1, 1


def quarter_of(iso: str) -> str:
    return f"{iso[:4]}-Q{(int(iso[5:7]) - 1) // 3 + 1}"
