import pytest

from ledgerkit.errors import LedgerError
from ledgerkit.money import format_cents, parse_amount, split_cents, sum_cents


@pytest.mark.parametrize("text,cents", [("12.34", 1234), ("-5", -500), ("0.5", 50), ("+3.07", 307)])
def test_parse(text, cents):
    assert parse_amount(text) == cents


@pytest.mark.parametrize("text", ["1.234", "abc", "", "1,50"])
def test_parse_rejects(text):
    with pytest.raises(LedgerError):
        parse_amount(text)


def test_format():
    assert format_cents(-5) == "-0.05"
    assert format_cents(123456) == "1234.56"
    assert format_cents(0) == "0.00"


def test_sum_and_split():
    assert sum_cents([1, 2, 3]) == 6
    assert split_cents(100, 3) == [34, 33, 33]
    with pytest.raises(LedgerError):
        sum_cents([1, 2.5])
