import pytest

from ledgerkit.errors import LedgerError
from ledgerkit.utils import args, dates, iterutils, tables, text


def test_dates():
    assert dates.to_iso("05.03.2026", "%d.%m.%Y") == "2026-03-05"
    assert list(dates.iter_months("2025-11-30", "2026-02-01")) == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert dates.quarter_of("2026-08-01") == "2026-Q3"
    assert dates.add_days("2026-02-28", 1) == "2026-03-01"


def test_text():
    assert text.clean_memo('  "COFFEE   SHOP CARD 1234" ') == "COFFEE SHOP"
    assert text.slugify("Café Münster") == "cafe-munster"
    assert text.first_word("  Rent May") == "rent"


def test_args():
    assert args.pop_option(["a", "--x", "1", "b"], "--x") == (["a", "b"], "1")
    assert args.pop_flag(["--json", "a"], "--json") == (["a"], True)
    with pytest.raises(LedgerError):
        args.pop_option(["--x"], "--x")


def test_iterutils_and_tables():
    assert list(iterutils.chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert iterutils.running_total([1, 2, 3]) == [1, 3, 6]
    out = tables.format_table(("k", "v"), [("a", "1"), ("bb", "22")])
    assert out.splitlines()[0] == "k    v"
