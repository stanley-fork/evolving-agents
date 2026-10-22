from ledgerkit.ledger import Ledger
from ledgerkit.rules import categorize, tagging
from ledgerkit.rules.matchers import AccountIs, AmountRange, all_of


def test_categorize_defaults():
    ledger = Ledger()
    rent = ledger.add("bank", -90000, "Rent March", date="2026-03-01")
    coffee = ledger.add("cash", -350, "coffee", date="2026-03-02")
    other = ledger.add("cash", -1, "zzz", date="2026-03-02")
    assert categorize.categorize(rent) == "rent"
    assert categorize.categorize(coffee) == "eating out"
    assert categorize.categorize(other) == "uncategorized"
    assert tagging.auto_tag(ledger) == 3
    assert tagging.with_tag(ledger, "rent") == [rent]


def test_matchers():
    ledger = Ledger()
    e = ledger.add("cash", -500, "x", date="2026-01-01")
    assert all_of(AccountIs("cash"), AmountRange(-1000, 0))(e)
