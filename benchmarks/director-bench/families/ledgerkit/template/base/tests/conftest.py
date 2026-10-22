import pytest

from ledgerkit.ledger import Ledger


@pytest.fixture
def ledger_factory():
    """ledger_factory(("cash", 1050, "lunch"), ("bank", -200, "rent", "2026-02-01")) -> Ledger"""

    def make(*rows):
        ledger = Ledger()
        for account, cents, *rest in rows:
            memo = rest[0] if rest else ""
            date = rest[1] if len(rest) > 1 else "2026-01-15"
            ledger.add(account, cents, memo, date=date)
        return ledger

    return make
