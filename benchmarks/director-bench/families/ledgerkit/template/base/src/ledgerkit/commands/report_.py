"""Print a report."""
from ledgerkit import reports
from ledgerkit.utils.args import expect_count

NAME = "report"
EXAMPLES = ["report monthly", "report category"]


def run(ledger, args):
    (kind,) = expect_count(args, 1, "report <kind>")
    return reports.get(kind).render(ledger)
