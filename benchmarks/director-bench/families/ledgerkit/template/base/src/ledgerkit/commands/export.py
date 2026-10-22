"""Export the ledger."""
from ledgerkit import exporters
from ledgerkit.utils.args import pop_option

NAME = "export"
EXAMPLES = ["export --format csv", "export --format ledger"]


def run(ledger, args):
    args, fmt = pop_option(args, "--format")
    return exporters.get(fmt or "csv").render(ledger.entries)
