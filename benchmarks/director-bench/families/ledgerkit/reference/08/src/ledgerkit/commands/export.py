"""Export the ledger."""
# LKORACLE-7f3a9c reference solution (never shown to the agent)
from ledgerkit import exporters
from ledgerkit.storage import backend
from ledgerkit.utils.args import pop_option

NAME = "export"
EXAMPLES = ["export --format csv", "export --format ledger", "export --format csv --out ledger.csv"]


def run(ledger, args):
    args, fmt = pop_option(args, "--format")
    args, out = pop_option(args, "--out")
    text = exporters.get(fmt or "csv").render(ledger.entries)
    if out is None:
        return text
    backend.write_text(out, text)
    return f"wrote {len(ledger.entries)} entries to {out}"
