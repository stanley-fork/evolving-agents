"""Entries <-> CSV text (round-trippable, unlike the csv exporter)."""
import csv
import io

from ledgerkit.ledger import Entry

FIELDS = ["id", "date", "account", "amount_cents", "memo", "tags"]


def dumps(entries) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(FIELDS)
    for e in entries:
        writer.writerow([e.id, e.date, e.account, e.amount_cents, e.memo, "|".join(e.tags)])
    return out.getvalue()


def loads(text: str) -> list:
    rows = list(csv.DictReader(io.StringIO(text)))
    return [Entry(int(r["id"]), r["date"], r["account"], int(r["amount_cents"]), r["memo"],
                  [t for t in r["tags"].split("|") if t]) for r in rows]
