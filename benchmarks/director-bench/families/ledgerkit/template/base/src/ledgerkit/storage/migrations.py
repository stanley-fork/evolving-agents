"""Upgrade stored ledgers to the current schema."""
from ledgerkit.errors import LedgerError

CURRENT = 3


def _v1_to_v2(rows):
    return [{**row, "tags": row.get("tags", [])} for row in rows]


def _v2_to_v3(rows):
    return [{**row, "date": row.get("date", "1970-01-01")} for row in rows]


STEPS = {1: _v1_to_v2, 2: _v2_to_v3}


def upgrade(doc):
    """doc = {"schema": n, "entries": [...]} (or a bare v1 list)."""
    if isinstance(doc, list):
        doc = {"schema": 1, "entries": doc}
    schema, rows = doc.get("schema", 1), doc.get("entries", [])
    if schema > CURRENT:
        raise LedgerError(422, f"ledger schema {schema} is newer than this ledgerkit ({CURRENT})")
    while schema < CURRENT:
        rows = STEPS[schema](rows)
        schema += 1
    return {"schema": CURRENT, "entries": rows}
