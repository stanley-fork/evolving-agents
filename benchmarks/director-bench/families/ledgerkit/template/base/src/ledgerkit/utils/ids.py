"""Identifiers."""
import hashlib


def short_hash(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:10]


def entry_fingerprint(date: str, account: str, amount_cents: int, memo: str) -> str:
    """Used by importers to skip rows that were already imported."""
    return short_hash(date, account, amount_cents, memo.lower())
