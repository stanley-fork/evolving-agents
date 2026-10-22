"""Tag entries in bulk."""
from ledgerkit.rules.categorize import categorize


def auto_tag(ledger, rules=None) -> int:
    """Add the category as a tag to untagged entries; return how many changed."""
    changed = 0
    for entry in ledger.entries:
        if not entry.tags:
            entry.tags.append(categorize(entry, rules))
            changed += 1
    return changed


def with_tag(ledger, tag: str) -> list:
    return [e for e in ledger.entries if tag in e.tags]
