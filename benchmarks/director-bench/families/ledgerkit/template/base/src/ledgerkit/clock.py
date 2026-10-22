"""Today's date, overridable for reproducible runs."""
import datetime
import os


def today() -> str:
    return os.environ.get("LEDGERKIT_TODAY") or datetime.date.today().isoformat()
