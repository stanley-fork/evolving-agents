"""Text helpers."""
import re
import unicodedata

_SPACE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def clean_memo(text: str) -> str:
    """Statement memos: collapse whitespace, drop surrounding quotes and card noise."""
    text = normalize_space(text.strip().strip('"'))
    text = re.sub(r"\bCARD \d{4}\b", "", text)
    return normalize_space(text)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


def first_word(text: str) -> str:
    parts = normalize_space(text).split(" ")
    return parts[0].lower() if parts and parts[0] else ""
