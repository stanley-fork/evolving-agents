"""The only place that touches the filesystem."""
import os
import tempfile


def exists(path: str) -> bool:
    return os.path.exists(path)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write_text(path: str, text: str) -> None:
    """Atomic write: temp file in the same directory, then rename."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".lk-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def remove(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)
