"""Advisory lock file so two ledgerkit processes do not write the same ledger."""
import os
from contextlib import contextmanager

from ledgerkit.errors import LedgerError


@contextmanager
def locked(path: str):
    lock = path + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LedgerError(409, f"{path} is locked by another process") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        os.unlink(lock)
