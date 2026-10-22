"""Errors raised by ledgerkit."""

ERROR_CODES = {
    400: "bad input",
    404: "not found",
    409: "conflict",
    422: "unprocessable",
}


class LedgerError(Exception):
    """User-facing error rendered as "LK-E<code>: <message>"."""

    def __init__(self, code: int, message: str):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code}")
        self.code = code
        self.message = message
        super().__init__(f"LK-E{code}: {message}")


class ImportFormatError(LedgerError):
    """A statement line that an importer cannot parse."""

    def __init__(self, importer: str, line: int, detail: str):
        super().__init__(422, f"{importer} line {line}: {detail}")
        self.importer = importer
        self.line = line
