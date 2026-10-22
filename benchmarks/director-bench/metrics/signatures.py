"""Failure signatures: type + file + symbol + normalized message.

Bump SIGNATURE_VERSION whenever normalization changes; reports recompute
signatures from the raw failure data stored in tasks.jsonl.
"""
from __future__ import annotations

import re

SIGNATURE_VERSION = "2"   # 2: task-specific names masked (family.yaml signature_terms)
MAX_MSG = 200

_SUBS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<id>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(\.\d+)?\b"), "<ts>"),
    # tmp dirs: keep a trailing file name (x.py), drop random directory names
    (re.compile(r"(?:/private)?/(?:tmp|var/folders)(?:/[^\s'\":,)/]+)*/(?=[\w\-]+\.\w+\b)"), "<tmp>/"),
    (re.compile(r"(?:/private)?/(?:tmp|var/folders)(?:/[^\s'\":,)/]+)*"), "<tmp>"),
    (re.compile(r"(?<![\w.<])(?:/[\w.@+\-]+){2,}/([\w.@+\-]+)"), r"\1"),       # abs path -> basename
    (re.compile(r"\bline \d+\b"), "line <n>"),
    (re.compile(r"(?<=[\w>]):\d+(?::\d+)?\b"), ":<n>"),                           # file.py:12:3
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    (re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b"), "<hex>"),
    (re.compile(r"\b\d+(\.\d+)?\s?(ms|s|sec|seconds)\b"), "<dur>"),
    (re.compile(r"\s+"), " "),
]
_EXC = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Exit))\b|\b(LK-[EL]\d+)\b")
_PYFILE = re.compile(r"([\w./\-]+\.(?:py|sh|toml|ya?ml|json|md))\b")


def _strip_roots(text: str, roots: list[str] | None) -> str:
    for root in sorted(roots or [], key=len, reverse=True):
        if root:
            text = text.replace(root.rstrip("/") + "/", "").replace(root.rstrip("/"), ".")
    return text


def normalize_message(msg: str, roots: list[str] | None = None) -> str:
    text = _strip_roots(msg or "", roots)
    for rx, rep in _SUBS:
        text = rx.sub(rep, text)
    return text.strip()[:MAX_MSG]


def normalize_file(path: str, roots: list[str] | None = None) -> str:
    if not path or path == "-":
        return ""
    return normalize_message(path, roots).lstrip("./") if path.startswith("/") else path.strip()


def signature(kind: str, file: str, symbol: str, msg: str, roots: list[str] | None = None) -> str:
    return "|".join([kind, normalize_file(file, roots), symbol if symbol != "-" else "",
                     normalize_message(msg, roots)])


_BENCH_FAIL = re.compile(r"^BENCH-FAIL type=(\S*) file=(\S*) symbol=(\S*) msg=(.*)$")


def verifier_signatures(lines: list[str], roots: list[str] | None = None) -> list[str]:
    out = []
    for line in lines:
        m = _BENCH_FAIL.match(line.strip())
        if m:
            out.append(signature("verify:" + m.group(1), m.group(2), m.group(3), m.group(4), roots))
        else:
            out.append(signature("verify:unknown", "", "", line, roots))
    return out


def tool_error_signature(err: dict, roots: list[str] | None = None) -> str:
    """err = {tool, file, message} from a tool_result with is_error=true."""
    msg = _strip_roots(err.get("message") or "", roots)
    lines = [ln.strip() for ln in msg.splitlines() if ln.strip() and not ln.strip().startswith("Exit code")]
    last = lines[-1] if lines else msg.strip()
    m = _EXC.search(msg)
    symbol = (m.group(1) or m.group(2)) if m else ""
    file = err.get("file") or ""
    if not file:
        fm = _PYFILE.search(last) or _PYFILE.search(msg)
        file = fm.group(1) if fm else ""
    return signature("tool:" + (err.get("tool") or "?"), file, symbol, last, roots)


_CI_SOURCE = re.compile(r"^ (\S+?): (.*)$")


def ci_signatures(lines: list[str], roots: list[str] | None = None) -> list[str]:
    """CI/reviewer feedback lines -> signatures. "CI LK-R03: msg" -> symbol LK-R03;
    "CI tools/check: msg" -> symbol tools/check; "CI: msg" -> no symbol."""
    out = []
    for line in lines:
        rest = line.strip()[2:] if line.strip().startswith("CI") else line.strip()
        m = _CI_SOURCE.match(rest)
        symbol, msg = (m.group(1), m.group(2)) if m else ("", rest.lstrip(": "))
        out.append(signature("ci", "", symbol, msg, roots))
    return out


def mask_terms(sig: str, terms: list[str] | None) -> str:
    """Replace a task's own names (the command it adds, its flags) with <task>, so the same mistake made on two
    tasks has one signature. Terms come from family.yaml `signature_terms`; longest first, whole words only."""
    for term in sorted(terms or [], key=len, reverse=True):
        sig = re.sub(rf"(?<![\w-]){re.escape(term)}(?![\w-])", "<task>", sig)
    return sig
