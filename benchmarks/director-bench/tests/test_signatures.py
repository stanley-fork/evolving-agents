from metrics.signatures import (SIGNATURE_VERSION, normalize_message, signature, tool_error_signature,
                                verifier_signatures)


def test_version_is_declared():
    assert SIGNATURE_VERSION == "2"


def test_normalizes_volatile_parts():
    a = normalize_message("NameError at /home/alice/run-1/work/ledgerkit/cli.py:12:4 at 2026-10-22T10:00:01Z "
                          "id 0x7ffde3 commit 3fa9c1e2b took 0.31s in /tmp/pytest-of-root/p3/x")
    b = normalize_message("NameError at /home/bob/run-7/work/ledgerkit/cli.py:99:1 at 2025-01-02 03:04:05 "
                          "id 0x1 commit deadbee1 took 12ms in /tmp/tmpq8r/x")
    assert a == b
    assert "alice" not in a and "12" not in a and "2026" not in a


def test_roots_become_relative():
    msg = "/sbx/run/work/lk/ledgerkit/registry.py:3: LK-L004 command 'x' is not registered"
    assert normalize_message(msg, ["/sbx/run/work/lk", "/sbx/run"]) == \
        "ledgerkit/registry.py:<n>: LK-L004 command 'x' is not registered"


def test_line_word_and_whitespace():
    assert normalize_message('File "x.py", line 41,   in  run') == 'File "x.py", line <n>, in run'


def test_signature_shape():
    assert signature("verify:lint", "ledgerkit/x.py", "LK-L003", "needs tag") == \
        "verify:lint|ledgerkit/x.py|LK-L003|needs tag"


def test_verifier_lines():
    sigs = verifier_signatures([
        "BENCH-FAIL type=test file=tests/test_a.py symbol=test_b msg=AssertionError: 1 != 2",
        "BENCH-FAIL type=hidden file=- symbol=hidden_03 msg='largest x' returned '7', want '0.07'",
        "something odd at /tmp/abc/q",
    ])
    assert sigs == ["verify:test|tests/test_a.py|test_b|AssertionError: 1 != 2",
                    "verify:hidden||hidden_03|'largest x' returned '7', want '0.07'",
                    "verify:unknown|||something odd at <tmp>"]


def test_tool_error_signature_uses_last_line_and_exception():
    err = {"tool": "Bash", "file": "", "message":
           "Exit code 1\nTraceback (most recent call last):\n  File \"/w/r/ledgerkit/cli.py\", line 9\n"
           "ModuleNotFoundError: No module named 'ledgerkit.registry'"}
    assert tool_error_signature(err, ["/w/r"]) == \
        "tool:Bash|ledgerkit/cli.py|ModuleNotFoundError|ModuleNotFoundError: No module named 'ledgerkit.registry'"
    edit = {"tool": "Edit", "file": "/w/r/ledgerkit/registry.py", "message": "String to replace not found in file."}
    assert tool_error_signature(edit, ["/w/r"]) == "tool:Edit|ledgerkit/registry.py||String to replace not found in file."


def test_same_error_different_runs_same_signature():
    e1 = {"tool": "Bash", "file": "", "message": "Exit code 1\n/sb/r1/work/lk/tools/lint.py:7: LK-L006 needs test"}
    e2 = {"tool": "Bash", "file": "", "message": "Exit code 2\n/sb/r2/work/lk/tools/lint.py:70: LK-L006 needs test"}
    assert tool_error_signature(e1, ["/sb/r1/work/lk"]) == tool_error_signature(e2, ["/sb/r2/work/lk"])


def test_ci_signatures():
    from metrics.signatures import ci_signatures
    assert ci_signatures(["CI LK-R03: packaging: command 'x' is not listed",
                          "CI tools/check: src/ledgerkit/commands/x.py:12: LK-L03 raw money",
                          "CI: behaviour test tag-saved failed"]) == [
        "ci||LK-R03|packaging: command 'x' is not listed",
        "ci||tools/check|src/ledgerkit/commands/x.py:<n>: LK-L03 raw money",
        "ci|||behaviour test tag-saved failed"]


def test_task_names_are_masked_so_the_same_mistake_repeats():
    from metrics.signatures import ci_signatures, mask_terms
    a = mask_terms(ci_signatures(["CI LK-R03: packaging: command 'balance' is not listed in ledgerkit.toml [commands]"])[0], ["balance"])
    b = mask_terms(ci_signatures(["CI LK-R03: packaging: command 'count' is not listed in ledgerkit.toml [commands]"])[0], ["count"])
    assert a == b and "<task>" in a
    assert mask_terms("ci||LK-R01|row lacks --memo-x", ["--memo"]) == "ci||LK-R01|row lacks --memo-x"   # whole words only
