"""The judge's reply is scored by `id` and `met` only; an unescaped quote in its free-text evidence (D5 [ran]) must not
drop the whole verdict — and a reply that names the wrong items still fails."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "d0"))

import run_d0 as R  # noqa: E402

IDS = ["c09/1", "c09/2"]


def test_valid_json_is_read_as_is():
    out = R.parse_items('{"items": [{"id": "c09/1", "met": true, "evidence": "a"}, {"id": "c09/2", "met": false, "evidence": "b"}]}', IDS)
    assert [(x["id"], x["met"]) for x in out] == [("c09/1", True), ("c09/2", False)]


def test_an_unescaped_quote_in_evidence_keeps_the_verdict():
    bad = '{"items": [{"id": "c09/1", "met": true, "evidence": "x"}, {"id": "c09/2", "met": false, "evidence": "Run `a`." (step 7)"}]}'
    assert [(x["id"], x["met"]) for x in R.parse_items(bad, IDS)] == [("c09/1", True), ("c09/2", False)]


def test_wrong_or_missing_items_still_fail():
    assert R.parse_items('{"items": [{"id": "c09/1", "met": true}]}', IDS) is None
    assert R.parse_items("no json at all", IDS) is None
