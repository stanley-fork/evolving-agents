import json

from ledgerkit import cli


def test_main_saves_only_mutating_commands(tmp_path, monkeypatch, capsys):
    path = tmp_path / "l.json"
    monkeypatch.setenv("LEDGERKIT_FILE", str(path))
    assert cli.main(["list"]) == 0 and not path.exists()
    assert cli.main(["add", "cash", "1.50", "x", "--date", "2026-01-02"]) == 0
    assert json.loads(path.read_text())["entries"][0]["amount_cents"] == 150
    assert cli.main(["nope"]) == 2
    assert "LK-E404" in capsys.readouterr().err
