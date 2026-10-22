"""The v0 family: deterministic repo, every verifier fails before and passes after,
the house rules are enforced, and there is headroom (static, no model calls)."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GIT_ENV, LEDGERKIT
from runner.config import load_family

FAM = load_family(LEDGERKIT)
sys.path.insert(0, str(LEDGERKIT))
import headroom  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    dest = tmp_path_factory.mktemp("lk") / "repo"
    subprocess.run([sys.executable, str(LEDGERKIT / "build_repo.py"), str(dest)], check=True)
    return dest


def snapshot(built, task, dest):
    """What the runner gives the agent: the snapshot tree as a fresh single-commit repo."""
    dest.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(built), "archive", FAM.snapshots[task]], check=True,
                         capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "snapshot"]):
        subprocess.run(["git", *args], cwd=dest, env=GIT_ENV, check=True, capture_output=True)
    return dest


def solve(work, task):
    shutil.copytree(LEDGERKIT / "reference" / task, work, dirs_exist_ok=True)
    subprocess.run([sys.executable, "tools/regen"], cwd=work, check=True, capture_output=True)


def verify(task, work):
    env = {**os.environ, "BENCH_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(["bash", str(FAM.verifier(task)), str(work)], cwd=work, env=env,
                          capture_output=True, text=True, timeout=300)


def lint_codes(out: str) -> set:
    return set(re.findall(r"BENCH-FAIL type=lint file=\S+ symbol=(LK-L\d+)", out))


def ci_lines(out: str) -> list:
    return [ln for ln in out.splitlines() if ln.startswith(("CI ", "CI:"))]


def test_build_is_deterministic(built, tmp_path):
    other = tmp_path / "again"
    subprocess.run([sys.executable, str(LEDGERKIT / "build_repo.py"), str(other)], check=True)
    for ref in ("task-01", "task-09"):
        a = subprocess.run(["git", "-C", str(built), "rev-parse", ref], capture_output=True, text=True).stdout
        b = subprocess.run(["git", "-C", str(other), "rev-parse", ref], capture_output=True, text=True).stdout
        assert a == b and a.strip()


def test_repo_is_too_large_to_read_in_one_go(built):
    files = subprocess.run(["git", "-C", str(built), "ls-tree", "-r", "--name-only", "task-01"],
                           capture_output=True, text=True, check=True).stdout.split()
    src = [f for f in files if f.startswith("src/") and f.endswith(".py")]
    lines = sum(len(subprocess.run(["git", "-C", str(built), "show", f"task-01:{f}"], capture_output=True,
                                   text=True).stdout.splitlines()) for f in src)
    assert len(src) >= 80 and lines >= 4000
    assert sum(f.startswith("tools/lintrules/l") for f in files) >= 10


def test_family_shape():
    assert len(FAM.tasks) == 10
    assert FAM.trap_task == "09" and FAM.retention_task == "10" and FAM.retention_of == "01"
    assert FAM.first == ["01"] and FAM.last == ["09", "10"]
    assert FAM.obsolete_marker == "CHANGELOG.d" and FAM.feedback_rounds == 1


@pytest.mark.parametrize("task", FAM.tasks)
def test_verifier_fails_before_and_passes_after(built, tmp_path, task):
    work = snapshot(built, task, tmp_path / "w")
    before = verify(task, work)
    assert before.returncode != 0 and "BENCH-FAIL" in before.stdout
    solve(work, task)
    after = verify(task, work)
    assert after.returncode == 0, after.stdout + after.stderr


def test_bare_pytest_hits_the_environment_quirk(built, tmp_path):
    work = snapshot(built, "01", tmp_path / "w")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work,
                         capture_output=True, text=True)
    assert out.returncode != 0 and "No module named 'ledgerkit._compat'" in out.stdout + out.stderr
    assert subprocess.run(["sh", "tools/check"], cwd=work, capture_output=True,
                          env={**os.environ, "PYTHON": sys.executable}).returncode == 0


NAIVE = '''"""Balance of one account."""
from ledgerkit.errors import LedgerError

NAME = "balance"


def run(ledger, args):
    if len(args) != 1:
        raise ValueError("usage: balance <account>")
    total = sum(e.amount_cents for e in ledger.for_account(args[0]))
    return f"balance {args[0]}: {total / 100:.2f}"
'''
NAIVE_TEST = '''from ledgerkit.cli import dispatch
from ledgerkit.ledger import Ledger


def test_balance():
    ledger = Ledger()
    ledger.add("cash", 1050, date="2026-01-01")
    assert dispatch(ledger, ["balance", "cash"]) == "balance cash: 10.50"
'''


def test_naive_attempt_trips_local_and_remote_rules(built, tmp_path):
    """README + obvious files: copy the add.py shape, hand-edit the registry, plain test."""
    work = snapshot(built, "01", tmp_path / "w")
    (work / "src/ledgerkit/commands/balance.py").write_text(NAIVE)
    reg = work / "src/ledgerkit/commands/_registry.py"
    text = reg.read_text().replace("from ledgerkit.commands import ", "from ledgerkit.commands import balance, ")
    reg.write_text(text.replace("COMMANDS = {", 'COMMANDS = {\n    "balance": balance.run,'))
    (work / "tests/commands/test_balance_cmd.py").write_text(NAIVE_TEST)
    out = verify("01", work).stdout
    assert lint_codes(out) == {"LK-L02", "LK-L03", "LK-L06"}
    ci = ci_lines(out)
    for code in ("LK-R01", "LK-R02", "LK-R03", "LK-R04"):
        assert any(ln.startswith(f"CI {code}:") for ln in ci), ci
    assert any(ln.startswith("CI tools/check: ") and "registry stale" in ln for ln in ci)


def test_a_reader_still_fails_the_remote_rules(built, tmp_path):
    """Everything readable in the repo done right (code, regen, marked test): remote CI still fails."""
    work = snapshot(built, "02", tmp_path / "w")
    for rel in ("src/ledgerkit/commands/count.py", "tests/commands/test_count_cmd.py"):
        shutil.copy(LEDGERKIT / "reference" / "02" / rel, work / rel)
    subprocess.run([sys.executable, "tools/regen"], cwd=work, check=True, capture_output=True)
    out = verify("02", work).stdout
    assert lint_codes(out) == set()
    assert sorted(ln.split(":")[0] for ln in ci_lines(out)) == ["CI LK-R01", "CI LK-R02", "CI LK-R03"]


@pytest.mark.parametrize("task", FAM.tasks)
def test_ci_feedback_never_leaks_the_oracle(built, tmp_path, task):
    work = snapshot(built, task, tmp_path / "w")
    (work / "src/ledgerkit/commands/broken.py").write_text('NAME = "broken"\n\n\ndef run(ledger, args):\n    return 1 / 0\n')
    out = verify(task, work).stdout
    ci = "\n".join(ci_lines(out))
    assert ci
    for secret in FAM.leak_markers + ["want", "returned", "raised", "LK-E4"]:
        assert secret not in ci, (secret, ci)


def test_trap_feedback_announces_the_new_convention_only_when_the_old_one_is_used(built, tmp_path):
    work = snapshot(built, "09", tmp_path / "w")
    solve(work, "09")
    (work / "changes" / "rename.toml").unlink()
    nothing = "\n".join(ci_lines(verify("09", work).stdout))
    assert "no change fragment for 'rename'" in nothing and "changes/" not in nothing
    (work / "CHANGELOG.d").mkdir()
    (work / "CHANGELOG.d" / "rename.md").write_text("### Added\n- rename\n")
    old = "\n".join(ci_lines(verify("09", work).stdout))
    assert "CHANGELOG.d/ was retired in 0.6.0" in old and "changes/<name>.toml" in old


def test_float_money_is_rejected(built, tmp_path):
    work = snapshot(built, "05", tmp_path / "w")
    solve(work, "05")
    mod = work / "src/ledgerkit/commands/transfer.py"
    mod.write_text(mod.read_text().replace("cents = parse_amount(text)", "cents = int(float(text) * 100)"))
    out = verify("05", work)
    assert out.returncode != 0 and "LK-L03" in lint_codes(out.stdout)


def test_undeclared_mutation_fails_a_behaviour_test_without_details(built, tmp_path):
    work = snapshot(built, "07", tmp_path / "w")
    solve(work, "07")
    mod = work / "src/ledgerkit/commands/tag.py"
    mod.write_text(mod.read_text().replace("MUTATES = True\n", ""))
    out = verify("07", work).stdout
    assert "not persisted" in out and ci_lines(out) == ["CI: behaviour test tag-saved failed"]


def test_hand_edited_registry_is_stale_even_if_correct(built, tmp_path):
    work = snapshot(built, "02", tmp_path / "w")
    solve(work, "02")
    reg = work / "src/ledgerkit/commands/_registry.py"
    lines = reg.read_text().splitlines(keepends=True)
    reg.write_text("# digest: 000000000000\n" + "".join(lines[1:]))
    assert "LK-L02" in lint_codes(verify("02", work).stdout)


def test_headroom_matrix():
    rows = headroom.matrix()
    assert all(len(r["remote"]) >= 3 for r in rows.values()), rows
    assert all(not r["remote_tells"] and not r["readme_mentions"] for r in rows.values())
    for rule in headroom.REMOTE:                                       # each remote rule: most tasks
        assert sum(rule in r["remote"] for r in rows.values()) >= 8
    assert len({tuple(r["local"] + r["remote"]) for r in rows.values()}) >= 6


def test_repo_snapshots_hold_no_oracle_content(built, tmp_path):
    for task in ("01", "09"):
        work = snapshot(built, task, tmp_path / task)
        blob = b"".join(p.read_bytes() for p in work.rglob("*") if p.is_file() and ".git" not in p.parts)
        for marker in FAM.leak_markers:
            assert marker.encode() not in blob


def test_every_oracle_and_reference_file_carries_a_leak_marker():
    files = [p for d in ("verify", "reference") for p in (LEDGERKIT / d).rglob("*") if p.is_file()
             and "__pycache__" not in p.parts]
    assert files
    for p in files:
        assert "LKORACLE-7f3a9c" in p.read_text(), p
