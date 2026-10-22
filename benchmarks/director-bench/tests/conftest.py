import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FAKE = ROOT / "tests" / "fake_claude.py"
LEDGERKIT = ROOT / "families" / "ledgerkit"

from runner.config import load_config, load_family  # noqa: E402
from runner.run import RunOptions, Runner  # noqa: E402

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull}


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True)


@pytest.fixture
def tiny_family(tmp_path):
    """4-task family: task NN passes iff done-NN.txt says 'ok'. Task 04 is the trap,
    whose snapshot has a CHANGED file; task 03 is the retention task of 01."""
    repo = tmp_path / "tiny-repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("tiny v1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v1")
    for t in ("01", "02", "03"):
        git(repo, "tag", f"t{t}")
    (repo / "README.md").write_text("tiny v2: conventions changed\n")
    git(repo, "commit", "-q", "-am", "v2")
    git(repo, "tag", "t04")

    fam = tmp_path / "families" / "tiny"
    (fam / "tasks").mkdir(parents=True)
    (fam / "verify").mkdir()
    for t in ("01", "02", "03", "04"):
        (fam / "tasks" / f"{t}-x.md").write_text(f"TASK-{t}: write done-{t}.txt\n")
        v = fam / "verify" / f"{t}.sh"
        v.write_text(f'#!/bin/sh\n# SECRET-ORACLE-{t}\ncd "$1"\n'
                     f'if [ "$(cat done-{t}.txt 2>/dev/null)" = ok ]; then exit 0; fi\n'
                     f'echo "BENCH-FAIL type=hidden file=done-{t}.txt symbol=check msg=missing at $1/done-{t}.txt"\n'
                     f'echo "CI LK-R01: review: done-{t}.txt is missing"\n'
                     f'exit 1\n')
        v.chmod(0o755)
    (fam / "family.yaml").write_text(f"""
name: tiny
repo: {{git: "{repo}"}}
snapshots: {{"01": t01, "02": t02, "03": t03, "04": t04}}
tasks: ["01", "02", "03", "04"]
order: {{first: ["01"], last: ["04"]}}
budget: {{max_turns: 10, max_tokens: 100000, wall_seconds: 60}}
trap: {{task: "04", obsolete_marker: "OLD-CONVENTION"}}
retention: {{task: "03", of: "01"}}
leak_markers: ["SECRET-ORACLE"]
""")
    return load_family(fam)


def write_config(tmp_path, name="T", **over):
    body = {
        "name": name, "home_scope": "run", "version_cmd": "echo mem-1",
        "session": {"env": {"FAKE_MEMORY": "${BENCH_MEMORY_DIR}/notes.txt"}},
        "hooks": {"setup": ['echo setup > "$BENCH_MEMORY_DIR/setup.txt"'],
                  "post_session": ['echo "$BENCH_TASK_INDEX $BENCH_TASK_ID $BENCH_PASSED $(basename "$BENCH_TRACE")" >> "$BENCH_MEMORY_DIR/outcomes.txt"']},
        "consolidate": {"command": ['echo "consolidated $BENCH_CONSOLIDATE_POSITIONS" >> "$BENCH_MEMORY_DIR/cons.txt"; '
                                    'echo \'{"input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.5}\''],
                        "every": 2},
    }
    body.update(over)
    path = tmp_path / f"{name}.yaml"
    import yaml
    path.write_text(yaml.safe_dump(body))
    return load_config(path)


@pytest.fixture
def make_run(tmp_path):
    def _make(family, config, spec, seed=1, **kw):
        spec_path = tmp_path / f"spec-{config.name}-{seed}-{len(list(tmp_path.iterdir()))}.json"
        spec_path.write_text(json.dumps(spec))
        log = tmp_path / f"log-{config.name}-{seed}.jsonl"
        extra = {"FAKE_CLAUDE_SPEC": str(spec_path), "FAKE_CLAUDE_LOG": str(log), **kw.pop("extra_env", {})}
        opts = RunOptions(family=family, config=config, seed=seed, model=kw.pop("model", "fake-model"),
                          results_dir=tmp_path / "results", claude_bin=str(FAKE),
                          sandbox_dir=tmp_path / "sandbox", extra_env=extra, **kw)
        runner = Runner(opts)
        return runner, log
    return _make


def read_jsonl(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
