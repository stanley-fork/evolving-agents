"""Turn your own instructions into a director's memory — the quickstart.

    python -m director.compile --claude-md ~/project/CLAUDE.md \\
        --memory-dir ~/.claude/projects/<your-project>/memory --out ~/.config/director/live/memory

Copies what you name (CLAUDE.md files, Claude Code's auto-memory directory, any folder of agents or skills) into a
sandbox, asks Claude Code (`claude -p`, Opus by default, read-only on your files) to compile it into pages of
atomic statements with an index of situations (`d0/compile.md`, `docs/MEMORY-FORMAT.md`), lints the result, and
writes a config for `python -m director.server` if none exists. It reads nothing you did not name, and writes only
under `--out` and the config. One compile of ~60 files cost about US$5 and 15 minutes (D0's, Opus 5.5).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from .core import Log, stream_claude
from .learn import lint

HERE = Path(__file__).resolve().parent
PROMPT = HERE.parent / "d0" / "compile.md"
FORMAT = HERE.parent / "docs" / "MEMORY-FORMAT.md"


def compile_memory(sources: list[Path], out: Path, model: str = "opus", budget: float = 10.0,
                   log_path: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="director-compile-") as tmp:
        work = Path(tmp)
        src = work / "sources"
        src.mkdir()
        for i, s in enumerate(sources):
            s = Path(s).expanduser()
            dest = src / f"{i:02d}-{s.name}"
            (shutil.copytree if s.is_dir() else shutil.copy)(s, dest)
        (work / "out").mkdir()
        log = Log(log_path or work / "compile-events.jsonl")
        r = stream_claude(PROMPT.read_text(), work, log, "director", "compile",
                          tools="Read,Grep,Glob,Write", model=model, max_turns=200, budget=budget)
        pages = sorted((work / "out").glob("*.md"))
        if not pages or not (work / "out" / "Index.md").exists():
            raise SystemExit(f"the compile wrote no pages or no Index.md: {r.get('result')!r}"[:400])
        out = Path(out).expanduser()
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(work / "out", out)
    shutil.copy(FORMAT, out / "README.md")
    statements = sum(len(re.findall(r"^## #", p.read_text(), re.M)) for p in out.glob("*.md") if p.name != "README.md")
    return {"pages": len(pages) - 1, "statements": statements, "lint": lint(out), "cost_usd": r.get("cost"),
            "turns": r.get("turns"), "out": str(out)}


def write_config(path: Path, memory: Path, workdir: Path) -> bool:
    path = Path(path).expanduser()
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "memory_dir": str(memory), "workdir": str(Path(workdir).expanduser()),
        "log": str(Path(memory).parent / "events.jsonl"),
        "local_base_url": "http://127.0.0.1:8792/v1", "local_model": "local",
        "director_model": "sonnet", "frontier_model": "sonnet", "frontier_tools": "Read,Grep,Glob",
        "plan_budget_usd": 1.0, "task_budget_usd": 1.0}, indent=1))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude-md", action="append", default=[], help="a CLAUDE.md (or AGENTS.md); repeatable")
    ap.add_argument("--memory-dir", action="append", default=[], help="Claude Code's auto-memory directory; repeatable")
    ap.add_argument("--extra", action="append", default=[], help="any other file or folder (agents, skills, notes)")
    ap.add_argument("--out", default="~/.config/director/live/memory")
    ap.add_argument("--config", default="~/.config/director/director.json")
    ap.add_argument("--workdir", default=".", help="where frontier tasks will run (your repository)")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--budget-usd", type=float, default=10.0)
    a = ap.parse_args()
    sources = [Path(p) for p in a.claude_md + a.memory_dir + a.extra]
    if not sources:
        ap.error("name at least one source: --claude-md, --memory-dir or --extra")
    missing = [str(p) for p in sources if not Path(p).expanduser().exists()]
    if missing:
        ap.error(f"not found: {', '.join(missing)}")
    res = compile_memory(sources, Path(a.out), a.model, a.budget_usd)
    res["config_written"] = write_config(Path(a.config), Path(a.out).expanduser(), Path(a.workdir))
    print(json.dumps(res, indent=1))
    print(f"\nnext: python -m director.server --config {a.config}   # then open http://127.0.0.1:8740")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
