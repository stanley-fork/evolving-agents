"""Loading and validating family.yaml and configs/*.yaml."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TODO_MARK = "TODO(confirm"
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class BenchError(Exception):
    """A user-facing error: bad config, mixed pins, failed hook."""


def expand(text: str, env: dict[str, str]) -> str:
    """Expand ${NAME} and ${NAME:-default}; an unset NAME without default is an error."""
    def sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in env and env[name] != "":
            return env[name]
        if default is not None:
            return default
        raise BenchError(f"variable ${{{name}}} is not set (needed by {text!r})")
    return _VAR.sub(sub, text)


def expand_tree(value, env: dict[str, str]):
    if isinstance(value, str):
        return expand(value, env)
    if isinstance(value, list):
        return [expand_tree(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_tree(v, env) for k, v in value.items()}
    return value


@dataclass
class Family:
    name: str
    dir: Path
    raw: dict
    tasks: list[str]
    snapshots: dict[str, str]
    first: list[str]
    last: list[str]
    max_turns: int
    max_tokens: int
    wall_seconds: int
    trap_task: str | None
    obsolete_marker: str | None
    retention_task: str | None
    retention_of: str | None
    leak_markers: list[str]
    feedback_rounds: int = 1

    def statement_path(self, task: str) -> Path:
        hits = sorted((self.dir / "tasks").glob(f"{task}-*.md"))
        if len(hits) != 1:
            raise BenchError(f"{self.name}: expected one tasks/{task}-*.md, found {len(hits)}")
        return hits[0]

    def statement(self, task: str) -> str:
        return self.statement_path(task).read_text(encoding="utf-8").strip()

    def verifier(self, task: str) -> Path:
        return self.dir / "verify" / f"{task}.sh"

    @property
    def main_tasks(self) -> list[str]:
        special = {self.trap_task, self.retention_task}
        return [t for t in self.tasks if t not in special]


def load_family(path: str | Path) -> Family:
    d = Path(path)
    if d.is_file():
        d = d.parent
    raw = yaml.safe_load((d / "family.yaml").read_text(encoding="utf-8"))
    tasks = [str(t) for t in raw["tasks"]]
    snaps = {str(k): str(v) for k, v in raw.get("snapshots", {}).items()}
    order = raw.get("order", {}) or {}
    budget = raw.get("budget", {}) or {}
    trap = raw.get("trap") or {}
    ret = raw.get("retention") or {}
    fam = Family(
        name=raw["name"], dir=d.resolve(), raw=raw, tasks=tasks, snapshots=snaps,
        first=[str(t) for t in order.get("first", [])],
        last=[str(t) for t in order.get("last", [])],
        max_turns=int(budget.get("max_turns", 60)),
        max_tokens=int(budget.get("max_tokens", 400_000)),
        wall_seconds=int(budget.get("wall_seconds", 3600)),
        trap_task=str(trap["task"]) if trap.get("task") else None,
        obsolete_marker=trap.get("obsolete_marker"),
        retention_task=str(ret["task"]) if ret.get("task") else None,
        retention_of=str(ret["of"]) if ret.get("of") else None,
        leak_markers=[str(m) for m in raw.get("leak_markers", [])],
        feedback_rounds=int((raw.get("feedback") or {}).get("rounds", 1)),
    )
    for t in tasks:
        if t not in snaps:
            raise BenchError(f"{fam.name}: task {t} has no snapshot")
        fam.statement_path(t)
        if not fam.verifier(t).exists():
            raise BenchError(f"{fam.name}: missing verify/{t}.sh")
    for t in fam.first + fam.last:
        if t not in tasks:
            raise BenchError(f"{fam.name}: order mentions unknown task {t}")
    return fam


@dataclass
class Config:
    name: str
    path: Path
    raw: dict
    text: str
    home_scope: str = "task"
    version_cmd: str | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    settings: dict | None = None
    setup: list[str] = field(default_factory=list)
    pre_session: list[str] = field(default_factory=list)
    post_session: list[str] = field(default_factory=list)
    consolidate: list[str] = field(default_factory=list)
    every: int = 2

    def todos(self) -> list[str]:
        return [ln.strip() for ln in self.text.splitlines() if TODO_MARK in ln]


def _cmds(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    session = raw.get("session") or {}
    hooks = raw.get("hooks") or {}
    cons = raw.get("consolidate") or {}
    cfg = Config(
        name=str(raw.get("name", p.stem)), path=p.resolve(), raw=raw, text=text,
        home_scope=raw.get("home_scope", "task"),
        version_cmd=raw.get("version_cmd"),
        extra_args=[str(a) for a in session.get("extra_args", []) or []],
        env={str(k): str(v) for k, v in (session.get("env") or {}).items()},
        settings=session.get("settings"),
        setup=_cmds(hooks.get("setup")),
        pre_session=_cmds(hooks.get("pre_session")),
        post_session=_cmds(hooks.get("post_session")),
        consolidate=_cmds(cons.get("command")),
        every=int(cons.get("every", 2)),
    )
    if cfg.home_scope not in ("task", "run"):
        raise BenchError(f"{p}: home_scope must be 'task' or 'run'")
    return cfg


def resolve(root: Path, kind: str, name: str) -> Path:
    """Accept a name (families/<name>, configs/<name>.yaml) or a path."""
    p = Path(name)
    if p.exists():
        return p
    cand = root / kind / (name if kind == "families" else f"{name}.yaml")
    if not cand.exists():
        raise BenchError(f"no {kind[:-1]} named {name!r} (looked for {cand})")
    return cand


def bench_root() -> Path:
    return Path(os.environ.get("BENCH_ROOT") or Path(__file__).resolve().parent.parent)
