"""One headless Claude Code session: launch, stream, enforce the budget, summarize."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


@dataclass
class SessionResult:
    status: str = "ok"            # ok | budget_tokens | timeout | max_turns | error
    exit_code: int | None = None
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    wall_seconds: float = 0.0
    tool_errors: list[dict] = field(default_factory=list)  # {tool, file, message}
    session_id: str | None = None  # from the stream's init/result line; used to --resume

    @property
    def tokens(self) -> int:
        """Budgeted tokens: new input + cache writes + output (cache reads excluded)."""
        return self.input_tokens + self.cache_creation_tokens + self.output_tokens

    @property
    def within_budget(self) -> bool:
        return self.status not in ("budget_tokens", "timeout")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = self.tokens
        d["within_budget"] = self.within_budget
        return d


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


class _Usage:
    """Usage from assistant messages, de-duplicated by message id (Claude Code may
    emit one line per content block, each repeating the message's usage)."""

    def __init__(self) -> None:
        self.by_id: dict[str, dict] = {}
        self.anon = 0

    def add(self, message: dict) -> None:
        usage = message.get("usage") or {}
        mid = message.get("id") or f"anon-{self.anon}"
        if not message.get("id"):
            self.anon += 1
        prev = self.by_id.get(mid, {})
        self.by_id[mid] = {k: max(int(usage.get(k) or 0), int(prev.get(k) or 0)) for k in USAGE_KEYS}

    def total(self, key: str) -> int:
        return sum(u[key] for u in self.by_id.values())


def combine(rounds: list[SessionResult]) -> SessionResult:
    """Sum a task's rounds (first attempt + feedback rounds): budget is shared across them."""
    total = SessionResult()
    for r in rounds:
        total.turns += r.turns
        total.input_tokens += r.input_tokens
        total.output_tokens += r.output_tokens
        total.cache_creation_tokens += r.cache_creation_tokens
        total.cache_read_tokens += r.cache_read_tokens
        total.wall_seconds = round(total.wall_seconds + r.wall_seconds, 3)
        total.tool_errors += r.tool_errors
        if r.cost_usd is not None:
            total.cost_usd = (total.cost_usd or 0.0) + r.cost_usd
        total.session_id = total.session_id or r.session_id
    over = [r.status for r in rounds if r.status in ("budget_tokens", "timeout")]
    total.status = over[0] if over else rounds[-1].status
    total.exit_code = rounds[-1].exit_code
    return total


def run_session(cmd: list[str], cwd: Path, env: dict[str, str], trace_path: Path,
                stderr_path: Path, max_tokens: int, wall_seconds: int, append: bool = False) -> SessionResult:
    res = SessionResult()
    usage = _Usage()
    tools: dict[str, tuple[str, dict]] = {}
    final: dict | None = None
    killed: list[str] = []
    start = time.monotonic()

    mode = "a" if append else "w"
    with open(trace_path, mode, encoding="utf-8") as trace, open(stderr_path, mode) as err:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=err,
                                stdin=subprocess.DEVNULL, text=True, start_new_session=True)

        def kill(reason: str) -> None:
            if not killed:
                killed.append(reason)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(wall_seconds, kill, args=("timeout",))
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                trace.write(line)
                trace.flush()
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("type")
                msg = ev.get("message") or {}
                if ev.get("session_id") and not res.session_id:
                    res.session_id = str(ev["session_id"])
                if kind == "assistant":
                    usage.add(msg)
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tools[block.get("id", "")] = (block.get("name", "?"), block.get("input") or {})
                    spent = (usage.total("input_tokens") + usage.total("cache_creation_input_tokens")
                             + usage.total("output_tokens"))
                    if spent > max_tokens:
                        kill("budget_tokens")
                        break
                elif kind == "user":
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                            name, inp = tools.get(block.get("tool_use_id", ""), ("?", {}))
                            res.tool_errors.append({
                                "tool": name,
                                "file": inp.get("file_path") or inp.get("path") or inp.get("notebook_path") or "",
                                "message": _text(block.get("content")),
                            })
                elif kind == "result":
                    final = ev
        finally:
            timer.cancel()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill("timeout")
                proc.wait()

    res.wall_seconds = round(time.monotonic() - start, 3)
    res.exit_code = proc.returncode
    res.input_tokens = usage.total("input_tokens")
    res.output_tokens = usage.total("output_tokens")
    res.cache_creation_tokens = usage.total("cache_creation_input_tokens")
    res.cache_read_tokens = usage.total("cache_read_input_tokens")
    res.turns = len(usage.by_id)
    if final is not None:
        u = final.get("usage") or {}
        # The result line is authoritative when present.
        if u:
            res.input_tokens = int(u.get("input_tokens") or 0)
            res.output_tokens = int(u.get("output_tokens") or 0)
            res.cache_creation_tokens = int(u.get("cache_creation_input_tokens") or 0)
            res.cache_read_tokens = int(u.get("cache_read_input_tokens") or 0)
        res.turns = int(final.get("num_turns") or res.turns)
        if final.get("total_cost_usd") is not None:
            res.cost_usd = float(final["total_cost_usd"])
    if killed:
        res.status = killed[0]
    elif final is not None and final.get("subtype") == "error_max_turns":
        res.status = "max_turns"
    elif final is None or proc.returncode != 0 or final.get("is_error"):
        res.status = "error"
    return res
