"""The director's loop (D1): plan by walking the operational memory, delegate each task to a lane, log everything.

One append-only event log is the only state the interface reads. Lanes:

    director   the planning call itself — Claude Code (`claude -p`) with DIRECTOR.md and the compiled memory;
               its walk (which pages it opened) is logged as it happens
    frontier   Claude Code as `claude -p --output-format stream-json`, read-only tools unless the config allows
               more; its transcript is logged as it streams
    local      an OpenAI-compatible server (llama.cpp, lora-kernel's gateway). The director never starts one:
               the user's machine runs a model only with the user's approval. If none answers, the task is
               logged `offline` and waits — it can be moved to the frontier.

Standard library only. `DIRECTOR_CLAUDE` overrides the `claude` executable (tests use a fake).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

PLAN_ASK = """{request}

Walk your memory first, as your instructions say. Then end your reply with the plan as one fenced JSON block:

```json
{{"summary": "<one line>",
  "tasks": [{{"id": "t1", "title": "<short>", "lane": "frontier" or "local", "why_lane": "<one line>",
             "prompt": "<the full instruction for whoever runs it>", "cites": ["Page#id", ...]}}]}}
```

Lanes: `local` only for a task that needs no tools, no files and no repository — drafting, summarising,
classifying text given in its prompt — and only if the local lane is online ({local_state}). Everything else is
`frontier`. Keep the plan to the tasks that are needed; order is execution order within a lane."""


class Log:
    """Append-only JSONL. Every event: {seq, t, lane, kind, task?, text?, ...}."""

    def __init__(self, path: Path):
        self.path, self.lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = sum(1 for _ in self.path.open()) if self.path.exists() else 0

    def emit(self, lane: str, kind: str, **fields) -> dict:
        with self.lock:
            ev = {"seq": self.seq, "t": time.strftime("%H:%M:%S"), "lane": lane, "kind": kind, **fields}
            with self.path.open("a") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.seq += 1
            return ev

    def since(self, seq: int) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for i, line in enumerate(self.path.open()):
            if i >= seq and line.strip():
                out.append(json.loads(line))
        return out


# ------------------------------------------------------------------ memory: what the director can do

def capabilities(memory_dir: Path) -> dict:
    """What the memory covers: pages, statements, and the situations of the index (where walks start)."""
    pages = {}
    for p in sorted(Path(memory_dir).glob("*.md")):
        if p.name in ("Index.md", "README.md"):
            continue
        txt = p.read_text()
        pages[p.stem] = {"statements": len(re.findall(r"^## #", txt, re.M)),
                         "about": (txt.splitlines()[1] if len(txt.splitlines()) > 1 else "").strip()}
    idx = Path(memory_dir) / "Index.md"
    situations = [re.sub(r"\s*→.*$", "", l[2:]).strip() for l in idx.read_text().splitlines()
                  if l.startswith("- ")] if idx.exists() else []
    return {"pages": pages, "statements": sum(v["statements"] for v in pages.values()),
            "situations": situations}


# ------------------------------------------------------------------ Claude Code, streamed

def claude_bin() -> str:
    return os.environ.get("DIRECTOR_CLAUDE", "claude")


def stream_claude(prompt: str, cwd: Path, log: Log, lane: str, task: str | None, *, system: str | None = None,
                  tools: str = "Read,Grep,Glob", model: str = "sonnet", max_turns: int = 20,
                  budget: float = 1.0) -> dict:
    """Run `claude -p` with stream-json and log its transcript as it arrives. Returns {result, cost, turns}."""
    argv = [claude_bin(), "-p", prompt, "--output-format", "stream-json", "--verbose", "--restricted",
            "--no-session-persistence", "--permission-mode", "dontAsk", "--tools", tools, "--allowedTools", tools,
            "--model", model, "--max-turns", str(max_turns), "--max-budget-usd", str(budget)]
    if system:
        argv += ["--append-system-prompt", system]
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                         text=True)
    final = {"result": None, "cost": None, "turns": None}
    for line in p.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "assistant":
            for c in m.get("message", {}).get("content", []):
                if c.get("type") == "text" and c.get("text", "").strip():
                    log.emit(lane, "text", task=task, text=c["text"])
                elif c.get("type") == "tool_use":
                    arg = c.get("input", {})
                    what = arg.get("file_path") or arg.get("pattern") or arg.get("path") or json.dumps(arg)[:120]
                    log.emit(lane, "tool", task=task, tool=c.get("name"), text=str(what))
        elif m.get("type") == "result":
            final = {"result": m.get("result"), "cost": m.get("total_cost_usd"), "turns": m.get("num_turns"),
                     "error": m.get("is_error")}
    p.wait()
    if p.returncode and final["result"] is None:
        final["error"] = True
        log.emit(lane, "error", task=task, text=p.stderr.read()[-400:])
    return final


# ------------------------------------------------------------------ the local lane

def local_online(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2).read()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_local(prompt: str, base_url: str, model: str, log: Log, task: str) -> dict:
    t0 = time.time()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    text = out["choices"][0]["message"]["content"]
    u = out.get("usage") or {}
    log.emit("local", "text", task=task, text=text)
    return {"result": text, "tokens": [u.get("prompt_tokens"), u.get("completion_tokens")],
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ the director

def parse_plan(text: str) -> dict | None:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", re.S)
    try:
        plan = json.loads(m[-1]) if m else None
    except ValueError:
        return None
    if not plan or not isinstance(plan.get("tasks"), list):
        return None
    for i, t in enumerate(plan["tasks"], 1):
        t.setdefault("id", f"t{i}")
        t["lane"] = "local" if t.get("lane") == "local" else "frontier"
    return plan


class Director:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.memory = Path(cfg["memory_dir"]).expanduser()
        self.workdir = Path(cfg.get("workdir", ".")).expanduser()
        self.log = Log(Path(cfg["log"]).expanduser())
        self.system = (HERE / "DIRECTOR.md").read_text()
        self.plan: dict | None = None
        self.busy = False

    def state(self) -> dict:
        return {"capabilities": capabilities(self.memory), "plan": self.plan, "busy": self.busy,
                "local_online": local_online(self.cfg.get("local_base_url")),
                "frontier_tools": self.cfg.get("frontier_tools", "Read,Grep,Glob")}

    def handle(self, request: str) -> None:
        """One request: plan (walking the memory), then run each lane's tasks in order, lanes in parallel."""
        self.busy = True
        try:
            self.log.emit("director", "request", text=request)
            online = local_online(self.cfg.get("local_base_url"))
            r = stream_claude(PLAN_ASK.format(request=request, local_state="online" if online else "OFFLINE"),
                              self.memory.parent, self.log, "director", None, system=self.system,
                              model=self.cfg.get("director_model", "sonnet"), max_turns=16,
                              budget=self.cfg.get("plan_budget_usd", 1.0))
            plan = parse_plan(r.get("result") or "")
            if plan is None:
                self.log.emit("director", "error", text="no plan could be read from the director's reply")
                return
            self.plan = {**plan, "request": request, "cost": r.get("cost")}
            self.log.emit("director", "plan", plan=self.plan)
            lanes = {"frontier": [t for t in plan["tasks"] if t["lane"] == "frontier"],
                     "local": [t for t in plan["tasks"] if t["lane"] == "local"]}
            threads = [threading.Thread(target=self._run_lane, args=(lane, ts)) for lane, ts in lanes.items() if ts]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.log.emit("director", "done", text=plan.get("summary", ""))
        finally:
            self.busy = False

    def _run_lane(self, lane: str, tasks: list[dict]) -> None:
        for t in tasks:
            self.log.emit(lane, "task_start", task=t["id"], text=t.get("title", ""))
            try:
                if lane == "local":
                    base = self.cfg.get("local_base_url")
                    if not local_online(base):
                        self.log.emit("local", "offline", task=t["id"],
                                      text="local lane offline: no server answers; the director does not start one")
                        continue
                    r = run_local(t["prompt"], base, self.cfg.get("local_model", "local"), self.log, t["id"])
                    self.log.emit("local", "task_end", task=t["id"], seconds=r["seconds"], tokens=r["tokens"])
                else:
                    r = stream_claude(t["prompt"], self.workdir, self.log, "frontier", t["id"],
                                      tools=self.cfg.get("frontier_tools", "Read,Grep,Glob"),
                                      model=self.cfg.get("frontier_model", "sonnet"),
                                      budget=self.cfg.get("task_budget_usd", 1.0))
                    self.log.emit("frontier", "task_end", task=t["id"], cost=r.get("cost"), turns=r.get("turns"),
                                  error=bool(r.get("error")))
            except Exception as e:                       # a lane's failure is an event, never a silent stop
                self.log.emit(lane, "error", task=t["id"], text=repr(e)[:400])
