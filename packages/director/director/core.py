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
import shutil
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

Lanes: {lane_rule} Everything else is `frontier`. Keep the plan to the tasks that are needed; order is execution
order within a lane."""

BARE_RULE = ("`local` only for a task that needs no tools, no files and no repository — drafting, summarising, "
             "classifying text given in its prompt — and only if the local lane is online ({state}).")
LK_RULE = ("`local` is lora-kernel: small local specialists, each trained on one domain with its own tools — {domains}. "
           "Use it for tasks in those domains, and only if it is online ({state}). A specialist abstains on anything "
           "outside its domain; such a task comes back to the frontier by itself. A specialist was trained on a team "
           "member's short requests, so write each local task's prompt as ONE short request in a team member's own "
           "words, one action and an explicit key per task (e.g. \"Details on HW-245?\", \"Move HW-245 to in "
           "review.\") — never an instruction paragraph; split two actions into two tasks.")
ABSTAIN = re.compile(r"\bOUT[ _-]OF[ _-]SCOPE\b", re.I)


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


def lk_token(lk: dict) -> str:
    """A lora-kernel gateway token for the configured user and role (the demo issuer: lora-kernel's own
    examples/common/tokens.py, imported from the configured checkout)."""
    import sys
    root = str(Path(lk["path"]).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
    from examples.common import tokens          # lora-kernel's, not ours
    return tokens.issue(lk["user"], lk["role"], lk["org"])


def run_local(prompt: str, base_url: str, model: str, log: Log, task: str, lk: dict | None = None) -> dict:
    """One local call. With `lk` (lora-kernel's gateway) the request carries a role token and opens a session of
    its own — a specialist's abstention was measured as a session's first turn, never mid-session — and the
    gateway's route is read: anything but `local` means the specialist abstained."""
    t0 = time.time()
    headers = {"Content-Type": "application/json"}
    if lk:
        headers["Authorization"] = f"Bearer {lk_token(lk)}"
        headers["X-Session-Id"] = f"director-{task}-{int(t0)}"
        model = "auto"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    text = out["choices"][0]["message"]["content"]
    u = out.get("usage") or {}
    route = out.get("x_route", "local")
    if route == "local" and lk and ABSTAIN.search(text or ""):
        route = "abstained"          # the member said OUT OF SCOPE in a form the gateway did not recognise
    for c in out.get("x_calls") or []:
        log.emit("local", "tool", task=task, tool=c.get("tool"),
                 text=f"{json.dumps(c.get('args'), ensure_ascii=False)[:100]} → {str(c.get('result', c.get('error', '')))[:120]}")
    if route == "local":
        log.emit("local", "text", task=task, text=text)
    return {"result": text, "route": route, "tokens": [u.get("prompt_tokens"), u.get("completion_tokens")],
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
    """The loop. States: idle → planning → awaiting_run (the user may move tasks between lanes, D3) → running →
    idle. After a task, `reflect` proposes memory edits (D2b) that the user accepts or rejects; an accepted edit
    is linted, versioned in agentvcs and becomes the live memory."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.memory = Path(cfg["memory_dir"]).expanduser()
        self.workdir = Path(cfg.get("workdir", ".")).expanduser()
        self.home = Path(cfg["log"]).expanduser().parent
        self.log = Log(Path(cfg["log"]).expanduser())
        self.system = (HERE / "DIRECTOR.md").read_text()
        self.plan: dict | None = None
        self.phase = "idle"
        self.last: dict | None = None            # the last finished task, for reflection
        self.proposals: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.store = None                        # agentvcs live store, opened on the first accepted edit

    @property
    def busy(self) -> bool:
        return self.phase in ("planning", "running", "reflecting", "guarding")

    def claim(self, phase: str, allowed: tuple[str, ...]) -> bool:
        with self.lock:
            if self.phase not in allowed:
                return False
            self.phase = phase
            return True

    def state(self) -> dict:
        return {"capabilities": capabilities(self.memory), "plan": self.plan, "busy": self.busy,
                "phase": self.phase, "local_online": local_online(self.cfg.get("local_base_url")),
                "frontier_tools": self.cfg.get("frontier_tools", "Read,Grep,Glob"),
                "can_reflect": self.phase == "idle" and self.last is not None,
                "proposals": {k: {x: v.get(x) for x in ("edits", "applied", "rejected", "lint", "status", "probes")}
                              for k, v in self.proposals.items()},
                "memory_version": self.store.head if self.store else None}

    # ------------------------------------------------------------------ plan, move, run

    def handle(self, request: str) -> None:
        """Plan by walking the memory; then wait for the user to run it (or run at once with `auto_run`)."""
        try:
            self.log.emit("director", "request", text=request)
            online = local_online(self.cfg.get("local_base_url"))
            r = stream_claude(PLAN_ASK.format(request=request, lane_rule=self.lane_rule(online)),
                              self.memory.parent, self.log, "director", None, system=self.system,
                              model=self.cfg.get("director_model", "sonnet"), max_turns=16,
                              budget=self.cfg.get("plan_budget_usd", 1.0))
            plan = parse_plan(r.get("result") or "")
            if plan is None:
                self.log.emit("director", "error", text="no plan could be read from the director's reply")
                self.phase = "idle"
                return
            for t in plan["tasks"]:
                t["planned_lane"], t["status"] = t["lane"], "queued"
            self.plan = {**plan, "request": request, "cost": r.get("cost"), "moves": []}
            self.log.emit("director", "plan", plan=self.plan)
            self.phase = "awaiting_run"
        except Exception as e:
            self.log.emit("director", "error", text=repr(e)[:400])
            self.phase = "idle"
            return
        if self.cfg.get("auto_run"):
            self.phase = "running"
            self.execute()

    def lane_rule(self, online: bool) -> str:
        state = "online" if online else "OFFLINE"
        lk = self.cfg.get("lora_kernel")
        if lk and self.cfg.get("local_kind") == "lora-kernel":
            return LK_RULE.format(domains=lk.get("domains", "its configured domains"), state=state)
        return BARE_RULE.format(state=state)

    def move(self, task_id: str, lane: str) -> dict:
        """D3: the user moves a queued task to the other lane before the plan runs. Logged, and kept for the
        reflection that follows the task — that is how the director learns where the user wants work done."""
        if lane not in ("frontier", "local"):
            raise ValueError("lane must be frontier or local")
        with self.lock:
            if self.phase != "awaiting_run" or not self.plan:
                raise ValueError("tasks can be moved only while the plan waits to run")
            t = next((t for t in self.plan["tasks"] if t["id"] == task_id), None)
            if t is None:
                raise ValueError(f"no task {task_id}")
            if t["lane"] == lane:
                return t
            mv = {"task": task_id, "title": t.get("title"), "prompt": t.get("prompt"), "from": t["lane"], "to": lane,
                  "request": self.plan["request"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t["lane"] = lane
            self.plan["moves"].append(mv)
        with (self.home / "lane-moves.jsonl").open("a") as f:
            f.write(json.dumps(mv, ensure_ascii=False) + "\n")
        self.log.emit("director", "lane_moved", task=task_id, text=f"{mv['from']} → {lane}", plan=self.plan)
        return t

    def execute(self) -> None:
        """Run each lane's tasks in order, lanes in parallel; keep the outputs for reflection."""
        try:
            outputs: dict[str, list[str]] = {}
            lanes = {"frontier": [t for t in self.plan["tasks"] if t["lane"] == "frontier"],
                     "local": [t for t in self.plan["tasks"] if t["lane"] == "local"]}
            threads = [threading.Thread(target=self._run_lane, args=(lane, ts, outputs))
                       for lane, ts in lanes.items() if ts]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.last = {"plan": self.plan, "outputs": outputs}
            self.log.emit("director", "done", text=self.plan.get("summary", ""))
        finally:
            self.phase = "idle"

    def _run_lane(self, lane: str, tasks: list[dict], outputs: dict) -> None:
        for t in tasks:
            t["status"] = "running"
            self.log.emit(lane, "task_start", task=t["id"], text=t.get("title", ""))
            try:
                if lane == "local":
                    base = self.cfg.get("local_base_url")
                    if not local_online(base):
                        t["status"] = "offline"
                        self.log.emit("local", "offline", task=t["id"],
                                      text="local lane offline: no server answers; the director does not start one")
                        continue
                    lk = self.cfg.get("lora_kernel") if self.cfg.get("local_kind") == "lora-kernel" else None
                    r = run_local(t["prompt"], base, self.cfg.get("local_model", "local"), self.log, t["id"], lk)
                    self.log.emit("local", "task_end", task=t["id"], seconds=r["seconds"], tokens=r["tokens"],
                                  route=r["route"])
                    if r["route"] != "local":
                        # the specialist abstained: lora-kernel's router decided this is not local work
                        t["lane"], t["abstained"] = "frontier", True
                        self.log.emit("local", "abstained", task=t["id"],
                                      text=f"lora-kernel abstained (route {r['route']}); the task goes to the frontier")
                        self.log.emit("director", "lane_moved", task=t["id"], text="local → frontier (abstained)",
                                      plan=self.plan)
                        self.log.emit("frontier", "task_start", task=t["id"], text=t.get("title", "") + " (abstained locally)")
                        r = stream_claude(t["prompt"], self.workdir, self.log, "frontier", t["id"],
                                          tools=self.cfg.get("frontier_tools", "Read,Grep,Glob"),
                                          model=self.cfg.get("frontier_model", "sonnet"),
                                          budget=self.cfg.get("task_budget_usd", 1.0))
                        self.log.emit("frontier", "task_end", task=t["id"], cost=r.get("cost"), turns=r.get("turns"),
                                      error=bool(r.get("error")))
                else:
                    r = stream_claude(t["prompt"], self.workdir, self.log, "frontier", t["id"],
                                      tools=self.cfg.get("frontier_tools", "Read,Grep,Glob"),
                                      model=self.cfg.get("frontier_model", "sonnet"),
                                      budget=self.cfg.get("task_budget_usd", 1.0))
                    self.log.emit("frontier", "task_end", task=t["id"], cost=r.get("cost"), turns=r.get("turns"),
                                  error=bool(r.get("error")))
                t["status"] = "done"
                outputs[t["id"]] = [lane, (r.get("result") or "")[:1500]]
            except Exception as e:                       # a lane's failure is an event, never a silent stop
                t["status"] = "error"
                self.log.emit(lane, "error", task=t["id"], text=repr(e)[:400])

    # ------------------------------------------------------------------ D2b: reflect, propose, accept

    def episode(self, correction: str) -> str:
        p, out = self.last["plan"], self.last["outputs"]
        lines = [f"Request: {p['request']}", f"Plan: {p.get('summary', '')}"]
        for t in p["tasks"]:
            lines.append(f"- {t['id']} [{t['lane']}] {t.get('title', '')} — follows {', '.join(t.get('cites') or []) or 'nothing'}")
            if t["id"] in out:
                lines.append(f"  result: {out[t['id']][1][:600]}")
        for m in p.get("moves") or []:
            lines.append(f"The user moved task {m['task']} ('{m['title']}') from the {m['from']} lane to the {m['to']} "
                         f"lane before it ran.")
        lines.append(f"Correction from the user: {correction.strip() or '(none)'}")
        return "\n".join(lines)

    def reflect(self, correction: str) -> None:
        from .learn import apply_edits, new_problems, reflect
        try:
            pid = f"p{len(self.proposals) + 1}"
            sb = self.home / "reflect-sandbox"
            if sb.exists():
                shutil.rmtree(sb)
            shutil.copytree(self.memory, sb / "memory")
            r = reflect(self.episode(correction), sb / "memory", self.log,
                        model=self.cfg.get("reflect_model", "opus"), budget=self.cfg.get("reflect_budget_usd", 2.0))
            cand = self.home / "candidates" / pid / "memory"
            res = apply_edits(self.memory, r["edits"] or [], cand)
            lint = new_problems(self.memory, cand)
            self.proposals[pid] = {"edits": r["edits"] or [], "applied": res["applied"], "rejected": res["rejected"],
                                   "lint": lint, "status": "proposed", "candidate": str(cand),
                                   "probes": r.get("probes") or [],
                                   "rationale": correction.strip() or self.last["plan"]["request"],
                                   "cost": r.get("cost")}
            self.log.emit("director", "proposal", task=pid, proposal=self.state()["proposals"][pid])
        except Exception as e:
            self.log.emit("director", "error", text=repr(e)[:400])
        finally:
            self.phase = "idle"

    def decide(self, pid: str, accept: bool) -> dict:
        """Accept: lint must be clean; the candidate becomes a gated agentvcs patch and the live memory. Reject:
        recorded, nothing changes."""
        p = self.proposals.get(pid)
        if p is None or p["status"] != "proposed":
            raise ValueError(f"no open proposal {pid}")
        if not accept:
            p["status"] = "rejected"
            self.log.emit("director", "proposal_rejected", task=pid)
            return {"status": "rejected"}
        if p["lint"] or not p["applied"]:
            raise ValueError("this proposal has lint problems or no edit applied; it cannot be accepted")
        g = self.cfg.get("guard")
        if g and g.get("enabled", True):
            # D0's guard before the edit is kept: the candidate memory re-plans D0's cases (k samples, blind judge)
            # and must hold `min`. Minutes and a few dollars, so it runs in the background.
            if not self.claim("guarding", ("idle",)):
                raise ValueError(f"the director is {self.phase}")
            p["status"] = "guarding"
            threading.Thread(target=self._guard_then_commit, args=(pid,), daemon=True).start()
            return {"status": "guarding"}
        return self._commit(pid, None)

    def _guard_then_commit(self, pid: str) -> None:
        from .guard import run_guard
        p, g = self.proposals[pid], self.cfg["guard"]
        try:
            self.log.emit("director", "guard_start", task=pid,
                          text=f"D0 guard: {g.get('k', 1)} sample(s) of D0's cases on the candidate memory, must hold "
                               f"{g.get('min', 0.86)}")
            # a fresh directory per guard run: proposal ids restart with the server, and the runner resumes what it finds
            res = run_guard(Path(p["candidate"]), self.home / "guards" / f"{pid}-{time.strftime('%Y%m%d-%H%M%S')}", g,
                            lambda line: self.log.emit("director", "guard_progress", task=pid, text=line),
                            current=self.memory)
            (Path(p["candidate"]).parent / "guard.json").write_text(json.dumps(res))
            if res.get("mode") == "targeted":
                text = (f"targeted guard: {len(res.get('relevant', []))} relevant case(s)"
                        + (f" · broke {', '.join(res['broke'])}" if res.get("broke") else " · none broke")
                        + f" · ${res.get('cost_usd', 0)}")
            else:
                vs = (f"vs current {res['current_mean']} (wins {res['wins']}, losses {res['losses']})"
                      if res.get("mode") == "paired" else f"(min {res['min']})")
                text = f"guard {res['mean']} {vs} · ${res['cost_usd']}"
            self.log.emit("director", "guard_done", task=pid, guard=res, text=text)
            if res.get("incomplete"):
                p["status"] = "proposed"            # not a verdict: the guard could not measure every case
                self.log.emit("director", "error", task=pid,
                              text="the D0 guard could not measure every case (plans or verdicts missing after retries); "
                                   "nothing was decided — Accept again to rerun it")
                return
            if not res["passed"]:
                p["status"] = "guard_failed"
                self.log.emit("director", "proposal_rejected", task=pid,
                              text=f"the D0 guard did not hold ({res.get('broke') or res.get('mean')}); the memory is unchanged")
                return
            if g.get("positive", True):
                # D7: the edit's own probes, current memory vs candidate — keep only a lesson that changes the plan
                from .guard import positive_probe
                self.log.emit("director", "guard_progress", task=pid,
                              text=f"[guard] positive probe: {len(p['probes'])} probe(s), current memory vs candidate")
                pr = positive_probe(p["probes"], self.memory, Path(p["candidate"]), self.home / "guards" / f"{pid}-probe-{time.strftime('%Y%m%d-%H%M%S')}", g,
                                    lambda line: self.log.emit("director", "guard_progress", task=pid, text=line))
                res["probe"] = pr
                (Path(p["candidate"]).parent / "guard.json").write_text(json.dumps(res))
                self.log.emit("director", "guard_done", task=pid, guard=res,
                              text=f"positive probe: {pr['verdict']} (current {pr.get('current')} → candidate {pr.get('candidate')})")
                if not pr["teaches"]:
                    p["status"] = pr["verdict"].lower().replace(" ", "_")
                    why = {"REDUNDANT": "the memory (or the model) already does this",
                           "NO EFFECT": "the edit did not change the plans its probes test",
                           "NO PROBE": "the reflection wrote no probe to test it",
                           "INCOMPLETE": "the probes could not be measured"}.get(pr["verdict"], pr["verdict"])
                    self.log.emit("director", "proposal_rejected", task=pid,
                                  text=f"not kept — {pr['verdict']}: {why}; the memory is unchanged")
                    return
            self._commit(pid, res)
        except Exception as e:
            p["status"] = "proposed"
            self.log.emit("director", "error", task=pid, text=f"guard: {e!r}"[:400])
        finally:
            self.phase = "idle"

    def _commit(self, pid: str, guard: dict | None) -> dict:
        from .memory_vcs import LiveStore
        p = self.proposals[pid]
        if self.store is None:
            self.store = LiveStore(self.home / "vcs", self.memory, self.cfg.get("agentvcs"))
        patch = self.store.commit(Path(p["candidate"]), p["rationale"], p["applied"], guard)
        shutil.rmtree(self.memory)
        shutil.copytree(p["candidate"], self.memory)
        p.update(status="accepted", patch=patch)
        self.log.emit("director", "memory_updated", task=pid, text=f"patch {patch['patch_id'][:15]}… applied · "
                      f"{', '.join(patch['dimensions'])}", patch=patch)
        return {"status": "accepted", **patch}
