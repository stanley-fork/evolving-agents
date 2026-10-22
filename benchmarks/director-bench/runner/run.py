"""Chained-session protocol for one (family, config, seed) run."""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from metrics.signatures import SIGNATURE_VERSION, ci_signatures, tool_error_signature, verifier_signatures
from runner import __version__
from runner.config import BenchError, Config, Family, expand, expand_tree
from runner.session import combine, run_session

# Host variables forwarded into the Claude session (plus --pass-env names).
SESSION_ENV_ALLOW = {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "USER", "LOGNAME", "SHELL",
                     "TMPDIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy",
                     "no_proxy", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE",
                     "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                     "IS_SANDBOX"}   # IS_SANDBOX=1: lets bypassPermissions run as root inside a container
SESSION_ENV_PREFIXES = ("ANTHROPIC_", "AWS_", "CLOUD_ML_", "GOOGLE_", "VERTEX_")
GIT_ID = {"GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@invalid",
          "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@invalid",
          "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}


@dataclass
class RunOptions:
    family: Family
    config: Config
    seed: int
    model: str
    results_dir: Path
    claude_bin: str = "claude"
    permission_mode: str = "bypassPermissions"
    run_id: str | None = None
    sandbox_dir: Path | None = None
    allow_todo: bool = False
    pass_env: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)   # tests / operator overrides
    keep_sandbox: bool = False
    max_tasks: int | None = None                               # smoke runs only


def task_order(family: Family, seed: int) -> list[str]:
    """first + seed-permuted middle + last. Depends only on (family, seed): the
    same seed gives the same order under every config."""
    middle = [t for t in family.tasks if t not in family.first and t not in family.last]
    random.Random(f"{family.name}/{seed}").shuffle(middle)
    return family.first + middle + family.last


def feedback_prompt(ci_lines: list[str], round_no: int) -> str:
    """The review-round prompt: only the verifier's CI lines, never its other output."""
    body = "\n".join(ci_lines)
    return (f"CI and code review on your change (round {round_no}) came back with:\n\n{body}\n\n"
            "Please address this feedback.")


def family_hash(family: Family) -> str:
    """Hash of every file in the family dir: results from edited families never mix."""
    h = hashlib.sha256()
    for p in sorted(family.dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(family.dir).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def sh(cmd: str, env: dict[str, str], cwd: Path, log: Path, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a hook command under bash with -euo pipefail; log stdout+stderr."""
    p = subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], env=env, cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"$ {cmd}\n{p.stdout}{p.stderr}[exit {p.returncode}]\n")
    return p


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)


def tree_contains(root: Path, needle: str) -> bool:
    data = needle.encode()
    return any(p.is_file() and data in p.read_bytes() for p in root.rglob("*")) if root.exists() else False


class Runner:
    def __init__(self, opts: RunOptions):
        self.o = opts
        fam, cfg = opts.family, opts.config
        if cfg.todos() and not opts.allow_todo:
            raise BenchError(f"config {cfg.name} has unresolved TODO markers (use --allow-todo to run anyway):\n  "
                             + "\n  ".join(cfg.todos()))
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.run_id = opts.run_id or f"{fam.name}-{cfg.name}-s{opts.seed}-{stamp}"
        self.out = opts.results_dir / self.run_id
        if self.out.exists():
            raise BenchError(f"results for run {self.run_id} already exist: {self.out}")
        base = opts.sandbox_dir or Path(tempfile.gettempdir()) / "director-bench-sandbox"
        # Outside the bench checkout, so the agent cannot stumble on verify/ or reference/.
        self.sandbox = (base / self.run_id).resolve()
        self.work = self.sandbox / "work" / fam.name     # same path every task (native memory keys on cwd)
        self.memory = self.sandbox / "memory"
        self.home = self.sandbox / "home"
        self.source = self.sandbox / "source"
        self.order = task_order(fam, opts.seed)[: opts.max_tasks or None]
        if os.sep in opts.claude_bin:   # sessions run in the workdir: make relative paths absolute
            opts.claude_bin = str(Path(opts.claude_bin).resolve())

    # ---------- environment ----------
    def bench_env(self, pos: int = 0, task: str = "", trace: str = "", passed: str = "",
                  **post: str) -> dict[str, str]:
        return {**post,
            "BENCH_RUN_ID": self.run_id, "BENCH_FAMILY": self.o.family.name,
            "BENCH_CONFIG": self.o.config.name, "BENCH_SEED": str(self.o.seed),
            "BENCH_TASK_ID": task, "BENCH_TASK_INDEX": str(pos) if pos else "",
            "BENCH_WORKDIR": str(self.work), "BENCH_MEMORY_DIR": str(self.memory),
            "BENCH_HOME": str(self.home), "BENCH_TRACE": trace, "BENCH_PASSED": passed,
            "BENCH_RESULTS_DIR": str(self.out), "BENCH_MODEL": self.o.model,
        }

    def expansion_env(self, bench: dict[str, str]) -> dict[str, str]:
        return {**os.environ, **self.o.extra_env, **bench}

    def hook_env(self, bench: dict[str, str], phase: str) -> dict[str, str]:
        xenv = self.expansion_env(bench)
        cfg_env = expand_tree(self.o.config.env, xenv)
        return {**xenv, **cfg_env, "BENCH_PHASE": phase, "HOME": str(self.home),
                "CLAUDE_CONFIG_DIR": str(self.home / ".claude")}

    def session_env(self, bench: dict[str, str]) -> dict[str, str]:
        host = {**os.environ, **self.o.extra_env}
        env = {k: v for k, v in host.items()
               if k in SESSION_ENV_ALLOW or k.startswith(SESSION_ENV_PREFIXES) or k in self.o.pass_env}
        env.update(self.o.extra_env)
        env.update(expand_tree(self.o.config.env, self.expansion_env(bench)))
        env.update({"HOME": str(self.home), "CLAUDE_CONFIG_DIR": str(self.home / ".claude")})
        return env

    # ---------- repo snapshots ----------
    def build_source(self) -> None:
        repo = self.o.family.raw["repo"]
        if "build" in repo:
            subprocess.run([sys.executable, str(self.o.family.dir / repo["build"]), str(self.source)],
                           check=True, capture_output=True)
        elif "git" in repo:
            subprocess.run(["git", "clone", "-q", "--mirror", str(repo["git"]), str(self.source)],
                           check=True, capture_output=True)
        else:
            raise BenchError("family.yaml repo needs `build:` or `git:`")

    def restore(self, task: str) -> None:
        """Fresh copy of the task's snapshot as a single-commit repo (no history, no other tags)."""
        ref = self.o.family.snapshots[task]
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        tar = subprocess.run(["git", "-C", str(self.source), "archive", ref], check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            try:
                tf.extractall(self.work, filter="tar")
            except TypeError:  # Python < 3.10.12 has no extraction filters
                tf.extractall(self.work)
        env = {**os.environ, **GIT_ID, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "snapshot"]):
            subprocess.run(["git", *args], cwd=self.work, env=env, check=True, capture_output=True)

    def fresh_home(self, reset: bool) -> None:
        if reset and self.home.exists():
            shutil.rmtree(self.home)
        (self.home / ".claude").mkdir(parents=True, exist_ok=True)
        (self.home / ".gitconfig").write_text("[user]\n\tname = bench\n\temail = bench@invalid\n")

    def snapshot_memory(self, label: str) -> Path:
        dst = self.out / "memory" / label
        copy_tree(self.memory, dst / "store")
        if self.o.config.home_scope == "run":
            copy_tree(self.home, dst / "home")
        dst.mkdir(parents=True, exist_ok=True)
        return dst

    # ---------- steps ----------
    def run_hooks(self, phase: str, cmds: list[str], bench: dict[str, str], tag: str) -> list[subprocess.CompletedProcess]:
        out = []
        for cmd in cmds:
            p = sh(cmd, self.hook_env(bench, phase), self.sandbox, self.out / "hooks" / f"{tag}.log")
            if p.returncode != 0:
                raise BenchError(f"{phase} hook failed (exit {p.returncode}): {cmd}\n{p.stderr[-2000:]}")
            out.append(p)
        return out

    def verify(self, task: str, label: str) -> tuple[bool, int, list[str], list[str]]:
        """Run the oracle on a throwaway copy so nothing it does reaches the workdir or memory.

        Returns (passed, exit code, BENCH-FAIL lines, CI lines). CI lines ("CI ...") are the
        CI-log/reviewer-style feedback a verifier may print; they are the only verifier output
        ever shown to the agent (in the feedback round)."""
        log = self.out / "verify" / f"{label}.log"
        with tempfile.TemporaryDirectory(prefix="bench-verify-") as tmp:
            copy = Path(tmp) / "work"
            shutil.copytree(self.work, copy, symlinks=True)
            env = {**os.environ, "BENCH_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1",
                   "BENCH_VERIFY_MODE": "ci"}
            try:
                p = subprocess.run(["bash", str(self.o.family.verifier(task)), str(copy)], cwd=copy,
                                   env=env, capture_output=True, text=True, timeout=900)
                # report paths as if the oracle had run in the workdir itself
                code, output = p.returncode, (p.stdout + p.stderr).replace(str(copy), str(self.work))
            except subprocess.TimeoutExpired:
                code, output = 124, "BENCH-FAIL type=verify file=- symbol=timeout msg=verifier timed out\n"
        log.write_text(output, encoding="utf-8")
        lines = [ln for ln in output.splitlines() if ln.startswith("BENCH-FAIL ")]
        if code != 0 and not lines:
            tail = [ln for ln in output.splitlines() if ln.strip()][-1:] or [f"exit {code}"]
            lines = [f"BENCH-FAIL type=verify file=- symbol=exit msg={tail[0]}"]
        ci = [ln.rstrip() for ln in output.splitlines() if ln.startswith(("CI ", "CI:"))]
        if code != 0 and not ci:
            ci = ["CI: checks failed"]
        return code == 0, code, lines, ci

    def consolidate(self, pos: int, covers: list[int]) -> dict:
        bench = self.bench_env(pos)
        bench["BENCH_CONSOLIDATE_POSITIONS"] = ",".join(map(str, covers))
        t0 = time.monotonic()
        procs = self.run_hooks("consolidate", self.o.config.consolidate, bench, f"consolidate-{pos:02d}")
        rec = {"after_position": pos, "covers": covers, "input_tokens": 0, "output_tokens": 0,
               "cost_usd": 0.0, "wall_seconds": round(time.monotonic() - t0, 3)}
        for p in procs:
            last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:]
            try:
                data = json.loads(last[0]) if last else {}
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                rec["input_tokens"] += int(data.get("input_tokens") or 0)
                rec["output_tokens"] += int(data.get("output_tokens") or 0)
                rec["cost_usd"] += float(data.get("cost_usd") or 0.0)
        return rec

    def claude_version(self) -> str:
        p = subprocess.run([self.o.claude_bin, "--version"], capture_output=True, text=True, timeout=60)
        if p.returncode != 0 or not p.stdout.strip():
            raise BenchError(f"`{self.o.claude_bin} --version` failed: {p.stderr.strip()}")
        return p.stdout.strip().splitlines()[0]

    def memory_version(self) -> str:
        cmd = self.o.config.version_cmd
        if not cmd:
            return "none"
        env = self.hook_env(self.bench_env(), "version")
        p = subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], env=env, capture_output=True, text=True)
        if p.returncode != 0 or not p.stdout.strip():
            raise BenchError(f"version_cmd failed for {self.o.config.name}: {cmd}\n{p.stderr.strip()}")
        return p.stdout.strip().splitlines()[-1]

    def write_json(self, name: str, data) -> None:
        (self.out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def append_jsonl(self, name: str, data: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, sort_keys=True) + "\n")

    # ---------- main loop ----------
    def run(self) -> Path:
        fam, cfg, o = self.o.family, self.o.config, self.o
        for d in ("traces", "verify", "hooks", "memory"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        if self.sandbox.exists():
            shutil.rmtree(self.sandbox)
        self.memory.mkdir(parents=True)
        self.fresh_home(reset=True)
        pins = {
            "model": o.model, "claude_version": self.claude_version(),
            "memory_system_version": self.memory_version(), "seed": o.seed,
            "permission_mode": o.permission_mode, "family": fam.name, "family_hash": family_hash(fam),
            "config": cfg.name, "config_hash": hashlib.sha256(cfg.text.encode()).hexdigest()[:16],
            "signature_version": SIGNATURE_VERSION, "bench_version": __version__,
        }
        self.write_json("pins.json", pins)
        meta = {"run_id": self.run_id, "family": fam.name, "config": cfg.name, "seed": o.seed,
                "order": self.order, "budget": {"max_turns": fam.max_turns, "max_tokens": fam.max_tokens,
                                                "wall_seconds": fam.wall_seconds},
                "consolidate_every": cfg.every if cfg.consolidate else 0, "home_scope": cfg.home_scope,
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running",
                "sandbox": str(self.sandbox), "pins": pins}
        self.write_json("run.json", meta)
        try:
            self.build_source()
            self.run_hooks("setup", cfg.setup, self.bench_env(), "setup")
            n = len(self.order)
            for pos, task in enumerate(self.order, start=1):
                self.run_task(pos, task)
                if cfg.consolidate and cfg.every > 0 and pos % cfg.every == 0 and pos < n:
                    covers = list(range(pos - cfg.every + 1, pos + 1))
                    self.append_jsonl("consolidations.jsonl", self.consolidate(pos, covers))
            self.snapshot_memory("final")
            meta["status"] = "complete"
        except BaseException as exc:
            meta["status"] = f"failed: {exc}"[:2000]
            raise
        finally:
            meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.write_json("run.json", meta)
            if not o.keep_sandbox and self.sandbox.exists():
                shutil.rmtree(self.sandbox, ignore_errors=True)
        return self.out

    def run_task(self, pos: int, task: str) -> None:
        fam, cfg, o = self.o.family, self.o.config, self.o
        label = f"{pos:02d}-{task}"
        trace = self.out / "traces" / f"{label}.jsonl"
        self.restore(task)                                  # 1. snapshot of task n
        self.fresh_home(reset=cfg.home_scope == "task")     #    HOME persists only if it is the store
        snap = self.snapshot_memory(label)                  #    what this session can see
        bench = self.bench_env(pos, task, str(trace))
        self.run_hooks("pre_session", cfg.pre_session, bench, label)

        xenv = self.expansion_env(bench)
        cmd = [o.claude_bin, "-p", fam.statement(task), "--output-format", "stream-json", "--verbose",
               "--max-turns", str(fam.max_turns), "--model", o.model, "--permission-mode", o.permission_mode]
        if cfg.settings:
            settings = self.sandbox / f"settings-{label}.json"
            settings.write_text(json.dumps(expand_tree(cfg.settings, xenv), indent=2))
            cmd += ["--settings", str(settings)]
        cmd += [expand(a, xenv) for a in cfg.extra_args]
        res = run_session(cmd, self.work, self.session_env(bench), trace,   # 2. headless session
                          self.out / "traces" / f"{label}.stderr", fam.max_tokens, fam.wall_seconds)
        if res.status == "error" and res.turns == 0:
            # Claude Code never started (auth, flags, root refusal): an infrastructure failure, not a task
            # failure. Counting it as a failed task would corrupt every metric, so the run stops.
            err = (self.out / "traces" / f"{label}.stderr")
            tail = err.read_text(errors="replace").strip()[-600:] if err.exists() else ""
            raise BenchError(f"session for task {task} did not start (exit {res.exit_code}): {tail}")

        ok, code, fail_lines, ci_lines = self.verify(task, label)      # 3. oracle
        first_pass, first_fail, first_ci = ok and res.within_budget, fail_lines, ci_lines
        rounds, feedback = [res], []
        # 3b. review round(s): resume the same session with the CI/reviewer feedback
        while not ok and len(feedback) < fam.feedback_rounds and res.within_budget and res.session_id:
            used = combine(rounds)
            turns_left, tokens_left = fam.max_turns - used.turns, fam.max_tokens - used.tokens
            wall_left = int(fam.wall_seconds - used.wall_seconds)
            if turns_left < 1 or tokens_left < 1 or wall_left < 1:
                break
            n = len(feedback) + 1
            text = feedback_prompt(ci_lines, n)
            feedback.append(text)
            with open(trace, "a", encoding="utf-8") as fh:   # the feedback turn, as the session saw it
                fh.write(json.dumps({"type": "user", "bench_feedback_round": n, "session_id": res.session_id,
                                     "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
                         + "\n")
            rcmd = [o.claude_bin, "-p", text, "--resume", res.session_id] + cmd[3:]
            rcmd[rcmd.index("--max-turns") + 1] = str(turns_left)
            res = run_session(rcmd, self.work, self.session_env(bench), trace,
                              self.out / "traces" / f"{label}.stderr", tokens_left, wall_left, append=True)
            if res.status == "error" and res.turns == 0:
                raise BenchError(f"feedback round {n} for task {task} did not start (exit {res.exit_code})")
            rounds.append(res)
            ok, code, fail_lines, ci_lines = self.verify(task, f"{label}-r{n}")
        total = combine(rounds)
        passed = ok and total.within_budget
        fb_file = self.out / "feedback" / f"{label}.txt"
        fb_file.parent.mkdir(exist_ok=True)
        fb_file.write_text("\n\n".join(feedback), encoding="utf-8")
        # What CI says about the final state when it still fails: CI lines only, as a person would see them on the
        # last push. Not shown to the session; offered to post-session hooks, so any memory with hooks can use it.
        final_ci = [] if passed else ci_lines
        final_file = self.out / "feedback" / f"{label}.final.txt"
        final_file.write_text("\n".join(final_ci), encoding="utf-8")
        kind = "trap" if task == fam.trap_task else "retention" if task == fam.retention_task else "main"
        roots = [str(self.work), str(self.sandbox)]
        record = {
            "position": pos, "task": task, "kind": kind, "snapshot": fam.snapshots[task],
            "passed": passed, "first_pass": first_pass, "rounds_used": len(feedback),
            "verifier_passed": ok, "verifier_exit": code,
            "verify_fail_lines": fail_lines,
            "verifier_signatures": verifier_signatures(fail_lines, roots),
            "first_verify_fail_lines": first_fail,
            "ci_feedback_lines": first_ci if not first_pass else [],
            "ci_feedback_signatures": ci_signatures(first_ci, roots) if not first_pass else [],
            "tool_errors": total.tool_errors,
            "tool_error_signatures": [tool_error_signature(e, roots) for e in total.tool_errors],
            "session": total.to_dict(), "rounds": [r.to_dict() for r in rounds],
            "final_ci_feedback_lines": final_ci,
            "trace": str(trace.relative_to(self.out)), "feedback": str(fb_file.relative_to(self.out)),
        }
        if kind == "trap" and fam.obsolete_marker:
            record["obsolete_marker_in_memory"] = tree_contains(snap, fam.obsolete_marker)
        self.append_jsonl("tasks.jsonl", record)
        post = self.bench_env(pos, task, str(trace), "true" if passed else "false",
                              BENCH_FIRST_PASS="true" if first_pass else "false",
                              BENCH_ROUNDS=str(len(feedback)), BENCH_FEEDBACK=str(fb_file),
                              BENCH_FINAL_FEEDBACK=str(final_file))
        self.run_hooks("post_session", cfg.post_session, post, label)
