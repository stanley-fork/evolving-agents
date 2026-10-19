"""Meta-recipes: Claude writes them from the memory, code runs them, a local model only picks and answers yes/no.

D15 (docs/BRIEF-D15.md) measured this split: gemma4:12b following Claude's recipes planned at 0.806 against 0.854 for
Sonnet walking the memory with tools (3/3/6 per case), in ~10 s and at no cost. Standard library only.

A recipe file (`recipes.json`, beside the memory's pages) is a JSON array of
    {"id": str, "trigger": str, "steps": [step, ...]}
where a step is either
    {"do": "<concrete action>", "cite": "Page#id"}                       always taken
    {"ask": "<yes/no about a fact in the request>", "yes": [...], "no": [...]}  a branch the model answers

    python -m director.recipes lint <memory_dir>
    python -m director.recipes stamp <memory_dir>      record the memory version the recipes were written from
    python -m director.recipes plan <memory_dir> "<request>" [--model gemma4:12b-mlx]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Protocol

from .memory.format import statement_ids, version

MAX_RECIPES_PICKED = 2
MAX_DEPTH = 2
DEFAULT_MODEL = os.environ.get("DIRECTOR_LOCAL_MODEL", "gemma4:12b-mlx")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


class Answerer(Protocol):
    def pick(self, request: str, triggers: list[str]) -> list[int]: ...
    def yes_no(self, request: str, question: str) -> bool: ...


class Ollama:
    """A local model behind Ollama, every answer constrained by a JSON schema."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA, temperature: float = 0.3):
        self.model, self.url, self.temperature = model, url, temperature
        self.calls, self.failures = 0, 0

    def _json(self, prompt: str, schema: dict):
        body = {"model": self.model, "stream": False, "think": False, "format": schema,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "num_ctx": 8192}}
        for _ in range(2):
            self.calls += 1
            req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                return json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"])
            except Exception:  # noqa: BLE001 — counted, and the step falls to its safe default
                continue
        self.failures += 1
        return None

    def pick(self, request: str, triggers: list[str]) -> list[int]:
        listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triggers))
        a = self._json(f"Request:\n{request}\n\nWhich of these types of request is it? Pick at most "
                       f"{MAX_RECIPES_PICKED}.\n{listing}\n\nReturn {{\"picked\": [numbers]}}.",
                       {"type": "object", "required": ["picked"], "properties": {"picked": {
                           "type": "array", "items": {"type": "integer", "minimum": 1, "maximum": len(triggers)},
                           "maxItems": MAX_RECIPES_PICKED}}})
        nums = (a or {}).get("picked", [])
        return list(dict.fromkeys(i - 1 for i in nums if isinstance(i, int) and 1 <= i <= len(triggers)))[
            :MAX_RECIPES_PICKED]

    def yes_no(self, request: str, question: str) -> bool:
        a = self._json(f"Request:\n{request}\n\nQuestion about the request: {question}\nAnswer yes or no.",
                       {"type": "object", "required": ["answer"], "properties": {"answer": {"enum": ["yes", "no"]}}})
        return (a or {}).get("answer") == "yes"


# ------------------------------------------------------------------ the file

def path_for(memory: Path) -> Path:
    return Path(memory) / "recipes.json"


def stamp_path(memory: Path) -> Path:
    return Path(memory) / "recipes.stamp.json"


def stamp(memory: Path) -> dict:
    """Record that the recipes were written from the memory as it is now."""
    rec = {"memory_version": version(memory), "stamped": time.strftime("%Y-%m-%dT%H:%M:%S")}
    stamp_path(memory).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def freshness(memory: Path) -> dict:
    """Whether the recipes were written from the memory as it is now. An accepted lesson changes the memory, and
    recipes written before it do not know it: `stale`, and /director:recipes should be run again."""
    cur, built = version(memory), None
    try:
        built = json.loads(stamp_path(memory).read_text(encoding="utf-8")).get("memory_version")
    except (OSError, ValueError):
        pass
    return {"memory_version": cur, "recipes_built_from": built, "unstamped": built is None,
            "stale": built is not None and built != cur}


def load(memory: Path) -> list[dict]:
    p = path_for(memory)
    if not p.is_file():
        raise FileNotFoundError(f"no recipes at {p}; run /director:recipes")
    return json.loads(p.read_text(encoding="utf-8"))


def lint(recipes: list[dict], known: set[str]) -> list[str]:
    """Problems that would make a recipe unrunnable or uncheckable; empty when clean."""
    problems, seen = [], set()

    def steps(xs, rid: str, depth: int) -> None:
        if not isinstance(xs, list):
            problems.append(f"{rid}: steps must be a list")
            return
        for s in xs:
            if not isinstance(s, dict):
                problems.append(f"{rid}: a step is not an object")
            elif "do" in s:
                if not s.get("cite"):
                    problems.append(f"{rid}: do-step without cite: {s['do'][:60]!r}")
                elif s["cite"] not in known:
                    problems.append(f"{rid}: cites a statement that does not exist: {s['cite']}")
            elif "ask" in s:
                if depth >= MAX_DEPTH:
                    problems.append(f"{rid}: branches nested deeper than {MAX_DEPTH}: {s['ask'][:60]!r}")
                for k in ("yes", "no"):
                    steps(s.get(k, []), rid, depth + 1)
            else:
                problems.append(f"{rid}: a step is neither do nor ask")

    if not isinstance(recipes, list) or not recipes:
        return ["recipes.json must be a non-empty JSON array"]
    for r in recipes:
        rid = r.get("id") or "?"
        if rid in seen:
            problems.append(f"duplicate recipe id {rid}")
        seen.add(rid)
        if not r.get("trigger"):
            problems.append(f"{rid}: no trigger")
        steps(r.get("steps"), rid, 0)
    return problems


def cited(recipes: list[dict]) -> set[str]:
    out: set[str] = set()

    def walk(xs):
        for s in xs or []:
            if "do" in s and s.get("cite"):
                out.add(s["cite"])
            elif "ask" in s:
                walk(s.get("yes"))
                walk(s.get("no"))

    for r in recipes:
        walk(r.get("steps"))
    return out


# ------------------------------------------------------------------ running one

def _run(steps: list[dict], request: str, who: Answerer, answers: list) -> list[tuple[str, str]]:
    out = []
    for s in steps:
        if "do" in s:
            out.append((s["do"], s.get("cite", "")))
        elif "ask" in s:
            yes = who.yes_no(request, s["ask"])
            answers.append({"ask": s["ask"], "answer": "yes" if yes else "no"})
            out += _run(s.get("yes" if yes else "no", []), request, who, answers)
    return out


def plan(request: str, recipes: list[dict], who: Answerer) -> dict:
    """The plan is the do-steps reached, in order, de-duplicated; the model writes none of its text."""
    picked = who.pick(request, [r["trigger"] for r in recipes])
    answers: list = []
    steps: list[tuple[str, str]] = []
    for i in picked:
        steps += _run(recipes[i]["steps"], request, who, answers)
    lines, seen = [], set()
    for do, cite in steps:
        if (do, cite) in seen:
            continue
        seen.add((do, cite))
        lines.append(f"{len(lines) + 1}. {do}" + (f" [[{cite}]]" if cite else ""))
    return {"plan": "\n".join(lines), "recipes": [recipes[i]["id"] for i in picked], "answers": answers,
            "cites": list(dict.fromkeys(c for _, c in steps if c)),
            "abstain": not picked}


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "lint":
        mem = Path(argv[1])
        rs = load(mem)
        problems = lint(rs, statement_ids(mem))
        known = statement_ids(mem)
        print(json.dumps({"recipes": len(rs), "problems": problems,
                          "statements_cited": len(cited(rs) & known), "statements": len(known),
                          **freshness(mem)}, indent=2))
        return 1 if problems else 0
    if len(argv) >= 2 and argv[0] == "stamp":
        print(json.dumps(stamp(Path(argv[1]))))
        return 0
    if len(argv) >= 3 and argv[0] == "plan":
        model = argv[argv.index("--model") + 1] if "--model" in argv else DEFAULT_MODEL
        print(json.dumps(plan(argv[2], load(Path(argv[1])), Ollama(model)), ensure_ascii=False, indent=2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
