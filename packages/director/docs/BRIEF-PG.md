# PG: does the director's memory carry a team's way of working to an agent that is not Claude?

**Written 2026-10-21, before any PG plan runs.** With Opus 5.5 planning, the director's measured edge on following
rules is gone ([D0O](BRIEF-D0O.md), [F1O](BRIEF-F1O.md)). What the person says the director is for is something else:
the team's operational memory **outside** the model — owned, visible, and usable by other models and agents. The
cheapest part of that to measure is the last one: give the same memory to a model from another provider and see
whether the director's form of it helps that model plan the team's way.

## The planner — another provider, no tools

**gemini-4-argon** (Google, through the person's AI Studio key), one call per plan, **no tools**: whatever memory an
arm has is given in the prompt, every file under its path. This is the simplest agent there is; an agent with tools
would only add a way to read the same files.

## Arms — the same sources, three forms

| arm | what the prompt holds |
|---|---|
| **GA — the notes** | F1's arm A, byte for byte: the workspace `CLAUDE.md`, lora-kernel's `CLAUDE.md`, agents and skills, and the memory files, with `MEMORY.md` as the system prompt (about 80k tokens) |
| **GT — the director's memory** | F1's arm T, byte for byte: the same sources compiled into pages of linked statements, with `DIRECTOR.md`'s walk as the system prompt, told the pages are below instead of on disk (about 33k tokens) |
| **GF — floor** | nothing: the situation alone |

Information parity holds by construction (T was compiled from A's files only, D0). Cases: **F1's 14** (situations
from after the memory was frozen), k = 2, the same request as every arm of D0 and F1. Sequence: **GA first**
(headroom), then GT; GF runs either way (cheap).

**Privacy.** The memory names private work and Google receives the prompt: every term of the private list is
replaced by a fixed marker in the system prompt, the files and the situation, and the whole request is checked for
any term left — and for anything shaped like a key — before each call; a leak stops the run. Plans, verdicts and the
list stay private; only scores are published.

## Judges

D0's `judge.md`, blind: **Claude Opus 5.5** — the primary judge, from another provider than the planner — and
gemini-4-argon, reported beside it (it would be judging its own model).

## What decides it, written before it runs

| result (Opus judge) | verdict |
|---|---|
| GA ≥ 0.85 | **STOP — no headroom**: Gemini already plans the team's way from the plain notes |
| **GT − GA ≥ +0.15** | **THE MEMORY PORTS** — the director's form of the team's way of working helps an agent that is not Claude |
| −0.15 < GT − GA < +0.15 | **PARITY** — the director's form carries the method as well as the notes, in 40 % of the tokens (said with its cost) |
| GT − GA ≤ −0.15 | **IT DOES NOT PORT** — the compiled form loses something the notes kept, for this model |

Beside it: GF (does any memory help this model), per-case wins, cost and input tokens per plan.

## Spend

Gemini: about US$9 for the three arms (GA ≈ 0.20, GT ≈ 0.11, GF ≈ 0.02 per plan), inside the key's ledger; this run
stops itself at **US$15** of Gemini spend. Claude: the Opus judge, about US$3. The person authorised the whole step
("hacé todo") on 2026-10-21.

## What it does not answer

- One other model; a local model (gemma4:12b) was measured on recipes in D15, not on this bank.
- No tools: an agent that reads files one by one might use the notes differently.
- F1's bank was written by Claude; a planner from another provider may read its phrasing differently, the same for
  every arm.
