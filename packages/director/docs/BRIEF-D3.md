# D3 — does the director learn where the user wants work done?

**Written 2026-10-14, before anything runs.** The user's instruction: "implementa lo que queda del plan" — which
includes running the local lane on the user's Mac (llama.cpp, Gemma 4 E4B Q8_0, ~8 GB, a few minutes, stopped
after; taken as the authorisation the plan asked for).

## What and why

D3 gives the user control of the lanes: a plan waits; any queued task can be moved between the frontier and
the local lane before it runs; the move is logged and becomes part of the episode the director reflects on
(D2b). The question: **after one move and one reflection, does the planner put similar tasks in the lane the
user chose — and only those?**

## The preference

The planner's built-in rule sends text-only work to the local lane when it is online. The user's preference,
taught by one move: **public-facing text (a release note, a README, the website, an announcement) is drafted on
the frontier, even when it is text-only**; private, internal text stays local.

## Run (one isolated director: a copy of the live memory, its own log and agentvcs store)

1. **R0** — the planner on 4 probes, k = 2, the local lane **online** (llama.cpp running). Lanes are read from
   the plan's JSON — no judge.
2. **Episode** — request "draft the release note for agentvcs's bisect goldens from these changes: …"; the plan
   puts the drafting on local; the user's move (scripted, standing in for the user) sends it to the frontier;
   the plan runs; reflection with the correction *public-facing text is drafted on the frontier*; the proposal
   is accepted only if it lints clean (the accept gate, versioned in agentvcs).
3. **R1** — the same 4 probes on the edited memory.

Probes (`d3/probes.jsonl`): three public-facing drafts (expected **frontier**) and one internal summary
(expected **local** — the control against over-generalising).

| verdict (fixed here) | condition |
|---|---|
| **STOP — no headroom** | R0 already sends ≥ 0.5 of the public-facing tasks to the frontier |
| **LEARNS THE LANE** | R1 sends ≥ 0.8 of the public-facing tasks to the frontier **and** keeps ≥ 0.75 of the internal one local |
| **OVER-GENERALISES** | the first holds, the control does not — it learned "frontier", not the preference |
| **DOES NOT LEARN** | otherwise |

Score of a probe sample = the fraction of its plan's tasks in the expected lane. Cap **US$15**.

## Result [ran] 2026-10-14 — **LEARNS THE LANE**

llama.cpp (Homebrew, Metal), Gemma 4 E4B Q8_0, the user's MacBook Air M4 16 GB, local lane online throughout.

| | R0 (before) | R1 (after one move + one reflection) |
|---|---|---|
| public-facing tasks planned on the frontier | **0.0** (0 / 6) | **1.0** (6 / 6) |
| internal summary kept local (the control) | 1.0 | **1.0** |

**Episode:** the release-note task was planned local; the scripted move sent it to the frontier; it ran there;
the reflection (Claude Opus 5.5) proposed one statement, `Task_Lanes#public_text_on_frontier` — *who reads it?
if anyone outside the lab, frontier; internal text stays local* — linked by itself to the confidentiality rules;
0 lint problems; accepted through the accept gate and applied as an agentvcs patch (dimensions `memory.index`,
`memory.task_lanes`). By the table fixed above: **LEARNS THE LANE**, no over-generalisation. US$7.15 of US$15.

**What this does and does not show.** One preference, taught once, on four probes with k = 2, read from the
plans' lanes (deterministic). The move and the correction were scripted *standing in for the user*; the D3
memory is an isolated copy, so the live memory does **not** carry this preference until the user teaches it
from the interface.

**The local lane, live [ran]:** after R1, the running director (not the isolated one) took an internal summary
request, planned it local, waited for Run, and the E4B on llama.cpp answered in 39 s; nothing left the machine.
llama.cpp was stopped afterwards.
