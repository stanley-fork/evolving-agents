# D9: do trajectory entries written by Claude make the memory walkable for the 12B?

**Written 2026-10-17, before anything runs.** Follows D8 (NOT NAVIGABLE: core recall 0.321, precision 0.894, 12/0
against BM25). D8 found that every missed core statement was an entry of some Index situation, but each case's core
spreads over 6–25 situations and the walk lets the 12B pick 4. The bottleneck is how the memory is organised, which is
Claude's job. D9 tests one restructuring: **trajectories**.

## The treatment

A **trajectory** is an Index line for a *type of task* (a workflow: start or resume a session, design an experiment,
launch a run, read a result, land a PR, publish…). It points at every statement that type of task needs, so one choice
of the 12B reaches what used to take several.

- **Where:** appended to `Index.md` under a `## Trajectories` heading, in the same line format
  (`- <task type> → [[Page#id]], …`). The walk's parser reads them like any situation. Pages and statements are not
  touched: 287 statements, same text.
- **Size:** at most 20 statements per trajectory, at most 20 trajectories. A trajectory that bundles everything would
  win recall by construction; the caps and the precision floor below are there against that.
- **Who writes them:** a **fresh Claude agent** (Opus 5.5) that sees only the memory's pages and Index. It is told not
  to open `d0/`, `d8/`, `~/.config/director/d0` or any plan, and to report every file it read.
- **Why a fresh agent:** the session that wrote this brief has seen D0's case list and one case's core (c01). If it wrote
  the trajectories, they could be tailored to the held-out cases. The fresh agent has not seen them.

## The walk (unchanged)

D8's runner, unchanged. It gains only `--memory` (default: D8's memory) and is pointed at a copy
(`~/.config/director/d9/memory`): same model, prompts, schema, `MAX_SITUACIONES = 4`, `SALTOS = 2`, temperature 0,
`num_ctx` 16384. Same reference (Sonnet's D0 citations: core = both samples, broad = either), same 14 cases, now
held out. Same BM25 baseline, with k = the number of statements the 12B picked in that case.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **NAVIGABLE** | mean core recall ≥ 0.70 **and** 12B beats BM25 in more cases than it loses **and** mean precision vs broad ≥ 0.70 |
| **PARTIAL** | mean core recall 0.40–0.70, with the other two conditions met |
| **NOT NAVIGABLE** | otherwise |

**Also reported, not scored:**
- paired against D8, per case, on core recall (wins / losses / ties, exact sign test);
- picks per case;
- JSON failures;
- wall time;
- which trajectories the 12B chose.

**One design, no redesign on these data.** If the verdict is not NAVIGABLE, the misses are reported, and the next
restructuring is briefed fresh, from material other than D0. If it is NAVIGABLE or PARTIAL, the next step is the plan
quality measurement with D0's judge (paid, briefed separately).

## Parameters

| parameter | value |
|---|---|
| model | `gemma4:12b-mlx` on Ollama 0.40.3 (MacBook Air M4, 16 GB) |
| runs | one deterministic run |
| cost | US$0 for the walk (local), plus the fresh agent's session |

Abort rule: as in D8 (more than 5 JSON failures after one retry each).
