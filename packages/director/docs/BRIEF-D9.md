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

## Outcome (2026-10-17)

**NOT NAVIGABLE** by the fixed rule. Mean core recall is **0.284**, below 0.40. Precision is 0.897, BM25 is beaten
10/0 (4 ties, p = 0.002), there were 0 JSON failures in 39 calls, and picks average 6.1 per case. Against D8, per case,
D9 has 4 wins and 7 losses (p = 0.55): **no detectable difference**, see the noise below.
Data: `~/.config/director/d9/{walks.jsonl,report.json,diagnosis.jsonl}`.

**Diagnosis, computed after the verdict on the same cases (not a redesign).** `d8/diagnose_walk.py` replays the walk
and logs every closed question.

| | D8 | D9 |
|---|---|---|
| ceiling: core reachable with the best 4 situations | 0.741 | **0.938** |
| core statements reachable after the 12B's situation pick (of 242) | 107 | 135 |
| statements shown at the first statement step | 222 | 545 |
| of them picked | 63 (28 %) | 98 (18 %) |
| core statements kept of those shown | 53 / 107 (50 %) | 71 / 135 (53 %) |

1. **The trajectories did their job structurally.** The ceiling rose from 0.74 to 0.94, and the 12B used them: 27 of
   its 52 situation picks were trajectories.
2. **Two closed questions lose the rest, about equally.** At the situation step, the 12B reaches 135 of the 242 core
   statements where 4 better picks would reach ~227. It sees 107 options and mixes trajectories with the old, narrow
   situations. At the statement step, it keeps about half of the core it is shown, as in D8, and it picks a smaller
   share of a list that is now 2.5× longer. The 12B behaves as a strict filter (precision ~0.9): it says yes to what
   clearly applies and leaves out the rest.
3. **The runs are not deterministic.** Replaying the same walk at temperature 0 changed the picks in 3–4 of 14 cases
   (c03 in D9 went from 0.19 to 0.56; c10 in D8 from 0.36 to 0.08). The replay means are D8 0.297 and D9 0.316, the
   reverse of the scored order. One run per arm cannot separate D8 from D9.

**Consequences for the next brief** (to be written fresh, not run on these data as a redesign):
- The walk strategy, not only the memory's content, has to fit a strict filter:
  - trajectories as their own first layer (about 20 options, then the old situations only as a fallback);
  - a statement step that asks for exclusion ("which do not apply") or uses short yes/no batches instead of one
    selection over 50 options.
- Every arm needs k ≥ 3 replicates per case, compared on per-case means.
