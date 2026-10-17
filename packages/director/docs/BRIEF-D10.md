# D10: a walk strategy that fits a strict filter

**Written 2026-10-17, before anything runs.** In D9, Claude's trajectories raised the ceiling of a 4-pick walk from
0.74 to 0.94, but the walk lost that gain at its two closed questions. The diagnosis found that the 12B behaves as a
strict filter (precision ~0.9): it says yes to what clearly applies and leaves out the rest. It also found that Ollama
MLX at temperature 0 is not deterministic (3–4 of 14 cases change between identical runs). D10 changes the **strategy**
(the code that asks the questions) and leaves the memory alone.

## Arms

Both arms walk D9's memory, unchanged (`~/.config/director/d9/memory`, 87 situations + 20 trajectories, 286
statements). Model, schema, prompts' wording style, temperature 0 and `num_ctx` 16384 are as in D8.

| arm | strategy |
|---|---|
| **S9** (control) | D9's walk, unchanged: ≤ 4 picks among all 107 Index lines; one inclusion question over all entry statements; 2 link hops by inclusion |
| **S10** (treatment) | three changes, below |

S10's changes:

1. **Layered entry.** The 12B first picks **≤ 3 of the 20 trajectories**. Then it picks **≤ 2 of the 87 situations**
   ("which add something specific"). Two short lists replace one list of 107.
2. **Exclusion over the entry statements, in batches of 10.** For each batch: "which of these do NOT apply?". Every
   statement it does not exclude is kept. A strict filter that leaves out what does not clearly apply should, under
   exclusion, keep what does not clearly fail.
3. **Link hops unchanged** (2 hops, inclusion): exclusion there would pull in the neighbourhood of everything kept.

## Replicates and pairing

Each arm runs **k = 3** times over the 14 cases. A case's score is the mean over its 3 replicates. Arms are compared per
case on that mean (wins, losses, ties, exact sign test). BM25 is computed per replicate with k = that replicate's
number of picks, as in D8.

## Verdicts (fixed here)

These are for S10, on the mean over cases of the per-case means:

| verdict | condition |
|---|---|
| **NAVIGABLE** | core recall ≥ 0.70 **and** S10 beats BM25 in more cases than it loses **and** precision vs broad ≥ 0.70 |
| **PARTIAL** | core recall 0.40–0.70, with the other two conditions met |
| **NOT NAVIGABLE** | otherwise |

Also reported, not scored:
- S10 vs S9 per case (sign test);
- picks per case;
- the spread across replicates (how many cases change);
- JSON failures;
- wall time.

## Caveat declared before running

S10 was designed from a diagnosis made on D0's cases (D9's step logs). The changes are generic: they name no case,
page or statement. But the held-out bank is no longer untouched by the design. A result that passes here should be
confirmed on a new bank written from other material before it is trusted.

## Parameters

| parameter | value |
|---|---|
| model | `gemma4:12b-mlx` on Ollama 0.40.3 (MacBook Air M4, 16 GB) |
| runs | 2 arms × 3 replicates × 14 cases |
| cost | US$0, ~1.5 h |

Abort rule: more than 5 JSON failures after one retry, per arm-replicate.

## Outcome (2026-10-17)

**NOT NAVIGABLE** by the fixed rule. S10's core recall would put it in PARTIAL, but its precision is below the floor.

| | S9 (D9's walk) | S10 |
|---|---|---|
| mean core recall (3 reps) | 0.286 | **0.452** |
| mean precision vs broad | 0.902 | **0.650** (floor 0.70) |
| picks per case | 6.1 | 14.6 |
| vs BM25 | 10/0, p = 0.002 | 12/0, p = 0.0005 |
| cases whose picks varied across the 3 reps | 2 | 0 |

**S10 vs S9:** 10 wins, 3 losses, 1 tie (p = 0.09). Data: `~/.config/director/d10/{walks.jsonl,report.json}`, log
`~/.config/director/d10.log`.

**Read before relying on it.**
- **Exclusion moved the 12B along the trade-off; it did not lift it.** Recall rose by 0.17 and precision fell by 0.25.
  The losses are where exclusion kept most of a long list: c04 (0.36), c13 (0.29), c02 (0.39). The 3 cases where S10
  lost to S9 (c10, c12, c13) are cases where the layered entry picked different trajectories.
- **Precision vs broad is a proxy.** It counts as wrong every statement Sonnet did not cite in either of its two plans,
  and Sonnet does not cite everything that applies. Whether 15 statements at 0.65 make a worse *plan* than 6 at 0.90 is
  the D0 judge's question, which this brief kept out of scope.
- **The held-out bank is now worn.** D9's diagnosis was made on it, and S10 was designed from that diagnosis. Any next
  step should be measured on a new bank.
