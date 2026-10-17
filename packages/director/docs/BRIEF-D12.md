# D12: does a plan composed by code close the gap between the 12B and Sonnet?

**Written 2026-10-17, before anything runs.** In D11 the 12B with the D10 walk (S10) had 0.90 of each case's grounding
statements in front of it, close to Sonnet's 0.95. It still planned at 0.68 against Sonnet's 0.92. It applied rules
partially, or let the user's request override them. Navigation is nearly solved; writing the plan is the gap. D12
moves the plan's composition into code, as D8 moved the walk.

## Arms (k = 2 samples per case, same judge for all)

| arm | how the plan is made |
|---|---|
| **F** (control) | D11's S10 arm, unchanged: S10 walk, then the 12B free-writes the plan from the statements |
| **C** (treatment) | S10 walk, then the plan is **composed by code** from closed answers (below) |
| **T** (reference) | Sonnet walking D0's memory sandbox, as in D0 and D11 |

**How C composes a plan:**

1. For each statement the walk kept, the 12B answers one JSON object, schema-constrained:
   - `applies` (bool): does this statement's When match the situation?
   - `conflicts` (bool): does what the user asked conflict with this statement's Do?
   - `step` (string): the concrete action that carries out this Do in this situation, in one sentence, keeping every
     requirement of the Do (files, commands, conditions, order).
2. One more JSON answer gives the task's remaining steps that no statement covers (at most 5, which must not
   contradict the steps above) and what to tell the user.
3. Code assembles the plan:
   1. **Tell the user first:** each conflict, as "you asked X; the lab's rule [[id]] says Do; I follow the rule unless
      you confirm otherwise";
   2. the steps of the statements that apply, in walk order, each citing `[[Page#id]]`;
   3. the remaining steps;
   4. what to tell the user.

Same model and settings as D11: `gemma4:12b-mlx`, `think` off, temperature 0 for the walk, 0.3 for the answers, D9's
memory.

## The bank

The bank is new: 12 cases, written by a fresh Claude agent under D11's rules. The agent reads only the memory pages, has
no access to D0–D11, and works under the same case requirements. One addition: no case may be grounded mainly in a
statement that grounded a D11 case. The agent is given only those statement ids, never D11's cases. The bank is
private: `~/.config/director/d12/cases.jsonl`.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **COMPOSITION CLOSES THE GAP** | C ≥ T − 0.10 **and** C beats F in more cases than it loses |
| **COMPOSITION HELPS** | C − F ≥ 0.10 with more wins than losses, but C < T − 0.10 |
| **NO GAIN** | otherwise |

**Also reported:**
- per-case scores;
- the grounding statements each arm had (as in D11);
- how many statements C marked `applies` and `conflicts`;
- wall time per plan.

## Cost and stopping

| item | estimate |
|---|---|
| bank | ~US$2 |
| T plans: 24 × ≤ US$0.60 | ~US$12 |
| judge: 72 verdicts | ~US$1 |
| **cap** | **US$20** |

The 12B arms are local. One design, no redesign on these data.
