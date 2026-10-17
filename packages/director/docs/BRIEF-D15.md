# D15: can a small local model follow a meta-recipe that Claude wrote?

**Written 2026-10-17, before anything runs.** The user clarified the design after D14. The local model is not meant to
walk the memory freely or to write the plan. It is meant to **follow a meta-recipe** that Claude creates and maintains in
the local operational memory. Following it should be simple enough to work even with a model smaller than the 12B.
D8–D14 did not test that: the model chose loose statements or wrote the plan itself. D15 does.

## The meta-memory: recipes

A **fresh Claude agent** (Opus 5.5) compiles D9's memory pages into recipes. It sees only the pages and the format; it
never sees D0–D14 or the new bank. The recipes live in a private JSON file, `~/.config/director/d15/recipes.json`.
Limits: at most 20 recipes, one per type of request.

Each recipe has:

- `id`;
- `trigger`: one line naming the type of request, the only text shown when choosing;
- `steps`: a list, run in order. A step is one of:
  - `{"do": "<concrete action>", "cite": "Page#id"}`: always taken;
  - `{"ask": "<yes/no question about a concrete, observable fact in the request>", "yes": [steps], "no": [steps]}`:
    a branch.

The questions must be about facts that can be read from the request ("does the request mention a CI check that
failed?"). They must not ask for judgement ("does this rule apply?"). D12 showed that a model asked for judgement says
yes to everything.

## Execution (code runs the recipe; the model only answers closed questions)

1. **Choose.** The model sees the request and the numbered triggers, and picks **at most 2** recipes (JSON schema).
2. **Run.** Code walks each chosen recipe's steps. At each `ask`, the model answers `yes` or `no` (JSON schema
   `enum`) and code takes that branch.
3. **The plan.** Code makes the plan from the `do` steps reached, in order, each citing `[[Page#id]]`, with duplicates
   removed. Nothing else is written. The model writes no text at all.

Every choice and answer is logged.

## Arms (k = 2 per case; same recipes, same code)

| arm | who answers the closed questions |
|---|---|
| **G12** | `gemma4:12b-mlx` |
| **Q4** | `qwen3.5:4b` |
| **Q2** | `qwen3.5:2b` |
| **RS** | Sonnet 5.5 via `claude -p`, no tools. It measures **the recipes themselves**: a strong answerer on the same recipes. |
| **T** | Sonnet walking D0's memory with tools, as in D11–D12 (the reference) |

The local models run on Ollama with `think` off and temperature 0.3.

## The bank

The bank is new: 12 cases written by another fresh agent under D11's rules, in `~/.config/director/d15/cases.jsonl`.
No item may rest mainly on a statement that grounded a D11 or D12 case. The two agents (bank and recipes) run apart and
never see each other's output.

## Judge

The judge is `d0/judge.md`, Sonnet, blind.

## Verdicts (fixed here)

| verdict | condition |
|---|---|
| **RECIPES WORK, EVEN SMALL** | Q4 or Q2 ≥ T − 0.10 |
| **RECIPES WORK** | G12 ≥ T − 0.10 (and no small model does) |
| **PARTIAL** | G12 ≥ T − 0.20 |
| **NOT YET** | otherwise |

How the outcome is read: if RS is also low, the recipes are the problem, and that is Claude's work. If RS is high and
the local arms are low, the answering is the problem.

**Also reported:**
- per-case scores;
- the grounds reached;
- recipe choices;
- answers that differ from RS's;
- calls and wall time per plan.

## Cost and stopping

| item | estimate |
|---|---|
| two agents | ~US$4 |
| T plans: 24 | ~US$12 |
| RS answers: ~24 × ≤ 10 calls | ~US$5 |
| judge: 120 verdicts | ~US$2 |
| **cap** | **US$30** |

One design, no redesign on these data.
