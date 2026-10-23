# evolving-agents — how this lab works

The active project is the **director** (`packages/director/`). Everything else is legacy, paused or frozen
(`legacy/`, `ai-os/`, `plugin/`, `packages/memory/`, `packages/agentvcs/`); do not change those unless asked.

## Measurements

- **A brief before any run.** Anything that spends money or calls a model on the lab's behalf gets
  `packages/director/docs/BRIEF-<ID>.md` first: what, why, model and provider, the verdict table, what would
  falsify it, the stopping condition and a spending cap. The result goes into the same brief as `## Result` /
  `## Outcome`, dated, with what was spent.
- **The verdict table is fixed before the run** and never moved after seeing the numbers. A result below the bar is
  published as flat; never reworded into a win.
- **Headroom first.** Before measuring an improvement, measure the baseline; if it is already near the ceiling
  (C0, D0O and PG stopped there), stop and say so.
- **Compare against the version it replaces**, never against a frontier model alone; say which model planned and
  which judged. Since 2026-10-21 the person plans with Opus; numbers measured with Sonnet are labelled as Sonnet's.
- A smoke run is not a measurement: one seed or a handful of cases decides nothing, and the write-up says so.

## Privacy

- The lab's memory, plans, transcripts and lanes config are private (`~/.config/director/`, a project's
  `.director/`). Public files use placeholders for private names; what a run publishes is its scores.
- Never commit a key or a private term. The plugin's public guard refuses such commits; do not work around it.

## The director's code

- `director/memory/` is standard library only, never calls a model and imports nothing from the rest of the
  package (`tests/test_memory_boundary.py` holds the line). Every interface accepts a lesson through
  `memory.gate`; a person's yes is required (`memory_accept`), except `director.headless`, which records
  `decided_by: policy:<name>`.
- Before pushing a director change: `cd packages/director && python -m pytest tests -q` (the 3 agentvcs tests
  skip without the Rust CLI) and, for the plugin, `claude plugin validate .` and `claude plugin test .`.
- A change to the plugin bumps `version` in `packages/director/.claude-plugin/plugin.json` **and** in the root
  `.claude-plugin/marketplace.json`, together.
- `benchmarks/director-bench/` is independent of the director: it must not import director code; the director is
  one configuration that calls its CLI. Its tests run in place (`cd benchmarks/director-bench && python -m pytest
  -q`), not in this repository's CI.

## Writing

- Docs and comments in plain words, outcome first, short. Numbers carry their brief link.
- When something ran, update the docs that cite it in the same change: the brief, `packages/director/README.md`,
  `ROADMAP.md`, `docs/EVOLUTION.md` and the RUNBOOK. The public site (`evolvingagentslabs.github.io`) checks its
  numbers nightly against the briefs.
