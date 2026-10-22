# director-bench

**Does Claude Code get better from one session to the next on recurring tasks — and how much
does a memory system add over Claude Code's own memory?**

director-bench runs chains of headless Claude Code sessions on families of similar tasks in the
same repo. Between sessions only the memory store survives. It measures the curve across
sessions (do repeated errors go away? do turns go down?), the harm (regressions, memory that
misleads after the repo changes) and the cost (session and consolidation tokens).

The thesis under test: *procedural memory consolidated only from verified trajectories reduces
repeated errors and cost per task more than native memory, without regressions.*

The benchmark knows no internals of any memory system. It only sees hooks, the memory store and
results. [director](https://github.com/EvolvingAgentsLabs/evolving-agents) is just one config
(three, with its variants) that calls director's CLI; nothing of director is imported here.
Adding another memory system means adding a YAML file.

## Configurations

All configs use the same model string, Claude Code version and permission mode.

| id | what | memory store |
|----|------|--------------|
| C0 | vanilla Claude Code, native memory off — the floor | none (fresh HOME every session) |
| C1 | Claude Code + native Auto Memory + Auto Dream — **the baseline to beat** | native memory dir + HOME |
| C2 | Claude Code + director, native memory off | director store |
| C3 | director + native memory | both |
| A1 | ablation: director without verifier gating (consolidates every trajectory) | director store |

Pinned per run and written to `results/<run-id>/pins.json`: model API string, `claude --version`,
memory system version, task-order seed (plus permission mode, family and config hashes, signature
version). `bench report` refuses to mix runs with different pins.

## Protocol (one run = family *f*, config *c*, seed *s*)

1. Restore the repo to task *n*'s snapshot: a fresh single-commit copy at the **same path** every
   task (Claude Code's native memory is keyed on the working directory).
2. New headless session:
   `claude -p "<task statement>" --output-format stream-json --verbose --max-turns 60 --model <pinned> --permission-mode <pinned> [--settings <config settings>] [config extra args]`.
   Budget: 60 turns or 400k tokens; the runner reads usage from the stream and kills the
   process when the token budget is exceeded (or when the wall-clock limit is hit).
3. After the session, run the verifier `verify/NN.sh <workdir>` (exit 0 = pass) **on a throwaway
   copy** of the workdir. Record pass/fail, tokens, turns, wall time, tool errors; save the full
   stream-json trace.
   **Review round.** If the verifier fails and `feedback.rounds` (family.yaml, default 1) remain,
   the runner resumes the **same** session —
   `claude -p "<feedback>" --resume <session_id> ...same flags...` (session id from the stream's
   init/result line; same HOME and workdir; `--max-turns` = turns left) — with the verifier's
   `CI ...` lines only (what a CI log or a reviewer would say), then verifies again. Turns, tokens
   and wall time are **shared** across rounds (the token kill uses what is left). The feedback
   turn is appended to the same trace (`traces/<pos>-<task>.jsonl`) as a
   `{"type": "user", "bench_feedback_round": n, ...}` line, followed by the resumed stream;
   round verifier logs are `verify/<pos>-<task>-r<n>.log`; the feedback text is saved as
   `feedback/<pos>-<task>.txt`. Recorded per task: `first_pass`, `passed` (final, within
   budget), `rounds_used`, the first attempt's `ci_feedback_lines` and `ci_feedback_signatures`,
   and per-round sessions. Why: rules that live in the repo are learned by *reading* (the DB0 live
   checks of both ledgerkit versions: a strong agent reads the checker before writing); the
   recurring knowledge a memory is for — what remote CI enforces, what reviewers ask for — only
   reaches the agent as feedback.
4. Every 2 tasks (configurable) run the config's consolidation. If its last stdout line is a JSON
   object with `input_tokens`/`output_tokens` (optionally `cost_usd`; count input as input + cache writes, cache reads apart, as a session is counted), that cost is added,
   prorated over the tasks the cycle covers. No consolidation after the last task.
5. Advance to *n*+1 keeping **only** the memory store.

Controls:

- **Seeds**: 5 per (config, family). The order is a function of (family, seed) only, so every
  config sees the same order for a seed. **Task 1 stays first** for every seed (a common cold
  start, and the retention task is compared against it); the middle tasks are permuted; the trap
  and the retention task close the run, in that order.
- **Retention task**: task 1 rewritten (different wording, same knowledge, task 1's snapshot).
- **Trap task**: its snapshot contradicts an earlier convention (the repo changed). Note that the
  retention task after it runs on the *old* snapshot: memory that blindly overwrote the old
  convention is penalized too; memory should be checked against the repo.
- **Isolation**: each run gets its own sandbox (default `$TMPDIR/director-bench-sandbox/<run-id>`,
  outside the checkout) with its own `HOME`, `CLAUDE_CONFIG_DIR=$HOME/.claude`, memory dir and
  workdir. Nothing is shared across configs or seeds. The Claude process gets an allowlisted
  environment (PATH, locale, proxy/CA vars, `ANTHROPIC_*`, cloud-provider credentials, plus
  `--pass-env NAME`), not the operator's whole environment.

### Hook environment

Every hook (`setup`, `pre_session`, `post_session`, `consolidate`, `version_cmd`) runs under
`bash -euo pipefail` from the sandbox root, with the operator's environment, the config's
`session.env`, `HOME`/`CLAUDE_CONFIG_DIR` pointing at the run's HOME, and:

| variable | meaning |
|----------|---------|
| `BENCH_RUN_ID` | run id (`<family>-<config>-s<seed>-<utc stamp>` unless `--run-id`) |
| `BENCH_FAMILY`, `BENCH_CONFIG`, `BENCH_SEED` | what is being run |
| `BENCH_TASK_ID` | task id (`"07"`); empty in `setup` |
| `BENCH_TASK_INDEX` | 1-based position in this run's order; for `consolidate`, the last covered position |
| `BENCH_WORKDIR` | the task's working copy (stable path) |
| `BENCH_MEMORY_DIR` | the memory store root (persists for the whole run) |
| `BENCH_HOME` | the run's HOME (fresh per session if `home_scope: task`) |
| `BENCH_TRACE` | path of the session's stream-json trace (`pre_session`: where it will be written) |
| `BENCH_PASSED` | `true`/`false`, final result — `post_session` only |
| `BENCH_FIRST_PASS` | `true`/`false`, first attempt (before any review round) — `post_session` only |
| `BENCH_ROUNDS` | review rounds used — `post_session` only |
| `BENCH_FEEDBACK` | file with the feedback given in-session (exactly what the session saw; empty if none) — `post_session` only |
| `BENCH_FINAL_FEEDBACK` | file with the CI lines on the final state when the task still fails after its rounds (not shown to the session; what a person would see on the last push; empty if it passed) — `post_session` only |
| `BENCH_CONSOLIDATE_POSITIONS` | comma-separated positions a consolidation covers |
| `BENCH_RESULTS_DIR`, `BENCH_MODEL`, `BENCH_PHASE` | run's results dir, pinned model, hook phase |

`${NAME}` and `${NAME:-default}` in `session.extra_args`, `session.env` and `session.settings`
are expanded by the runner from the same variables; an unset variable without a default is an
error. Hook commands are expanded by bash (`set -u` makes unset variables fail loudly).

The verifier never runs inside the workdir and its output only goes to `results/`. Only its
`CI ...` lines reach the session (review round) and hooks (`BENCH_FEEDBACK`); post-session hooks
run after the final round, and `BENCH_TRACE` then holds every round including the feedback turn.

## Config YAML schema

```yaml
name: C2                         # id used in results and reports
description: ...
home_scope: task                 # task: fresh HOME per session | run: HOME persists (part of the store)
version_cmd: 'printf "%s\n" "${DIRECTOR_VERSION}"'   # last stdout line = memory system version; null -> "none"
session:
  extra_args: ["--plugin-dir", "${DIRECTOR_PLUGIN_DIR}"]   # appended to the claude command
  env: {DIRECTOR_MEMORY: "${BENCH_MEMORY_DIR}/.director/memory"}   # session + hooks
  settings: {autoMemoryEnabled: false}                      # JSON passed via --settings <file>
hooks:
  setup: [<cmd>, ...]            # once, before the first task
  pre_session: [<cmd>, ...]      # before each session
  post_session: [<cmd>, ...]     # after each session + verifier (BENCH_PASSED set)
consolidate:
  command: <cmd> | [<cmd>, ...] | null   # last stdout line may be {"input_tokens":..,"output_tokens":..,"cost_usd":..}
  every: 2                       # cadence in tasks
```

Any line containing `TODO(confirm` makes the runner refuse the config unless `--allow-todo`.

## Family layout

```
families/<family>/
  family.yaml          # repo source, snapshots, order, budget, trap/retention ids, obsolete marker, leak markers
  build_repo.py        # (v0) builds the repo deterministically, one git tag per task snapshot
  tasks/NN-*.md        # task statements (what the agent sees, verbatim)
  verify/NN.sh         # binary oracle: `NN.sh <workdir>`, exit 0 = pass; hidden checks live here
  reference/NN/        # reference solution overlay (never shown to the agent)
  headroom.py          # (ledgerkit) static per-task rule matrix
```

`family.yaml` keys: `name`, `repo: {build: <script>}` or `repo: {git: <url-or-path>}`,
`snapshots: {task: git-ref}`, `tasks: [ids]`, `order: {first: [...], last: [...]}`,
`budget: {max_turns, max_tokens, wall_seconds}`, `trap: {task, obsolete_marker}`,
`retention: {task, of}`, `leak_markers: [...]`, `feedback: {rounds: 1}`.

Verifiers should print one line per failure,
`BENCH-FAIL type=<t> file=<f> symbol=<s> msg=<m>`, which becomes the failure signature
(otherwise the last output line is used), plus `CI <source>: <message>` / `CI: <message>` lines:
the feedback a CI log or a reviewer would give. CI lines are shown to the agent, so they must
never contain verifier code, reference content, hidden-check details or leak markers (hidden
behaviour checks report only `CI: behaviour test <name> failed`). The verifier runs with
`BENCH_VERIFY_MODE=ci`.

### The v0 family: `ledgerkit`

A synthetic personal-finance package (built by `families/ledgerkit/build_repo.py` from
`template/` plus generated modules; nothing binary is committed, no private code).

**Redesigned twice after live C0 checks** (Sonnet 5.5, Claude Code 2.1.308). DB0: the first
version could be read whole in one turn and named its check command, so nothing repeated. The
second version hid seven rules in an undocumented lint, but a strong agent reads `tools/` before
writing (turn 2: `cat tools/lint.toml tools/lintrules/*`) and passed every task first time: rules
that live in the repo are learned by reading, and memory could only save reading tokens. The
knowledge that recurs, and that memory exists for, lives **outside** the repository — what remote
CI enforces, what reviewers ask for, team decisions. So this version, for any memory system:

1. **Remote rules, only in the team's CI** (`verify/oracle.py`; no rule code, doc or config in
   the repo), consistent across tasks, each needed by 8–10 of the 10 tasks:
   `LK-R01` a `docs/commands.md` row per command (usage lists every `--flag`; new rows say
   `since` = `unreleased`, the release bot stamps versions), `LK-R02` the release bot's change
   fragment `CHANGELOG.d/<name>.md` with `### Added|Changed|Fixed`, `LK-R03` a packaging entry in
   `ledgerkit.toml` `[commands]`, `LK-R04` review: a `@pytest.mark.cmd("<name>")` test taking
   `ledger_factory`, and every new `--flag` tested. The data files exist as realistic examples
   (existing docs rows, toml entries, an empty `CHANGELOG.d/`, existing marked tests); the rules
   do not.
2. **Local rules, readable** (`./tools/check`): `LK-L02 registry stale` (generated registry; run
   `./tools/regen`), `LK-L03 raw money`, `LK-L06` errors, `LK-L12 direct io`; and a behaviour
   rule readable in `cli.py` (commands that change the ledger set `MUTATES = True` or nothing is
   saved). `tools/lintrules/` has 11 modules, 7 disabled/legacy.
3. **Review round**: the first attempt's failures come back as CI lines, e.g.
   `CI LK-R01: docs/commands.md has no row for 'balance'`,
   `CI LK-R02: release bot: no changelog fragment for 'balance'`,
   `CI tools/check: src/ledgerkit/commands/x.py:12: LK-L03 raw money`,
   `CI: behaviour test tag-saved failed`.
4. **Environment quirk**: bare `pytest` (what the README says) fails with
   `No module named 'ledgerkit._compat'`; `./tools/check` sets the path.
5. **Size**: ~120 modules / ~4.5k lines of `src/` (~190 KB); the commands package is one corner.
6. **Tasks**: 10 × "add a command or an option, with its tests", touching different rule subsets
   (`python families/ledgerkit/headroom.py`: 1–4 local and 3–4 remote rules per task, 3.8 remote
   on average; it also checks that no snapshot states a remote rule).
7. **Trap (task 09)**: the release bot now wants `changes/<name>.toml` with
   `type = "added|changed|fixed"`; the feedback names the new convention only when the old one
   is used (`CHANGELOG.d/ was retired in 0.6.0 ...`); `obsolete_marker: CHANGELOG.d`.
   **Retention (task 10)**: task 01 retold, on task 01's snapshot.

Every verifier fails on its untouched snapshot and passes on its reference solution (applied with
`./tools/regen`); a "reader" attempt (everything readable done right) still fails the remote rules
(`tests/test_family_ledgerkit.py`).

**Declared bias:** this family was designed by the same lab that builds director. The v0 plan
still requires at least one family chosen or reviewed by a third party.

## Metrics

Per family, then the unweighted mean across families. 95 % bootstrap CIs resample **seeds**
(with replacement, within each family; paired across configs); deterministic given
`--boot-seed`.

- **RER — repeated-error rate (primary).** Failures are the unique failure signatures of each
  task session: the **first attempt's CI feedback** (`ci|<source>|<message>`), the final verifier
  failures of failed tasks, and tool errors (`tool_result.is_error`) of every round. A failure is *repeated* if its signature appeared in an earlier session of the same
  run. RER = repeated / failures, pooled over the seeds of a family. Signature = failure type +
  file + symbol + normalized message (absolute paths, sandbox roots, line numbers, timestamps,
  hex ids, tmp dirs, durations stripped). `SIGNATURE_VERSION` versions the normalizer; reports
  recompute signatures from raw failure data.
- **Success rate within budget**: final verifier green (after review rounds) and the session not
  killed for tokens/time.
- **First-pass rate**: verifier green on the first attempt. **Rounds per task**: review rounds used.
- **Tokens per successful task**: (session tokens + consolidation tokens) / successes. Session
  tokens = input + cache creation + output; cache reads are recorded separately (and the
  `total_cost_usd` Claude Code reports is kept as `cost_per_success`).
- **Turns to first green verifier**: turns of all rounds, censored at `max_turns` for failed tasks.
- **Learning slope**: OLS slope of (censored) turns on task position, main tasks only.
- **Retention**: success and turns on the retention task minus those on task 1.
- **Regression rate**: tasks C0 solves that the config doesn't, same seed / tasks C0 solves.
- **Memory-induced error on the trap** (v0 heuristic): trap failed **and** the memory store the
  trap session saw contains the family's `obsolete_marker`. `labels/<run-id>.json` overrides it
  (see `labels/README.md`: an LLM-judge label with 20 % manual review). The bench never calls an LLM.

## Win criteria vs C1 (pre-registered — see [DECISION.md](DECISION.md))

1. RER relative reduction ≥ 30 % with the 95 % CI excluding 0;
2. success rate not worse than −2 pp;
3. regression rate ≤ C1 + 1 pp;
4. memory-induced error ≤ C1;
5. tokens per successful task ≤ C1 + 10 %.

Mechanism: A1 significantly worse than C2 in RER or memory-induced error. `bench report` prints
each condition as PASS/FAIL whenever C1 and another config are present, plus an advisory
Go / Iterate / Pivot verdict.

Leak control: memory stores must never see verifiers or reference solutions. Before every
session the store is snapshotted to `results/<run>/memory/`; `bench audit --run <id>` greps all
snapshots for the family's `leak_markers` and exits 1 on any hit.

## Quickstart

```bash
pip install -e '.[test]'            # Python >= 3.10, PyYAML; or run `python -m runner ...`
pytest -q                           # fully offline, uses tests/fake_claude.py

# real run (C0 = no memory). The Claude process gets an isolated HOME, so authenticate by env:
export ANTHROPIC_API_KEY=...        # or CLAUDE_CODE_OAUTH_TOKEN
bench run --family ledgerkit --config C0 --seed 1 --model <pinned-model-id>
bench report --run <run-id>         # or several --run, or no --run for all of results/
bench audit --run <run-id>

# director configs need, in the operator's environment:
export DIRECTOR_PLUGIN_DIR=/path/to/director/plugin   # passed as --plugin-dir
export DIRECTOR_PYTHON=/path/to/venv/bin/python       # a python that can `import director`
export DIRECTOR_VERSION=$(git -C /path/to/evolving-agents rev-parse --short HEAD)
bench run --family ledgerkit --config C2 --seed 1 --model <pinned-model-id>
```

Offline smoke with the fake claude: `bench run ... --claude-bin tests/fake_claude.py --env FAKE_CLAUDE_SPEC=<spec.json>`
(see `tests/test_smoke_ledgerkit.py` for a spec).

**Caution.** Sessions run with `--permission-mode bypassPermissions` by default. The workdir and
sandbox live outside the checkout, but an agent with shell access could still roam the
filesystem and find `families/*/verify`. For runs that count, use a container or VM where the
bench checkout is not readable by the agent's user.

## Adding a memory system

Write `configs/<ID>.yaml` (schema above): how the memory is switched on (`session.env`,
`session.settings`, `session.extra_args`), its hooks, its consolidation command and cadence, and a
`version_cmd`. Put everything it stores under `${BENCH_MEMORY_DIR}` (or use `home_scope: run`
if it lives in HOME) so isolation, snapshots and the audit cover it.

## Adding a family

Create `families/<name>/` with the layout above: 8–12 tasks of the same kind on the same repo,
offline deterministic binary verifiers from a fixed snapshot, one trap task, one retention task,
reference solutions, and leak markers that occur only in `verify/` and `reference/`. Add a test
like `tests/test_family_ledgerkit.py` proving each verifier fails before and passes after.
Freeze families **before** any director work; at least one must be chosen or reviewed by a third
party (see [docs/THREATS.md](docs/THREATS.md)).

## Declared limits

Built for single-developer repos with recurring tasks. Not for teamwork, giant repos or one-off
tasks. Memory-induced error is a heuristic in v0.

## Cost

Full v0 scale: 4 families × 10 tasks × 5 configs × 5 seeds = **1,000 sessions**, plus
consolidation. Run a pilot first: 1 family × C0/C1/C2 × 3 seeds = **90 sessions**, to measure
variance and C0's headroom. If C0 almost never repeats errors there is no room to measure: stop.

## Status

- Week 1 (this commit): runner, metrics, report/audit, C0 on one family (`ledgerkit`), offline
  test suite with a fake `claude`.
- C1 pending confirmation of native memory controls: the settings keys were verified in the
  pinned binary, but no headless on-demand Auto Dream trigger was found (see `configs/C1.yaml`).
- C2/C3/A1 wired to director's headless CLI interface; to be run once director's CLI lands.
- Three more families to be frozen before director work, at least one chosen or reviewed by a
  third party.

## License

MIT — see [LICENSE](LICENSE).
