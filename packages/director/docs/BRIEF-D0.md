# D0 — does walking the operational memory beat the session as we run it today?

**Written 2026-10-14, before any arm runs.** The user's decisions the same day: the director lives in
`packages/director/` of this repository, code public and memory private, and D0 measures before anything
else is built.

## What and why

The director is a meta-agent that keeps the lab's **operational memory** — how we work, as atomic,
linked statements — walks it as a trajectory to plan a task, and delegates the plan to Claude Code
(frontier) or to local models. Before building its loop, its interface or its learning, D0 asks the one
question that can make all of that unnecessary: **does a session that plans by walking the memory follow
our way of working better than the session we run today?**

Today's way is the baseline, never a frontier model: Claude Code with the workspace's `CLAUDE.md` loaded
and the auto-memory index (`MEMORY.md`) in context, the memory files and the project instructions one
`Read` away. The rules are already written down; this week's lapses happened with them loaded. D0 asks
whether *organisation and navigation* — not new information — close that gap.

## Arms (sequence, not a grid)

| arm | what the model gets | runs |
|---|---|---|
| **A — today** | a sandbox with the workspace `CLAUDE.md` (auto-loaded), `lora-kernel/CLAUDE.md` and its `.claude/agents` + `.claude/skills`, the memory files under `memory/` with `MEMORY.md` appended to the system prompt, as Claude Code loads it | **first** — the headroom arm |
| **T — director** | a sandbox with the same sources **compiled** into atomic-statement pages (`docs/MEMORY-FORMAT.md`) and `DIRECTOR.md` as the appended system prompt: find the entry statement, follow the links, plan citing the statements walked | only if A leaves headroom |

**Information parity:** T's pages are compiled only from A's sources (the compiler reads that copy and
nothing else), so a difference is organisation and navigation, not knowledge. Asserted by the compiler's
input list in the run record.

Both arms: `claude -p --restricted`, tools `Read, Grep, Glob` only (nothing executes), the same model
(**Claude Sonnet 5.5**, provider Anthropic through the local Claude Code), `--max-turns 12`,
`--max-budget-usd 0.60` per call, k = 2 samples per case. The prompt asks for the concrete next actions in
order and what will be told to the user; it says nothing about rules or memory.

## Cases

`d0/cases.jsonl`: **14 situations**, each written as a task in flight — the request, the state it arrives
in — never as a quiz. Each comes from a real lapse or near-lapse in the lab's own history where the rule
that should have changed the plan **was already written** in the sources. Each carries 2–3 checklist items
about **actions** (merge or not, ask or not, read the code or not, write the brief first or not), never about
wording: a check that can fail while the behaviour is right measures phrasing. Private names are
placeholders filled at run time from the private config (`{PRIVATE_PROJECT}`), so this file is public.

## Judge

`d0/judge.md`, Claude Sonnet 5.5, no tools, **blind to the arm**: the case, its checklist and the plan;
one verdict per item, `met` or not, with the sentence that decides it. Score of a sample = items met /
items; of an arm = mean over cases of the mean over its samples.

## Verdict (fixed here)

| | condition |
|---|---|
| **STOP — no headroom** | A ≥ 0.85: today's session already plans our way on these cases; T is not built. One redesign is allowed, with the user's sign-off: the same cases behind a long realistic session prefix (the lapses happened deep in long sessions). |
| **GO to T** | A < 0.85 |
| **T PASSES** | T − A ≥ +0.15 **and** T wins more discordant cases than it loses (exact sign test on case-level means, p < 0.05 one-sided) |
| **T FAILS** | otherwise — the memory organised as trajectories does not change the plan; reported as is |

Beside the verdict: per-case scores, cost, turns, and which statements T cited (does it walk, or answer
from the prompt?).

## Cost and stopping

A: 14 × 2 = 28 calls, ≈ US$3–8; T adds the compile (≈ US$2–5) and 28 calls. Cap **US$25** for D0, read
from each call's `total_cost_usd`. Results persist per call (`results/d0/*.jsonl`, private), resumable, one
progress line per call. Transcripts stay private (they quote the memory); only scores are published.

## Not in D0

The director's loop, the interface, the local lane, learning and agentvcs versioning (D1–D3). A local
model as the planner. Long-session pressure (only as the one redesign above).

## Arm A [ran] 2026-10-14 — 0.589: GO to T

28/28 plans, 0 errors, 28/28 verdicts parsed, US$1.17, 1.36 turns per plan on average (today's session
plans from its prompt and rarely opens a memory file). Per case: c01, c02, c04, c05, c11, c14 at 1.0;
c06, c08 0.75; c07 0.5; c03 0.25; **c09, c10, c12, c13 at 0** — c13's rule is written in the very
`CLAUDE.md` the session had loaded, and both samples planned the edit to the subtree. Scores only are
published; plans and verdicts stay private.

**Noted on the judge, before T runs:** c12 sample 0 proposed the right command but waited for the user's
go, and was judged not met. The judge is the same for both arms and blind to them; its strictness is a
level, not a bias between arms. Kept as is (changing it after seeing A would be a redesign of the instrument).

## Added before T runs: the attribution arm

T differs from A in two ways at once: the memory is **organised** as linked atomic statements, and the
system prompt **tells** the session to walk it before planning. If T passes, one more arm separates them:
**A+** = A's sandbox and sources with the same walk-before-planning instruction, pointed at `MEMORY.md`
instead of `Index.md`. A+ ≈ T means the instruction carried it; T > A+ means the organisation did. Bought
only if T passes (sequence, not a grid).

## Arm T [ran] 2026-10-14 — 0.929: **T PASSES**

The compile (Claude Opus 5.5, A's 57 files and nothing else, parity asserted): 20 pages, 286 statements, an
index of 88 situations, US$4.97, 88 turns. T: 28/28 plans, 0 errors, 28/28 verdicts parsed, US$12.56,
9.8 turns per plan, **every plan cites the statements it walked**.

| | A (today) | T (director) |
|---|---|---|
| mean | 0.589 | **0.929** |
| cases at 1.0 | 6 / 14 | 12 / 14 |
| c09 mirror · c13 subtree · c03 brief | 0 · 0 · 0.25 | 1.0 · 1.0 · 1.0 |
| c10 grid · c12 tail | 0 · 0 | 0.5 · 0.5 |
| cost per plan · turns | US$0.04 · 1.4 | US$0.45 · 9.8 |

T − A = **+0.339**; case-level wins/losses **8 : 0** (6 ties at 1.0), exact one-sided sign test
**p = 0.0039**. By the table fixed above: **T PASSES**. D0 total US$19.34 of the US$25 cap.

**Read before believing it.**
- Two T plans read by hand (c13, c09): they cite the statement that decides the case and act on it (refuse
  the subtree edit and ask; mirror + `check-mirrors.py` before the commit). The verdicts are not the judge
  rewarding length — but T's plans are longer (4,473 vs 2,645 characters on average) and the judge saw them.
- **T costs ~11× more per plan** and walks ~7× more turns. Part of the gain may be *effort* — reading
  memory at all — rather than the organisation. That is exactly what the attribution arm A+ separates.
- `DIRECTOR.md` as run pointed at `docs/MEMORY-FORMAT.md`, which the sandbox does not have (a `sed` that
  did not match a wrapped line); plans noted it and read the pages directly. Fixed in the file for the next
  run; it can only have cost T, not helped it.
- 14 cases written by the same session that compiled nothing but knows the rules; parity holds (both arms
  see the same rules), but the cases are not a held-out sample of future work.

**Next, by the sequence written before T ran: A+**, the attribution arm. It needs ≈ US$8–12, past the
D0 cap — the user's call.

## A+ — the attribution arm, written before it runs

**The user's decision, 2026-10-14: the D0 cap rises to US$35 for A+**; after it, D1. A+ = A's sandbox byte for
byte, A's system prompt plus `DIRECTOR.md`'s five steps pointed at `MEMORY.md` and the `CLAUDE.md` files
instead of `Index.md` (`run_d0.APLUS_WALK`). Same model, tools, turns, budget, k, judge.

| reading (fixed here) | condition |
|---|---|
| **the organisation carries it** | T − A+ ≥ +0.10 and T wins more cases than A+ |
| **the instruction carries it** | A+ ≥ T − 0.05 — the director can walk today's files; compiling is not needed for D1 |
| **in between** | otherwise — both matter; D1 keeps the compiled memory, and the gap is reported |

## A+ [ran] 2026-10-14 — 0.768: **the organisation carries it**

28/28 plans, 0 errors, US$8.48, 6.0 turns per plan; **10 of 28 plans took one turn**, i.e. ignored the walk
instruction and planned from the prompt. D0's total: US$28.13 of the US$35 cap.

| | A | A+ (A + the walk instruction) | T (compiled memory, walked) |
|---|---|---|---|
| mean | 0.589 | 0.768 | **0.929** |
| c13 subtree · c10 grid | 0 · 0 | 0 · 0 | 1.0 · 0.5 |
| c09 mirror · c07 goldens | 0 · 0.5 | 1.0 · 1.0 | 1.0 · 1.0 |

T − A+ = **+0.161**, T wins 4 cases, A+ wins 0 (10 ties) → by the reading fixed above, **the organisation
carries it**: the instruction alone buys about half the gain (+0.18 of +0.34), and the compiled, indexed
pages buy the rest. With 4 discordant cases the sign test is p = 0.0625 — the reading was fixed on the
gap and the wins, not on p, and it is reported beside them. c13 is the clearest case: the subtree rule sits in
the very `CLAUDE.md` A and A+ had loaded, and both planned the edit; T found it from the index's situation
"working in ai-os / evolving-agents".

**For D1:** the director keeps the compiled memory and walks it from `Index.md`.
