# Where the director should beat other memories, and why

**Written 2026-10-22, before director-bench has measured anything that counts.** The lab rule is a brief before a
run. This page is the step before that: what the director does, what each of its mechanisms should buy against
the alternatives, and on which kind of work. Each hypothesis says what it predicts in director-bench's metrics, so a
run can contradict it. The family the bench runs on was redesigned with these properties in mind. It was designed
by the same lab that builds the director, and that is author bias: the bench's plan still requires one family
chosen or reviewed by someone outside the lab.

## 1. How the director works, in one pass

1. **Compile.** Your `CLAUDE.md`, `AGENTS.md` and Claude Code's memory become atomic statements:
   **When** (a situation an agent recognises mid-task), **Do** (a checkable action), **Because** (the incident or
   decision behind it). Each statement links to the statements that apply next. An **Index** names the situations
   where a walk starts.
2. **Walk.** At the start of a session the plugin's hook puts the walk into the context. Before planning, the agent
   reads the Index, follows the matching statements and their links, and cites what each step follows.
3. **Journal.** What happens is appended to one file: the person's messages, plans and lane moves, and — with no
   person present — each session's verifier verdict, with a digest of what the session did. Nothing that plans reads
   the journal; only the reflection does.
4. **Reflect / consolidate.** Claude Code reads the journal since the last reflection and drafts lessons:
   - corrections;
   - decisions (a model, a cap, "from now on …");
   - **methods**: a way of doing a kind of task that a session reached only after failed attempts, each step with
     the check that caught the failure, without that task's values.

   Each lesson comes as edits to statements, a modify preferred over a duplicate, plus two probes of its own.
5. **Gate.** A lesson enters only through `memory.gate`:
   - the lint is clean (links resolve, every statement is reachable from the Index);
   - optionally, a check shows the lesson changes the plans its probes test (TEACHES, against REDUNDANT and NO
     EFFECT);
   - someone decides: a person's yes, or with no person present a declared policy (`verified` keeps lessons only
     from sessions whose verifier passed).

   Each accepted edit is a versioned patch, and the journal records who decided.

## 2. What it competes with

| approach | how it learns | what it loads |
|---|---|---|
| no memory | — | the repository, every time |
| a hand-written `CLAUDE.md` | a person edits it | the whole file |
| **Claude Code's native memory** | **Auto Memory**: the agent writes notes to `MEMORY.md` *during* the session, whenever it judges something worth keeping; **Auto Dream**: a background consolidation every ~24 h / 5 sessions (server-tunable; it may never fire in a short run) | the head of `MEMORY.md` at session start |
| retrieval over past sessions | stores trajectories, finds similar ones | the nearest past episodes — measured flat three times in the 2025 toolkit (−4, +0, −6 against an oracle retriever) |

The baseline to beat is the native memory, and it is strong: it learns in the session, for free, with no
consolidation step to pay for. The director adds three things it lacks:
- a gate between "something happened" and "it is now a rule";
- a structure (situations → statements → links) instead of a flat note file;
- edits to existing statements instead of appended notes.

The hypotheses are about when those three matter.

## 3. Hypotheses, each with what would show it

**H1 — Verified gating keeps wrong lessons out.** A session that fails can still write a confident note: a wrong
cause, a workaround that only looked like a fix. Native memory writes during the session, before any verifier has
spoken. The director's `verified` policy never shows a failed session to the reflection.

*Predicts:* fewer memory-induced errors and regressions for C2 than for C1, and than for A1 (the director learning
from every session). This is the bench's mechanism condition. *Matters when* sessions fail in ways that look like
success from inside: an unlisted rule enforced only by a later check, or a flaky step.

**H2 — Methods with their checks cut repeated failures.** What recurs in real work is procedural, and learned by
failing: run the generator before the check, write the fragment the release tool expects, use the project's money
type. A method statement keeps the steps in order *and the check that showed each one had gone wrong*, so the next
session does the step instead of failing it first.

*Predicts:* a lower repeated-error rate (RER) and fewer turns to green for C2 than for C0. Against C1 the difference
is format and selection, not capability: Claude's own notes can record the same procedure. If C1 matches C2 on RER,
this hypothesis adds nothing over native memory, and that is the likeliest way for the director to lose.

**H3 — Editing in place survives a changed convention.** When the repository changes a rule, a flat note file
tends to keep the old note next to the new one. The director's reflection is told to modify the statement that
covers a lesson, with the new incident as its Because.

*Predicts:* fewer memory-induced errors on the trap task for C2 than for C1. *Risk, declared:* the retention task
runs on the old snapshot, so a memory that updated correctly is tested on the old rule. That measures whether the
memory says *when* each rule applies, not only *what* it is.

**H4 — Recall at the moment it applies.** The walk brings the right statement to the plan. With Sonnet planning
that was the whole effect: 0.589 → 0.929 ([D0](BRIEF-D0.md)). With Opus there was none: 0.964 alone
([D0O](BRIEF-D0O.md)).

*Predicts:* C2's advantage over C1 is larger with Sonnet than with Opus. With Opus, expect H4 to add nothing; that
is what the lab's own results say.

**H5 — Memory instead of exploration lowers cost, once the repository is large.** Reading a statement is cheaper
than rediscovering a convention from code. In C2 the same score came at −34 % per plan with Opus
([C2](BRIEF-C2.md)). Consolidation is not free: in the first smoke run, on a tiny repository, it made C2 cost +70 %
per task.

*Predicts:* tokens per successful task below C0 only when the repository is too large to read whole and the run is
long enough to amortise consolidation. *Falsified if* C2 stays above C1 + 10 % on tokens (the bench's own threshold).

**Not measured by the bench, and still the director's:**
- a lesson can be read, diffed, blamed and reverted;
- who decided is on record;
- the memory travels to another model as plain notes (Gemini followed it at 0.875 — [PG](BRIEF-PG.md)).

These are reasons to choose it at equal scores, not scores.

## 4. Where it should not win (said before running)

- **Small, self-documenting repositories.** The first smoke run: no memory solved every task first time in four
  turns. Nothing to remember, so nothing to gain, and the consolidation is pure cost.
- **One-off tasks.** A lesson pays only when its situation recurs.
- **Conventions the model re-derives cheaply** from the code in front of it. The stronger the model, the more of
  them.
- **Rules written anywhere in the repository.** Measured in DB0's second live check: Sonnet read the checker's rule
  modules in its second turn and passed every task first time. What is in the repository is learned by reading;
  memory only saves the reading.
- **Short runs.** Native memory learns in the session; the director learns every second task. Over ten tasks,
  that lag costs it the early sessions.

## 5. What this means for the bench's family

These are the properties a family needs for **any** memory to have something to learn. None of them is specific to
the director:
- rules learned by failing, consistent across tasks, **absent from the repository** — enforced by a remote CI or a
  reviewer whose feedback reaches the session in a review round (rules kept in the repository are read, not
  failed: DB0's second live check);
- an environment quirk that costs turns;
- a repository too large to read whole;
- tasks of one kind that each touch a different subset of the rules;
- one rule that changes (the trap);
- a retention task on the old snapshot.

Before any configuration runs at scale, C0 must show it repeats errors on the family (DB0's headroom gate).

If C1 then matches C2 on repeated errors and harm, the director's case reduces to cost, auditability and
portability, and that is what gets published.
