---
description: Turn the user's corrections and decisions, and the methods a task had to find, since the last reflection into gated edits of the director's memory
argument-hint: "[the correction, if not already said]"
allowed-tools: Read, Grep, Glob, mcp__plugin_director_director__memory_lint, mcp__plugin_director_director__memory_propose, mcp__plugin_director_director__memory_check, mcp__plugin_director_director__memory_status, mcp__plugin_director_director__memory_accept, mcp__plugin_director_director__memory_reject, mcp__plugin_director_director__journal_read, mcp__plugin_director_director__journal_note
---

A task in this session has ended. The user's correction, if they give one here: $ARGUMENTS

1. Call `journal_read`: what happened since the last reflection — the user's own messages (`person`, captured as they
   were sent), plans proposed, moved, cancelled and run, delegations, lessons decided, and the corrections and outcomes
   noted. From it, list what the memory may lack:
   - **corrections** — the user said a plan or a result was wrong (from above, from a `person` message, or a
     `correction` entry); a task the user moved to another lane on the band (`plan.moved`) is one too: "this kind of
     work runs on that lane";
   - **decisions** — the user chose, capped, ruled or preferred something that later work must follow and could not
     work out on its own: a model, a provider, a spending cap, a tool, a layout, "from now on …", "don't …", "use X,
     not Y". A decision is a lesson even when nothing went wrong;
   - **methods** — a way of doing a kind of task that the session reached only after attempts that failed (an
     `outcome` or `correction` that lists what went wrong before it went right), and that the memory has no statement
     for: the next agent would rediscover it the same way. A method is a lesson even when the user corrected nothing.
   Check each against the memory first; drop what it already holds. If nothing is left, call `journal_note` with kind
   `reflected`, say so and stop.
2. For each correction, decision or method left, decide what the memory (the directory the director's context names, with
   `Index.md`) must hold so that the next time an agent is in a situation like this — not this exact one — its plan
   comes out right. For a decision, **When** is the situation where it applies, **Do** the decision itself, exactly
   (the number, the name, the choice), and **Because** the user's reason if they gave one, with the date they decided
   it. For a method, **When** is the kind of task, **Do** its steps in order, each with the check that showed the
   step had gone wrong (the overlay, the second estimate, the test) — without the values of this one task (its files,
   slice numbers, paths, results), which belong beside the task's data, not in the memory — and **Because** the
   attempts that failed. Search the memory for
   statements that already cover it first: if one does, **modify** it (sharpen its When, Do or Because) rather than
   adding a duplicate; add a statement only for what none covers. Phrase **When** as a situation an agent recognises
   mid-task, general enough to recur in other tools and files; **Do** as a concrete, checkable action; **Because**
   with the incident as evidence. Link it to the statements it applies with, and name the Index situations that
   should reach it (reuse an existing situation line's text exactly when one fits).
3. Write **two probes**: each a short situation — a task in flight, in the second person — where the lesson should
   change the plan, set apart from this incident (another file, tool or goal; none of its sentences), with one to
   three checklist items that each name an **action** the plan must take or refuse. Do not say in the situation what
   the right action is.
4. Call `memory_propose` with the edits (`op` add|modify, `page`, `id`, `when`, `do`, `because`, `links` as
   `Page#id`, `situations`; `about` only for a new page), the probes, a one-line `rationale`, and the report as
   `episode`. If it reports lint problems or rejected edits, fix them and propose again. A proposal you replace with a
   new one — and any proposal `memory_status` calls `stale` (the memory changed after it was made; accepting it would
   undo what came since) — is closed with `memory_reject`, and you say so.
5. Show the user the diff, and ask: accept, check first, or reject. **Check** runs the positive probe
   (`memory_check`, minutes and a few dollars): it keeps a lesson only if it changes the plans its probes test —
   TEACHES; REDUNDANT means the memory or the model already does it; NO EFFECT means the edit changed nothing.
   Poll `memory_status` until it lands and report the verdict.
6. Only on the user's explicit yes, call `memory_accept` with `user_said_yes: true`. Report whether it was checked and
   versioned, and where the previous memory was kept. A refusal leaves the memory as it was; say why. On the user's
   no, call `memory_reject`.
   Where Claude Code shows the director's band above the prompt, the user can also decide there (**Diff**, **Accept**,
   **Reject**); their decision reaches you with their next message. Do not call `memory_accept` again for a proposal
   they already accepted or rejected on the band.
7. Call `journal_note` with kind `reflected` and one line on what was learned, so the next reflection starts here.
