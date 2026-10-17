---
description: After a task, turn what went wrong (or the user's correction) into a gated edit of the director's memory
argument-hint: "[the correction, if not already said]"
allowed-tools: Read, Grep, Glob, mcp__plugin_director_director__memory_lint, mcp__plugin_director_director__memory_propose, mcp__plugin_director_director__memory_check, mcp__plugin_director_director__memory_status, mcp__plugin_director_director__memory_accept
---

A task in this session has ended. The user's correction, if they give one here: $ARGUMENTS

1. Write the task report for yourself: what was asked, the plan and which memory statements it cited, what was done,
   and the correction (from above, or from what the user said during the task). If there was no correction and
   nothing went wrong, say so and stop: there is nothing to learn.
2. Decide what the memory (the directory the director's context names, with `Index.md`) must hold so that the next
   time an agent is in a situation like this — not this exact one — its plan comes out right. Search the memory for
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
   `episode`. If it reports lint problems or rejected edits, fix them and propose again.
5. Show the user the diff, and ask: accept, check first, or reject. **Check** runs the positive probe
   (`memory_check`, minutes and a few dollars): it keeps a lesson only if it changes the plans its probes test —
   TEACHES; REDUNDANT means the memory or the model already does it; NO EFFECT means the edit changed nothing.
   Poll `memory_status` until it lands and report the verdict.
6. Only on the user's explicit yes, call `memory_accept` with `user_said_yes: true`. Report whether it was checked and
   versioned, and where the previous memory was kept. A refusal leaves the memory as it was; say why.
