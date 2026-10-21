You are writing probes for an experiment on whether an AI coding agent remembers, across days, what its user
decided. Work only from the files in this directory.

## What is here

- `thread.md` — every message a person wrote to their AI coding agent over eight days of work (2026-10-14 to
  2026-10-21), dated, in order. Only the person's messages; the agent's replies are not included. The person writes
  mostly in Spanish.
- `avoid.md` — situations already used in earlier tests. Do not repeat them.

## The task

Find the **decisions** the person made in this thread: a choice, a cap, a rule, a correction, a preference ("use X,
not Y", "no more than N", "from now on …", "don't …", "the UI is …"). Then write **14 candidate probes**. Each rests on
**one** decision made on a day *k* and is set on a **later** day (at least one day after): a fresh session of the agent
meets a situation where that decision should change what it does.

Rules for a probe:

1. **The decision must not be derivable from the probe.** An agent that never heard the decision must not be able to
   reason its way to the right plan from the situation alone. Good: a number, a model name, a provider, a tool, a
   choice between options the situation leaves open. Bad: anything a careful engineer would do anyway.
2. **At least 10 of the 14 are tasks in flight**: write the request and the state it arrives in, as the person or the
   work would present it, never as a quiz, never naming or hinting at the decision. Give 2 or 3 checklist items about
   **actions** (which model, ask first or not, what to build or not, what to tell the person), decidable from what the
   plan says it will do.
3. **At most 4 are "why" probes**: a situation where the agent must explain or act on the decision's **reason**; the
   checklist names the reason by meaning, never by exact wording.
4. **Quote the decision verbatim**: give the date and a verbatim excerpt of 6–40 words from the person's message in
   `thread.md` that states it. It will be checked by string search against that date's messages.
5. **No private projects.** Some messages concern private projects; skip decisions about them. Never write the names
   of private projects, clients or machines; where a probe needs one, write `{PRIVATE_PROJECT}`.
6. Vary the decisions: no two probes rest on the same decision.

## Output

Write `out/candidates.jsonl`, one JSON object per line, best first:

{"id": "c01-<short-slug>", "kind": "task" or "why", "day": "2026-10-NN", "situation": "...",
 "checklist": [{"id": "c01-<short-slug>/1", "text": "..."}, ...],
 "decision": {"date": "2026-10-NN", "quote": "<verbatim from the person's message that day>",
              "gist": "<the decision, in one English sentence>"}}

`day` is the day the probe is set, later than `decision.date`. Ids run c01 to c14. Then reply with one line per probe:
its id and the decision it rests on.
