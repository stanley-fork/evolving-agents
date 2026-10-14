# The operational memory's format

The memory is a small wiki of **operational** knowledge — how the lab works — in the shape of the lab's
memory design: a **page** is a collection of **atomic statements**, the smallest verifiable unit; links live
inside statements, and a task is planned by walking from an entry statement along its links. Encyclopedic
facts (what a project is, what a number was) appear only as support for an operational statement.

## Files

```
memory/
  Index.md            situations → entry statements (where a walk starts)
  <Page_Name>.md      one page per subject: a practice, a tool, a project, a kind of task
```

## A page

```markdown
# Experiment_Briefs
How an experiment is set up before anything runs.

## #brief_before_run
**When:** a run on Colab, a GPU, a paid API or the user's machine is about to start.
**Do:** write `results/<run>/BRIEF.md` first — what, why, model, provider, what falsifies it, the verdict table, the stopping condition.
**Because:** the briefing goes before the `nohup`, not beside the report (user's rule).
**Links:** [[Measurement#headroom_first]] · [[Colab#one_session_one_training]]
```

- A statement id is `Page_Name#snake_case`, unique, stable once written.
- **When** is the trigger, phrased as a situation an agent will recognise mid-task, not as a topic.
- **Do** is an action, concrete enough to check: a command, a file, a question to the user, a refusal.
- **Because** is the reason or the evidence (a run, an incident, the user's decision), so a statement can
  be verified or retired.
- **Links** name the statements that apply next or alongside; a walk follows them while they apply.

## Index.md

One line per situation, pointing at the statements a walk should start from:

```markdown
- About to launch a run (Colab, GPU, paid API, local model) → [[Experiment_Briefs#brief_before_run]], [[Local_Hardware#ask_before_heavy_runs]]
```

## What a walk is

Index → the matching entry statements → their links, while each applies → a plan whose actions cite the
statements they follow (`[[Page#id]]`). A plan that cites nothing either found no applicable statement and
says so, or did not walk.
