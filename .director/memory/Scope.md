# Scope
Which parts of the repository are active and which must be left alone.

## #active_project_director
**When:** starting a task in the evolving-agents repository and deciding where the work belongs.
**Do:** Do the work in the director, `packages/director/`; it is the active project.
**Because:** The lab states the director is the active project; everything else is legacy, paused or frozen.
**Links:** [[Scope#leave_legacy_alone]] · [[Director_Code#run_tests_before_push]]

## #leave_legacy_alone
**When:** about to edit a file under `legacy/`, `ai-os/`, `plugin/`, `packages/memory/` or `packages/agentvcs/`.
**Do:** Do not change it unless the person explicitly asked for that change.
**Because:** These directories are legacy, paused or frozen.
**Links:** [[Scope#active_project_director]]
