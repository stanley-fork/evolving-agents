# labels/

Optional per-run labels that override the v0 heuristic for **memory-induced error** on the trap
task. One file per run: `labels/<run-id>.json`

```json
{"trap_memory_induced": true, "labeler": "llm-judge:<model>", "manual_review": false, "notes": "..."}
```

- The bench never calls an LLM. Labels are produced outside it (e.g. an LLM judge reading the
  trap trace and the memory snapshot `results/<run-id>/memory/<pos>-<trap>/`).
- Review 20 % of judge labels manually (`"manual_review": true`, `"labeler": "manual"` when a
  person overrides). If judge/manual agreement is below 85 %, review all labels manually.
- Without a label file, `bench report` uses the heuristic: trap failed **and** the memory store
  the trap session saw contains the family's `obsolete_marker`.
