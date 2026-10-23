# Measurements
How runs that spend money or call models are planned, compared, judged and recorded.

## #brief_before_run
**When:** about to launch anything that spends money or calls a model on the lab's behalf (a run, eval, benchmark).
**Do:** First write `packages/director/docs/BRIEF-<ID>.md`; do not start the run until it exists.
**Because:** The lab requires a brief before any run.
**Links:** [[Measurements#brief_contents]] · [[Measurements#verdict_fixed_before_run]] · [[Measurements#headroom_first]]

## #brief_contents
**When:** writing a `BRIEF-<ID>.md` before a run.
**Do:** Include: what is run, why, the model and provider, the verdict table, what would falsify it, the stopping condition and a spending cap.
**Because:** These are the required parts of a brief.
**Links:** [[Measurements#brief_before_run]] · [[Measurements#verdict_fixed_before_run]] · [[Measurements#name_planner_and_judge]]

## #record_result_in_brief
**When:** a run has finished (or been stopped) and its result is known.
**Do:** Write the result into the same brief under `## Result` / `## Outcome`, dated, stating what was spent.
**Because:** The brief is the single record of a run, and the public site checks its numbers against the briefs.
**Links:** [[Writing#update_citing_docs]] · [[Measurements#flat_result_published_flat]] · [[Measurements#smoke_run_not_measurement]]

## #verdict_fixed_before_run
**When:** setting the bar that decides whether a run succeeded, or tempted to adjust it after seeing numbers.
**Do:** Fix the verdict table in the brief before the run, and never move it after seeing the numbers.
**Because:** The lab rule: the verdict table is fixed before the run.
**Links:** [[Measurements#brief_contents]] · [[Measurements#flat_result_published_flat]]

## #flat_result_published_flat
**When:** a result came in below the bar of the verdict table and you are writing it up.
**Do:** Publish it as flat; never reword it into a win.
**Because:** The lab rule: a result below the bar is published as flat.
**Links:** [[Measurements#verdict_fixed_before_run]] · [[Measurements#record_result_in_brief]]

## #headroom_first
**When:** about to measure whether a change improves something.
**Do:** Measure the baseline first, before measuring the improvement.
**Because:** If the baseline is already near the ceiling there is no room to show an improvement.
**Links:** [[Measurements#stop_at_ceiling]] · [[Measurements#brief_before_run]]

## #stop_at_ceiling
**When:** the baseline you just measured is already near the ceiling.
**Do:** Stop the measurement and say so in the write-up.
**Because:** C0, D0O and PG were stopped there for this reason.
**Links:** [[Measurements#headroom_first]] · [[Measurements#record_result_in_brief]]

## #compare_against_replaced
**When:** choosing what to compare a new version against in a measurement.
**Do:** Compare against the version it replaces; never against a frontier model alone.
**Because:** The lab rule for measurements.
**Links:** [[Measurements#name_planner_and_judge]] · [[Measurements#brief_contents]]

## #name_planner_and_judge
**When:** writing up a measurement or its brief.
**Do:** State which model planned and which model judged.
**Because:** Results depend on the models used; since 2026-10-21 the person plans with Opus.
**Links:** [[Measurements#label_sonnet_numbers]] · [[Measurements#compare_against_replaced]]

## #label_sonnet_numbers
**When:** reporting or citing numbers that were measured with Sonnet.
**Do:** Label them as Sonnet's numbers.
**Because:** Since 2026-10-21 the person plans with Opus, so Sonnet numbers must not be read as the current setup's.
**Links:** [[Measurements#name_planner_and_judge]]

## #smoke_run_not_measurement
**When:** a run used one seed or only a handful of cases (a smoke run) and you are writing it up.
**Do:** Treat it as deciding nothing, and say in the write-up that it is a smoke run, not a measurement.
**Because:** One seed or a handful of cases decides nothing.
**Links:** [[Measurements#record_result_in_brief]]
