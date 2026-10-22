# Privacy
What stays private, what may be published, and how commits are guarded.

## #private_stays_private
**When:** about to save or move the lab's memory, plans, transcripts or lanes config.
**Do:** Keep them in the private locations, `~/.config/director/` or a project's `.director/`, never in public files.
**Because:** The lab's memory, plans, transcripts and lanes config are private.
**Links:** [[Privacy#placeholders_for_private_names]] · [[Privacy#never_commit_key_or_private_term]]

## #placeholders_for_private_names
**When:** writing a public file that would mention a private name.
**Do:** Use a placeholder instead of the private name.
**Because:** Public files use placeholders for private names.
**Links:** [[Privacy#publish_scores_only]] · [[Privacy#never_commit_key_or_private_term]]

## #publish_scores_only
**When:** publishing what a run produced.
**Do:** Publish its scores; keep transcripts, plans and memory private.
**Because:** What a run publishes is its scores.
**Links:** [[Privacy#private_stays_private]] · [[Measurements#record_result_in_brief]]

## #never_commit_key_or_private_term
**When:** about to commit any change.
**Do:** Check that it contains no key and no private term; never commit one.
**Because:** Keys and private terms must not become public.
**Links:** [[Privacy#respect_public_guard]] · [[Director_Code#run_tests_before_push]]

## #respect_public_guard
**When:** the plugin's public guard refuses a commit.
**Do:** Remove the key or private term it flagged; do not work around the guard.
**Because:** The guard exists to refuse such commits.
**Links:** [[Privacy#never_commit_key_or_private_term]]
