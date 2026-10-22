# Director_Code
Rules for changing the director's code, plugin and the independent benchmark.

## #memory_stdlib_only
**When:** editing or adding an import in `director/memory/`.
**Do:** Use only the Python standard library.
**Because:** `director/memory/` is standard library only.
**Links:** [[Director_Code#memory_no_model_calls]] · [[Director_Code#memory_no_package_imports]]

## #memory_no_model_calls
**When:** writing code in `director/memory/`.
**Do:** Never call a model from it.
**Because:** The memory module never calls a model.
**Links:** [[Director_Code#memory_stdlib_only]] · [[Director_Code#memory_no_package_imports]]

## #memory_no_package_imports
**When:** about to import something from the rest of the director package inside `director/memory/`.
**Do:** Don't; `director/memory/` imports nothing from the rest of the package. Keep `tests/test_memory_boundary.py` passing.
**Because:** `tests/test_memory_boundary.py` holds this boundary.
**Links:** [[Director_Code#memory_stdlib_only]] · [[Director_Code#run_tests_before_push]]

## #lessons_through_gate
**When:** writing or changing an interface that accepts a lesson into memory.
**Do:** Route the lesson through `memory.gate`.
**Because:** Every interface accepts a lesson through `memory.gate`.
**Links:** [[Director_Code#person_yes_required]] · [[Director_Code#headless_records_policy]]

## #person_yes_required
**When:** a lesson is about to be accepted into memory from an interactive interface.
**Do:** Require the person's yes (`memory_accept`) before accepting it.
**Because:** A person's yes is required to accept a lesson.
**Links:** [[Director_Code#lessons_through_gate]] · [[Director_Code#headless_records_policy]]

## #headless_records_policy
**When:** a lesson is accepted through `director.headless`, where no person answers.
**Do:** Record `decided_by: policy:<name>` on it instead of a person's yes.
**Because:** `director.headless` is the one exception to the person's-yes rule.
**Links:** [[Director_Code#person_yes_required]]

## #run_tests_before_push
**When:** about to push a change to the director.
**Do:** Run `cd packages/director && python -m pytest tests -q`; expect the 3 agentvcs tests to skip if the Rust CLI is absent.
**Because:** Required before pushing a director change; the 3 agentvcs tests skip without the Rust CLI.
**Links:** [[Director_Code#validate_plugin_before_push]] · [[Privacy#never_commit_key_or_private_term]]

## #validate_plugin_before_push
**When:** about to push a change that touches the plugin.
**Do:** Also run `claude plugin validate .` and `claude plugin test .`.
**Because:** Required before pushing a plugin change.
**Links:** [[Director_Code#run_tests_before_push]] · [[Director_Code#bump_both_versions]]

## #bump_both_versions
**When:** making any change to the plugin.
**Do:** Bump `version` in `packages/director/.claude-plugin/plugin.json` and in the root `.claude-plugin/marketplace.json`, together in the same change.
**Because:** The two versions must move together.
**Links:** [[Director_Code#validate_plugin_before_push]]

## #bench_no_director_imports
**When:** editing code in `benchmarks/director-bench/`.
**Do:** Never import director code; drive the director only as one configuration that calls its CLI.
**Because:** The benchmark is independent of the director.
**Links:** [[Director_Code#bench_tests_in_place]]

## #bench_tests_in_place
**When:** testing a change to `benchmarks/director-bench/`.
**Do:** Run its tests in place: `cd benchmarks/director-bench && python -m pytest -q`; don't expect the repository's CI to run them.
**Because:** Its tests are not part of this repository's CI.
**Links:** [[Director_Code#bench_no_director_imports]]
