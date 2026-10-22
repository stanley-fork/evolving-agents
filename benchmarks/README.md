# benchmarks/

**Temporary home.** [`director-bench/`](director-bench/) is staged here until its own repository,
`EvolvingAgentsLabs/director-bench`, exists; then it moves there with this folder removed. It is independent of the
director by design (it knows no director internals; the director is one configuration that calls its CLI), and
nothing in this repository imports it. Its tests are not in this repository's CI: run them in place with
`cd benchmarks/director-bench && python -m pytest -q`.

`director-bench/HISTORY.md` keeps the commit history it had before it was staged here. `results/` holds the
summaries of DB0's third live check (no traces), cited by
[BRIEF-DB0](../packages/director/docs/BRIEF-DB0.md).
