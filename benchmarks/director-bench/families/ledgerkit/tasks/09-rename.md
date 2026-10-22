Add a `rename` command to ledgerkit.

`rename <old> <new>` moves every entry of account `<old>` to account `<new>`
and returns `renamed <n> entries from <old> to <new>` (the change must be saved
to the ledger file when run from the command line). If `<old>` has no
entries it is a not-found error; if `<new>` already has entries it is a
conflict error. Missing arguments are a bad-input error.

Include tests, and follow the project's conventions.
