Add a `largest` command to ledgerkit.

`largest <account>` returns the largest single entry amount of that account,
formatted with two decimals (for example `10.50`). If the account has no
entries it is a not-found error. Calling it without an account is a bad-input
error.

Include tests, and follow the project's conventions.
