Add a `balance` command to ledgerkit.

`balance <account>` returns the sum of all entries of that account, formatted
with two decimals, as `balance <account>: <amount>` — for example
`balance cash: 10.25`. An account with no entries has balance `0.00`.
Calling it without an account is a bad-input error.

Include tests, and follow the project's conventions.
