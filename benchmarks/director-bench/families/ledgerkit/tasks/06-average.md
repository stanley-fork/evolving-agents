Add an `average` command to ledgerkit.

`average [account]` returns the mean entry amount — of that account, or of the
whole ledger when no account is given — rounded down (towards negative
infinity) to a whole cent and formatted with two decimals: entries 1.00 and
1.01 give `1.00`; entries -0.01 and -0.02 give `-0.02`. When there are no
entries it returns `0.00`.

Include tests, and follow the project's conventions.
