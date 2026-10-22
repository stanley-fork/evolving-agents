"""Argument helpers for commands."""
from ledgerkit.errors import LedgerError


def pop_option(args, flag):
    """Remove `flag VALUE` from args; return (rest, value or None)."""
    rest, value, i = [], None, 0
    while i < len(args):
        if args[i] == flag:
            if i + 1 >= len(args):
                raise LedgerError(400, f"{flag} needs a value")
            value = args[i + 1]
            i += 2
            continue
        rest.append(args[i])
        i += 1
    return rest, value


def pop_flag(args, flag):
    """Remove a boolean flag; return (rest, present)."""
    rest = [a for a in args if a != flag]
    return rest, len(rest) != len(args)


def expect_count(args, count, usage):
    if len(args) != count:
        raise LedgerError(400, f"usage: {usage}")
    return args
