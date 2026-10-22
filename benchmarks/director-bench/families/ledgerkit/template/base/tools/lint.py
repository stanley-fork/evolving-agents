#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from lintrules import Context, enabled_rules  # noqa: E402


def main() -> int:
    ctx = Context(ROOT)
    problems = []
    for rule in enabled_rules(ROOT / "tools" / "lint.toml"):
        problems += rule.check(ctx)
    for path, line, code, msg in sorted(set(problems)):
        print(f"{path}:{line}: {code} {msg}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
