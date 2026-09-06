#!/usr/bin/env python3
"""Every number the demo shows, resolved to the artifact it is attributed to.

The redesign in `doc/20` put two real projects on the demo's first screen and
gave the desk an inspector that cites what it read. Both of those are worth
exactly as much as the citations are true, and a citation is only true until
somebody regenerates an artifact.

So this resolves them. Three checks, and they fail the build rather than warn:

1. **The published demo carries hemo's attested numbers.**  Every constant the
   hemodynamics scope displays is read out of
   `projects/hemo-verified/gates/reports/h0.json` and looked for in the built
   HTML. `ai-ui/test/hemo-demo.test.ts` already checks the *source*; this checks
   the *artifact that ships*, which is a different thing — a demo can be built
   from correct source and then edited, and the website serves the file.

2. **The published demo carries coclea's.**  The correct membrane chain's worst
   relative error is computed in TypeScript by an independent eigensolver and
   recorded in Python by LAPACK against a sympy closed form. They must agree to
   the precision the page prints.

3. **Every address the inspector can cite exists.**  A citation pointing at a
   path that is not in the repository is worse than no citation: it looks like
   evidence, it survives review, and the only way to find it is to click it.

Run:  python3 scripts/check-demo-provenance.py [--demo <path>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H0 = ROOT / "projects" / "hemo-verified" / "gates" / "reports" / "h0.json"
COCLEA_A01 = (
    ROOT
    / "projects"
    / "coclea-sr"
    / "gates"
    / "reports"
    / "report_A01_test_eigenvalues_match_the_closed_form.json"
)
DEMO = ROOT.parent / "evolvingagentslabs.github.io" / "demo" / "index.html"
INSPECTOR = ROOT / "ai-ui" / "src" / "inspector.ts"
DESK = ROOT / "ai-ui" / "src" / "desk.ts"

problems: list[str] = []
checked = 0


def fail(msg: str) -> None:
    problems.append(msg)


def load(path: Path) -> dict:
    """A missing artifact is a failure, never a skip.

    Collapsing the two is how four drift tests in this repository turned into
    silent passes on 2026-08-14: the seam broke and the suite stayed green.
    """
    if not path.exists():
        fail(f"{path.relative_to(ROOT.parent)} does not exist; nothing to check against")
        return {}
    return json.loads(path.read_text())


def expect(html: str, needle: str, what: str, where: str) -> None:
    """The demo must contain this string, and it must come from `where`."""
    global checked
    checked += 1
    if needle not in html:
        fail(f"{what}: the demo does not contain {needle!r}, which {where} records")


def check_hemo(html: str) -> None:
    a = load(H0)
    if not a:
        return

    # Printed to four places by `hemo-demo.ts`. The source of truth is the full
    # float; the page shows the rounding, so that is what is looked for.
    expect(html, f"{a['auc_composite']:.4f}", "H0 composite AUC", "h0.json")
    expect(html, f"{a['spearman_composite']:.4f}", "H0 Spearman", "h0.json")
    expect(html, f"{a['auc_per_oracle']['A3']:.3f}", "A3 alone", "h0.json")
    expect(html, f"{a['auc_per_oracle']['A5']:.3f}", "A5 alone", "h0.json")
    expect(html, f"{a['auc_per_oracle']['A4']:.3f}", "A4 alone", "h0.json")
    expect(html, str(a["n"]), "the number of rows", "h0.json")
    expect(html, str(a["kill_threshold"]), "the kill threshold", "h0.json")
    expect(
        html,
        f"{a['false_accept_rate'] * 100:.1f}%",
        "the false-accept rate",
        "h0.json",
    )
    for k in ("python", "numpy", "scipy"):
        expect(html, a["environment"][k], f"the {k} the run used", "h0.json")

    # A4's two readings are the finding, and they must stay distinct. If the
    # artifact were regenerated on the first machine again, the flow's prose
    # would be describing a disagreement that no longer exists.
    global checked
    checked += 1
    if round(a["auc_per_oracle"]["A4"], 3) == 0.706:
        fail(
            "h0.json's A4 is now 0.706, the first machine's reading. "
            "The A4 flow in the demo describes two machines disagreeing; "
            "with one reading there is nothing to describe."
        )


def check_coclea(html: str) -> None:
    a = load(COCLEA_A01)
    if not a:
        return
    global checked

    # The demo's eigensolver is a third implementation — TypeScript, Sturm
    # bisection, no dependencies — next to LAPACK and sympy. `2.592e-4` is the
    # *defect* chain and has no Python report, because the suite does not gate a
    # scheme it does not ship; that number is held by
    # `ai-ui/test/cochlea-demo.test.ts`. What is checkable here is the correct
    # chain's worst error, which both sides compute.
    #
    # Both notations, because the two languages disagree about the exponent and
    # neither is wrong. Python's `:.3e` pads to two digits — `9.278e-06`;
    # JavaScript's `toExponential(3)` does not — `9.278e-6`. Comparing the
    # strings without allowing for that fails on a number that matches, which is
    # the most expensive kind of false alarm: it teaches whoever hits it that
    # this check is noise.
    want = f"{a['max_relative_error']:.3e}"
    alt = re.sub(r"e([+-])0(\d)$", r"e\1\2", want)
    checked += 1
    if want not in html and alt not in html:
        fail(
            f"the correct membrane chain: the demo does not contain {want!r} or {alt!r}, "
            f"which report_A01_test_eigenvalues_match_the_closed_form.json records "
            f"as max_relative_error"
        )

    checked += 1
    if not a.get("passed"):
        fail("GATE-A01 is red on the correct chain; the demo says it freezes")


def check_citations() -> None:
    """Every `at:` the inspector can emit must resolve.

    Two schemes are legitimate and neither is a file path:

    - `flow:<id>#step-<n>` — the flow store's own record, which the desk
      resolves out of the state it already holds and prints verbatim.
    - `agents/<NAME>.md` — a per-project path, so it is checked for shape rather
      than existence: the demo's scopes are simulated and have no directory.

    Everything else must be a path in the repository, and this is the check that
    only exists here: nothing in the TypeScript suite knows what a file is.
    """
    global checked
    src = INSPECTOR.read_text() + DESK.read_text()

    # Addresses written as literals in the shipped rules.
    lits = set(re.findall(r"at:\s*'([^']+)'", src))
    lits |= set(re.findall(r'at:\s*"([^"]+)"', src))
    # And the ones the desk's artifact reader claims to know about.
    lits |= set(re.findall(r"return '(projects/[^\\\n']+)", src))

    for at in sorted(lits):
        checked += 1
        if at.startswith("flow:") or at.startswith("agents/"):
            continue
        if at in ("gate.report", "run.reply"):
            # The flow store's own source labels, not paths. They name where an
            # observation came from inside the runner.
            continue
        if not (ROOT / at).exists():
            fail(f"the inspector can cite {at!r} and no such path exists in the repository")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=DEMO)
    args = ap.parse_args()

    if not args.demo.exists():
        print(f"the built demo is not at {args.demo}", file=sys.stderr)
        print("build it first: cd ai-ui && node scripts/build-demo.ts --out <path>", file=sys.stderr)
        return 2

    html = args.demo.read_text()
    check_hemo(html)
    check_coclea(html)
    check_citations()

    if problems:
        print(f"demo provenance: {len(problems)} problem(s)\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"demo provenance: {checked} claim(s) resolved to the artifact they name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
