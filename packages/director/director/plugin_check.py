"""The plugin's background check (`memory_check`): D7's positive probe on one proposal, its verdict written to
`<proposal>/check.json`.

    python -m director.plugin_check <live memory> <proposal dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .guard import positive_probe


def main() -> int:
    memory, d = Path(sys.argv[1]), Path(sys.argv[2])
    rec = json.loads((d / "proposal.json").read_text())
    work = d / f"check-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True)
    cfg = work / "config.json"           # the runner's config: everything private stays under the proposal
    cfg.write_text(json.dumps({"private_dir": str(work), "workspace": str(memory.parent), "memory_dir": str(memory),
                               "placeholders": {}}))
    try:
        res = positive_probe(rec["probes"], memory, d / "memory", work, {"config": str(cfg)},
                             lambda line: print(line, flush=True))
    except BaseException as e:           # an account limit raises SystemExit: not a verdict
        res = {"verdict": "INCOMPLETE", "teaches": False, "error": repr(e)[:400]}
    (d / "check.json").write_text(json.dumps(res, indent=1))
    (d / "check.running").unlink(missing_ok=True)
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
