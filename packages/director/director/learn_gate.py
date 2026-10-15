"""The accept gate's command (`LiveStore.commit`): materialise the candidate manifest's pages, lint them, and
report `lint_problems`; `human_accepted` is 1 because this gate runs only from the user's Accept. With a path to
the D0 guard's result (argv[1]), it also reports `guard`, the candidate memory's mean on D0's cases."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director.learn import lint  # noqa: E402

m = json.loads(Path(os.environ["AGENTVCS_MANIFEST_FILE"]).read_text())
with tempfile.TemporaryDirectory() as d:
    for name, dim in m["dimensions"].items():
        text = dim["content"]["template"]
        page = text.splitlines()[0].lstrip("# ").strip() if text.startswith("# ") else name.split(".", 1)[1]
        (Path(d) / f"{'Index' if name == 'memory.index' else page}.md").write_text(text)
    problems = lint(Path(d))
metrics = {"lint_problems": len(problems), "human_accepted": 1}
if len(sys.argv) > 1:
    g = json.loads(Path(sys.argv[1]).read_text())
    metrics["guard"] = g["mean"] if g.get("n_cases") == 14 else 0.0
    metrics["guard_passed"] = 1 if g.get("passed") else 0
print(json.dumps({"metrics": metrics}))
