"""D2's deterministic half: edits applied to a copy, the index updated, and the lint gate that must be able to
fail — a link to nothing, an unreachable statement, a duplicate id."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from director.learn import apply_edits, lint, new_problems, statements  # noqa: E402


def mem(tmp: Path) -> Path:
    m = tmp / "m"
    m.mkdir()
    (m / "Index.md").write_text("# Index\n- about to commit a document → [[Docs#mirror]]\n")
    (m / "Docs.md").write_text("# Docs\nDocs.\n\n## #mirror\n**When:** a\n**Do:** b\n**Because:** c\n**Links:** \n")
    return m


ADD = {"op": "add", "page": "Shell", "id": "verify_edit", "about": "Shell pitfalls.", "when": "after a sed",
       "do": "grep the new text", "because": "a sed matched nothing", "links": ["Docs#mirror"],
       "situations": ["about to commit a document", "just ran a scripted edit"]}


def test_add_creates_the_page_links_it_and_indexes_it(tmp_path):
    src = mem(tmp_path)
    r = apply_edits(src, [ADD], tmp_path / "out")
    assert r == {"applied": ["add Shell#verify_edit"], "rejected": []}
    out = tmp_path / "out"
    assert "verify_edit" in statements((out / "Shell.md").read_text())
    idx = (out / "Index.md").read_text()
    assert "- about to commit a document → [[Docs#mirror]], [[Shell#verify_edit]]" in idx
    assert "- just ran a scripted edit → [[Shell#verify_edit]]" in idx
    assert lint(out) == [] and new_problems(src, out) == []
    assert "Shell.md" not in [p.name for p in src.iterdir()]          # the source is never touched


def test_modify_keeps_unspecified_fields_and_add_refuses_a_duplicate(tmp_path):
    src = mem(tmp_path)
    r = apply_edits(src, [{"op": "modify", "page": "Docs", "id": "mirror", "do": "update docs/es in the same commit"},
                          {"op": "add", "page": "Docs", "id": "mirror", "when": "x", "do": "y", "because": "z"}],
                    tmp_path / "out")
    assert r["applied"] == ["modify Docs#mirror"] and "exists" in r["rejected"][0]["why"]
    block = statements((tmp_path / "out" / "Docs.md").read_text())["mirror"]
    assert "**Do:** update docs/es in the same commit" in block and "**When:** a" in block


def test_lint_fails_on_a_dangling_link_and_an_unreachable_statement(tmp_path):
    src = mem(tmp_path)
    bad = {**ADD, "links": ["Nowhere#nothing"], "situations": []}
    apply_edits(src, [bad], tmp_path / "out")
    probs = new_problems(src, tmp_path / "out")
    assert "Shell: link to missing [[Nowhere#nothing]]" in probs
    assert "unreachable Shell#verify_edit" in probs


def test_memory_manifest_has_one_lowercase_prompt_dimension_per_page_and_skips_the_format(tmp_path):
    from director.memory_vcs import manifest
    m = mem(tmp_path)
    (m / "README.md").write_text("# format")
    man = manifest(m, "v0")
    assert sorted(man["dimensions"]) == ["memory.docs", "memory.index"]      # agentvcs: names are lowercase
    assert man["dimensions"]["memory.docs"]["kind"] == "prompt"
    assert man["dimensions"]["memory.docs"]["content"]["template"].startswith("# Docs")


def test_compile_copies_only_what_it_is_given_and_lints_the_result(tmp_path, monkeypatch):
    import os
    from director import compile as C
    fake = tmp_path / "claude"
    fake.write_text("""#!/usr/bin/env python3
import json, os
os.makedirs("out", exist_ok=True)
assert sorted(os.listdir("sources")) == ["00-CLAUDE.md", "01-memory"], os.listdir("sources")
open("out/Index.md", "w").write("# Index\\n- about to commit → [[Git#always_pr]]\\n")
open("out/Git.md", "w").write("# Git\\nGit.\\n\\n## #always_pr\\n**When:** a\\n**Do:** b\\n**Because:** c\\n**Links:** \\n")
print(json.dumps({"type": "result", "result": "1 page, 1 statement", "total_cost_usd": 0.01, "num_turns": 3}))
""")
    fake.chmod(0o755)
    monkeypatch.setenv("DIRECTOR_CLAUDE", str(fake))
    (tmp_path / "CLAUDE.md").write_text("Always open a PR.")
    (tmp_path / "memory").mkdir(); (tmp_path / "memory" / "MEMORY.md").write_text("- x")
    res = C.compile_memory([tmp_path / "CLAUDE.md", tmp_path / "memory"], tmp_path / "out" / "memory")
    assert res["pages"] == 1 and res["statements"] == 1 and res["lint"] == []
    assert (tmp_path / "out" / "memory" / "README.md").exists()
    assert C.write_config(tmp_path / "cfg.json", tmp_path / "out" / "memory", tmp_path) is True
    assert C.write_config(tmp_path / "cfg.json", tmp_path / "out" / "memory", tmp_path) is False   # never overwrites
