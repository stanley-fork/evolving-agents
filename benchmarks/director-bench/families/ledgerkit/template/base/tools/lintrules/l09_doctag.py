import ast
import re

CODE = "LK-L09"


def check(ctx):
    out = []
    for mod in ctx.command_modules():
        doc = ast.get_docstring(mod.tree) or ""
        if not re.search(rf"^Command: {re.escape(mod.name or '')}$", doc, re.M):
            out.append((mod.rel, 1, CODE, "doc tag"))
    return out
