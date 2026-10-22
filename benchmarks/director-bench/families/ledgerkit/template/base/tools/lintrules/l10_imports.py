import ast

CODE = "LK-L10"


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        names = [n.module or "" for n in mod.tree.body if isinstance(n, ast.ImportFrom)]
        if names != sorted(names):
            out.append((mod.rel, 1, CODE, "import order"))
    return out
