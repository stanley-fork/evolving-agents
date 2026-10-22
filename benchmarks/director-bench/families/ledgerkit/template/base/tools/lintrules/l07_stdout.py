import ast

CODE = "LK-L07"


def check(ctx):
    out = []
    for mod in ctx.command_modules():
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                out.append((mod.rel, node.lineno, CODE, "stdout"))
            if isinstance(node, ast.Attribute) and node.attr in ("stdout", "stderr"):
                out.append((mod.rel, node.lineno, CODE, "stdout"))
    return out
