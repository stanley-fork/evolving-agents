import ast

CODE = "LK-L06"


def _codes(ctx):
    tree = ast.parse((ctx.pkg / "errors.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ERROR_CODES":
            return {k.value for k in node.value.keys}
    return set()


def check(ctx):
    out, codes = [], _codes(ctx)
    for mod in ctx.command_modules():
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if getattr(exc, "id", getattr(exc, "attr", None)) != "LedgerError":
                    out.append((mod.rel, node.lineno, CODE, "raw error"))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LedgerError":
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Constant) and first.value in codes):
                    out.append((mod.rel, node.lineno, CODE, "bad code"))
            if isinstance(node, ast.Assert):
                out.append((mod.rel, node.lineno, CODE, "raw error"))
    return out
