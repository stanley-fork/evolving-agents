import ast

CODE = "LK-L03"
BANNED_CALLS = {"float", "round", "Decimal"}


def _bad(node) -> bool:
    if isinstance(node, ast.Constant) and type(node.value) is float:
        return True
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in BANNED_CALLS:
        return True
    if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "decimal" for a in node.names):
        return True
    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "decimal":
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return True
    if isinstance(node, ast.FormattedValue) and isinstance(node.format_spec, ast.JoinedStr):
        spec = "".join(str(v.value) for v in node.format_spec.values if isinstance(v, ast.Constant))
        return spec.rstrip().endswith(("f", "%", "e", "g"))
    return False


def check(ctx):
    return [(mod.rel, getattr(node, "lineno", 1), CODE, "raw money")
            for mod in ctx.command_modules() for node in ast.walk(mod.tree) if _bad(node)]
