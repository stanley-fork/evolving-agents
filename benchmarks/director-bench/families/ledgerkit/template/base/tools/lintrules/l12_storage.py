import ast

CODE = "LK-L12"
IO_METHODS = {"write_text", "read_text", "write_bytes", "read_bytes", "open"}


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        if mod.rel.startswith("src/ledgerkit/storage/"):
            continue
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                out.append((mod.rel, node.lineno, CODE, "direct io"))
            elif (isinstance(func, ast.Attribute) and func.attr in IO_METHODS
                  and not (isinstance(func.value, ast.Name) and func.value.id == "backend")):
                out.append((mod.rel, node.lineno, CODE, "direct io"))
    return out
