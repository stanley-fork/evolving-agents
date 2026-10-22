import ast

CODE = "LK-L15"
LIMIT = 12
BRANCHES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp)


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.FunctionDef):
                score = 1 + sum(isinstance(n, BRANCHES) for n in ast.walk(node))
                if score > LIMIT:
                    out.append((mod.rel, node.lineno, CODE, "complex"))
    return out
