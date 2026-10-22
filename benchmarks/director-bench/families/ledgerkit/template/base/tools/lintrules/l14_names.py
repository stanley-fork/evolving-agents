import ast
import re

CODE = "LK-L14"
SNAKE = re.compile(r"^_?[a-z][a-z0-9_]*$")


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.FunctionDef) and not SNAKE.match(node.name):
                out.append((mod.rel, node.lineno, CODE, "name"))
    return out
