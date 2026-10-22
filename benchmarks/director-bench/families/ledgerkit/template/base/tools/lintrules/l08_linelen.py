CODE = "LK-L08"
LIMIT = 100


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        for n, line in enumerate(mod.source.splitlines(), start=1):
            if len(line) > LIMIT:
                out.append((mod.rel, n, CODE, "long line"))
    return out
