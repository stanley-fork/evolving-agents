import re

CODE = "LK-L11"
MARK = re.compile(r"#\s*(TODO|FIXME|XXX)\b")


def check(ctx):
    out = []
    for mod in ctx.modules(ctx.pkg):
        for n, line in enumerate(mod.source.splitlines(), start=1):
            if MARK.search(line):
                out.append((mod.rel, n, CODE, "todo"))
    return out
