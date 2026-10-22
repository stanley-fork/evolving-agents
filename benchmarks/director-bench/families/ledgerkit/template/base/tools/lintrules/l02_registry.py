from lintrules import _regen

CODE = "LK-L02"


def check(ctx):
    out = [(m.rel, 1, CODE, "no NAME") for m in ctx.command_modules() if not m.name]
    path = _regen.target(ctx.root)
    expected, _ = _regen.render(ctx.root)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current != expected:
        out.append((path.relative_to(ctx.root).as_posix(), 1, CODE, "registry stale"))
    return out
