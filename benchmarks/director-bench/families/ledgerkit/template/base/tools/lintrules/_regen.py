import hashlib
from pathlib import Path

from lintrules import Context


def render(root: Path):
    ctx = Context(root)
    mods = [(m.path.stem, m.name) for m in ctx.command_modules() if m.name]
    body = [f"from ledgerkit.commands import {', '.join(stem for stem, _ in mods)}", "", "COMMANDS = {"]
    body += [f'    "{name}": {stem}.run,' for stem, name in sorted(mods, key=lambda m: m[1])]
    body.append("}")
    text = "\n".join(body) + "\n"
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    return f"# digest: {digest}\n{text}", [name for _, name in mods]


def target(root: Path) -> Path:
    return Path(root) / "src" / "ledgerkit" / "commands" / "_registry.py"


def write(root: Path):
    text, names = render(root)
    target(root).write_text(text, encoding="utf-8")
    return target(root), names
