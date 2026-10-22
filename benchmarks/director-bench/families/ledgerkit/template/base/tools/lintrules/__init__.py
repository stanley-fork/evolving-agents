import ast
import importlib
import re
import subprocess
from pathlib import Path


class Module:
    def __init__(self, root: Path, path: Path):
        self.path = path
        self.rel = path.relative_to(root).as_posix()
        self.source = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, self.rel)

    def assigned(self, name: str):
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value
        return None

    @property
    def name(self):
        value = self.assigned("NAME")
        return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


class Context:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.pkg = self.root / "src" / "ledgerkit"
        self._changed = None

    def modules(self, base: Path):
        return [Module(self.root, p) for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts]

    def command_modules(self):
        return [Module(self.root, p) for p in sorted((self.pkg / "commands").glob("*.py"))
                if not p.name.startswith("_")]

    def test_modules(self):
        return self.modules(self.root / "tests")

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True).stdout

    def changed(self) -> set:
        if self._changed is None:
            try:
                base = self.git("rev-list", "--max-parents=0", "HEAD").split()[-1]
                files = self.git("diff", "--name-only", base).split()
                files += self.git("ls-files", "--others", "--exclude-standard").split()
            except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
                files = None
            self._changed = set(files) if files is not None else None
        return self._changed


def enabled_rules(path: Path):
    section, names = None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if re.fullmatch(r"\[\w+\]", line):
            section = line[1:-1]
        elif section == "rules" and "=" in line:
            key, value = (s.strip() for s in line.split("=", 1))
            if value == "true":
                names.append(key)
    return [importlib.import_module(f"lintrules.{name}") for name in names]
