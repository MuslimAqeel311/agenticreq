"""Project symbol table: which file a name comes from, followed across local imports.

Agents rarely define their tools in the file that builds the agent. This module finds the
project modules an entry file imports (transitively), and resolves a name or ``mod.attr`` to the
function / class / constant it refers to. Third-party packages are never loaded: only ``.py``
files under the project's search roots count. Pure ``ast`` — nothing is imported or executed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from agenticreq.pathutil import is_within

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "site-packages",
    ".agenticreq",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".tox",
}
_ROOT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")
MAX_RESOLVE_DEPTH = 8
#: framework / library packages. Their source is never followed unless it sits inside the scanned root,
#: even when a repo vendors it (an example inside the smolagents repo imports the smolagents source).
FRAMEWORK_PACKAGES = {
    "agents",
    "agno",
    "anthropic",
    "autogen",
    "autogen_agentchat",
    "autogen_core",
    "autogen_ext",
    "beeai_framework",
    "camel",
    "claude_agent_sdk",
    "crewai",
    "crewai_tools",
    "dspy",
    "google",
    "haystack",
    "langchain",
    "langchain_anthropic",
    "langchain_community",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "litellm",
    "llama_index",
    "metagpt",
    "openai",
    "pydantic",
    "pydantic_ai",
    "semantic_kernel",
    "smolagents",
    "strands",
}

FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class FileCtx:
    path: Path
    rel: str
    tree: ast.Module
    imports: set[str] = field(default_factory=set)  # top-level package names (framework detection)
    consts: dict[str, ast.expr] = field(default_factory=dict)  # module-level NAME = <expr>
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    functions: dict[str, FuncDef] = field(default_factory=dict)
    #: local name -> (module path, attribute or None). ``from m import a as b`` -> b: (m, a);
    #: ``import m as b`` -> b: (m, None)
    import_table: dict[str, tuple[Path | None, str | None]] = field(default_factory=dict)
    #: assignments to a plain name anywhere in the file (function locals included): name -> [(line, value)]
    local_assigns: dict[str, list[tuple[int, ast.expr]]] = field(default_factory=dict)
    #: True when the file was reached by following an import from the scanned root
    via_import: bool = False


@dataclass
class Resolved:
    """What a name resolves to. ``kind`` is function | class | const | module."""

    kind: str
    fc: FileCtx
    name: str
    node: ast.AST | None = None


def iter_py_files(root: Path, include_tests: bool = True) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    from agenticreq.ast.classify import is_test_file

    return [
        p
        for p in sorted(root.rglob("*.py"))
        if not (SKIP_DIRS & set(p.relative_to(root).parts))
        and (include_tests or not is_test_file(str(p.relative_to(root))))
    ]


def _project_root(start: Path) -> Path:
    cur = start if start.is_dir() else start.parent
    for cand in [cur, *cur.parents][:4]:
        if any((cand / m).exists() for m in _ROOT_MARKERS):
            return cand
    return cur


class SymbolTable:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.base = root if root.is_dir() else root.parent
        self.project_root = _project_root(root)
        self.by_path: dict[Path, FileCtx] = {}
        self.errors: list[tuple[str, int | None, str]] = []

    # ------------------------------------------------------------ loading
    def load(self) -> list[FileCtx]:
        """Parse the scanned files, then the local modules they import (transitively)."""
        # test code configures models, bounds and fake tools too, but it is not what ships
        queue = [(p, False) for p in iter_py_files(self.root, include_tests=False)]
        while queue:
            p, via = queue.pop(0)
            p = p.resolve()
            if p in self.by_path:
                continue
            fc = self._parse(p, via)
            if fc is None:
                continue
            self.by_path[p] = fc
            for target, _ in fc.import_table.values():
                if target is not None and target not in self.by_path:
                    queue.append((target, True))
        return list(self.by_path.values())

    def _rel(self, p: Path) -> str:
        from agenticreq.pathutil import relative

        for b in (self.base, self.project_root):
            rel = relative(p, b)
            if rel is not None:
                return str(rel)
        return str(p)

    def _parse(self, p: Path, via: bool) -> FileCtx | None:
        rel = self._rel(p)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"), filename=str(p))
        except SyntaxError as e:
            self.errors.append((rel, e.lineno, f"SyntaxError: {e.msg}"))
            return None
        except ValueError as e:  # null bytes (3.10 / 3.11 raise ValueError, 3.12+ SyntaxError)
            self.errors.append((rel, None, f"not parseable as Python: {e}"))
            return None
        except OSError as e:
            self.errors.append((rel, None, f"unreadable: {e}"))
            return None
        fc = FileCtx(path=p, rel=rel, tree=tree, via_import=via)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                fc.imports |= {a.name.split(".")[0] for a in node.names}
                for a in node.names:
                    mod = self.find_module(a.name, p)
                    if mod is None:
                        continue
                    if a.asname:
                        fc.import_table[a.asname] = (mod, None)
                        continue
                    # `import a.b.c` binds `a`; keep the dotted spelling too so `a.b.c.f` resolves
                    fc.import_table[a.name] = (mod, None)
                    top = self.find_module(a.name.split(".")[0], p)
                    if top is not None:
                        fc.import_table.setdefault(a.name.split(".")[0], (top, None))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    fc.imports.add(node.module.split(".")[0])
                pkg = self.find_module(node.module or "", p, node.level)
                for a in node.names:
                    if a.name == "*":
                        continue
                    sub = self.find_module(f"{node.module}.{a.name}" if node.module else a.name, p, node.level)
                    if sub is not None:  # from pkg import submodule
                        fc.import_table[a.asname or a.name] = (sub, None)
                    elif pkg is not None:
                        fc.import_table[a.asname or a.name] = (pkg, a.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                fc.local_assigns.setdefault(node.targets[0].id, []).append((node.lineno, node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                fc.local_assigns.setdefault(node.target.id, []).append((node.lineno, node.value))
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        fc.consts[t.id] = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                fc.consts[stmt.target.id] = stmt.value
            elif isinstance(stmt, ast.ClassDef):
                fc.classes[stmt.name] = stmt
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fc.functions[stmt.name] = stmt
        return fc

    def find_module(self, dotted: str, importer: Path, level: int = 0) -> Path | None:
        """Path of a project module, or None for anything not in the project (stdlib, site-packages)."""
        parts = [x for x in dotted.split(".") if x]
        if level:
            anchor = importer.parent
            for _ in range(level - 1):
                anchor = anchor.parent
            roots = [anchor]
        else:
            if not parts:
                return None
            roots = list(dict.fromkeys([importer.parent, self.base, self.project_root, self.project_root / "src"]))
        for r in roots:
            if (
                not level
                and parts[0] in FRAMEWORK_PACKAGES
                and not is_within(r.joinpath(parts[0]).resolve(), self.base.resolve())
            ):
                continue
            cand = r.joinpath(*parts)
            for p in (cand.with_suffix(".py") if parts else None, cand / "__init__.py"):
                if p is not None and p.is_file() and not (SKIP_DIRS & set(p.parts)):
                    return p.resolve()
        return None

    # ------------------------------------------------------------ resolution
    def resolve(self, fc: FileCtx, node: ast.AST, depth: int = 0) -> Resolved | None:
        """Resolve ``name`` or ``mod.attr`` (as written in ``fc``) to its definition."""
        if depth > MAX_RESOLVE_DEPTH:
            return None
        if isinstance(node, ast.Name):
            found = self._lookup(fc, node.id, depth)
            if found is None and node.id in fc.local_assigns:
                # a local variable (e.g. `tools = [...]` inside main()): the nearest assignment above the use
                line = getattr(node, "lineno", 10**9)
                before = [v for ln, v in fc.local_assigns[node.id] if ln <= line] or [
                    v for _, v in fc.local_assigns[node.id]
                ]
                found = Resolved("const", fc, node.id, before[-1])
            return found
        if isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
            dotted = ast.unparse(node.value)
            if dotted in fc.import_table and fc.import_table[dotted][1] is None:
                target = self.by_path.get(fc.import_table[dotted][0])  # type: ignore[arg-type]
                return self._lookup(target, node.attr, depth + 1) if target is not None else None
            owner = self.resolve(fc, node.value, depth + 1)
            if owner is not None and owner.kind == "module":
                return self._lookup(owner.fc, node.attr, depth + 1)
        return None

    def _lookup(self, fc: FileCtx, name: str, depth: int) -> Resolved | None:
        if name in fc.functions:
            return Resolved("function", fc, name, fc.functions[name])
        if name in fc.classes:
            return Resolved("class", fc, name, fc.classes[name])
        if name in fc.consts:
            return Resolved("const", fc, name, fc.consts[name])
        if name in fc.import_table:
            path, attr = fc.import_table[name]
            target = self.by_path.get(path) if path is not None else None
            if target is None:
                return None
            if attr is None:
                return Resolved("module", target, target.rel)
            if depth > MAX_RESOLVE_DEPTH:
                return None
            return self._lookup(target, attr, depth + 1)  # follows re-exports
        return None
