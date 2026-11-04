"""A number the project names after the limit it sets: ``MAX_NO_PROGRESS_STEPS = 5``, ``MAX_CHARS = 6000``,
``APPROVAL_TIMEOUT_S = 4 * 3600``. Read for the limits agenticreq cannot otherwise find in code (a no-progress stop,
an output cap, an approval timeout), so a value the code really sets is never reported as "the code sets no limit".
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agenticreq.names import norm

_SKIP = frozenset(
    {".git", ".agenticreq", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__", "tests", "test"}
)
#: a name's unit suffix -> seconds
_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hours": 3600,
    "ms": 0.001,
}


def _value(node: ast.AST) -> float | None:
    """A number written as a literal or simple arithmetic of literals (4 * 3600)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        a, b = _value(node.left), _value(node.right)
        if a is None or b is None:
            return None
        return a * b if isinstance(node.op, ast.Mult) else a + b if isinstance(node.op, ast.Add) else a - b
    return None


@lru_cache(maxsize=16)
def _constants(root: str) -> tuple[tuple[str, float, str], ...]:
    """(name, value, file:line) of every module-level numeric constant in the project."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = Path(dirpath) / f
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            # a module constant, or a field of a module-level class (a dataclass or settings default)
            body = [*tree.body, *(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body)]
            for node in body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    got = _default_in(node.value)
                    v = got[0] if got else None
                    for t in targets:
                        if isinstance(t, ast.Name) and v is not None:
                            rel = os.path.relpath(p, root).replace(os.sep, "/")
                            out.append((t.id, v, f"{rel}:{node.lineno}"))
    return tuple(out)


@dataclass(frozen=True)
class Several:
    """The project gives this limit different values and nothing says which one the agent uses."""

    name: str
    files: tuple[str, ...]

    def text(self) -> str:
        return f"several values for {self.name} found in {', '.join(self.files)}; cannot pick"


def lookup(
    root: str | Path, *words: str, seconds: bool = False, near: tuple[str, ...] = ()
) -> tuple[float, str, str] | Several | None:
    """(value, name, file:line) of the project constant whose name has all ``words``; ``seconds`` reads its unit suffix
    (APPROVAL_TIMEOUT_H = 4 -> 14400). With different values, the ones defined in ``near`` (the files that hold the
    agent) win; when that leaves more than one, the result says so."""
    hits = []
    for name, v, where in _constants(str(Path(root).resolve())):
        parts = norm(name).split("_")
        if all(w in parts for w in words):
            unit = next((_UNITS[x] for x in reversed(parts) if x in _UNITS), 1) if seconds else 1
            hits.append((v * unit, name, where))
    if not hits:
        return None
    if len({h[0] for h in hits}) == 1:
        return hits[0]
    close = [h for h in hits if h[2].rsplit(":", 1)[0] in near]
    if close and len({h[0] for h in close}) == 1:
        return close[0]
    return Several(hits[0][1], tuple(sorted({h[2].rsplit(":", 1)[0] for h in close or hits})))


def find(root: str | Path, *words: str, seconds: bool = False) -> tuple[float, str, str] | None:
    """``lookup`` for a caller with no use for "several": None when there is none, or more than one value."""
    hit = lookup(root, *words, seconds=seconds)
    return None if isinstance(hit, Several) else hit


@lru_cache(maxsize=64)
def setting_default(root: str | Path, attr: str) -> tuple[float, str] | None:
    """The default a settings object gives ``attr`` in code: a class field (``max_messages: int = 120``) or the
    value it is built with (``max_messages=_num("MAX_MESSAGES", 120, int)``, ``int(os.getenv("X", "120"))``).
    (value, how it was found); None when it is not found, or found with different values."""
    found: set[tuple[float, str]] = set()
    for dirpath, dirs, files in os.walk(str(root)):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse((Path(dirpath) / f).read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            for n in ast.walk(tree):
                value: ast.AST | None = None
                if isinstance(n, ast.keyword) and n.arg == attr:
                    value = n.value
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == attr:
                    value = n.value
                got = _default_in(value) if value is not None else None
                if got is not None:
                    found.add(got)
    values = {v for v, _ in found}
    if len(values) != 1:
        return None
    v, env = sorted(found)[0]
    return v, (f"default in code of env {env}" if env else f"default in code of settings.{attr}")


@lru_cache(maxsize=64)
def plain_field(root: str | Path, attr: str) -> bool:
    """``attr`` is a field of a plain class or dataclass, so its default is what the code runs with. A settings class
    (pydantic BaseSettings) can be overridden from the environment."""
    found = False
    for dirpath, dirs, files in os.walk(str(root)):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse((Path(dirpath) / f).read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            for c in ast.walk(tree):
                if not isinstance(c, ast.ClassDef):
                    continue
                if any(
                    isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == attr
                    for n in c.body
                ):
                    if any("Settings" in ast.unparse(b) for b in c.bases):
                        return False
                    found = True
    return found


@lru_cache(maxsize=64)
def setting_text(root: str | Path, attr: str) -> str | None:
    """The text default of a class field ``attr: str = "x"``; None when it is not found, or found with two values."""
    found: set[str] = set()
    for dirpath, dirs, files in os.walk(str(root)):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse((Path(dirpath) / f).read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            for c in ast.walk(tree):
                if isinstance(c, ast.ClassDef):
                    for n in c.body:
                        if (
                            isinstance(n, ast.AnnAssign)
                            and isinstance(n.target, ast.Name)
                            and n.target.id == attr
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str)
                        ):
                            found.add(n.value.value)
    return next(iter(found)) if len(found) == 1 else None


def setting_site(root: str | Path, attr: str) -> str | None:
    """file:line of the first place a settings field ``attr`` gets its default."""
    for name, _, where in _constants(str(Path(root).resolve())):
        if name == attr:
            return where
    return None


def _default_in(node: ast.AST) -> tuple[float, str] | None:
    """A number, or the default of an env read around it: (value, env name or "")."""
    v = _value(node)
    if v is not None:
        return v, ""
    if isinstance(node, ast.Call):
        env = next(
            (
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.isupper()
            ),
            "",
        )
        for a in [*node.args, *(k.value for k in node.keywords)]:
            d = _value(a)
            if d is None and isinstance(a, ast.Constant) and isinstance(a.value, str):
                try:
                    d = float(a.value)
                except ValueError:
                    d = None
            if d is None and isinstance(a, ast.Call):
                inner = _default_in(a)
                if inner is not None:
                    return inner[0], env or inner[1]
            if d is not None:
                return d, env
    return None
