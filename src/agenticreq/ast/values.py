"""What number a limit argument holds, read from code without running it, shared by every rule that reads one.

The extractor already reads literals, module and imported constants (``from config import X``, ``config.X``),
simple one-hop variables, dataclass / settings defaults and ``os.getenv("X", "6")`` defaults. This module adds what
was left as "not a literal":

* a class attribute (``Limits.MAX_TURNS``);
* a parameter of the function the limit is set in, read at every place the project calls that function with a
  constant (``def go(task, turns): Runner.run(..., max_turns=turns)`` and ``go(task, 6)``);
* a name the project defines more than once with different values: never one picked at random, never "no limit",
  but "several values ... cannot pick" (a sign-off);
* a value that only exists at run time (``sys.argv``, ``input()``, an environment read with no default, a parameter
  nothing calls with a constant): said so, with the expression, for a sign-off.

The library never proposes a number: every value here is one written in the project, with where it is written.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenticreq.ast.schema import _name, literal
from agenticreq.ast.symbols import FileCtx

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor

_MAX_DEPTH = 3


@dataclass
class Unresolved:
    """Why a limit's value is not one number: several definitions (``several`` = their file:line), or a value
    that exists only at run time (``runtime`` = the expression)."""

    several: list[str] = field(default_factory=list)
    runtime: str | None = None


def _number(node: ast.AST | None) -> float | None:
    ok, val = literal(node) if node is not None else (False, None)
    if ok and isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def module_definitions(fc: FileCtx, name: str) -> list[tuple[float | None, int]]:
    """(value, line) of every module-level assignment of ``name`` in ``fc``, including inside a top-level if /
    try / with block (``if ENV == "prod": MAX = 6 else: MAX = 9``)."""
    out: list[tuple[float | None, int]] = []

    def visit(stmts: list[ast.stmt]) -> None:
        for st in stmts:
            if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    out.append((_number(st.value), st.lineno))
            elif isinstance(st, ast.If):
                visit(st.body)
                visit(st.orelse)
            elif isinstance(st, ast.Try):
                visit(st.body)
                for h in st.handlers:
                    visit(h.body)
                visit(st.orelse)
            elif isinstance(st, ast.With):
                visit(st.body)

    visit(fc.tree.body)
    return out


def _branch_value(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> tuple[float | None, str]:
    v = _number(node)
    where = f"{fc.rel}:{getattr(node, 'lineno', 0)}"
    if v is None and isinstance(node, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, node)
        if r is not None and r.kind == "const":
            v = _number(r.node)
            where = f"{r.fc.rel}:{getattr(r.node, 'lineno', 0)}"
    return v, where


def several_definitions(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> list[str]:
    """file:line of each definition when the name ``node`` refers to is set to different values: in its module
    (``if prod: MAX = 6 else: MAX = 9``), or by a conditional expression (``a.MAX if x else b.MAX``)."""
    if isinstance(node, ast.IfExp):
        a, b = _branch_value(ex, fc, node.body), _branch_value(ex, fc, node.orelse)
        return [a[1], b[1]] if a[0] is not None and b[0] is not None and a[0] != b[0] else []
    if not isinstance(node, (ast.Name, ast.Attribute)):
        return []
    r = ex.symbols.resolve(fc, node)
    if r is None or r.kind != "const":
        return []
    if isinstance(r.node, ast.IfExp):  # turns = a.MAX if x else b.MAX
        return several_definitions(ex, r.fc, r.node)
    defs = module_definitions(r.fc, r.name)
    values = {v for v, _ in defs}
    if len(defs) > 1 and len(values) > 1:
        return [f"{r.fc.rel}:{line}" for _, line in defs]
    return []


def class_attribute(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> tuple[float, str] | None:
    """``Limits.MAX_TURNS``: the value assigned to that attribute in the class body, with its file:line."""
    if not isinstance(node, ast.Attribute):
        return None
    r = ex.symbols.resolve(fc, node.value)
    cls = r.node if r is not None and r.kind == "class" else None
    if cls is None and isinstance(node.value, ast.Name):
        cls = fc.classes.get(node.value.id)
        r_fc = fc
    else:
        r_fc = r.fc if r is not None else fc
    if not isinstance(cls, ast.ClassDef):
        return None
    for st in cls.body:
        targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == node.attr for t in targets):
            v = _number(getattr(st, "value", None))
            if v is not None:
                return v, f"{r_fc.rel}:{st.lineno}"
    return None


def _param_index(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> tuple[int, bool] | None:
    """(position, is_method) of parameter ``name``; None when the function has no such parameter."""
    args = [*fn.args.posonlyargs, *fn.args.args]
    for i, a in enumerate(args):
        if a.arg == name:
            return i, bool(args) and args[0].arg in ("self", "cls")
    if any(a.arg == name for a in fn.args.kwonlyargs):
        return -1, False
    return None


def _calls_named(ex: ProjectExtractor, name: str) -> list[tuple[FileCtx, ast.Call]]:
    """Every call of a function named ``name`` in the project: one walk of every file, kept on the extractor (a big
    project would otherwise be walked once per value read)."""
    index: dict[str, list[tuple[FileCtx, ast.Call]]] | None = getattr(ex, "_calls_by_name", None)
    if index is None:
        index = {}
        for fc in ex.files:
            for c in ast.walk(fc.tree):
                if isinstance(c, ast.Call):
                    n = _name(c.func)
                    if n:
                        index.setdefault(n, []).append((fc, c))
        ex._calls_by_name = index  # type: ignore[attr-defined]
    return index.get(name, [])


def from_call_sites(
    ex: ProjectExtractor, fc: FileCtx, node: ast.AST, call: ast.AST | None, depth: int = 0
) -> tuple[float | None, list[str], str | None]:
    """A limit set from a parameter of the function it is written in: the values the project passes for that
    parameter. Returns (value, several, where): one value with the file:line of the call that passes it, or the
    file:line of every call when they pass different values, or (None, [], None) when no call passes a constant."""
    if not isinstance(node, ast.Name) or call is None or depth > _MAX_DEPTH:
        return None, [], None
    fn = ex._enclosing_fn.get(id(call))
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None, [], None
    idx = _param_index(fn, node.id)
    if idx is None:
        return None, [], None
    pos, is_method = idx
    default = None
    defaults = fn.args.defaults
    plain = [*fn.args.posonlyargs, *fn.args.args]
    if pos >= 0 and defaults and pos >= len(plain) - len(defaults):
        default = defaults[pos - (len(plain) - len(defaults))]
    found: list[tuple[float, str]] = []
    unreadable = False
    for other, c in _calls_named(ex, fn.name):
        arg: ast.AST | None = next((k.value for k in c.keywords if k.arg == node.id), None)
        if arg is None and pos >= 0:
            p = pos - (1 if is_method and isinstance(c.func, ast.Attribute) else 0)
            if 0 <= p < len(c.args) and not any(isinstance(a, ast.Starred) for a in c.args[: p + 1]):
                arg = c.args[p]
        if arg is None:
            arg = default
        v = _number(arg)
        if v is None and isinstance(arg, (ast.Name, ast.Attribute)):
            r = ex.symbols.resolve(other, arg)
            v = _number(r.node) if r is not None and r.kind == "const" else None
            if v is None:
                v2, sev, _ = from_call_sites(ex, other, arg, c, depth + 1)
                v = v2 if not sev else None
        if v is None:
            unreadable = True
            continue
        found.append((v, f"{other.rel}:{c.lineno}"))
    if not found:
        return None, [], None
    values = {v for v, _ in found}
    if len(values) > 1 or unreadable:
        return None, [w for _, w in found], None
    return found[0][0], [], found[0][1]


#: calls / names whose value is decided when the program runs
_RUNTIME_CALLS = {"input", "getpass"}
_RUNTIME_ATTRS = {"argv", "environ", "stdin"}


def unread_expression(node: ast.AST) -> tuple[str, bool]:
    """(expression text, read at run time?) for a value no static rule could read. ``True`` when the value is
    decided when the program runs: sys.argv, input(), os.environ[...], an environment read with no default."""
    text = ast.unparse(node)[:80]
    for n in ast.walk(node):
        nm = (_name(n.func) or "") if isinstance(n, ast.Call) else ""
        if nm in _RUNTIME_CALLS or (isinstance(n, ast.Attribute) and n.attr in _RUNTIME_ATTRS):
            return text, True
        if isinstance(n, ast.Call) and nm in ("getenv", "get") and len(n.args) == 1 and not n.keywords:
            if nm == "getenv" or "environ" in ast.unparse(n.func):
                return text, True
    return text, False
