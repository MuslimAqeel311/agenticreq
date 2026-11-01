"""Caps on what a tool returns or takes, in the forms code really writes them.

A tool is capped when its body (or a project helper it calls) limits a collection, a text or an input:

* slicing: ``rows[:200]``, ``text[:5000]``, ``items[-50:]``; ``itertools.islice(it, 200)``;
* a counter that stops the loop: ``if len(rows) >= 200: break`` / ``return``, ``for i, x in enumerate(..): if i >= N``;
* ``min(x, 200)`` (``max(1, min(n, 10))``), passed on as a bound;
* a size check that refuses: ``if len(content) > MAX_BYTES: raise ...`` (or returns an error);
* SQL ``LIMIT N`` (also in f-strings and with a query parameter), ``.limit(N)``, ``.head(N)``, ``.nlargest(N)``;
* a capping keyword on a call: ``limit=``, ``max_results=``, ``top_k=``, ``page_size=``, ``per_page=``, ``n=``, ``k=``;
* a project helper that does one of these with its own parameter (``truncate(text, 6000)``), followed through
  imports.

Numbers are read through module constants (``MAX_ROWS = 200``), class attributes, imported constants and local
assignments. A value read from the environment or a settings object makes the limit exist with its value unknown
(``source`` names it): the report sends it to a mock run, never to a gap and never to a pass.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agenticreq.ast.classify import BOUND_PARAMS

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor
    from agenticreq.ast.symbols import FileCtx

#: calls whose first argument is a row count: df.head(10), q.limit(5), df.nlargest(3, "x"), cur.fetchmany(50),
#: resp.read(MAX_BYTES)
_COUNT_CALLS = {"limit", "head", "tail", "nlargest", "nsmallest", "take", "top", "fetchmany", "first_n", "read"}
_SQL_LIMIT = re.compile(r"\bLIMIT\s+(\d+|\{\s*(\w+)\s*\}|\?|%s|:\w+|\$\d+)", re.IGNORECASE)
_ENV_CALLS = {"getenv", "get"}
#: names of objects that hold settings read at run time (settings.max_rows, config["limit"], cfg.top_k)
_CONFIG_NAME = re.compile(r"(settings|config|conf|cfg|options|opts|env|params)$", re.IGNORECASE)
_HELPER_DEPTH = 3


@dataclass(frozen=True)
class Limit:
    """One cap. ``value`` None = set at run time (``source`` says where: "env MAX_ROWS", "settings.max_rows")."""

    form: str  # slice | islice | counter | min() | size check | SQL LIMIT | .head() | limit= | helper ...
    value: float | None
    line: int | None
    source: str = ""  # MAX_ROWS / argument max_chars / env MAX_ROWS / settings.max_rows

    @property
    def runtime(self) -> bool:
        return self.value is None and not self.source.startswith("argument ")

    def describe(self, file: str) -> str:
        where = f"{file}:{self.line}" if self.line else file
        if self.runtime:
            return f"limit set from {self.source} ({self.form}, {where})"
        n = f"{int(self.value)}" if self.value is not None and float(self.value).is_integer() else f"{self.value}"
        if self.source.startswith("argument "):  # the caller sets it: say so, with the default it has
            arg = self.source.removeprefix("argument ")
            return (
                f"capped by the {arg} argument"
                + (f" (default {n})" if self.value is not None else "")
                + f" ({self.form}, {where})"
            )
        via = f" = {self.source}" if self.source else ""
        return f"capped at {n}{via} ({self.form}, {where})"


class _Ctx:
    def __init__(self, ex: ProjectExtractor | None, fc: FileCtx | None, fn: ast.FunctionDef | ast.AsyncFunctionDef):
        self.ex, self.fc, self.fn = ex, fc, fn
        args = [*fn.args.posonlyargs, *fn.args.args]
        defaults = dict(zip([a.arg for a in args][len(args) - len(fn.args.defaults) :], fn.args.defaults))
        defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None})
        self.params = [a.arg for a in [*args, *fn.args.kwonlyargs]]
        self.defaults = defaults
        self.locals: dict[str, ast.AST] = {}
        for n in ast.walk(fn):  # the last plain assignment of each local (max_results = max(1, min(...)))
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self.locals[n.targets[0].id] = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                self.locals[n.target.id] = n.value


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def value_of(c: _Ctx, node: ast.AST | None, seen: frozenset[str] = frozenset()) -> tuple[float | None, str] | None:
    """(number, source) a bound expression stands for, or None when it is not a bound at all.
    number None + source "env X" / "settings.x" = set at run time; source "argument p" = the caller sets it."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        v = _num(node.value)
        return (v, "") if v is not None else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = value_of(c, node.operand, seen)
        return (-inner[0], inner[1]) if inner and inner[0] is not None else inner
    if isinstance(node, ast.BinOp):
        a, b = value_of(c, node.left, seen), value_of(c, node.right, seen)
        if a and b and a[0] is not None and b[0] is not None:
            try:
                v = {
                    ast.Mult: a[0] * b[0],
                    ast.Add: a[0] + b[0],
                    ast.Sub: a[0] - b[0],  # half = limit // 2
                    ast.FloorDiv: a[0] // b[0] if b[0] else None,
                    ast.Div: a[0] / b[0] if b[0] else None,
                }.get(type(node.op))
            except Exception:  # noqa: BLE001
                v = None
            return (v, a[1] or b[1]) if v is not None else None
        return a or b
    if isinstance(node, ast.Call):
        fname = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if fname in ("int", "float", "abs", "round") and node.args:
            return value_of(c, node.args[0], seen)
        if fname == "min" and node.args:  # the smallest fixed value caps; a runtime one keeps its source
            vals = [value_of(c, a, seen) for a in node.args]
            fixed = [v for v in vals if v and v[0] is not None and not v[1].startswith("argument ")]
            if fixed:
                return min(fixed, key=lambda v: v[0] or 0)
            return next((v for v in vals if v), None)
        if fname == "max" and node.args:  # max(1, n) is a floor: the cap is whatever n is
            vals = [value_of(c, a, seen) for a in node.args if not (isinstance(a, ast.Constant))]
            return next((v for v in vals if v), None)
        if fname in _ENV_CALLS and _is_env(node):
            key = str(node.args[0].value) if node.args and isinstance(node.args[0], ast.Constant) else "?"
            return None, f"env {key}"
        return None
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        key = str(node.slice.value) if isinstance(node.slice, ast.Constant) else "?"
        return None, f"env {key}"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and _CONFIG_NAME.search(node.value.id):
        key = str(node.slice.value) if isinstance(node.slice, ast.Constant) else "?"
        return None, f"{node.value.id}[{key!r}]"
    if isinstance(node, ast.Name):
        name = node.id
        if name in seen:
            return None
        seen = seen | {name}
        if name in c.locals:
            return value_of(c, c.locals[name], seen)
        if name in c.params:
            d = c.defaults.get(name)
            dv = value_of(c, d, seen) if d is not None else None
            return (dv[0] if dv else None), f"argument {name}"
        resolved = _const(c, node, seen)
        return resolved
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            return _class_attr(c, node.attr, seen)
        owner = ast.unparse(node.value)
        if _CONFIG_NAME.search(owner.split(".")[-1].split("(")[0]) or owner.endswith("()"):
            return None, f"{owner}.{node.attr}"
        resolved = _const(c, node, seen)
        if resolved is not None:
            return resolved
        if node.attr.isupper():  # SomeClass.MAX_ROWS / module.MAX_ROWS not in the project: a named constant
            return None, f"{owner}.{node.attr}"
    return None


def _is_env(call: ast.Call) -> bool:
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr == "getenv":
        return True
    return isinstance(f, ast.Attribute) and f.attr == "get" and _is_environ(f.value)


def _is_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        or isinstance(node, ast.Name)
        and node.id == "environ"
    )


def _const(c: _Ctx, node: ast.AST, seen: frozenset[str]) -> tuple[float | None, str] | None:
    """A module constant (MAX_ROWS = 200), here or imported from another project module."""
    if c.ex is None or c.fc is None:
        return None
    r = c.ex.symbols.resolve(c.fc, node)
    if r is None or r.kind != "const" or not isinstance(r.node, ast.AST):
        return None
    inner = _Ctx(c.ex, r.fc, c.fn)
    inner.locals, inner.params = {}, []
    v = value_of(inner, r.node, seen)
    label = ast.unparse(node)
    if v is None:
        return None
    return v[0], (v[1] if v[1] and v[0] is None else label)


def _class_attr(c: _Ctx, attr: str, seen: frozenset[str]) -> tuple[float | None, str] | None:
    if c.fc is None:
        return None
    for cls in c.fc.classes.values():
        if not any(m is c.fn for m in getattr(cls, "body", [])):
            continue
        for st in cls.body:
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in st.targets):
                v = value_of(c, st.value, seen)
                return (v[0], f"{cls.name}.{attr}") if v and v[0] is not None else v
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == attr:
                v = value_of(c, st.value, seen)
                return (v[0], f"{cls.name}.{attr}") if v and v[0] is not None else v
    return None


def _bound(c: _Ctx, form: str, node: ast.AST | None, at: ast.AST) -> Limit | None:
    v = value_of(c, node)
    if v is None:
        return None
    value, source = v
    if value is not None and value <= 0:
        return None  # rows[:0] / x[:-1] trims, it does not cap
    return Limit(form=form, value=value, line=getattr(at, "lineno", None), source=source)


def _stops(body: list[ast.stmt]) -> bool:
    return any(isinstance(x, (ast.Break, ast.Return, ast.Raise)) for st in body for x in ast.walk(st))


def _own_nodes(fn: ast.AST) -> list[ast.AST]:
    """Every node of the function, but not of functions defined inside it."""
    out: list[ast.AST] = []
    stack = list(getattr(fn, "body", None) or ast.iter_child_nodes(fn))  # the body: not decorators or defaults
    while stack:
        n = stack.pop()
        out.append(n)
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))
    return out


def find_limits(
    ex: ProjectExtractor | None,
    fc: FileCtx | None,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> list[Limit]:
    c = _Ctx(ex, fc, fn)
    out: list[Limit] = []
    nodes = _own_nodes(fn)
    # a slice inside a test ("<html" in body[:500], if name[:3] == "abc") inspects a value, it caps nothing
    tested = {id(x) for n in nodes if isinstance(n, (ast.Compare, ast.BoolOp)) for x in ast.walk(n)}
    tested |= {id(x) for n in nodes if isinstance(n, (ast.If, ast.While, ast.IfExp)) for x in ast.walk(n.test)}
    loops = [n for n in nodes if isinstance(n, (ast.For, ast.AsyncFor, ast.While))]
    in_loop = {id(x) for lp in loops for st in lp.body for x in ast.walk(st)}
    for n in nodes:
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and id(n) not in tested:
            s = n.slice
            if s.upper is not None and not (isinstance(s.upper, ast.UnaryOp) and isinstance(s.upper.op, ast.USub)):
                lim = _bound(c, "slice", s.upper, n)
            elif s.lower is not None and isinstance(s.lower, ast.UnaryOp) and isinstance(s.lower.op, ast.USub):
                lim = _bound(c, "slice", s.lower.operand, n)  # items[-50:]
            else:
                lim = None
            if lim:
                out.append(lim)
        elif (
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and len(n.test.ops) == 1
            and isinstance(n.test.ops[0], (ast.GtE, ast.Gt, ast.Eq))
            and _stops(n.body)
        ):
            left = n.test.left
            sized = isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len"
            counter = isinstance(left, ast.Name) and (id(n) in in_loop)
            if sized or counter:
                form = "counter" if id(n) in in_loop else "size check"
                lim = _bound(c, form, n.test.comparators[0], n)
                if lim:
                    out.append(lim)
        elif isinstance(n, ast.Call):
            fname = (
                n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
            )
            if fname == "islice" and len(n.args) >= 2:
                lim = _bound(c, "islice", n.args[-1], n)
                if lim:
                    out.append(lim)
            elif fname in _COUNT_CALLS and isinstance(n.func, ast.Attribute) and n.args:
                lim = _bound(c, f".{fname}()", n.args[0], n)
                if lim:
                    out.append(lim)
            for a in [*n.args, *(k.value for k in n.keywords)]:  # params={"limit": 5}, {"hitsPerPage": n}
                d = c.locals.get(a.id) if isinstance(a, ast.Name) else a
                if isinstance(d, ast.Dict):
                    for key, val in zip(d.keys, d.values):
                        if (
                            isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                            and key.value.replace("-", "_") in BOUND_PARAMS | {"hitsPerPage", "pageSize", "maxResults"}
                        ):
                            lim = _bound(c, f"{key.value}=", val, n)
                            if lim:
                                out.append(lim)
            for k in n.keywords:
                if k.arg and k.arg in BOUND_PARAMS:
                    lim = _bound(c, f"{k.arg}=", k.value, n)
                    if lim:
                        out.append(lim)
            for a in n.args[:1]:  # SQL text: "SELECT ... LIMIT 50", f"... LIMIT {n}"
                out += _sql_limits(c, a, n)
            if depth < _HELPER_DEPTH:
                out += _helper_limits(c, n, depth, seen | {id(fn)})
    return out


def _sql_limits(c: _Ctx, node: ast.AST, at: ast.AST) -> list[Limit]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        m = _SQL_LIMIT.search(node.value)
        if m and m.group(1).isdigit():
            return [Limit("SQL LIMIT", float(m.group(1)), getattr(at, "lineno", None))]
        if m:  # LIMIT ? / %s / :n: the number is a query parameter
            return [Limit("SQL LIMIT", None, getattr(at, "lineno", None), "a query parameter")]
    if isinstance(node, ast.JoinedStr):
        text = "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
            for v in node.values
            if isinstance(v, (ast.Constant, ast.FormattedValue))
        )
        m = _SQL_LIMIT.search(text)
        if m:
            if m.group(1).isdigit():
                return [Limit("SQL LIMIT", float(m.group(1)), getattr(at, "lineno", None))]
            lim = _bound(c, "SQL LIMIT", ast.Name(id=m.group(2)) if m.group(2) else None, at)
            return [lim] if lim else [Limit("SQL LIMIT", None, getattr(at, "lineno", None), "a query parameter")]
    if isinstance(node, ast.Name) and node.id in c.locals:
        return _sql_limits(c, c.locals[node.id], at)
    return []


def _helper_limits(c: _Ctx, call: ast.Call, depth: int, seen: frozenset[int]) -> list[Limit]:
    """A project helper that caps one of its own parameters: the caller's argument for it is the cap."""
    if c.ex is None or c.fc is None:
        return []
    r = c.ex.symbols.resolve(c.fc, call.func)
    if (
        r is None
        or r.kind != "function"
        or not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef))
        or id(r.node) in seen
    ):
        return []
    inner = find_limits(c.ex, r.fc, r.node, depth + 1, seen)
    if not inner:
        return []
    hp = [a.arg for a in [*r.node.args.posonlyargs, *r.node.args.args, *r.node.args.kwonlyargs]]
    bound_args: dict[str, ast.AST] = dict(zip(hp, call.args))
    bound_args.update({k.arg: k.value for k in call.keywords if k.arg})
    out = []
    name = r.node.name
    for lim in inner:
        if lim.source.startswith("argument "):
            p = lim.source.removeprefix("argument ")
            if p in bound_args:  # truncate(text, 6000): the helper's limit is what the caller passes
                got = _bound(c, f"{name}()", bound_args[p], call)
                if got:
                    out.append(got)
                continue
            if lim.value is not None:  # the helper's own default applies
                out.append(Limit(f"{name}()", lim.value, getattr(call, "lineno", None), lim.source))
        else:
            out.append(Limit(f"{name}()", lim.value, getattr(call, "lineno", None), lim.source))
    return out


def best(limits: list[Limit]) -> Limit | None:
    """The limit to report: a fixed number first (the smallest), then one the caller sets, then a runtime one."""
    fixed = [x for x in limits if x.value is not None and not x.source.startswith("argument ")]
    if fixed:
        return min(fixed, key=lambda x: x.value or 0)
    arg = [x for x in limits if x.source.startswith("argument ")]
    if arg:
        return arg[0]
    return limits[0] if limits else None
