"""Static partial evaluation: what an expression is, without running the code.

Agents are often built from configuration rather than written out one by one::

    cfg = yaml.safe_load(open(CONFIG_DIR / "agents.yaml"))
    for key, c in cfg.items():
        Agent(role=c["role"], tools=self._tools_for(key), max_iter=c.get("max_iter", 6))

To see four agents there (and which tools each one holds), the extractor needs the iterable's
rows and, per row, what ``c["role"]`` and ``self._tools_for(key)`` reduce to. This module does that
on the AST: it follows constants across imports, ``self.x`` assignments, small helper functions and
methods (inlined with their arguments), ``dict[key]`` / ``.get()`` lookups, and configuration files
read with yaml / json / toml (paths built from ``__file__``, ``Path``, ``os.path``). Everything it
cannot see stays an unreduced expression; callers treat that as unknown, never as a value.
"""

from __future__ import annotations

import ast
import copy
import json
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agenticreq.ast.schema import _name, literal
from agenticreq.ast.symbols import FileCtx, FuncDef, SymbolTable

UNK: Any = object()  # "not known statically"
#: stands for a parameter no call binds: it must never resolve to a same-named variable elsewhere in the file
_UNBOUND = "__agenticreq_unbound__"
_MAX_DEPTH = 10
_MAX_ACTIVE = 40  # reduce() calls nested through any path at once
_MAX_STEPS = 400  # reduce() calls one top-level evaluation may make: static analysis must stay fast
_MAX_FILE_BYTES = 1_000_000
_MAX_INLINE_STMTS = 25  # a helper longer than this is not a config lookup: it is not inlined
_LOADERS = {"safe_load", "load", "full_load", "unsafe_load", "loads", "safe_load_all"}
_LOADER_MODULES = {"yaml", "json", "tomllib", "toml", "tomli", "ruamel", "orjson", "ujson"}
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml")
#: CrewAI @CrewBase reads these class attributes as paths relative to the class's file
_CREW_CONFIG_ATTRS = {"agents_config": "config/agents.yaml", "tasks_config": "config/tasks.yaml"}
_COMPARE: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


@dataclass(frozen=True)
class Frame:
    """Where an expression is evaluated: its file and, for ``self.x``, its class."""

    fc: FileCtx
    cls: ast.ClassDef | None = None


def to_ast(value: Any) -> ast.expr:
    """A plain value (from a config file) as a literal AST node."""
    if isinstance(value, dict):
        return ast.Dict(keys=[to_ast(k) for k in value], values=[to_ast(v) for v in value.values()])
    if isinstance(value, (list, tuple)):
        return ast.List(elts=[to_ast(v) for v in value], ctx=ast.Load())
    if value is None or isinstance(value, (str, int, float, bool)):
        return ast.Constant(value=value)
    return ast.Constant(value=str(value))  # dates and other scalars yaml produces


class _Subst(ast.NodeTransformer):
    def __init__(self, env: dict[str, ast.expr]) -> None:
        self.env = env

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.env and isinstance(node.ctx, ast.Load):
            return ast.copy_location(copy.deepcopy(self.env[node.id]), node)
        return node


def subst(node: ast.AST, env: dict[str, ast.expr]) -> Any:
    """``node`` with every free use of an ``env`` name replaced by its value (a deep copy; the node
    itself when no ``env`` name occurs in it, since nothing here mutates a tree)."""
    if not env or not any(isinstance(n, ast.Name) and n.id in env for n in ast.walk(node)):
        return node
    return _Subst(env).visit(copy.deepcopy(node))


def bind_target(target: ast.AST, value: ast.expr) -> dict[str, ast.expr] | None:
    """``for <target> in ...`` bindings for one element; None when the shapes do not match."""
    if isinstance(target, ast.Name):
        return {target.id: value}
    if isinstance(target, (ast.Tuple, ast.List)):
        if not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) != len(target.elts):
            return None
        out: dict[str, ast.expr] = {}
        for t, v in zip(target.elts, value.elts):
            b = bind_target(t, v)
            if b is None:
                return None
            out.update(b)
        return out
    return None


class Evaluator:
    def __init__(
        self, symbols: SymbolTable, base: Path, functions: Callable[[str], list[tuple[FileCtx, FuncDef]]]
    ) -> None:
        self.symbols = symbols
        self.base = base.resolve()
        self.functions = functions  # name -> project definitions (for methods called on self)
        self._files: dict[Path, ast.expr | None] = {}
        self._parents: dict[str, dict[int, ast.AST]] = {}
        #: nesting of reduce() across every path (value / path / loaders restart their own depth count,
        #: so a self-referential constant like `cfg = json.load(cfg)` would otherwise never stop)
        self._active = 0
        self._returns: dict[int, tuple[FuncDef, ast.expr | None]] = {}  # id(fn) -> its return template
        self._steps = 0
        #: config files read since the caller last cleared it, for evidence ("loop over config/agents.yaml")
        self.touched: set[str] = set()
        #: config files that could not be read: (file, reason). The caller reports them (AS-902), never drops them.
        self.errors: list[tuple[str, str]] = []

    # ------------------------------------------------------------ structure
    def parents(self, fc: FileCtx) -> dict[int, ast.AST]:
        if fc.rel not in self._parents:
            self._parents[fc.rel] = {id(c): p for p in ast.walk(fc.tree) for c in ast.iter_child_nodes(p)}
        return self._parents[fc.rel]

    def enclosing(self, fc: FileCtx, node: ast.AST, kinds: tuple[type, ...]) -> ast.AST | None:
        par = self.parents(fc)
        if id(node) not in par and node is not fc.tree:
            return self._enclosing_by_line(fc, node, kinds)
        cur = par.get(id(node))
        while cur is not None and not isinstance(cur, kinds):
            cur = par.get(id(cur))
        return cur if isinstance(cur, ast.AST) else None

    @staticmethod
    def _enclosing_by_line(fc: FileCtx, node: ast.AST, kinds: tuple[type, ...]) -> ast.AST | None:
        """For a copy of a node (an inlined method body is substituted into a deep copy, so it is not in the
        tree): the innermost ``kinds`` definition whose lines contain it. Copies keep their line numbers."""
        line = getattr(node, "lineno", None)
        if line is None:
            return None
        best: tuple[int, ast.AST] | None = None
        for n in ast.walk(fc.tree):
            start = getattr(n, "lineno", 0)
            if (
                isinstance(n, kinds)
                and start <= line <= (getattr(n, "end_lineno", None) or start)
                and (best is None or start >= best[0])
            ):
                best = (start, n)
        return best[1] if best else None

    def frame_of(self, fc: FileCtx, node: ast.AST) -> Frame:
        fn = self.enclosing(fc, node, (ast.FunctionDef, ast.AsyncFunctionDef))
        cls = self.enclosing(fc, fn, (ast.ClassDef,)) if fn is not None else self.enclosing(fc, node, (ast.ClassDef,))
        return Frame(fc, cls if isinstance(cls, ast.ClassDef) else None)

    # ------------------------------------------------------------ values
    def _row_field(self, fr: Frame, node: ast.Attribute, d: int) -> tuple[Frame, ast.expr] | None:
        """``X(...).attr`` for a project dataclass / NamedTuple ``X``: the keyword, or the positional argument in the
        order the class declares its fields."""
        if isinstance(node.value, ast.Dict) and d <= 20:  # a row read from a file, or an enum member
            hit = self._index(node.value, node.attr)
            return self.reduce(fr, hit, d + 1) if hit is not None else None
        if not isinstance(node.value, ast.Call) or d > 20:
            return None
        call = node.value
        for k in call.keywords:
            if k.arg == node.attr:
                return self.reduce(fr, k.value, d + 1)
        r = self.symbols.resolve(fr.fc, call.func)
        if r is None or not isinstance(r.node, ast.ClassDef):
            return None
        fields = [s.target.id for s in r.node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        if node.attr in fields and fields.index(node.attr) < len(call.args):
            return self.reduce(fr, call.args[fields.index(node.attr)], d + 1)
        return None

    def value(self, fr: Frame, node: ast.AST | None) -> Any:
        """The Python value of ``node``, or UNK."""
        if node is None:
            return UNK
        rfr, r = self.reduce(fr, node)
        s = self._str(rfr, r)
        if s is not None:
            return s
        ok, v = literal(r)
        return v if ok else UNK

    def string(self, fr: Frame, node: ast.AST | None) -> str | None:
        v = self.value(fr, node)
        return v.strip() if isinstance(v, str) and v.strip() else None

    def _str(self, fr: Frame, node: ast.AST, depth: int = 0) -> str | None:
        if depth > _MAX_DEPTH:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a = self._str(*self.reduce(fr, node.left), depth + 1)
            b = self._str(*self.reduce(fr, node.right), depth + 1)
            return a + b if a is not None and b is not None else None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                inner = v.value if isinstance(v, ast.FormattedValue) else v
                rfr, r = self.reduce(fr, inner)
                p = self._str(rfr, r, depth + 1)
                if p is None:
                    ok, lv = literal(r)
                    if not ok or isinstance(lv, (dict, list, tuple, set)):
                        return None
                    p = str(lv)
                parts.append(p)
            return "".join(parts)
        return None

    # ------------------------------------------------------------ reduction
    def reduce(self, fr: Frame, node: ast.AST, depth: int = 0) -> tuple[Frame, Any]:
        """Rewrite ``node`` as far as static knowledge allows. The result is a (frame, node) pair: a
        reduced node may come from another file (a constant, an inlined helper)."""
        if self._active == 0:
            self._steps = 0
        self._steps += 1
        if depth > _MAX_DEPTH or self._active > _MAX_ACTIVE or self._steps > _MAX_STEPS:
            return fr, node
        self._active += 1
        try:
            return self._reduce(fr, node, depth)
        finally:
            self._active -= 1

    def _reduce(self, fr: Frame, node: ast.AST, depth: int) -> tuple[Frame, Any]:
        d = depth + 1
        if isinstance(node, ast.Name):
            if node.id == _UNBOUND:
                return fr, node
            if node.id == "__file__":
                return fr, ast.Constant(value=str(fr.fc.path))
            r = self.symbols.resolve(fr.fc, node)
            if r is not None and r.kind == "const" and r.node is not None and r.node is not node:
                return self.reduce(Frame(r.fc, None), r.node, d)
            return fr, node
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls") and fr.cls is not None:
                got = self._self_attr(fr, node.attr, d)
                return got if got is not None else (fr, node)
            r = self.symbols.resolve(fr.fc, node)
            if r is not None and r.kind == "const" and r.node is not None:
                return self.reduce(Frame(r.fc, None), r.node, d)
            field = self._row_field(fr, node, d)  # spec.name where spec is a row: AgentSpec("web_researcher", ...)
            return field if field is not None else (fr, node)
        if isinstance(node, ast.Subscript):
            cfr, coll = self.reduce(fr, node.value, d)
            key = self.value(fr, node.slice)
            if key is UNK:
                return fr, node
            picked = self._index(coll, key)
            return self.reduce(cfr, picked, d) if picked is not None else (fr, node)
        if isinstance(node, ast.Call):
            return self._reduce_call(fr, node, d)
        if isinstance(node, ast.IfExp):  # `A if DEBUG else B` with a static condition
            cond = self.value(fr, node.test)
            return (fr, node) if cond is UNK else self.reduce(fr, node.body if cond else node.orelse, d)
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.UnaryOp)):  # `key == "researcher"`, `name in (...)`
            v = self._truth(fr, node)
            return (fr, node) if v is UNK else (fr, ast.Constant(value=v))
        return fr, node

    def _truth(self, fr: Frame, node: ast.AST) -> Any:
        """A comparison / boolean expression over statically known values, or UNK."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            v = self.value(fr, node.operand)
            return UNK if v is UNK else not v
        if isinstance(node, ast.BoolOp):
            vals = [self.value(fr, x) for x in node.values]
            if isinstance(node.op, ast.And):
                if any(v is not UNK and not v for v in vals):
                    return False
                return UNK if any(v is UNK for v in vals) else True
            if any(v is not UNK and v for v in vals):
                return True
            return UNK if any(v is UNK for v in vals) else False
        if not isinstance(node, ast.Compare):
            return UNK
        left = self.value(fr, node.left)
        if left is UNK:
            return UNK
        for op, comp in zip(node.ops, node.comparators):
            right = self.value(fr, comp)
            if right is UNK or isinstance(op, (ast.Is, ast.IsNot)) and right not in (None, True, False):
                return UNK
            try:
                ok = bool(_COMPARE[type(op)](left, right))
            except (KeyError, TypeError):
                return UNK
            if not ok:
                return False
            left = right
        return True

    @staticmethod
    def _index(coll: Any, key: Any) -> ast.expr | None:
        if isinstance(coll, ast.Dict):
            for k, v in zip(coll.keys, coll.values):
                if k is not None and literal(k) == (True, key):
                    return v
            return None
        if isinstance(coll, (ast.List, ast.Tuple)) and isinstance(key, int) and not isinstance(key, bool):
            try:
                return coll.elts[key]
            except IndexError:
                return None
        return None

    def _reduce_call(self, fr: Frame, node: ast.Call, depth: int) -> tuple[Frame, Any]:
        func = node.func
        fname = _name(func)
        if isinstance(func, ast.Attribute) and fname in ("get", "pop", "setdefault") and node.args:
            cfr, coll = self.reduce(fr, func.value, depth)
            key = self.value(fr, node.args[0])
            if isinstance(coll, ast.Dict) and key is not UNK:
                picked = self._index(coll, key)
                if picked is not None:
                    return self.reduce(cfr, picked, depth)
                return (fr, node.args[1]) if len(node.args) > 1 else (fr, ast.Constant(value=None))
            return fr, node
        if fname in ("str", "dict", "list", "tuple") and len(node.args) == 1 and not node.keywords:
            rfr, r = self.reduce(fr, node.args[0], depth)
            if fname == "str" and isinstance(r, ast.Constant):
                return rfr, ast.Constant(value=str(r.value))
            if isinstance(r, (ast.Dict, ast.List, ast.Tuple, ast.Constant)):
                return rfr, r
            return fr, node
        # a project function / method: inline its return value
        target = self._resolve_function(fr, node)
        if target is None and fname in _LOADERS:
            loaded = self._load_call(fr, node)
            return (fr, loaded) if loaded is not None else (fr, node)
        if target is not None:
            tfr, fn, bound_self = target
            got = self._inline(fr, tfr, fn, node, bound_self, depth)
            if got is not None:
                return got
        return fr, node

    # ------------------------------------------------------------ functions
    def _resolve_function(self, fr: Frame, call: ast.Call) -> tuple[Frame, FuncDef, bool] | None:
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("self", "cls")
            and fr.cls is not None
        ):
            m = self._method(fr.fc, fr.cls, func.attr)
            return (Frame(m[0], m[1]), m[2], True) if m is not None else None
        r = self.symbols.resolve(fr.fc, func)
        if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return Frame(r.fc, None), r.node, False
        return None

    def _method(
        self, fc: FileCtx, cls: ast.ClassDef, name: str, depth: int = 0
    ) -> tuple[FileCtx, ast.ClassDef, FuncDef] | None:
        for st in cls.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == name:
                return fc, cls, st
        if depth > 4:
            return None
        for b in cls.bases:  # project base classes
            r = self.symbols.resolve(fc, b)
            if r is not None and r.kind == "class" and isinstance(r.node, ast.ClassDef):
                found = self._method(r.fc, r.node, name, depth + 1)
                if found is not None:
                    return found
        return None

    def call_env(self, caller: Frame, fn: FuncDef, call: ast.Call, bound_self: bool) -> dict[str, ast.expr]:
        """Parameters of ``fn`` bound to the call's arguments (reduced in the caller's frame) and defaults."""
        params = [*fn.args.posonlyargs, *fn.args.args]
        if bound_self and params and params[0].arg in ("self", "cls"):
            params = params[1:]
        env: dict[str, ast.expr] = {}
        defaults = fn.args.defaults
        for p, dflt in zip(params[len(params) - len(defaults) :], defaults):
            env[p.arg] = dflt
        for p, kw_dflt in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if kw_dflt is not None:
                env[p.arg] = kw_dflt
        names = {p.arg for p in params} | {p.arg for p in fn.args.kwonlyargs}
        for n in names | {a.arg for a in (fn.args.vararg, fn.args.kwarg) if a is not None}:
            env.setdefault(n, ast.Name(id=_UNBOUND, ctx=ast.Load()))
        for p, a in zip(params, call.args):
            if isinstance(a, ast.Starred):
                break
            env[p.arg] = self._portable(caller, a)
        for k in call.keywords:
            if k.arg in names:
                env[k.arg] = self._portable(caller, k.value)
        return env

    def _portable(self, fr: Frame, node: ast.expr) -> ast.expr:
        """An argument as it can be read inside another function: its literal value when it has one.
        Calls are left as written (reduced later, only if the inlined body actually needs them)."""
        if any(
            isinstance(n, (ast.Call, ast.Lambda, ast.ListComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(node)
        ):
            return node
        v = self.value(fr, node)
        if v is not UNK:
            return to_ast(v)
        rfr, r = self.reduce(fr, node)
        return r if isinstance(r, (ast.Dict, ast.List, ast.Tuple)) and rfr.fc is fr.fc else node

    def _inline(
        self, caller: Frame, tfr: Frame, fn: FuncDef, call: ast.Call, bound_self: bool, depth: int
    ) -> tuple[Frame, Any] | None:
        cached = self._returns.get(id(fn))
        if cached is None or cached[0] is not fn:
            short = sum(1 for _ in ast.walk(fn) if isinstance(_, ast.stmt)) <= _MAX_INLINE_STMTS
            cached = (fn, self.return_expr(fn, {}) if short else None)  # locals substituted once
            self._returns[id(fn)] = cached
        if cached[1] is None:
            return None
        env = self.call_env(caller, fn, call, bound_self)
        return self.reduce(tfr, subst(cached[1], env), depth)

    @staticmethod
    def return_expr(fn: FuncDef, env: dict[str, ast.expr]) -> ast.expr | None:
        """The value ``fn`` returns, with its parameters and straight-line locals substituted:
        ``x = ...``, ``with open(p) as fh:`` and ``return``. A branch that returns (``if`` / ``elif`` /
        ``else``, ``match`` / ``case``) becomes ``A if <test> else <rest>``, which reduce() decides once the
        test is known (``key == "researcher"`` after the caller's argument is bound)."""

        def walk(stmts: list[ast.stmt], env: dict[str, ast.expr]) -> ast.expr | None:
            env = dict(env)
            for i, st in enumerate(stmts):
                if isinstance(st, ast.Return):
                    return subst(st.value, env) if st.value is not None else ast.Constant(value=None)
                if isinstance(st, ast.Assign) and len(st.targets) == 1:
                    b = bind_target(st.targets[0], subst(st.value, env))
                    if b is not None:
                        env.update(b)
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                    env[st.target.id] = subst(st.value, env)
                elif isinstance(st, (ast.With, ast.AsyncWith)):
                    for item in st.items:
                        if isinstance(item.optional_vars, ast.Name):
                            env[item.optional_vars.id] = subst(item.context_expr, env)
                    got = walk(st.body, env)
                    if got is not None:
                        return got
                elif isinstance(st, ast.If):
                    body = walk(st.body, env)
                    if body is None:
                        continue  # a branch that does not return: read on, as before
                    rest = walk(st.orelse + stmts[i + 1 :], env)
                    return ast.IfExp(
                        test=subst(st.test, env),
                        body=body,
                        orelse=rest if rest is not None else ast.Constant(value=None),
                    )
                elif isinstance(st, ast.Match):
                    return Evaluator._match_expr(st, stmts[i + 1 :], env, walk)
                elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                    continue  # docstring
            return None

        return walk(fn.body, env)

    @staticmethod
    def _match_expr(
        st: ast.Match,
        rest: list[ast.stmt],
        env: dict[str, ast.expr],
        walk: Callable[[list[ast.stmt], dict[str, ast.expr]], ast.expr | None],
    ) -> ast.expr | None:
        """``match key: case "a": return A ...`` as ``A if key == "a" else ...``. A pattern other than a
        literal, ``a | b`` or ``_`` gets an unknown test, so its branch is possible, never assumed."""
        subj = subst(st.subject, env)
        unknown = ast.Name(id=_UNBOUND, ctx=ast.Load())

        def test(p: ast.pattern) -> ast.expr | None:  # None: always matches
            if isinstance(p, ast.MatchAs) and p.pattern is None:
                return None
            if isinstance(p, ast.MatchValue):
                return ast.Compare(left=subj, ops=[ast.Eq()], comparators=[p.value])
            if isinstance(p, ast.MatchSingleton):
                return ast.Compare(left=subj, ops=[ast.Is()], comparators=[ast.Constant(value=p.value)])
            if isinstance(p, ast.MatchOr):
                parts = [test(x) for x in p.patterns]
                return (
                    None
                    if any(x is None for x in parts)
                    else ast.BoolOp(op=ast.Or(), values=[x for x in parts if x is not None])
                )
            return unknown

        out: ast.expr | None = walk(rest, env)
        for case in reversed(st.cases):
            body = walk(case.body, env)
            if body is None:
                return None  # a case that does not return: the function's value is not one expression
            t = test(case.pattern)
            if case.guard is not None:
                t = unknown
            out = body if t is None else ast.IfExp(test=t, body=body, orelse=out or ast.Constant(value=None))
        return out

    # ------------------------------------------------------------ self.x
    def _self_attr(self, fr: Frame, attr: str, depth: int) -> tuple[Frame, Any] | None:
        cls = fr.cls
        assert cls is not None
        found: ast.expr | None = None
        owner: FuncDef | None = None
        for st in cls.body:  # class-level `agents_config = "config/agents.yaml"`
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in st.targets):
                found = st.value
            elif (
                isinstance(st, ast.AnnAssign)
                and isinstance(st.target, ast.Name)
                and st.target.id == attr
                and st.value is not None
            ):
                found = st.value
        methods = sorted(
            (m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))),
            key=lambda m: m.name != "__init__",
        )
        for m in methods:
            if found is not None:
                break
            for n in ast.walk(m):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
                if (
                    any(
                        isinstance(t, ast.Attribute)
                        and t.attr == attr
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        for t in tgts
                    )
                    and getattr(n, "value", None) is not None
                ):
                    found, owner = getattr(n, "value"), m
                    break
        crew = any(_name(d.func if isinstance(d, ast.Call) else d) == "CrewBase" for d in cls.decorator_list)
        if found is None:
            if crew and attr in _CREW_CONFIG_ATTRS:
                found = ast.Constant(value=_CREW_CONFIG_ATTRS[attr])
            else:
                return None
        if owner is not None:  # `self.cfg = _load(name)` with locals of __init__ before it
            pre = self._locals_before(owner, found)
            found = subst(found, pre)
            if found is None:
                return None
            params = {a.arg for a in [*owner.args.posonlyargs, *owner.args.args, *owner.args.kwonlyargs]}
            if any(isinstance(n, ast.Name) and n.id in params - {"self", "cls"} for n in ast.walk(found)):
                return None  # `self.config = config`: whatever the caller passed, not known here
        rfr, r = self.reduce(fr, found, depth)
        if crew and attr in _CREW_CONFIG_ATTRS:
            s = self._str(rfr, r)
            if s is not None and s.endswith(_CONFIG_SUFFIXES):
                data = self._read(fr.fc, Path(s))
                if data is not None:
                    return fr, data
        return rfr, r

    @staticmethod
    def _locals_before(fn: FuncDef, stop: ast.AST) -> dict[str, ast.expr]:
        env: dict[str, ast.expr] = {}
        for st in fn.body:
            if any(n is stop for n in ast.walk(st)):
                break
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                env[st.targets[0].id] = subst(st.value, env)
        return env

    # ------------------------------------------------------------ config files
    def _load_call(self, fr: Frame, call: ast.Call) -> ast.expr | None:
        """yaml.safe_load(open(p)) / json.load(fh) / json.loads(Path(p).read_text()) / tomllib.load(...)"""
        func = call.func
        owner = _name(func.value) if isinstance(func, ast.Attribute) else ""
        if isinstance(func, ast.Attribute) and owner not in _LOADER_MODULES:
            return None
        if isinstance(func, ast.Name) and not fr.fc.imports & _LOADER_MODULES:
            return None  # a bare `load(...)` counts only in a file that imports yaml / json / toml
        if not call.args:
            return None
        path = self._source_path(fr, call.args[0])
        return self._read(fr.fc, path) if path is not None else None

    def _source_path(self, fr: Frame, node: ast.AST, depth: int = 0) -> Path | None:
        """The file an ``open(...)`` / ``.read_text()`` / file-handle expression reads."""
        if depth > _MAX_DEPTH:
            return None
        rfr, node = self.reduce(fr, node)
        if isinstance(node, ast.Name):  # `with open(p) as f: CFG = json.load(f)`
            handle = self._with_target(rfr.fc, node)
            return self._source_path(rfr, handle, depth + 1) if handle is not None else None
        if isinstance(node, ast.Call):
            fname = _name(node.func)
            if fname == "open" and isinstance(node.func, ast.Name) and node.args:
                return self.path(rfr, node.args[0])
            if fname in ("read_text", "read_bytes", "open", "read") and isinstance(node.func, ast.Attribute):
                inner = node.func.value
                if isinstance(inner, ast.Call) and _name(inner.func) == "open" and isinstance(inner.func, ast.Name):
                    return self._source_path(rfr, inner, depth + 1)
                return self.path(rfr, inner)
        return None

    @staticmethod
    def _with_target(fc: FileCtx, name: ast.Name) -> ast.expr | None:
        """The expression a ``with ... as <name>`` binds, nearest above the use."""
        line = getattr(name, "lineno", 10**9)
        best: tuple[int, ast.expr] | None = None
        for n in ast.walk(fc.tree):
            if isinstance(n, (ast.With, ast.AsyncWith)) and n.lineno <= line:
                for item in n.items:
                    if (
                        isinstance(item.optional_vars, ast.Name)
                        and item.optional_vars.id == name.id
                        and (best is None or n.lineno > best[0])
                    ):
                        best = (n.lineno, item.context_expr)
        return best[1] if best else None

    def path(self, fr: Frame, node: ast.AST, depth: int = 0) -> Path | None:
        """A filesystem path built statically: ``Path(__file__).parent / "config" / name``,
        ``os.path.join(os.path.dirname(__file__), "x.yaml")``, constants and f-strings of those."""
        if depth > _MAX_DEPTH:
            return None
        d = depth + 1
        if isinstance(node, ast.Name) and node.id == "__file__":
            return fr.fc.path
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            p = self.path(fr, node.value, d)
            return p.parent if p is not None else None
        if isinstance(node, ast.Attribute) and node.attr in ("parents",):
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            a, b = self.path(fr, node.left, d), self.path(fr, node.right, d)
            return a / b if a is not None and b is not None else None
        if isinstance(node, ast.Call):
            fname = _name(node.func)
            if fname in ("Path", "PurePath", "PosixPath") and node.args:
                parts = [self.path(fr, a, d) for a in node.args]
                if any(p is None for p in parts):
                    return None
                out = parts[0]
                for p in parts[1:]:
                    out = out / p  # type: ignore[operator]
                return out
            if fname in ("resolve", "absolute", "expanduser") and isinstance(node.func, ast.Attribute):
                return self.path(fr, node.func.value, d)
            if fname == "join" and isinstance(node.func, ast.Attribute) and _name(node.func.value) == "path":
                parts = [self.path(fr, a, d) for a in node.args]
                if not parts or any(p is None for p in parts):
                    return None
                return Path(os.path.join(*[str(p) for p in parts]))
            if fname in ("dirname",) and node.args:
                p = self.path(fr, node.args[0], d)
                return p.parent if p is not None else None
            if fname in ("abspath", "realpath", "normpath", "fspath", "str") and len(node.args) == 1:
                return self.path(fr, node.args[0], d)
            if fname == "joinpath" and isinstance(node.func, ast.Attribute):
                base = self.path(fr, node.func.value, d)
                parts = [self.path(fr, a, d) for a in node.args]
                if base is None or any(p is None for p in parts):
                    return None
                for p in parts:
                    base = base / p  # type: ignore[operator]
                return base
        rfr, r = self.reduce(fr, node, d)
        if r is not node:
            return self.path(rfr, r, d)
        s = self._str(fr, node)
        return Path(s) if s is not None else None

    def _read(self, fc: FileCtx, p: Path) -> ast.expr | None:
        """A config file's content as a literal AST. Relative paths are tried against the file that
        reads it, then the project root (the usual working directory). Only files inside the project."""
        cands = [p] if p.is_absolute() else [fc.path.parent / p, self.base / p]
        for c in cands:
            try:
                c = c.resolve()
            except OSError:
                continue
            if c in self._files:
                if self._files[c] is not None:
                    self.touched.add(self._rel(c))
                return self._files[c]
            if not c.is_file() or c.suffix not in _CONFIG_SUFFIXES or self.base not in c.parents:
                continue
            data = None
            try:
                if c.stat().st_size <= _MAX_FILE_BYTES:
                    text = c.read_text(encoding="utf-8-sig")
                    if c.suffix == ".json":
                        data = json.loads(text)
                    elif c.suffix == ".toml":
                        try:
                            import tomllib  # type: ignore[import-not-found,import-untyped,unused-ignore]
                        except ImportError:  # Python 3.10 has no tomllib; `tomli` is the same parser
                            import tomli as tomllib  # type: ignore[no-redef,import-not-found,import-untyped,unused-ignore]

                        data = tomllib.loads(text)
                    else:
                        import yaml

                        data = yaml.safe_load(text)
            except Exception as e:  # noqa: BLE001  (a broken config file is unknown, not a crash, but it is reported)
                data = None
                self.errors.append((self._rel(c), f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:120]}"))
            node = to_ast(data) if data is not None else None
            self._files[c] = node
            if node is not None:
                self.touched.add(self._rel(c))
            return node
        return None

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.base))
        except ValueError:
            return str(p)

    # ------------------------------------------------------------ iteration
    def _enum_members(self, fr: Frame, it: ast.AST) -> list[ast.expr] | None:
        """``for r in Role``: one row per member, with its ``.name`` and ``.value``."""
        if not isinstance(it, (ast.Name, ast.Attribute)):
            return None
        r = self.symbols.resolve(fr.fc, it)
        if r is None or not isinstance(r.node, ast.ClassDef):
            return None
        if not any("Enum" in (_name(b) or "") for b in r.node.bases):
            return None
        return [
            ast.Dict(
                keys=[ast.Constant(value="name"), ast.Constant(value="value")],
                values=[ast.Constant(value=s.targets[0].id), s.value],
            )
            for s in r.node.body
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
        ]

    def rows(self, fr: Frame, it: ast.AST, depth: int = 0) -> list[ast.expr] | None:
        """The elements ``for x in it`` visits, when ``it`` is known statically; else None."""
        if depth > _MAX_DEPTH:
            return None
        d = depth + 1
        if isinstance(it, ast.Call):
            fname = _name(it.func)
            if isinstance(it.func, ast.Attribute) and fname in ("items", "keys", "values") and not it.args:
                _, coll = self.reduce(fr, it.func.value, d)
                if not isinstance(coll, ast.Dict):
                    return None
                pairs = [(k, v) for k, v in zip(coll.keys, coll.values) if k is not None]
                if fname == "items":
                    return [ast.Tuple(elts=[k, v], ctx=ast.Load()) for k, v in pairs]
                return [v for _, v in pairs] if fname == "values" else [k for k, _ in pairs]
            if fname == "enumerate" and it.args:
                inner = self.rows(fr, it.args[0], d)
                start = self.value(fr, it.args[1]) if len(it.args) > 1 else 0
                if inner is None or not isinstance(start, int):
                    return None
                return [ast.Tuple(elts=[ast.Constant(value=i + start), e], ctx=ast.Load()) for i, e in enumerate(inner)]
            if fname == "zip" and it.args:
                cols = [self.rows(fr, a, d) for a in it.args]
                if any(c is None for c in cols):
                    return None
                return [ast.Tuple(elts=list(t), ctx=ast.Load()) for t in zip(*cols)]
            if fname in ("list", "tuple", "sorted", "reversed", "set") and len(it.args) == 1:
                inner = self.rows(fr, it.args[0], d)
                if inner is None:
                    return None
                if fname == "reversed":
                    return inner[::-1]
                if fname == "sorted":
                    vals = [literal(e) for e in inner]
                    if all(ok for ok, _ in vals):
                        try:
                            order = sorted(range(len(inner)), key=lambda i: vals[i][1])
                            return [inner[i] for i in order]
                        except TypeError:
                            return inner
                return inner
            if fname == "range":
                args = [self.value(fr, a) for a in it.args]
                if args and all(isinstance(a, int) for a in args) and len(args) <= 3:
                    rng = range(*args)
                    return [ast.Constant(value=i) for i in rng] if len(rng) <= 200 else None
                return None
        members = self._enum_members(fr, it)
        if members is not None:
            return members
        rfr, coll = self.reduce(fr, it, d)
        if isinstance(coll, ast.Dict):
            return [k for k in coll.keys if k is not None]
        if isinstance(coll, (ast.List, ast.Tuple, ast.Set)):
            return (
                [e for e in coll.elts if not isinstance(e, ast.Starred)]
                if not any(isinstance(e, ast.Starred) for e in coll.elts)
                else None
            )
        if coll is not it and isinstance(coll, ast.Call):
            return self.rows(rfr, coll, d)
        return None
