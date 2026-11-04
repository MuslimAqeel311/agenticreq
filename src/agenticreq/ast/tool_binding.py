"""Which tools are bound to which agent: tool lists, loops, comprehensions, handoffs, clones.

Moved out of ``extractor.py`` unchanged; each function takes the extractor as ``ex``.
"""

from __future__ import annotations

import ast
import re
from agenticreq.ast.classify import is_pinned
from agenticreq.ast.classify import looks_like_model
from agenticreq.ast.evaluate import bind_target
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.ast.symbols import FileCtx
from agenticreq.ast.symbols import FuncDef
from agenticreq.models.observed import BoundRef
from agenticreq.models.observed import DeclaredAgent
from agenticreq.models.observed import GateRef
from agenticreq.models.observed import ModelRef
from agenticreq.models.observed import SecretRef
from agenticreq.models.observed import UnresolvedTool
from agenticreq.ast.extract_support import (
    BIND_CALLS,
    BOUND_KWARGS,
    DYNAMIC_TOOL_CALLS,
    DYNAMIC_TOOL_KWARGS,
    EXCLUDE_KEY,
    GATE_DECORATORS,
    MCP_SERVER_SIDE,
    TOOL_LIST_KWARGS,
    _LIVE_SECRET,
    _MAX_DEPTH,
    _PLACEHOLDER,
    _SECRET_NAME,
    _appends_to,
    _kw,
    _ref,
    _str,
    _subst_all,
)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor


def _tools_from_expr(
    ex: ProjectExtractor,
    fc: FileCtx,
    node: ast.AST | None,
    target: str | None = None,
    *,
    in_tool_list: bool = False,
    depth: int = 0,
) -> list[str]:
    """Tool names an expression denotes, registering tools it creates on the way.

    ``in_tool_list``: the expression sits where a framework expects tools (``tools=[...]``,
    ``bind_tools(...)``), so a bare function there *is* a tool (pydantic-ai, ADK, Agno, AutoGen ...).
    """
    if node is None or depth > _MAX_DEPTH:
        return []

    def rec(n: ast.AST | None, f: FileCtx = fc) -> list[str]:
        return ex._tools_from_expr(f, n, in_tool_list=in_tool_list, depth=depth + 1)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [t for el in node.elts for t in rec(el.value if isinstance(el, ast.Starred) else el)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return rec(node.left) + rec(node.right)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)) and len(node.generators) == 1:
        gen = node.generators[0]
        bindings = ex._loop_bindings(fc, gen.target, gen.iter, depth + 1)
        if bindings is None:  # a config file / helper / self.x: `[TOOLS[n] for n in cfg["tools"]]`
            rows = ex.ev.rows(ex.ev.frame_of(fc, node), gen.iter)
            if rows is None:  # `os.environ.get("TOOLS", "a,b").split(",")`: what runs unless the deployment overrides
                rows = _env_split_rows(ex, fc, gen.iter)
            envs = [bind_target(gen.target, r) for r in rows] if rows is not None else None
            if envs is not None and all(e is not None for e in envs):
                bindings = [(fc, e) for e in envs if e is not None]
        if bindings is None:
            ex._mark_unresolved_template(fc, node.elt, "comprehension over a value that is not a literal")
            return []
        ex._mark_unrolled(fc, node.elt)
        # each element is resolved in the file it was written in (TOOLS may live in another module); an `if` filter
        # is evaluated per element: False drops it, "cannot tell" keeps it but records that it is only conditional
        kept: list[str] = []
        for el_fc, env in bindings:
            verdict = _all_conditions(ex, el_fc, gen.ifs, env) if gen.ifs else True
            if verdict is False:
                continue
            found = rec(_subst_all(node.elt, env), el_fc)
            if verdict is None:
                ex.obs.conditional_tools += [t for t in found if t not in ex.obs.conditional_tools]
            kept += found
        return kept
    if isinstance(node, ast.IfExp):  # a branch of `_tools_for(key)`: decided when the test is known
        rfr, red = ex.ev.reduce(ex.ev.frame_of(fc, node), node)
        if red is not node:
            return rec(red, rfr.fc)
        both = rec(node.body) + rec(node.orelse)  # undecided: either side may be the agent's, never dropped
        ex.obs.conditional_tools += [t for t in both if t not in ex.obs.conditional_tools]
        return list(dict.fromkeys(both))
    if isinstance(node, ast.Call):
        if _name(node.func) in ("list", "tuple") and node.args:
            return rec(node.args[0])
        if isinstance(node.func, (ast.Subscript, ast.Call)):  # `REGISTRY[name]()`, `REGISTRY.get(name)()`
            rfr, rfunc = ex.ev.reduce(ex.ev.frame_of(fc, node), node.func)
            if rfunc is not node.func and isinstance(rfunc, (ast.Name, ast.Attribute)):
                return rec(
                    ast.copy_location(ast.Call(func=rfunc, args=node.args, keywords=node.keywords), node), rfr.fc
                )
        if ex._wrapper_parts(node) is not None:
            return ex._wrapper_tool(fc, node, target)
        if as_tool := ex._agent_as_tool(fc, node, target):
            return as_tool
        r = ex.symbols.resolve(fc, node.func)  # MyTool() instance of a tool class
        if r is not None and r.kind == "class" and isinstance(r.node, ast.ClassDef):
            name = ex._class_tool(r.fc, r.node)
            if name:
                return [ex._named_instance(fc, node, name) or name]
            # a plugin class (Semantic Kernel): every method decorated as a tool is one of the agent's tools
            return [
                ex.node_tool[id(m)]
                for m in r.node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(m) in ex.node_tool
            ]
        if r is None and ex._is_library_tool_class(fc, node.func):
            return [ex._library_tool(fc, node, target)]
        if (
            r is not None
            and r.kind == "function"
            and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and id(r.node) not in ex.node_tool
        ):
            # a project factory: `make_search_tool(...)` returns StructuredTool.from_function(...)
            fid = id(r.node)
            if fid in ex._factory_tools:  # read once: the same factory is reached from many call sites
                found = ex._factory_tools[fid]
            elif fid in ex._factory_active:  # recursion: the outer reading already covers it
                found = []
            else:
                ex._factory_active.add(fid)
                try:
                    returns = [n for n in ast.walk(r.node) if isinstance(n, ast.Return)]
                    per_return = [
                        ex._tools_from_expr(r.fc, ret.value, depth=depth + 1) if ret.value is not None else []
                        for ret in returns
                    ]
                    found = [t for ts in per_return for t in ts]
                    if len(per_return) > 1:  # a tool only some branches return is held only on those paths
                        ex.obs.conditional_tools += [
                            t
                            for t in dict.fromkeys(found)
                            if t not in ex.obs.conditional_tools and not all(t in ts for ts in per_return)
                        ]
                    if not found:  # `built = []; for n, f in SPECS.items(): built.append(tool(n)(f)); return built`
                        found = [
                            t
                            for ret in returns
                            if isinstance(ret.value, ast.Name)
                            for t in ex._local_list_tools(r.fc, r.node, ret.value.id, depth + 1)
                        ]
                finally:
                    ex._factory_active.discard(fid)
                if depth + 1 < _MAX_DEPTH:  # a reading cut short by the depth limit is not the factory's answer
                    ex._factory_tools[fid] = found
            if found:
                return found
        if _name(node.func) in ex.var_to_tool:
            return [ex.var_to_tool[_name(node.func)]]
        # `self._tools_for(key)`: a method whose return value picks the tools (project functions were
        # followed above; a method on self needs the class, which the evaluator knows)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("self", "cls")
        ):
            rfr, red = ex.ev.reduce(ex.ev.frame_of(fc, node), node)
            return rec(red, rfr.fc) if red is not node else []
        return []
    if isinstance(node, ast.Subscript):  # TOOLS_BY_AGENT["research"], {...}[key]
        rfr, red = ex.ev.reduce(ex.ev.frame_of(fc, node), node)
        return rec(red, rfr.fc) if red is not node else []
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and in_tool_list:
        # a tool named by a string (config files list tool names): only names the project declares
        name = node.value.rsplit("__", 1)[-1] if node.value.startswith("mcp__") else node.value  # Claude Agent SDK
        t = ex.var_to_tool.get(name, name)
        if t not in ex.obs.declared_tools and (builtin := ex._builtin_tool(fc, node)):
            return [builtin]
        return [t] if t in ex.obs.declared_tools else []
    if isinstance(node, ast.Dict):
        ok, d = literal(node)
        if ok and isinstance(d, dict):
            n = d.get("name") or (d.get("function") or {}).get("name")
            return [n] if isinstance(n, str) else []
        return []
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = node.id if isinstance(node, ast.Name) else node.attr
        if isinstance(node, ast.Name) and key in ex.tool_list_vars:
            return ex.tool_list_vars[key]
        r = ex.symbols.resolve(fc, node)
        if (
            r is None
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "cls")
        ):  # `self.wiki = WikipediaTool()` set in __init__
            rfr, red = ex.ev.reduce(ex.ev.frame_of(fc, node), node)
            if red is not node:
                return rec(red, rfr.fc)
        if r is None:
            if key in ex.var_to_tool:
                return [ex.var_to_tool[key]]
            return [key] if key in ex.obs.declared_tools else []
        if r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if id(r.node) in ex.node_tool:
                return [ex.node_tool[id(r.node)]]
            if in_tool_list:
                return [ex._function_tool(r.fc, r.node, None, kind="function")]
            return []
        if r.kind == "class" and isinstance(r.node, ast.ClassDef):
            name = ex._class_tool(r.fc, r.node) or (ex._schema_class_tool(r.fc, r.node) if in_tool_list else None)
            return [name] if name else []
        if r.kind == "const":
            if r.name in ex.var_to_tool:
                return [ex.var_to_tool[r.name]]
            base = ex._tools_from_expr(r.fc, r.node, r.name, in_tool_list=in_tool_list, depth=depth + 1)
            if isinstance(r.node, (ast.List, ast.Tuple)) and r.name:  # a list grown after it was built
                for x in ast.walk(r.fc.tree):
                    if _appends_to(x, r.name):
                        base = base + ex._tools_from_expr(r.fc, x.args[0], in_tool_list=True, depth=depth + 1)
                    elif isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name) and x.target.id == r.name:
                        base = base + ex._tools_from_expr(r.fc, x.value, in_tool_list=True, depth=depth + 1)
            return list(dict.fromkeys(base))
    return []


def _env_split_rows(ex: ProjectExtractor, fc: FileCtx, it: ast.AST) -> list[ast.expr] | None:
    """Names in a comma-separated environment default: ``ENABLED = os.environ.get("T", "a,b").split(",")``."""
    if isinstance(it, ast.Name):
        r = ex.symbols.resolve(fc, it)
        if r is None or r.kind != "const" or not isinstance(r.node, ast.expr):
            return None
        fc, it = r.fc, r.node
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "split"):
        return None
    ok, sep = literal(it.args[0]) if it.args else (True, None)
    assumed = ex._env_assumed(fc, it.func.value)
    if not ok or assumed is None or not isinstance(assumed.value, str) or not (sep is None or isinstance(sep, str)):
        return None
    return [ast.Constant(value=p.strip()) for p in assumed.value.split(sep) if p.strip()]


def _loop_bindings(
    ex: ProjectExtractor, fc: FileCtx, target: ast.AST, it: ast.AST, depth: int
) -> list[tuple[FileCtx, dict[str, ast.expr]]] | None:
    """Every iteration of ``for <target> in <it>`` as {name: value node}, when ``it`` is a literal
    (list / tuple / dict, its ``.items()`` / ``.keys()`` / ``.values()``, possibly through a constant).
    None when the iterable is a runtime value."""
    names = (
        [target.id]
        if isinstance(target, ast.Name)
        else [e.id for e in target.elts if isinstance(e, ast.Name)]
        if isinstance(target, ast.Tuple)
        else []
    )
    if not names or isinstance(target, ast.Tuple) and len(names) != len(target.elts):
        return None
    method = None
    if (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Attribute)
        and it.func.attr in ("items", "keys", "values")
        and not it.args
    ):
        method, it = it.func.attr, it.func.value
    src_fc, coll = fc, it
    if isinstance(it, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, it)
        if r is None or r.kind != "const" or r.node is None:
            return None
        src_fc, coll = r.fc, r.node
    if isinstance(coll, ast.Dict):
        pairs = [(k, v) for k, v in zip(coll.keys, coll.values) if k is not None]
        if method == "items":
            rows: list[list[ast.expr]] = [[k, v] for k, v in pairs]
        elif method == "values":
            rows = [[v] for _, v in pairs]
        else:
            rows = [[k] for k, _ in pairs]
    elif isinstance(coll, (ast.List, ast.Tuple, ast.Set)) and method is None:
        rows = [list(e.elts) if len(names) > 1 and isinstance(e, ast.Tuple) else [e] for e in coll.elts]
    else:
        return None
    if len(names) == 1:
        rows = [[ast.Tuple(elts=row, ctx=ast.Load())] if len(row) > 1 else row for row in rows]
    if any(len(row) != len(names) for row in rows):
        return None
    return [(src_fc, dict(zip(names, row))) for row in rows]


def _local_list_tools(ex: ProjectExtractor, fc: FileCtx, fn: FuncDef, var: str, depth: int) -> list[str]:
    """Tools in a list a function builds and returns: ``var = [...]``, ``var.append(x)``,
    ``var.extend([...])``, ``var += [...]``, also inside ``for`` loops over a literal table."""
    cache_key = (id(fn), var)
    if cache_key in ex._local_list_cache:
        return ex._local_list_cache[cache_key]
    out: list[str] = []

    def visit(stmts: list[ast.stmt], env: dict[str, ast.expr], efc: FileCtx) -> None:
        for st in stmts:
            if isinstance(st, (ast.For, ast.AsyncFor)):
                bindings = ex._loop_bindings(fc, st.target, st.iter, depth)
                if bindings is None:
                    for x in ast.walk(st):
                        if _appends_to(x, var):
                            ex._mark_unresolved_template(fc, x, "appended in a loop over a runtime value")
                    continue
                for x in ast.walk(st):
                    if _appends_to(x, var):
                        ex._mark_unrolled(fc, x)
                for bfc, b in bindings:
                    visit(st.body, {**env, **b}, bfc)
                continue
            for x in ast.walk(st):
                added: list[ast.expr] = []
                if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in x.targets):
                    added = [x.value]
                elif isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name) and x.target.id == var:
                    added = [x.value]
                elif _appends_to(x, var):
                    added = list(x.args[:1])
                for a in added:
                    out.extend(
                        ex._tools_from_expr(
                            efc if env else fc, _subst_all(a, env) if env else a, in_tool_list=True, depth=depth + 1
                        )
                    )

    visit(fn.body, {}, fc)
    if depth < _MAX_DEPTH:
        ex._local_list_cache[cache_key] = out
    return out


def _dynamic_source(ex: ProjectExtractor, fc: FileCtx, x: ast.AST) -> str | None:
    """Why this node hands the agent tools the code does not name (MCP server, toolset, registry), else None."""
    if isinstance(x, ast.keyword) and x.arg in DYNAMIC_TOOL_KWARGS:
        if _registry_names(ex, x.value) is not None:
            return None  # a toolset the code fills itself: its tools are read from the registrations, not unknown
        return f"`{x.arg}=` supplies tools at run time"
    if isinstance(x, ast.Dict):  # a hosted MCP server: {"type": "mcp", "server_url": ...}
        for k, v in zip(x.keys, x.values):
            if (
                isinstance(k, ast.Constant)
                and k.value == "type"
                and isinstance(v, ast.Constant)
                and isinstance(v.value, str)
                and "mcp" in v.value.lower()
            ):
                return "a hosted MCP server supplies tools at run time"
        return None
    if not isinstance(x, ast.Call):
        return None
    name = _name(x.func) or ""
    low = name.lower()
    if low in MCP_SERVER_SIDE:
        return None
    if "mcp" in low:
        return f"`{name}(...)` loads tools from an MCP server"
    if low in DYNAMIC_TOOL_CALLS:
        if isinstance(x.func, ast.Attribute) or low in ("toolbox", "toolboxclient", "toolboxtoolset"):
            recv = x.func.value if isinstance(x.func, ast.Attribute) else None
            if isinstance(recv, ast.Name) and recv.id in ("self", "cls"):
                return None
            if ex.symbols.resolve(fc, x.func) is None:  # a project function is followed, not guessed at
                return f"`{name}(...)` returns tools that are decided at run time"
    return None


def _mark_dynamic_tool_sources(ex: ProjectExtractor, fc: FileCtx) -> None:
    """Record every MCP / toolset / registry construct as an unresolved tool source, so that "tool X is absent"
    can never be claimed while the agent can receive tools the code does not list."""
    for x in ast.walk(fc.tree):
        why = _dynamic_source(ex, fc, x)
        if why is None:
            continue
        anchor = x.value if isinstance(x, ast.keyword) else x
        ref = _ref(fc.rel, anchor)
        ex._unresolved.setdefault(str(ref), UnresolvedTool(code_ref=ref, reason=why, kind="dynamic"))


def _mark_unrolled(ex: ProjectExtractor, fc: FileCtx, template: ast.AST) -> None:
    for x in ast.walk(template):
        if isinstance(x, ast.Call):
            ex._unrolled.add(str(_ref(fc.rel, x)))


def _mark_unresolved_template(ex: ProjectExtractor, fc: FileCtx, template: ast.AST, why: str) -> None:
    for x in ast.walk(template):
        if isinstance(x, ast.Call) and ex._wrapper_parts(x) is not None:
            ref = _ref(fc.rel, x)
            ex._unresolved.setdefault(str(ref), UnresolvedTool(code_ref=ref, reason=f"tool built {why}"))
            return


def _literal_elements(ex: ProjectExtractor, fc: FileCtx, node: ast.AST, depth: int) -> list[tuple[FileCtx, ast.expr]]:
    """(file, element) of a literal list/tuple, possibly through a constant (``TOOLS = (a, b)``)."""
    if depth > _MAX_DEPTH:
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [(fc, e) for e in node.elts]
    if isinstance(node, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, node)
        if r is not None and r.kind == "const" and r.node is not None:
            return ex._literal_elements(r.fc, r.node, depth + 1)
    return []


#: calls that run the model: a `for _ in range(N)` around one of them bounds the agent loop
_MODEL_CALL_ATTRS = {
    "create",
    "invoke",
    "ainvoke",
    "chat",
    "generate",
    "generate_content",
    "complete",
    "run",
    "arun",
    "stream",
    "predict",
    "respond",
    "kickoff",
    "send_message",
    "responses",
}


def _is_model_literal(v: ast.AST | None) -> bool:
    return (
        isinstance(v, ast.Constant)
        and isinstance(v.value, str)
        and bool(re.search(r"\d", v.value))
        and looks_like_model(v.value)
    )


#: a value that NAMES where the secret lives instead of being the secret: "env:OPENAI_API_KEY", "${KEY}", "MY_API_KEY"
_SECRET_POINTER = re.compile(r"^(env:|[$%{<]|os[.]environ|vault:|secret:|aws:)", re.IGNORECASE)
_ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$")


def _is_literal_credential(name: str, v: ast.AST | None) -> bool:
    """A string literal, long enough, not a placeholder and not a pointer to a secret, under a secret-like name
    (api_key=, "token": ...)."""
    if not (_SECRET_NAME.search(name) and isinstance(v, ast.Constant) and isinstance(v.value, str)):
        return False
    s = v.value
    pointer = _SECRET_POINTER.match(s) or _ENV_VAR_NAME.match(s)
    return len(s) >= 16 and " " not in s and not _PLACEHOLDER.search(s) and not pointer


def _note_excluded(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> None:
    """``disallowed_tools=[...]`` / ``{"excluded_functions": [...]}``: the code takes these out of the model's choice."""
    ok, val = literal(node)
    if not ok and isinstance(node, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, node)
        if r is not None and r.kind == "const" and r.node is not None:
            ok, val = literal(r.node)
    if ok and isinstance(val, (list, tuple, set, frozenset)):
        for n in val:
            if isinstance(n, str) and n not in ex.obs.excluded_tools:
                ex.obs.excluded_tools.append(n)


def _is_retry_loop(node: ast.For) -> bool:
    """``for attempt in range(3): try: return call() except E: continue``: the loop repeats one call that failed, it
    does not take steps. The structure decides (a try that leaves the loop on success); the name only helps."""
    for t in ast.walk(node):
        if isinstance(t, ast.Try) and t.handlers:
            success = [*t.body, *t.orelse]
            if any(isinstance(x, (ast.Return, ast.Break)) for s in success for x in ast.walk(s)):
                return True
    target = node.target.id.lower() if isinstance(node.target, ast.Name) else ""
    return any(w in target for w in ("attempt", "retry", "retries"))


def _range_bound(ex: ProjectExtractor, fc: FileCtx, node: ast.For) -> float | None:
    """N of ``for _ in range(N)`` when the body calls the model, else None."""
    it = node.iter
    if not (isinstance(it, ast.Call) and _name(it.func) == "range" and it.args):
        return None
    b = ex._bound_ref(fc, "max_steps", "range", it.args[1] if len(it.args) >= 2 else it.args[0])
    if b.value is None:
        return None
    runs_model = any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in _MODEL_CALL_ATTRS
        for c in ast.walk(node)
    )
    return b.value if runs_model else None


def _while_bound(ex: ProjectExtractor, fc: FileCtx, node: ast.While) -> float | None:
    """N of ``while i < N:`` when ``i`` is incremented in the body and the body calls the model (the structure,
    not the counter's name, is the evidence)."""
    t = node.test
    if not (
        isinstance(t, ast.Compare)
        and len(t.ops) == 1
        and isinstance(t.ops[0], (ast.Lt, ast.LtE))
        and isinstance(t.left, ast.Name)
    ):
        return None
    b = ex._bound_ref(fc, "max_steps", "while", t.comparators[0])
    if b.value is None:
        return None
    counts = any(
        isinstance(n, ast.AugAssign)
        and isinstance(n.op, ast.Add)
        and isinstance(n.target, ast.Name)
        and n.target.id == t.left.id
        for n in ast.walk(node)
    )
    runs_model = any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in _MODEL_CALL_ATTRS
        for c in ast.walk(node)
    )
    return b.value if counts and runs_model else None


_UNK = object()


def _cond_value(ex: ProjectExtractor, fc: FileCtx, node: ast.AST, env: dict[str, ast.expr]) -> Any:
    """The value of one side of a comprehension filter, or _UNK. ``t.name`` of a tool is its tool name."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and node.attr in ("name", "__name__", "func_name"):
        base = env.get(node.value.id, node.value) if isinstance(node.value, ast.Name) else node.value
        names = ex._tools_from_expr(fc, base, in_tool_list=True)
        return names[0] if len(names) == 1 else _UNK
    if isinstance(node, ast.Name) and node.id in env:
        return _cond_value(ex, fc, env[node.id], {})
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        ok, val = literal(node)
        return val if ok else _UNK
    if isinstance(node, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, node)
        if r is not None and r.kind == "const" and r.node is not None:
            ok, val = literal(r.node)
            return val if ok else _UNK
    return _UNK


def _cond(ex: ProjectExtractor, fc: FileCtx, t: ast.AST, env: dict[str, ast.expr], depth: int = 0) -> bool | None:
    """Three-valued truth of a comprehension filter: True / False, or None when the scan cannot tell."""
    if depth > 6:
        return None
    if isinstance(t, ast.BoolOp):
        vals = [_cond(ex, fc, v, env, depth + 1) for v in t.values]
        if isinstance(t.op, ast.And):
            return False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
        return True if any(v is True for v in vals) else (False if all(v is False for v in vals) else None)
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        v = _cond(ex, fc, t.operand, env, depth + 1)
        return None if v is None else not v
    if isinstance(t, ast.Compare) and len(t.ops) == 1 and len(t.comparators) == 1:
        left = _cond_value(ex, fc, t.left, env)
        right = _cond_value(ex, fc, t.comparators[0], env)
        if left is _UNK or right is _UNK:
            return None
        op = t.ops[0]
        try:
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
            if isinstance(op, ast.Eq):
                return bool(left == right)
            if isinstance(op, ast.NotEq):
                return bool(left != right)
        except TypeError:
            return None
    return None


def _all_conditions(ex: ProjectExtractor, fc: FileCtx, tests: list[ast.expr], env: dict[str, ast.expr]) -> bool | None:
    verdict: bool | None = True
    for t in tests:
        v = _cond(ex, fc, t, env)
        if v is False:
            return False
        if v is None:
            verdict = None
    return verdict


#: a call that hands a tool to an agent: ``agent.tool(fn)``, ``register_function(fn, caller=agent)``,
#: ``team.add_tool(fn, agent=a)`` ... the name says it registers, an agent variable says for whom
_REGISTER_HINT = re.compile(r"(register|attach|bind|install|equip|enable|provide|give|add|use|tool)", re.IGNORECASE)
_HOLDER_KW = {"caller", "agent", "assistant", "to", "on", "target", "owner", "for_agent", "bot"}


def _attach_by_call(ex: ProjectExtractor, fc: FileCtx, node: ast.Call) -> None:
    name = _name(node.func) or ""
    if not _REGISTER_HINT.search(name):
        return
    holders: list[str] = []
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ex.agent_vars
    ):
        holders.append(node.func.value.id)  # agent.tool(fn)
    holders += [
        k.value.id
        for k in node.keywords
        if k.arg in _HOLDER_KW and isinstance(k.value, ast.Name) and k.value.id in ex.agent_vars
    ]
    if not holders:
        return
    names: list[str] = []
    for a in [
        *node.args,
        *(k.value for k in node.keywords if k.arg in (None, "f", "fn", "func", "function", "tool", "tools")),
    ]:
        if isinstance(a, ast.Name) and a.id in ex.agent_vars:
            continue
        names += ex._tools_from_expr(fc, a, in_tool_list=True)
    given = ex._static_str(fc, _kw(node, "name"))
    if given and given in ex.obs.declared_tools:
        names = [given]
    for h in dict.fromkeys(holders):
        ex._attach_tools(fc, ex.agent_vars[h], list(dict.fromkeys(names)), node)


def _bindings_and_facts(ex: ProjectExtractor, fc: FileCtx) -> None:
    # innermost enclosing function of every call (outer functions are walked first, inner ones overwrite)
    ex._enclosing_fn = {}
    ex._method_owner = {}  # id(method) -> tool name of its class (BaseTool._run, Tool.forward)
    for fn in ast.walk(fc.tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    ex._enclosing_fn[id(n)] = fn
        elif isinstance(fn, ast.ClassDef) and id(fn) in ex.node_tool:
            for m in fn.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ex._method_owner[id(m)] = ex.node_tool[id(fn)]
    for name, value in fc.consts.items():
        if isinstance(value, (ast.List, ast.Tuple, ast.ListComp)):
            lst = ex._tools_from_expr(fc, value, in_tool_list=ex._used_as_tool_list(name))
            if isinstance(value, (ast.List, ast.Tuple)):  # grown later: TOOLS.append(x) / TOOLS.extend([...]) / +=
                for x in ast.walk(fc.tree):
                    if _appends_to(x, name):
                        lst = lst + ex._tools_from_expr(fc, x.args[0], in_tool_list=True)
                    elif isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name) and x.target.id == name:
                        lst = lst + ex._tools_from_expr(fc, x.value, in_tool_list=True)
                lst = list(dict.fromkeys(lst))
            if lst:
                ex.tool_list_vars[name] = lst
    assigned_agents: dict[int, str] = {}
    for node in ast.walk(fc.tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.targets[0], ast.Name):
            assigned_agents[id(node.value)] = node.targets[0].id
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.targets[0], ast.Attribute)
            and _name(node.value.func) in ex.agent_ctors
        ):
            assigned_agents[id(node.value)] = node.targets[0].attr  # self.researcher = Agent(...)
    for node in ast.walk(fc.tree):  # a registry: {"billing": Agent(...)} names the agent, whatever it says of itself
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(v, ast.Call) and _name(v.func) in ex.agent_ctors and (key := _str(k)):
                    assigned_agents[id(v)] = key
    for node in ast.walk(fc.tree):
        if isinstance(node, ast.Call):
            ex._call_facts(fc, node, assigned_agents.get(id(node)))
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                ks = _str(k)
                if ks in BOUND_KWARGS:
                    ex.obs.bounds.append(ex._bound_ref(fc, BOUND_KWARGS[ks], ks, v))
                elif (
                    ks == "model" and isinstance(v, ast.Constant) and isinstance(v.value, str) and _is_model_literal(v)
                ):  # AutoGen llm_config={"config_list": [{"model": "gpt-4o"}]}
                    mref = _ref(fc.rel, v)
                    if not any(m.name == v.value and m.code_ref == mref for m in ex.obs.models):
                        ex.obs.models.append(ModelRef(name=v.value, pinned=is_pinned(v.value), code_ref=mref))
                elif ks and EXCLUDE_KEY.match(ks):  # {"excluded_functions": ["revoke_permit"]}
                    _note_excluded(ex, fc, v)
                elif (
                    ks
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                    and _is_literal_credential(ks, v)
                    and not _LIVE_SECRET.match(v.value)
                ):  # {"api_key": "3f9c..."}
                    ex.obs.secrets.append(SecretRef(name=ks, code_ref=_ref(fc.rel, v), preview=v.value[:4] + "…"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            ex._secret_and_model_assign(fc, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dn = _name(dec.func if isinstance(dec, ast.Call) else dec)
                if dn in GATE_DECORATORS:
                    ex.obs.gates.append(GateRef(kind=GATE_DECORATORS[dn], tools=None, code_ref=_ref(fc.rel, dec)))
        elif isinstance(node, ast.While):  # `while turns < 10: turns += 1; client...create(...)` is a step bound
            steps = _while_bound(ex, fc, node)
            if steps is not None:
                ex.obs.bounds.append(
                    BoundRef(
                        kind="max_steps",
                        name="counted while loop around the model call",
                        value=float(steps),
                        code_ref=_ref(fc.rel, node),
                    )
                )
        elif isinstance(node, ast.For):  # `for _ in range(5): client.chat...create(...)` is a step bound
            steps = _range_bound(ex, fc, node)
            if steps is not None:
                retry = _is_retry_loop(node)  # a retry loop never feeds max_steps: N attempts are N - 1 retries
                ex.obs.bounds.append(
                    BoundRef(
                        kind="max_retries" if retry else "max_steps",
                        name="range() retry loop around the model call"
                        if retry
                        else "range() loop around the model call",
                        value=float(steps) - 1 if retry else float(steps),
                        code_ref=_ref(fc.rel, node),
                    )
                )
        if isinstance(node, ast.keyword) and node.arg and EXCLUDE_KEY.match(node.arg):  # disallowed_tools=[...]
            _note_excluded(ex, fc, node.value)
        if (
            isinstance(node, ast.keyword)
            and node.arg
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _is_literal_credential(node.arg, node.value)
            and not _LIVE_SECRET.match(node.value.value)
        ):
            # api_key="3f9c…" : a literal credential under a secret-like keyword, whatever its format
            ex.obs.secrets.append(
                SecretRef(name=node.arg, code_ref=_ref(fc.rel, node.value), preview=node.value.value[:4] + "…")
            )
        if isinstance(node, ast.keyword) and node.arg and isinstance(node.value, (ast.BinOp, ast.JoinedStr)):
            built = ex._static_str(fc, node.value)  # api_key="sk-" + "proj-" + "..." (assembled, still hard-coded)
            if built and _LIVE_SECRET.match(built):
                ex.obs.secrets.append(
                    SecretRef(name=node.arg, code_ref=_ref(fc.rel, node.value), preview=built[:6] + "…")
                )
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _LIVE_SECRET.match(node.value):
            ex.obs.secrets.append(
                SecretRef(name="<literal>", code_ref=_ref(fc.rel, node), preview=node.value[:6] + "…")
            )


def _used_as_tool_list(ex: ProjectExtractor, var: str) -> bool:
    """``TOOLS = [a, b]`` counts bare functions only when TOOLS is later passed where tools go."""
    if ex._tool_list_names is None:  # one pass over the project, not one per list constant
        names: set[str] = set()
        for fc in ex.files:
            for n in ast.walk(fc.tree):
                if not isinstance(n, ast.Call):
                    continue
                args = [k.value for k in n.keywords if k.arg in TOOL_LIST_KWARGS]
                if _name(n.func) in BIND_CALLS and n.args:
                    args.append(n.args[0])
                if _name(n.func) == "create_react_agent" and len(n.args) > 1:
                    args.append(n.args[1])
                names |= {x.id for a in args for x in ast.walk(a) if isinstance(x, ast.Name)}
        ex._tool_list_names = names
    return var in ex._tool_list_names


def _handoff_target(el: ast.AST) -> str:
    """``x``, ``"writer"``, ``handoff(x)``, ``handoff(agent=x, ...)``, ``Handoff(target="x")`` -> 'x'."""
    if isinstance(el, ast.Constant) and isinstance(el.value, str):
        return el.value  # AutoGen Swarm handoffs=["writer"], LlamaIndex can_handoff_to=["writer"]
    if isinstance(el, ast.Call):
        target = _kw(el, "agent") or _kw(el, "target") or (el.args[0] if el.args else None)
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            return target.value
        return _name(target) if target is not None else ""
    return _name(el)


def _late_agent_wiring(ex: ProjectExtractor, fc: FileCtx) -> None:
    """``triage.handoffs = [a, b]`` / ``agent.tools.append(t)`` after the agent was constructed;
    ``writer.clone(tools=[...])`` builds another agent from one."""
    for node in ast.walk(fc.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "clone":
            ex._clone_agent(fc, node)
        elif isinstance(node, ast.Call):
            _attach_by_call(ex, fc, node)  # register_function(fn, caller=a) / a.tool(fn) / team.add_tool(fn, agent=a)
    for node in ast.walk(fc.tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            tgt = node.targets[0]
            owner = tgt.value.id if isinstance(tgt.value, ast.Name) else None
            ag = ex.agent_vars.get(owner or "")
            if ag is None:
                continue
            if tgt.attr in ("handoffs", "sub_agents") and isinstance(node.value, (ast.List, ast.Tuple)):
                for el in node.value.elts:
                    h = ex._handoff_target(el)
                    if h and h not in ag.handoffs:
                        ag.handoffs.append(h)
            elif tgt.attr in TOOL_LIST_KWARGS:
                ex._attach_tools(fc, ag, ex._tools_from_expr(fc, node.value, in_tool_list=True), node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("initiate_chat", "a_initiate_chat")
            and isinstance(node.func.value, ast.Name)
        ):
            # AutoGen 0.2: user_proxy.initiate_chat(assistant, ...) — the sender drives the recipient
            ag = ex.agent_vars.get(node.func.value.id)
            target = node.args[0] if node.args else _kw(node, "recipient")
            if ag is not None and isinstance(target, ast.Name) and target.id not in ag.handoffs:
                ag.handoffs.append(target.id)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Attribute)
            and node.targets[0].value.attr in TOOL_LIST_KWARGS
        ):
            # smolagents: ``trader.tools["transfer_out"] = TransferOut()`` after the agent was built
            sub = node.targets[0].value
            ag = ex.agent_vars.get(sub.value.id if isinstance(sub.value, ast.Name) else "")
            if ag is not None:
                ex._attach_tools(fc, ag, ex._tools_from_expr(fc, ast.List(elts=[node.value]), in_tool_list=True), node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in TOOL_LIST_KWARGS
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            # ``agent.tools.update({"name": Tool()})``
            sub = node.func.value
            ag = ex.agent_vars.get(sub.value.id if isinstance(sub.value, ast.Name) else "")
            if ag is not None:
                ex._attach_tools(
                    fc, ag, ex._tools_from_expr(fc, ast.List(elts=list(node.args[0].values)), in_tool_list=True), node
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend")
            and isinstance(node.func.value, ast.Attribute)
        ):
            inner = node.func.value
            owner = inner.value.id if isinstance(inner.value, ast.Name) else None
            ag = ex.agent_vars.get(owner or "")
            if ag is None or not node.args:
                continue
            if inner.attr in TOOL_LIST_KWARGS:
                arg = node.args[0] if node.func.attr == "extend" else ast.List(elts=[node.args[0]])
                ex._attach_tools(fc, ag, ex._tools_from_expr(fc, arg, in_tool_list=True), node)
            elif inner.attr == "handoffs":
                items = (
                    node.args[0].elts
                    if node.func.attr == "extend" and isinstance(node.args[0], (ast.List, ast.Tuple))
                    else [node.args[0]]
                )
                for e in items:
                    h = ex._handoff_target(e)
                    if h and h not in ag.handoffs:
                        ag.handoffs.append(h)


def _clone_agent(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> None:
    """OpenAI Agents SDK ``base.clone(name=..., tools=[...], instructions=...)``: a new agent when the
    call changes what it can do (its tools, handoffs, instructions or name); a model swap is the same agent."""
    base = ex.agent_vars.get(_name(call.func.value))  # type: ignore[attr-defined]
    changed = {k.arg for k in call.keywords} & {"name", "tools", "handoffs", "instructions"}
    if base is None or not changed:
        return
    ref = _ref(fc.rel, call)
    if any(a.code_ref == ref for a in ex.obs.declared_agents):
        return
    tools = list(base.tools)
    if "tools" in changed:
        tools = ex._tools_from_expr(fc, _kw(call, "tools"), in_tool_list=True)
        for t in tools:
            ex.obs.bound_tools.setdefault(t, []).append(ref)
    handoffs = list(base.handoffs)
    ho = _kw(call, "handoffs")
    if isinstance(ho, (ast.List, ast.Tuple)):
        handoffs = [h for h in (ex._handoff_target(el) for el in ho.elts) if h]
    var = ex._site_name(fc, call)
    name = ex._static_str(fc, _kw(call, "name")) or var or f"{base.name} (clone)"
    prompt = ex._static_str(fc, _kw(call, "instructions")) if "instructions" in changed else base.system_prompt
    if name == base.name and tools == base.tools and handoffs == base.handoffs:
        name = var if var and var != base.name else f"{base.name} (clone)"
    ag = DeclaredAgent(
        name=name,
        tools=tools,
        system_prompt=prompt,
        handoffs=handoffs,
        kind=base.kind,
        delegates=list(base.delegates),
        var=var,
        allow_delegation=base.allow_delegation,
        code_ref=ref,
        built_from=f"clone of {base.name}",
    )
    ex.obs.declared_agents.append(ag)
    if var:
        ex.agent_vars.setdefault(var, ag)


def _attach_tools(ex: ProjectExtractor, fc: FileCtx, ag: DeclaredAgent, names: list[str], node: ast.AST) -> None:
    for t in names:
        if t not in ag.tools:
            ag.tools.append(t)
        ex.obs.bound_tools.setdefault(t, []).append(_ref(fc.rel, node))


def _registry_names(ex: ProjectExtractor, value: ast.AST) -> list[str] | None:
    """Tools of a toolset / registry object whose tools the code registers itself (``@ops.tool`` on ``ops =
    FunctionToolset()``), when ``value`` is that object (or a list of them); None when it is anything else."""
    owners: dict[str, list[str]] = {}
    for owner, tool, _where in ex._registry_bindings:
        owners.setdefault(owner, []).append(tool)
    vals = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    names = [v.id for v in vals if isinstance(v, ast.Name)]
    if not vals or len(names) != len(vals) or not all(n in owners for n in names):
        return None
    return [t for n in names for t in owners[n]]


def _bound_list(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> list[str] | None:
    node = ex._bound_node(call)
    names = ex._tools_from_expr(fc, node, in_tool_list=True) if node is not None else []
    for k in call.keywords:  # toolsets=[ops]: the tools the code registered on `ops` are the agent's
        if k.arg in DYNAMIC_TOOL_KWARGS:
            names = names + (_registry_names(ex, k.value) or [])
    if node is None and not names:
        return None
    return list(dict.fromkeys(names))
