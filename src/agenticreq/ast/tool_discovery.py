"""How a Python callable becomes a tool: decorators, wrappers, tool classes, dict schemas, MCP.

Moved out of ``extractor.py`` unchanged; each function takes the extractor as ``ex``.
"""

from __future__ import annotations

import ast
import re
from agenticreq.ast.paths import gate_coverage
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.ast.schema import params_from_function
from agenticreq.ast.schema import params_from_model_class
from agenticreq.ast.schema import params_from_smolagents_inputs
from agenticreq.ast.schema import parse_docstring
from agenticreq.ast.symbols import FileCtx
from agenticreq.ast.symbols import FuncDef
from agenticreq.jsonschema_util import params_from_json_schema
from agenticreq.models.finding import SourceRef
from agenticreq.models.observed import GateRef
from agenticreq.models.observed import ParamSchema
from agenticreq.models.observed import ToolSchema
from agenticreq.models.observed import UnresolvedTool
from agenticreq.ast.extract_support import (
    GATE_CALLS,
    HUMAN_GATE_CALL,
    TOOL_BASE_CLASSES,
    TOOL_DECORATORS,
    TOOL_WRAPPERS,
    _AGENT_TOOL_PKGS,
    _BROWSER_USE_INJECTED,
    _FUNC_KWARGS,
    FUNC_KW_PATTERN,
    _MAX_DEPTH,
    _kw,
    _ref,
    _str,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor


def _tools(ex: ProjectExtractor, fc: FileCtx) -> None:
    ex._index_scopes(fc)
    for node in ast.walk(fc.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ex._decorated_tool(fc, node)
        elif isinstance(node, ast.ClassDef):
            ex._class_tool(fc, node)
        elif isinstance(node, ast.Dict):
            ex._dict_schema_tool(fc, node)
        elif isinstance(node, ast.Call) and ex._wrapper_parts(node) is not None:
            # a wrapper anywhere declares a tool: `return StructuredTool.from_function(...)`,
            # `return [types.Tool(name=...), ...]`, `register(Tool(...))`
            ex._tools_from_expr(fc, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            var = tgt.id if isinstance(tgt, ast.Name) else None
            names = ex._tools_from_expr(fc, node.value, var)
            if var and len(names) == 1:
                ex.var_to_tool.setdefault(var, names[0])  # inv_t = tool(...)(get_inventory); t = MyTool()
        elif isinstance(node, (ast.Assign, ast.AugAssign)) and isinstance(node.value, (ast.List, ast.Tuple)):
            # `tools = [WriteFileTool(), make_tool()]`: tool *instances* in any list are tools, wherever
            # the list goes later; bare functions there are not (they need a tools= context)
            names = ex._tools_from_expr(fc, node.value)
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name):  # `tools = [...]` and `tools += [...]` in one function: one context
                ex._group(fc, node, f"var:{tgt.id}", names)


def _decorated_tool(ex: ProjectExtractor, fc: FileCtx, fn: FuncDef) -> None:
    for dec in fn.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        dname = _name(call.func if call else dec)
        if dname not in TOOL_DECORATORS and not (dname == "action" and "browser_use" in fc.imports):
            continue  # browser-use: @controller.action("desc") / @self.registry.action("desc")
        skip_ctx = (
            dname == "tool" and isinstance((call.func if call else dec), ast.Attribute) and "pydantic_ai" in fc.imports
        )
        # @mcp.tool / @app.tool, or a plain @tool alias, in a FastMCP server file (unless the file also takes
        # `tool` from an agent framework)
        mcp_server = (
            dname == "tool"
            and ex._is_mcp_file(fc)
            and (isinstance(call.func if call else dec, ast.Attribute) or not (_AGENT_TOOL_PKGS & fc.imports))
        )
        name = ex._function_tool(
            fc, fn, call, kind="mcp-server tool" if mcp_server else f"@{dname}", var=fn.name, skip_ctx=skip_ctx
        )
        if mcp_server:  # @mcp.tool / @app.tool on a FastMCP server: provided to clients, not used by an agent
            t = ex.obs.declared_tools.get(name)
            if t is not None and "mcp_server" not in t.features:
                t.features = sorted({*t.features, *(ex._mcp_features(fc, call) or ["mcp_server"])})
        target = dec if call is None else call.func
        if isinstance(target, ast.Attribute):  # @self.registry.action / @agent.tool: one registry, one context
            ex._group(fc, fn, f"registry:{ast.unparse(target.value)}", [name])
            if isinstance(target.value, ast.Name):
                ex._registry_bindings.append((target.value.id, name, _ref(fc.rel, dec)))
        if call is not None:
            ex._decorator_tool[id(call)] = name  # its approval kwargs scope to this tool (_call_facts)
        return


def _function_tool(
    ex: ProjectExtractor,
    fc: FileCtx,
    fn: FuncDef,
    opts: ast.Call | None,
    *,
    kind: str,
    var: str | None = None,
    skip_ctx: bool = False,
    at: SourceRef | None = None,
) -> str:
    """A tool backed by a Python function. ``opts`` is the decorator / wrapper call carrying name= etc."""
    name = fn.name
    desc_override = None
    if opts is not None:
        first = _str(opts.args[0]) if opts.args else None
        if first and _name(opts.func) == "action":  # browser-use: @action("description", ...)
            desc_override = first
        elif first:  # crewai tool("Name"), claude-agent-sdk @tool("name", ...)
            name = first
        for k in ("name", "name_override", "tool_name"):
            if _str(_kw(opts, k)):
                name = _str(_kw(opts, k)) or name
        desc_override = _str(_kw(opts, "description")) or _str(_kw(opts, "description_override")) or desc_override
    if id(fn) in ex.node_tool and name == ex.node_tool[id(fn)]:
        if var:
            ex.var_to_tool.setdefault(var, name)
        return name  # already registered (same function wrapped again, e.g. once per agent)
    summary, doc_params = parse_docstring(ast.get_docstring(fn))
    params = params_from_function(fn, doc_params, skip_ctx)
    # LangChain @tool(args_schema=Model) / StructuredTool.from_function(args_schema=Model): the model is the schema
    schema_node = _kw(opts, "args_schema") if opts is not None else None
    if schema_node is not None and (sc := ex._resolve_class(fc, schema_node)) is not None:
        params = params_from_model_class(sc) or params
    if "browser_use" in fc.imports:  # browser-use fills these special parameters itself
        params = [p for p in params if p.name not in _BROWSER_USE_INJECTED]
    t = ToolSchema(
        name=name,
        description=desc_override or summary,
        params=params,
        source="ast",
        code_ref=at or _ref(fc.rel, fn),
        kind=kind,
        func_name=fn.name,
        features=ex._body_features(fn, fc),
        body_bounds=ex._body_bounds(fn),
    )
    name = ex._add_tool(t, var, fn)
    for inner in ast.walk(fn):
        if isinstance(inner, ast.Call) and (gate := _gate_kind(ex, fc, inner)):
            ref = _ref(fc.rel, inner)
            if not any(g.code_ref == ref for g in ex.obs.gates):
                cov, why = gate_coverage(fn, inner)
                ex.obs.gates.append(
                    GateRef(
                        kind=gate,
                        tools=[name],
                        code_ref=ref,
                        coverage=cov,
                        coverage_note=why,
                        waits_forever=waits_forever(ex, fc, inner),
                    )
                )
    guard = guard_call(fn)
    if guard is not None and not any(g.tools == [name] for g in ex.obs.gates):
        ex.obs.gates.append(
            GateRef(
                kind="guard_call",
                tools=[name],
                code_ref=_ref(fc.rel, guard),
                coverage="unverified",
                coverage_note=f"{_name(guard.func)}() can stop {name} before it acts; nothing shows it asks a person",
            )
        )
    return name


#: calls that test a value, never a decision: a guard around these is input handling, not approval
_NOT_A_GUARD = frozenset(dir(__import__("builtins"))) | {"get", "startswith", "endswith", "strip", "lower", "upper"}
_VALIDATION = re.compile(r"valid|sanitize|parse|format", re.IGNORECASE)


def guard_call(fn: ast.AST) -> ast.Call | None:
    """The call that can stop the tool before it acts, read from behaviour rather than a name: a statement of the
    tool's own body, ``if not f(...): return`` / ``raise``. Once the tool's effects are read, one that comes after
    the write is dropped (ast/extractor.py). It shows a check exists, not that a person makes it, so the result is a
    gate to verify and never a pass."""
    for st in getattr(fn, "body", []):
        if not isinstance(st, ast.If) or not any(isinstance(x, (ast.Return, ast.Raise)) for x in st.body):
            continue
        for c in ast.walk(st.test):
            nm = _name(c.func) if isinstance(c, ast.Call) else None
            if nm and nm not in _NOT_A_GUARD and not (_VALIDATION.search(nm) and "approv" not in nm.lower()):
                assert isinstance(c, ast.Call)
                return c
    return None


#: names in a function that put a time limit on waiting for a person
_TIME_LIMIT = re.compile(r"timeout|alarm|select|timer|wait_for|deadline|expires?", re.IGNORECASE)


def waits_forever(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> bool:
    """A gate call that blocks until a person answers, with no time limit: input() itself, or a project function
    whose body calls input() and nothing that limits the wait (a timeout, signal.alarm, select, a Timer)."""
    if _name(call.func) == "input":
        return True
    if any(k.arg and _TIME_LIMIT.search(k.arg) for k in call.keywords):
        return False
    r = ex.symbols.resolve(fc, call.func)
    if r is None or r.kind != "function" or not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    names = [
        (_name(n.func) or "")
        if isinstance(n, ast.Call)
        else n.id
        if isinstance(n, ast.Name)
        else n.attr
        if isinstance(n, ast.Attribute)
        else ""
        for n in ast.walk(r.node)
    ]
    return "input" in names and not any(_TIME_LIMIT.search(x) for x in names if x)


def _asks_human(ex: ProjectExtractor, fn: ast.AST, depth: int = 0) -> bool:
    """Does this project function itself ask a person (input(), a known gate call, or a call that says so)?"""
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            nm = _name(n.func) or ""
            if (
                nm == "input"
                or nm in GATE_CALLS
                or (HUMAN_GATE_CALL.match(nm) and depth == 0 and nm != getattr(fn, "name", ""))
            ):
                return True
    return False


def _gate_kind(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> str | None:
    """The kind of human gate this call is, or None. Exact names first; then a call named like a human check
    (confirm_with_user, ask_user_to_approve ...). A project function with such a name only counts when its body
    really asks someone (confirm_order() that books an order is not a gate); a library function counts when the
    name says human / user / approval / consent, never on a bare `confirm`."""
    nm = _name(call.func) or ""
    if nm in GATE_CALLS:
        return GATE_CALLS[nm] if _is_gate_call(ex, fc, call) else None
    if not HUMAN_GATE_CALL.match(nm):
        return None
    r = ex.symbols.resolve(fc, call.func)
    if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "approval_call" if _asks_human(ex, r.node) else None
    return "approval_call" if re.search(r"human|user|approv|consent|sign_?off", nm, re.IGNORECASE) else None


def _is_gate_call(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> bool:
    """A call that really is a human gate. `interrupt` counts only when it is LangGraph's: the file imports
    langgraph and the project does not define its own function of that name (a home-made `interrupt()` that
    returns True gates nothing)."""
    if _name(call.func) != "interrupt":
        return True
    return "langgraph" in fc.imports and ex.symbols.resolve(fc, call.func) is None


def _class_tool(ex: ProjectExtractor, fc: FileCtx, cls: ast.ClassDef) -> str | None:
    if id(cls) in ex.node_tool:
        return ex.node_tool[id(cls)]
    if not ex._is_tool_class(fc, cls):
        return None
    attrs: dict[str, ast.expr] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            attrs[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            attrs[stmt.target.id] = stmt.value
    name = _attr_name(attrs.get("name")) or _inherited_name(ex, fc, cls) or cls.name
    desc = _attr_name(attrs.get("description")) or parse_docstring(ast.get_docstring(cls))[0]
    params = []
    kind = "BaseTool"
    run_fn = next(
        (
            s
            for s in cls.body
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
            and s.name in ("_run", "_arun", "forward", "run", "__call__", "invoke")
        ),
        None,
    )
    if "inputs" in attrs:  # smolagents Tool
        params = params_from_smolagents_inputs(attrs["inputs"])
        kind = "smolagents.Tool"
    elif "args_schema" in attrs:
        schema_cls = ex._resolve_class(fc, attrs["args_schema"])
        if schema_cls is not None:
            params = params_from_model_class(schema_cls)
    elif run_fn is not None:
        params = params_from_function(run_fn, parse_docstring(ast.get_docstring(run_fn))[1])
    t = ToolSchema(
        name=name,
        description=desc,
        params=params,
        source="ast",
        code_ref=_ref(fc.rel, cls),
        kind=kind,
        func_name=None,
        features=ex._body_features(run_fn, fc) if run_fn else [],
        body_bounds=ex._body_bounds(run_fn) if run_fn else {},
    )
    ex._tool_classes[name] = cls.name
    return ex._add_tool(t, cls.name, cls)


def _attr_name(node: ast.expr | None) -> str | None:
    """A class attribute's text: ``name = "x"`` or ``name: str = Field(default="x")``."""
    if isinstance(node, ast.Call) and _name(node.func) == "Field":
        return _str(_kw(node, "default")) or (_str(node.args[0]) if node.args else None)
    return _str(node)


def _inherited_name(ex: ProjectExtractor, fc: FileCtx, cls: ast.ClassDef, depth: int = 0) -> str | None:
    """``name`` set on a project base class."""
    for b in cls.bases:
        r = ex.symbols.resolve(fc, b)
        if r is None or r.kind != "class" or not isinstance(r.node, ast.ClassDef) or depth >= _MAX_DEPTH:
            continue
        for s in r.node.body:
            if not isinstance(s, (ast.Assign, ast.AnnAssign)):
                continue
            target = s.targets[0] if isinstance(s, ast.Assign) else s.target
            if isinstance(target, ast.Name) and target.id == "name":
                return _attr_name(s.value)
        if got := _inherited_name(ex, r.fc, r.node, depth + 1):
            return got
    return None


def named_instance(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, cls_tool: str) -> str | None:
    """``_Tool(name="save_file")``: this instance is its own tool, whatever its class is called."""
    explicit = _str(_kw(call, "name"))
    if not explicit:
        return None
    base = ex.obs.declared_tools[cls_tool]
    ex._instanced.add(cls_tool)
    if explicit in ex.obs.declared_tools:
        return explicit
    desc = _str(_kw(call, "description")) or base.description
    ex.obs.declared_tools[explicit] = base.model_copy(
        update={"name": explicit, "description": desc, "code_ref": _ref(fc.rel, call)}
    )
    return explicit


def _is_tool_class(ex: ProjectExtractor, fc: FileCtx, cls: ast.ClassDef, depth: int = 0) -> bool:
    """Direct or inherited (across files) subclass of a known tool base class."""
    for b in cls.bases:
        bn = _name(b)
        if bn in TOOL_BASE_CLASSES or (bn.endswith("Tool") and bn[0].isupper()):  # RagTool, SerperDevTool ...
            return True
        if depth < _MAX_DEPTH:
            r = ex.symbols.resolve(fc, b)
            if (
                r is not None
                and r.kind == "class"
                and isinstance(r.node, ast.ClassDef)
                and ex._is_tool_class(r.fc, r.node, depth + 1)
            ):
                return True
    return False


def _is_mcp_file(fc: FileCtx) -> bool:
    return bool({"mcp", "fastmcp"} & fc.imports)


def _mcp_features(ex: ProjectExtractor, fc: FileCtx, opts: ast.Call | None) -> list[str]:
    """MCP tool annotations a server declares (``annotations=ToolAnnotations(readOnlyHint=True, ...)``
    or a dict). Clients use them to decide whether to ask the user before a call."""
    if opts is None or not ex._is_mcp_file(fc):
        return []
    feats = ["mcp_server"]
    ann = _kw(opts, "annotations")
    hints: dict[str, object] = {}
    if isinstance(ann, ast.Call):
        hints = {k.arg: literal(k.value)[1] for k in ann.keywords if k.arg}
        for kw in ann.keywords:  # ToolAnnotations(**{"category": "X", "readOnlyHint": True})
            ok, d = literal(kw.value) if kw.arg is None else (False, None)
            if ok and isinstance(d, dict):
                hints.update(d)
    elif isinstance(ann, ast.Dict):
        ok, d = literal(ann)
        hints = d if ok and isinstance(d, dict) else {}
    for k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
        if k in hints:
            feats.append(f"mcp:{k}={'true' if hints[k] is True else 'false'}")
    if ann is not None and not hints:
        feats.append("mcp:annotations=dynamic")
    return feats


def _is_library_tool_class(ex: ProjectExtractor, fc: FileCtx, func: ast.AST) -> bool:
    """``WriteFileTool()`` imported from a package (not defined in the project), named ``*Tool``."""
    n = _name(func)
    return (
        bool(n)
        and n[0].isupper()
        and n.endswith("Tool")
        and n not in TOOL_WRAPPERS | TOOL_BASE_CLASSES
        and (
            n in fc.import_table
            or n
            in {a.asname or a.name for imp in ast.walk(fc.tree) if isinstance(imp, ast.ImportFrom) for a in imp.names}
        )
    )


def _library_tool(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, target: str | None) -> str:
    """A tool class from a library. Its wire name is not in the project; LangChain's convention
    (``WriteFileTool`` -> ``write_file``) is used, unless ``name=`` is passed. Heuristic."""
    cls = _name(call.func)
    name = ex._static_str(fc, _kw(call, "name")) or re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls[: -len("Tool")]).lower()
    t = ToolSchema(
        name=name,
        description=None,
        params=[],
        source="ast",
        code_ref=_ref(fc.rel, call),
        kind=f"library tool {cls} (name from a code pattern)",
    )
    name = ex._add_tool(t, target)
    ex.var_to_tool.setdefault(cls, name)
    return name


def _is_tool_module(ex: ProjectExtractor, fc: FileCtx) -> bool:
    """Most of the module's public functions are tools somewhere (a shared_tools.py), not a utils.py."""
    public = [f for n, f in fc.functions.items() if not n.startswith("_")]
    return bool(public) and sum(id(f) in ex.node_tool for f in public) * 2 >= len(public)


def _agent_as_tool(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, target: str | None) -> list[str]:
    """An agent exposed as a tool: OpenAI Agents ``agent.as_tool(tool_name=...)``, ADK ``AgentTool(agent=a)``."""
    if isinstance(call.func, ast.Attribute) and call.func.attr == "as_tool":
        agent_node: ast.AST | None = call.func.value
        name = ex._static_str(fc, _kw(call, "tool_name")) or ex._static_str(fc, _kw(call, "name"))
        desc = ex._static_str(fc, _kw(call, "tool_description")) or ex._static_str(fc, _kw(call, "description"))
    elif _name(call.func) == "AgentTool":
        agent_node = _kw(call, "agent") or (call.args[0] if call.args else None)
        name, desc = None, None
    else:
        return []
    if name is None and isinstance(agent_node, ast.Name):
        ag = ex.agent_vars.get(agent_node.id)
        name = ag.name if ag else agent_node.id
    if not name:
        return []
    # the framework gives an agent-as-tool one string parameter: OpenAI Agents `input`, ADK `request`
    param = ParamSchema(name="input" if _name(call.func) == "as_tool" else "request", type="string", required=True)
    agent_ref = _name(agent_node) if agent_node is not None else ""
    t = ToolSchema(
        name=name,
        description=desc,
        params=[param],
        source="ast",
        code_ref=_ref(fc.rel, call),
        kind="agent-as-tool",
        raw={"agent": agent_ref} if agent_ref else None,
    )
    return [ex._add_tool(t, target)]


def _schema_class_tool(ex: ProjectExtractor, fc: FileCtx, cls: ast.ClassDef) -> str | None:
    """A pydantic model / TypedDict passed where tools go: LangChain ``bind_tools([ConductResearch])``."""
    if id(cls) in ex.node_tool:
        return ex.node_tool[id(cls)]
    if not {_name(b) for b in cls.bases} & {"BaseModel", "TypedDict"}:
        return None
    t = ToolSchema(
        name=cls.name,
        description=parse_docstring(ast.get_docstring(cls))[0],
        params=params_from_model_class(cls),
        source="ast",
        code_ref=_ref(fc.rel, cls),
        kind="schema-class",
    )
    return ex._add_tool(t, cls.name, cls)


def _wrapper_parts(ex: ProjectExtractor, call: ast.Call) -> tuple[str, ast.Call, ast.Call] | None:
    """(wrapper name, options call, call holding the wrapped callable) for ``W(f)`` / ``W(**o)(f)``."""
    if isinstance(call.func, ast.Call) and _name(call.func.func) in TOOL_WRAPPERS:
        return _name(call.func.func), call.func, call  # curried: tool(parse_docstring=True)(f)
    if _name(call.func) in TOOL_WRAPPERS:
        return _name(call.func), call, call
    return None


def _wrapped_callable(ex: ProjectExtractor, holder: ast.Call) -> ast.expr | None:
    found: ast.expr | None = None
    for k in _FUNC_KWARGS:
        if (v := _kw(holder, k)) is not None:
            found = v
            break
    if found is None:  # async_fn= / run_fn= / handler= ... : a function-shaped keyword
        found = next(
            (
                k.value
                for k in holder.keywords
                if k.arg
                and FUNC_KW_PATTERN.search(k.arg)
                and isinstance(k.value, (ast.Name, ast.Attribute, ast.Lambda))
            ),
            None,
        )
    if found is None:
        found = next((a for a in holder.args if not (isinstance(a, ast.Constant) and isinstance(a.value, str))), None)
    # functools.partial(f, dep=...) binds dependencies; the tool is still f
    while isinstance(found, ast.Call) and _name(found.func) == "partial" and found.args:
        found = found.args[0]
    return found


def _wrapper_tool(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, target: str | None) -> list[str]:
    parts = ex._wrapper_parts(call)
    if parts is None:
        return []
    wname, opts, holder = parts
    callable_node = ex._wrapped_callable(holder)
    ref = _ref(fc.rel, call)
    if callable_node is not None:
        r = ex.symbols.resolve(fc, callable_node)
        for _ in range(_MAX_DEPTH):  # bound = partial(f, ...); coroutine=bound  ->  f
            if r is None or r.kind != "const" or not isinstance(r.node, (ast.Call, ast.Name, ast.Attribute)):
                break
            inner: ast.expr = r.node
            while isinstance(inner, ast.Call) and _name(inner.func) == "partial" and inner.args:
                inner = inner.args[0]
            if inner is r.node and isinstance(inner, ast.Call):
                break
            r = ex.symbols.resolve(r.fc, inner)
        if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = ex._function_tool(r.fc, r.node, opts, kind=f"{wname}(...)", var=target)
            if wname == "register_function":
                ex.obs.bound_tools.setdefault(name, []).append(ref)
            return [name]
        if r is not None and r.kind == "class" and isinstance(r.node, ast.ClassDef):
            cls_tool = ex._class_tool(r.fc, r.node)
            return [cls_tool] if cls_tool else []
        # lambda / unresolvable callable: keep the tool if it is explicitly named
    explicit = _str(_kw(opts, "name")) or (_str(opts.args[0]) if opts.args else None)
    if not explicit:
        if callable_node is not None or opts.args or _kw(opts, "name") is not None:
            # a tool is built here, but its name and function are runtime values: it exists, unseen
            ex._unresolved.setdefault(
                str(ref),
                UnresolvedTool(
                    code_ref=ref, reason=f"{wname}(...) with a dynamic name / callable ({ast.unparse(call)[:80]})"
                ),
            )
        return []
    name = explicit
    params = []
    schema_node = _kw(opts, "args_schema")
    if schema_node is not None and (sc := ex._resolve_class(fc, schema_node)) is not None:
        params = params_from_model_class(sc)
    input_schema = _kw(opts, "inputSchema") or _kw(opts, "input_schema") or _kw(opts, "parameters")
    if input_schema is not None and not params:  # MCP types.Tool(inputSchema={...}), Haystack Tool(parameters=)
        ok, js = literal(input_schema)
        params = params_from_json_schema(js) if ok and isinstance(js, dict) else []
    mcp = ex._mcp_features(fc, opts) if input_schema is not None or _kw(opts, "annotations") else []
    t = ToolSchema(
        name=name,
        description=ex._static_str(fc, _kw(opts, "description")),
        params=params,
        source="ast",
        code_ref=ref,
        kind="mcp-server tool" if mcp else f"{wname}(...)",
        features=mcp,
    )
    return [ex._add_tool(t, target)]


def _dict_schema_tool(ex: ProjectExtractor, fc: FileCtx, node: ast.Dict) -> None:
    ok, d = literal(node)
    if not ok or not isinstance(d, dict):
        return
    if isinstance(d.get("name"), str) and isinstance(d.get("input_schema"), dict):  # anthropic
        name, desc, params = d["name"], d.get("description"), d["input_schema"]
    elif d.get("type") == "function" and isinstance(d.get("function"), dict) and d["function"].get("name"):
        f = d["function"]
        name, desc, params = f["name"], f.get("description"), f.get("parameters")
    elif d.get("type") == "function" and isinstance(d.get("name"), str) and "parameters" in d:  # responses
        name, desc, params = d["name"], d.get("description"), d.get("parameters")
    else:
        return
    if name in ex.obs.declared_tools:
        return
    ex._add_tool(
        ToolSchema(
            name=name,
            description=desc,
            params=params_from_json_schema(params),
            source="ast",
            code_ref=_ref(fc.rel, node),
            kind="dict-schema",
        ),
        None,
    )
    ex.obs.bound_tools.setdefault(name, []).append(_ref(fc.rel, node))


def _resolve_class(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> ast.ClassDef | None:
    r = ex.symbols.resolve(fc, node)
    if r is not None and r.kind == "class" and isinstance(r.node, ast.ClassDef):
        return r.node
    name = _name(node)
    return next((f.classes[name] for f in ex.files if name in f.classes), None)
