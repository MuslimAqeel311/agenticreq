"""Static extraction: declared tools + schemas, bindings, agents, models, bounds, gates, secrets.

Pure ``ast`` — the user's code is parsed, never imported or executed. Local imports are followed
(``ast.symbols``), so a tool defined in ``tools.py`` and wrapped in ``agent.py`` is still found.

Every way of turning a Python callable into a tool goes through one resolver,
``_tools_from_expr``: decorators, ``tool(f)``, curried ``tool(**opts)(f)``, factory calls
(``StructuredTool.from_function``, ``FunctionTool.from_defaults``, ``Tool(func=...)``), tool
classes, comprehensions (``[tool(f) for f in (a, b)]``), and bare functions in a ``tools=`` list.

One agent constructor call can build several agents: in a loop over a table or a config file
(``for key, cfg in yaml.safe_load(...).items(): Agent(role=cfg["role"], ...)``), in a factory function
called once per agent (``make_agent("researcher", [search])``), or from CrewAI ``config=`` entries.
``_agent_instances`` expands every such call into one agent per row / call site (``ast.evaluate``
does the static evaluation). When the count cannot be known it is reported, never guessed.
"""

from __future__ import annotations

import ast
from agenticreq.ast.classify import is_test_file
from agenticreq.ast.envvalues import Assumed, env_default, load_dotenv
from agenticreq.ast.evaluate import UNK, Evaluator
from agenticreq.ast.schema import _name, literal
from agenticreq.ast.schema import params_from_function
from agenticreq.ast.symbols import FileCtx
from agenticreq.ast.symbols import FuncDef
from agenticreq.ast.symbols import SKIP_DIRS
from agenticreq.ast.symbols import SymbolTable
from agenticreq.ast.symbols import iter_py_files
from agenticreq.models.finding import SourceRef
from agenticreq.models.observed import BoundRef
from agenticreq.models.observed import DeclaredAgent
from agenticreq.models.observed import ExtractionError
from agenticreq.models.observed import ObservedAgent
from agenticreq.models.observed import ToolSchema
from agenticreq.models.observed import UnresolvedTool
from pathlib import Path
from agenticreq.ast import agent_discovery, body_analysis, signals, tool_binding, tool_discovery
from agenticreq.ast.extract_support import (
    AGENT_CTORS,
    _merge_groups,
    _ref,
)

#: body facts a tool inherits from the project functions it calls; whether an argument reaches them is not tracked
_PROPAGATED_EFFECTS = {"fs_write", "fs_read", "exec", "http", "non_idempotent", "self_modify", "logs"}
_EFFECT_DEPTH = 4

__all__ = ["ProjectExtractor", "extract", "iter_py_files", "SKIP_DIRS"]


class ProjectExtractor:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.base = self.root if self.root.is_dir() else self.root.parent
        self.obs = ObservedAgent(root=str(self.root))
        self.symbols = SymbolTable(self.root)
        self.files: list[FileCtx] = []
        self.var_to_tool: dict[str, str] = {}  # python identifier -> tool name
        self.node_tool: dict[int, str] = {}  # id(def / class node) -> tool name
        self.agent_vars: dict[str, DeclaredAgent] = {}
        self.tool_list_vars: dict[str, list[str]] = {}
        self.agent_ctors: set[str] = set(AGENT_CTORS)
        self._tool_list_names: set[str] | None = None
        self._groups: dict[tuple[str, int, str], set[str]] = {}  # (file, scope node id, key) -> tools
        self._scope: dict[int, int] = {}  # id(node) -> id(innermost enclosing function / class)
        self._decorator_tool: dict[int, str] = {}  # id(decorator call) -> the tool it declares
        self._enclosing_fn: dict[int, FuncDef] = {}  # id(call) -> innermost function around it (per file)
        self._method_owner: dict[int, str] = {}
        self._fn_index: dict[str, list[tuple[FileCtx, FuncDef]]] = {}  # function / method name -> definitions
        self._run_receivers: set[str] = set()  # names that hold a runnable agent / graph / team
        self._registry_bindings: list[tuple[str, str, SourceRef]] = []  # (@agent.tool owner var, tool, where)
        #: tool constructions whose name / callable is dynamic: reported unless a loop unrolls them
        self._unresolved: dict[str, UnresolvedTool] = {}
        self._unrolled: set[str] = set()
        #: tools a project factory function returns, per function node (a factory is read once, not once per caller)
        self._factory_tools: dict[int, list[str]] = {}
        self._factory_active: set[int] = set()  # factories being read right now: recursion stops here
        self._local_list_cache: dict[tuple[int, str], list[str]] = {}
        self._effects_cache: dict[int, frozenset[str]] = {}
        self._dotenv_cache: dict[str, tuple[str, str]] | None = None
        self._default_provider: str | None = None  # set_default_openai_client(<client with a known provider>)
        self.ev = Evaluator(self.symbols, self.base, lambda n: self._fn_index.get(n, []))
        self._sites: dict[int, list[tuple[FileCtx, ast.Call]]] = {}  # id(function) -> its call sites
        self._agent_bounds: set[int] = set()  # id(agent call) whose bounds were read per instance

    # ------------------------------------------------------------ entry
    def extract(self) -> ObservedAgent:
        self.files = self.symbols.load()
        for rel, line, msg in self.symbols.errors:
            self.obs.extraction_errors.append(ExtractionError(file=rel, line=line, message=msg))
        self.obs.files = [fc.rel for fc in self.files]
        # `class BrowserUseAgent(Agent)`: constructing a project subclass of an agent class builds an agent
        self.agent_ctors = set(AGENT_CTORS)
        for fc in self.files:
            for c in fc.classes.values():
                if {_name(b) for b in c.bases} & AGENT_CTORS:
                    self.agent_ctors.add(c.name)
        for fc in self.files:
            for n in ast.walk(fc.tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._fn_index.setdefault(n.name, []).append((fc, n))
        for fc in self.files:
            self._tools(fc)
        for fc in self.files:  # register tools created inline in tools=[...] before any fact needs them
            for node in ast.walk(fc.tree):
                if isinstance(node, ast.Call):
                    self._bound_list(fc, node)
                    if _name(node.func) == "register_function":  # AutoGen: a statement, not an assignment
                        self._wrapper_tool(fc, node, None)
        for fc in self.files:
            self._bindings_and_facts(fc)
        for fc in self.files:
            self._late_agent_wiring(fc)
        self._run_receivers = self._find_run_receivers()
        for fc in self.files:
            self._signals(fc)
        self._env_file_signals()
        self._helper_flows()
        for owner, tool, where in self._registry_bindings:  # @agent.tool / @agent.tool_plain bind to `agent`
            ag = self.agent_vars.get(owner)
            if ag is not None:
                self.obs.bound_tools.setdefault(tool, []).append(where)
                if tool not in ag.tools:
                    ag.tools.append(tool)
        self._tool_body_delegation()
        for fc in self.files:
            tool_binding._mark_dynamic_tool_sources(self, fc)
        self.obs.unresolved_tools = [u for k, u in sorted(self._unresolved.items()) if k not in self._unrolled]
        if self._default_provider:  # agents SDK: set_default_openai_client(AsyncOpenAI(base_url=<groq>))
            for m in self.obs.models:
                m.provider = m.provider or self._default_provider
        self.obs.tool_vars = dict(self.var_to_tool)
        self.obs.tool_groups = _merge_groups(list(self._groups.values()))
        self._framework()
        seen_bounds: set[tuple[str, float | None, str, int]] = set()
        unique: list[BoundRef] = []
        for b in self.obs.bounds:  # the same bound reached through two spellings on one line is one bound
            k = (b.kind, b.value, b.code_ref.file if b.code_ref else "", b.code_ref.line or 0 if b.code_ref else 0)
            if k not in seen_bounds:
                seen_bounds.add(k)
                unique.append(b)
        self.obs.bounds = unique
        if not self.obs.framework and not self.obs.declared_tools:
            self._loose_functions()
        self._attach_limits()
        for rel, why in self.ev.errors:  # a config file the code reads could not be parsed: say so (AS-902)
            self.obs.extraction_errors.append(
                ExtractionError(
                    file=rel, message=f"config file could not be read ({why}); loops over it are not expanded"
                )
            )
        return self.obs

    def _tool_function(self, t: ToolSchema) -> tuple[FileCtx, FuncDef] | None:
        """The function a tool runs, found by the file and name it was declared with."""
        if not t.func_name or t.code_ref is None:
            return None
        fc = next((f for f in self.files if f.rel == t.code_ref.file), None)
        if fc is None:
            return None
        fn = fc.functions.get(t.func_name)
        if fn is None:
            for cls in fc.classes.values():
                fn = next(
                    (
                        m
                        for m in cls.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == t.func_name
                    ),
                    None,
                )
                if fn is not None:
                    break
        return (fc, fn) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

    def _attach_limits(self) -> None:
        """Each tool's caps, in any of the forms ast/limits.py reads, with where they are."""
        from agenticreq.ast.limits import best, find_limits

        from agenticreq.ast.effects import classify_body

        for t in [*self.obs.declared_tools.values(), *self.obs.loose_functions]:
            found = self._tool_function(t)
            if found is None:
                continue
            fc, fn = found
            eff = classify_body(self, fc, fn)
            t.effect, t.effect_reason, t.effect_line = eff.effect, eff.reason, eff.line
            lims = find_limits(self, fc, fn)
            if not lims:
                continue
            top = best(lims)
            fixed = [x for x in lims if not x.runtime]
            if fixed:
                t.limits = [x.describe(fc.rel) for x in sorted(fixed, key=lambda x: (x.value is None, x.value or 0))][
                    :3
                ]
                if "bounded_output" not in t.features:
                    t.features = sorted({*t.features, "bounded_output"})
            elif top is not None:  # every cap found reads its value at run time: not shown bounded, sent to a mock run
                t.limit_runtime = top.source
                t.limits = [top.describe(fc.rel)]
                t.features = sorted({*t.features, "bounded_output_runtime"} - {"bounded_output"})

    def _loose_functions(self) -> None:
        """No framework, no tools: still look at what each public function does, framework-agnostically."""
        risky = {"exec", "url_from_arg", "unbounded_read"}
        for fc in self.files:
            if is_test_file(fc.rel):
                continue
            for fn in fc.tree.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name.startswith("_"):
                    continue
                feats = self._body_features(fn, fc)
                if risky & set(feats):
                    self.obs.loose_functions.append(
                        ToolSchema(
                            name=fn.name,
                            source="ast",
                            code_ref=_ref(fc.rel, fn),
                            kind="function (no framework)",
                            func_name=fn.name,
                            features=feats,
                            params=params_from_function(fn, {}),
                        )
                    )

    def _add_tool(self, t: ToolSchema, var: str | None, node: ast.AST | None = None) -> str:
        """Register a tool; a second declaration with the same name keeps the first. Returns the name."""
        name = t.name
        if name not in self.obs.declared_tools:
            self.obs.declared_tools[name] = t
        if var:
            self.var_to_tool.setdefault(var, name)
        if node is not None:
            self.node_tool.setdefault(id(node), name)
        return name

    # ------------------------------------------------------------ tool declarations
    def _index_scopes(self, fc: FileCtx) -> None:
        def visit(node: ast.AST, scope: int) -> None:
            for child in ast.iter_child_nodes(node):
                self._scope[id(child)] = scope
                inner = id(child) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else scope
                visit(child, inner)

        visit(fc.tree, id(fc.tree))

    def _group(self, fc: FileCtx, node: ast.AST, key: str, names: list[str], scope_of: ast.AST | None = None) -> None:
        if names:
            scope = self._scope.get(id(scope_of or node), id(fc.tree))
            self._groups.setdefault((fc.rel, scope, key), set()).update(names)

    def _tools(self, fc: FileCtx) -> None:
        tool_discovery._tools(self, fc)

    def _decorated_tool(self, fc: FileCtx, fn: FuncDef) -> None:
        tool_discovery._decorated_tool(self, fc, fn)

    def _function_tool(
        self,
        fc: FileCtx,
        fn: FuncDef,
        opts: ast.Call | None,
        *,
        kind: str,
        var: str | None = None,
        skip_ctx: bool = False,
        at: SourceRef | None = None,
    ) -> str:
        return tool_discovery._function_tool(self, fc, fn, opts, kind=kind, var=var, skip_ctx=skip_ctx, at=at)

    def _class_tool(self, fc: FileCtx, cls: ast.ClassDef) -> str | None:
        return tool_discovery._class_tool(self, fc, cls)

    def _is_tool_class(self, fc: FileCtx, cls: ast.ClassDef, depth: int = 0) -> bool:
        return tool_discovery._is_tool_class(self, fc, cls, depth)

    _is_mcp_file = staticmethod(tool_discovery._is_mcp_file)

    def _mcp_features(self, fc: FileCtx, opts: ast.Call | None) -> list[str]:
        return tool_discovery._mcp_features(self, fc, opts)

    def _is_library_tool_class(self, fc: FileCtx, func: ast.AST) -> bool:
        return tool_discovery._is_library_tool_class(self, fc, func)

    def _library_tool(self, fc: FileCtx, call: ast.Call, target: str | None) -> str:
        return tool_discovery._library_tool(self, fc, call, target)

    def _is_tool_module(self, fc: FileCtx) -> bool:
        return tool_discovery._is_tool_module(self, fc)

    def _agent_as_tool(self, fc: FileCtx, call: ast.Call, target: str | None) -> list[str]:
        return tool_discovery._agent_as_tool(self, fc, call, target)

    def _schema_class_tool(self, fc: FileCtx, cls: ast.ClassDef) -> str | None:
        return tool_discovery._schema_class_tool(self, fc, cls)

    def _wrapper_parts(self, call: ast.Call) -> tuple[str, ast.Call, ast.Call] | None:
        return tool_discovery._wrapper_parts(self, call)

    def _wrapped_callable(self, holder: ast.Call) -> ast.expr | None:
        return tool_discovery._wrapped_callable(self, holder)

    def _wrapper_tool(self, fc: FileCtx, call: ast.Call, target: str | None) -> list[str]:
        return tool_discovery._wrapper_tool(self, fc, call, target)

    def _dict_schema_tool(self, fc: FileCtx, node: ast.Dict) -> None:
        tool_discovery._dict_schema_tool(self, fc, node)

    def _resolve_class(self, fc: FileCtx, node: ast.AST) -> ast.ClassDef | None:
        return tool_discovery._resolve_class(self, fc, node)

    # ------------------------------------------------------------ values from the environment
    def _env_assumed(self, fc: FileCtx, node: ast.AST | None) -> Assumed | None:
        """`int(os.getenv("MAX", "10"))` / a name holding it / a `.env` entry -> the value the code runs with unless
        the deployment overrides it (always reported as assumed)."""
        if self._dotenv_cache is None:
            self._dotenv_cache = load_dotenv(self.base, self.symbols.project_root)
        return env_default(self.symbols, fc, node, self._dotenv_cache)

    def _bound_ref(self, fc: FileCtx, kind: str, name: str, node: ast.expr, call: ast.Call | None = None) -> BoundRef:
        ok, val = literal(node)
        if not ok and isinstance(node, (ast.Name, ast.Attribute)):
            rr = self.symbols.resolve(fc, node)
            if rr is not None and rr.kind == "const":
                ok, val = literal(rr.node)
        if not ok and call is not None:
            v = self.ev.value(self.ev.frame_of(fc, call), node)
            ok, val = v is not UNK, v
        assumed: str | None = None
        if not (ok and isinstance(val, (int, float)) and not isinstance(val, bool)):
            a = self._env_assumed(fc, node)
            if a is not None and isinstance(a.value, (int, float)):
                ok, val, assumed = True, a.value, a.label
            elif isinstance(node, ast.Attribute):  # config.settings.max_messages: the settings' default in code
                from agenticreq.ast.constants import setting_default

                got = setting_default(self.obs.root, node.attr)
                if got is not None:
                    ok, val, assumed = True, got[0], got[1]
            elif isinstance(
                node, ast.Call
            ):  # _num("MAX_MESSAGES", 120, int): a project helper reading env with a default
                from agenticreq.ast.constants import _default_in

                hit = _default_in(node)
                if hit is not None:
                    ok, val, assumed = True, hit[0], f"default in code of env {hit[1]}" if hit[1] else "default in code"
        return BoundRef(
            kind=kind,
            name=name,
            value=float(val) if ok and isinstance(val, (int, float)) and not isinstance(val, bool) else None,
            assumed_from=assumed,
            code_ref=_ref(fc.rel, node),
        )

    # ------------------------------------------------------------ tool body facts
    _non_idempotent = staticmethod(body_analysis._non_idempotent)

    def _body_features(self, fn: FuncDef | None, fc: FileCtx | None = None) -> list[str]:
        """The function's own body facts, plus the effects of the project functions it calls (save_report ->
        _write -> open(..., "w") is a file write of save_report)."""
        feats = set(body_analysis._body_features(fn))
        if fn is not None and fc is not None:
            feats |= self._callee_effects(fc, fn)
        return sorted(feats)

    def _callee_effects(self, fc: FileCtx, fn: FuncDef) -> frozenset[str]:
        if id(fn) not in self._effects_cache:
            self._effects_cache[id(fn)] = frozenset(self._follow_effects(fc, fn, {id(fn)}, 0))
        return self._effects_cache[id(fn)]

    def _follow_effects(self, fc: FileCtx, fn: FuncDef, seen: set[int], depth: int) -> set[str]:
        out: set[str] = set()
        if depth >= _EFFECT_DEPTH:
            return out
        owner = next((c for c in fc.classes.values() if any(m is fn for m in c.body)), None)
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call):
                continue
            callee: FuncDef | None = None
            callee_fc = fc
            f = n.func
            if (
                owner is not None
                and isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in ("self", "cls")
            ):
                callee = next(
                    (
                        m
                        for m in owner.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == f.attr
                    ),
                    None,
                )
            else:
                r = self.symbols.resolve(fc, f)
                if (
                    r is not None
                    and r.kind == "function"
                    and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ):
                    callee, callee_fc = r.node, r.fc
            if callee is None or id(callee) in seen or is_test_file(callee_fc.rel):
                continue
            seen.add(id(callee))
            out |= set(body_analysis._body_features(callee)) & _PROPAGATED_EFFECTS
            out |= self._follow_effects(callee_fc, callee, seen, depth + 1)
        return out

    _egress_guarded = staticmethod(body_analysis._egress_guarded)

    _fails_open = staticmethod(body_analysis._fails_open)

    _body_bounds = staticmethod(body_analysis._body_bounds)

    # ------------------------------------------------------------ the tool-expression resolver
    def _tools_from_expr(
        self,
        fc: FileCtx,
        node: ast.AST | None,
        target: str | None = None,
        *,
        in_tool_list: bool = False,
        depth: int = 0,
    ) -> list[str]:
        return tool_binding._tools_from_expr(self, fc, node, target, in_tool_list=in_tool_list, depth=depth)

    def _loop_bindings(
        self, fc: FileCtx, target: ast.AST, it: ast.AST, depth: int
    ) -> list[tuple[FileCtx, dict[str, ast.expr]]] | None:
        return tool_binding._loop_bindings(self, fc, target, it, depth)

    def _local_list_tools(self, fc: FileCtx, fn: FuncDef, var: str, depth: int) -> list[str]:
        return tool_binding._local_list_tools(self, fc, fn, var, depth)

    def _mark_unrolled(self, fc: FileCtx, template: ast.AST) -> None:
        tool_binding._mark_unrolled(self, fc, template)

    def _mark_unresolved_template(self, fc: FileCtx, template: ast.AST, why: str) -> None:
        tool_binding._mark_unresolved_template(self, fc, template, why)

    def _literal_elements(self, fc: FileCtx, node: ast.AST, depth: int) -> list[tuple[FileCtx, ast.expr]]:
        return tool_binding._literal_elements(self, fc, node, depth)

    # ------------------------------------------------------------ bindings, agents, models, bounds, gates
    def _bindings_and_facts(self, fc: FileCtx) -> None:
        # innermost enclosing function of every call (outer functions are walked first, inner ones overwrite)
        tool_binding._bindings_and_facts(self, fc)

    def _used_as_tool_list(self, var: str) -> bool:
        return tool_binding._used_as_tool_list(self, var)

    _handoff_target = staticmethod(tool_binding._handoff_target)

    def _late_agent_wiring(self, fc: FileCtx) -> None:
        tool_binding._late_agent_wiring(self, fc)

    def _clone_agent(self, fc: FileCtx, call: ast.Call) -> None:
        tool_binding._clone_agent(self, fc, call)

    def _attach_tools(self, fc: FileCtx, ag: DeclaredAgent, names: list[str], node: ast.AST) -> None:
        tool_binding._attach_tools(self, fc, ag, names, node)

    def _bound_list(self, fc: FileCtx, call: ast.Call) -> list[str] | None:
        return tool_binding._bound_list(self, fc, call)

    def _call_facts(self, fc: FileCtx, call: ast.Call, assigned_to: str | None) -> None:
        agent_discovery._call_facts(self, fc, call, assigned_to)

    # ------------------------------------------------------------ agents built in loops / factories / config
    #: (bindings, name hint, how built, file of the factory call site whose arguments are bound)
    _Instance = tuple[dict[str, ast.expr], "str | None", "str | None", "FileCtx | None"]

    def _agent_instances(self, fc: FileCtx, node: ast.AST, depth: int = 0) -> tuple[list[_Instance], str | None]:
        return agent_discovery._agent_instances(self, fc, node, depth)

    def _call_sites(self, fn: FuncDef) -> list[tuple[FileCtx, ast.Call]]:
        return agent_discovery._call_sites(self, fn)

    def _site_name(self, fc: FileCtx, call: ast.Call) -> str | None:
        return agent_discovery._site_name(self, fc, call)

    def _declared_agent(
        self,
        fc: FileCtx,
        call: ast.Call,
        fname: str,
        assigned_to: str | None,
        env: dict[str, ast.expr],
        hint: str | None,
        how: str | None,
        bound: list[str] | None,
        ref: SourceRef,
        site: FileCtx | None = None,
    ) -> DeclaredAgent:
        return agent_discovery._declared_agent(self, fc, call, fname, assigned_to, env, hint, how, bound, ref, site)

    def _bound_node(self, call: ast.Call) -> ast.expr | None:
        return agent_discovery._bound_node(self, call)

    # ------------------------------------------------------------ resolution helpers
    _assigned_value = staticmethod(agent_discovery._assigned_value)

    def _default_of(self, fc: FileCtx, node: ast.AST, call: ast.Call | None) -> str | None:
        return agent_discovery._default_of(self, fc, node, call)

    def _code_executor(self, fc: FileCtx, call: ast.Call, fname: str, agent: str, ref: SourceRef) -> None:
        agent_discovery._code_executor(self, fc, call, fname, agent, ref)

    def _group_agent(self, fc: FileCtx, call: ast.Call, fname: str, assigned_to: str | None, ref: SourceRef) -> None:
        agent_discovery._group_agent(self, fc, call, fname, assigned_to, ref)

    def _tool_body_delegation(self) -> None:
        agent_discovery._tool_body_delegation(self)

    def _find_run_receivers(self) -> set[str]:
        return agent_discovery._find_run_receivers(self)

    def _awaits_run(self, e: ast.AST, depth: int = 0) -> bool:
        return agent_discovery._awaits_run(self, e, depth)

    def _runs_agent(self, call: ast.Call) -> bool:
        return agent_discovery._runs_agent(self, call)

    def _env_file_signals(self) -> None:
        signals._env_file_signals(self)

    def _helper_flows(self) -> None:
        signals._helper_flows(self)

    def _url_param_functions(self) -> dict[str, set[int]]:
        return signals._url_param_functions(self)

    def _static_str(self, fc: FileCtx, node: ast.AST | None, depth: int = 0) -> str | None:
        return signals._static_str(self, fc, node, depth)

    def _model_from_node(
        self, fc: FileCtx, node: ast.AST, ref: SourceRef, provider: str | None = None, call: ast.Call | None = None
    ) -> None:
        signals._model_from_node(self, fc, node, ref, provider, call)

    _BASE_URL_KWARGS = ("base_url", "openai_api_base", "api_base", "endpoint", "base_api_url")

    def _provider_of_expr(self, fc: FileCtx, v: ast.AST, depth: int = 0) -> str | None:
        return signals._provider_of_expr(self, fc, v, depth)

    def _provider_of(self, fc: FileCtx, call: ast.Call, fname: str) -> str | None:
        return signals._provider_of(self, fc, call, fname)

    def _secret_and_model_assign(self, fc: FileCtx, node: ast.Assign | ast.AnnAssign) -> None:
        signals._secret_and_model_assign(self, fc, node)

    # ------------------------------------------------------------ Level-2 project signals
    def _signals(self, fc: FileCtx) -> None:
        signals._signals(self, fc)

    _use_index = staticmethod(signals._use_index)

    def _traces(self, fc: FileCtx, value: ast.AST) -> bool:
        return signals._traces(self, fc, value)

    # ------------------------------------------------------------ framework + execution model
    def _framework(self) -> None:
        signals._framework(self)


def extract(path: str | Path) -> ObservedAgent:
    return ProjectExtractor(path).extract()
