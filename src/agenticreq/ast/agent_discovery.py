"""Agent constructor calls: one call can build several agents; models, bounds, gates, executors.

Moved out of ``extractor.py`` unchanged; each function takes the extractor as ``ex``.
"""

from __future__ import annotations

import ast
import re
from agenticreq.ast.classify import is_test_file
from agenticreq.ast.classify import looks_like_model
from agenticreq.ast.evaluate import Frame
from agenticreq.ast.evaluate import UNK
from agenticreq.ast.evaluate import bind_target
from agenticreq.ast.evaluate import subst
from agenticreq.ast.paths import gate_coverage
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.ast.symbols import FileCtx
from agenticreq.ast.symbols import FuncDef
from agenticreq.models.finding import SourceRef
from agenticreq.models.observed import BoundRef
from agenticreq.models.observed import CodeExecRef
from agenticreq.models.observed import DeclaredAgent
from agenticreq.models.observed import GateRef
from agenticreq.models.observed import UnresolvedAgent
from typing import Any
from agenticreq.ast.tool_discovery import _gate_kind
from agenticreq.ast.extract_support import (
    BIND_CALLS,
    BOUND_CALLS,
    BOUND_KWARGS,
    GATE_HOOK_KWARG,
    GATE_KWARGS,
    MODEL_KWARGS,
    TOOL_LIST_KWARGS,
    TOOL_WRAPPERS,
    _GENERIC_BUILDER,
    _GROUP_CTORS,
    _NAMED_FIRST_ARG,
    _RUN_BUILDERS,
    _RUN_CALLS,
    _TOOL_SCOPED_GATES,
    _kw,
    _ref,
    _str,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor


_Instance = tuple[dict[str, ast.expr], "str | None", "str | None", "FileCtx | None"]

#: registering a guard: kernel.add_filter(...), app.use_middleware(...), agent.register_hook(...)
_REGISTER_GUARD = re.compile(
    r"^(add|register|use|attach|set|install|append)_([a-z_]*_)?"
    r"(filter|middleware|hook|interceptor|guard|guardrail|policy)s?$",
    re.IGNORECASE,
)
#: a function that asks a person: input(), interrupt(), ask_human(), request_approval(), confirm() ...
_HUMAN_SIGNAL = re.compile(r"approv|confirm|human|hitl|consent|ask_user|permission|interrupt")


def _filter_gate(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, fname: str, ref: SourceRef) -> None:
    """A registered filter / hook whose body asks a human is an approval gate for the tools it names. Whether it
    covers every path is code, not config, so the coverage stays unverified (UNKNOWN, never a PASS)."""
    if not _REGISTER_GUARD.match(fname or ""):
        return
    for a in [*call.args, *(k.value for k in call.keywords)]:
        r = ex.symbols.resolve(fc, a) if isinstance(a, (ast.Name, ast.Attribute)) else None
        fn: ast.AST | None = None
        if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = r.node
        elif isinstance(a, ast.Lambda):
            fn = a
        if fn is None:
            continue
        asks, tools = False, []
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and _HUMAN_SIGNAL.search((_name(n.func) or "").lower()):
                asks = True
            if isinstance(n, ast.Call) and _name(n.func) == "input":
                asks = True
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in ex.obs.declared_tools:
                tools.append(n.value)
        if asks:
            ex.obs.gates.append(
                GateRef(
                    kind="approval_call",
                    tools=sorted(set(tools)) or None,
                    code_ref=ref,
                    coverage="unverified",
                    coverage_note="a registered filter/hook asks a human; that it covers every path is not provable",
                )
            )


def _seq_elements(ev: Any, fr: Any, node: ast.AST | None, depth: int = 0) -> list[ast.expr]:
    """Elements of a sequence expression, through list()/tuple(), `+`, `*x` and names the evaluator can follow."""
    if node is None or depth > 6:
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[ast.expr] = []
        for e in node.elts:
            out += _seq_elements(ev, fr, e.value, depth + 1) if isinstance(e, ast.Starred) else [e]
        return out
    if isinstance(node, ast.Call) and _name(node.func) in ("list", "tuple", "set", "sorted") and node.args:
        return _seq_elements(ev, fr, node.args[0], depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _seq_elements(ev, fr, node.left, depth + 1) + _seq_elements(ev, fr, node.right, depth + 1)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        rfr, red = ev.reduce(fr, node)
        if red is not node:
            return _seq_elements(ev, rfr, red, depth + 1)
    return []


#: keywords whose text is prose or an identifier of the agent, never a model id (`name="claude-helper-2"`)
_NOT_A_MODEL_KW = {
    "name",
    "role",
    "description",
    "instructions",
    "instruction",
    "system_message",
    "system_prompt",
    "prompt",
    "goal",
    "backstory",
    "content",
    "text",
    "message",
    "query",
    "title",
    "label",
}


def _call_facts(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, assigned_to: str | None) -> None:
    fname = _name(call.func)
    ref = _ref(fc.rel, call)
    # direct Python calls of a tool function (not via the model), followed across imports
    r = ex.symbols.resolve(fc, call.func)
    if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if (
            id(r.node) not in ex.node_tool
            and r.fc is not fc
            and ex._is_tool_module(r.fc)
            and ast.get_docstring(r.node)
            and not ex._is_mcp_file(r.fc)
        ):
            # (a tool has a description; an MCP server's other functions are its transport plumbing)
            # a function from a tool module (its siblings are tools elsewhere) called straight from
            # Python: the model never sees it, so it is invisible at the boundary by construction
            ex._function_tool(r.fc, r.node, None, kind="tool-module function (code pattern)")
        if id(r.node) in ex.node_tool:
            ex.obs.direct_calls.setdefault(ex.node_tool[id(r.node)], []).append(ref)
    # tool binding
    bound = ex._bound_list(fc, call)
    if bound:
        for t in bound:
            ex.obs.bound_tools.setdefault(t, []).append(ref)
        ex._group(fc, call, f"bind:{id(call)}", bound)
    _filter_gate(ex, fc, call, fname, ref)
    if fname.startswith("from_") and isinstance(call.func, ast.Attribute) and _name(call.func.value) in ex.agent_ctors:
        fname = _name(call.func.value)  # ReActAgent.from_tools(...) builds a ReActAgent
    # agents
    if fname in ex.agent_ctors:
        instances, why = ex._agent_instances(fc, call)
        if why is not None:  # how many agents this builds is not static: one placeholder, reported
            instances = [({}, None, None, None)]
        built: list[DeclaredAgent] = []
        for env, hint, how, site in instances:
            ag = ex._declared_agent(
                fc, call, fname, assigned_to if len(instances) == 1 else None, env, hint, how, bound, ref, site
            )
            built.append(ag)
        seen: set[tuple[Any, ...]] = set()  # a factory called twice with the same arguments builds the same agent
        unique = []
        for a in built:
            sig = (a.name, tuple(a.tools), a.system_prompt, tuple(a.handoffs), tuple(a.delegates))
            if sig not in seen:
                seen.add(sig)
                unique.append(a)
        built = unique
        names = [a.name for a in built]
        for i, ag in enumerate(built):  # rows the code does not tell apart stay distinct nodes
            if names.count(ag.name) > 1:
                ag.name = f"{ag.name}#{i + 1}"
        for ag in built:
            if any(a.name == ag.name and a.tools == ag.tools and a.kind == ag.kind for a in ex.obs.declared_agents):
                continue  # the same agent from another constructor call of one factory (a try / fallback branch)
            ex.obs.declared_agents.append(ag)
            if ag.var:
                ex.agent_vars[ag.var] = ag
            if fname == "CodeAgent":
                ex.obs.prompt_only_tools += [t for t in ag.tools if t not in ex.obs.prompt_only_tools]
        if why is not None:
            ex.obs.unresolved_agents.append(UnresolvedAgent(code_ref=ref, reason=why, name=built[0].name))
        ex._code_executor(fc, call, fname, built[0].name, ref)
    elif fname in _GROUP_CTORS:
        ex._group_agent(fc, call, fname, assigned_to, ref)
    # models
    provider = ex._provider_of(fc, call, fname)
    for kw in MODEL_KWARGS:
        v = _kw(call, kw)
        if v is not None:
            ex._model_from_node(fc, v, ref, provider, call)
    for k in call.keywords:  # any keyword holding a model id (Agno `id=`, `engine=`, ...): the value says what it is
        if (
            k.arg
            and k.arg not in MODEL_KWARGS
            and k.arg not in _NOT_A_MODEL_KW
            and isinstance(k.value, ast.Constant)
            and isinstance(k.value.value, str)
            and re.search(r"\d", k.value.value)
            and looks_like_model(k.value.value)
        ):
            ex._model_from_node(fc, k.value, ref, provider, call)
    for k in call.keywords:  # LiteLLMModel(**model_params) with model_params = {"model_id": model_id, ...}
        if k.arg is None and isinstance(k.value, ast.Name):
            d = ex._assigned_value(fc, k.value.id)
            if isinstance(d, ast.Dict):
                for dk, dv in zip(d.keys, d.values):
                    if _str(dk) in MODEL_KWARGS:
                        ex._model_from_node(fc, dv, ref, provider, call)
    if (
        fname
        in (
            "ChatOpenAI",
            "ChatAnthropic",
            "OpenAIServerModel",
            "LiteLLMModel",
            "LLM",
            "init_chat_model",
            "OpenAIModel",
            "AnthropicModel",
            "OpenAIChatModel",
            "Agent",
            "LiteLlm",
            "create_react_agent",
            "create_agent",
            "ChatGoogleGenerativeAI",
            "ChatMistralAI",
            "ChatGroq",
            "GroqModel",
            "GoogleModel",
            "GeminiModel",
            "MistralModel",
            "CohereModel",
            "BedrockConverseModel",
            "OpenAIResponsesModel",
            "HuggingFaceModel",
        )
        and call.args
    ):
        ex._model_from_node(fc, call.args[0], ref, provider)
    # bounds (an expanded agent call recorded its own, one per agent). A bound on an agent's own constructor belongs
    # to that agent; one on a run / graph call applies to every agent.
    owner = built[0].name if fname in ex.agent_ctors and len(built) == 1 else None
    for k in call.keywords if id(call) not in ex._agent_bounds else []:
        if k.arg in BOUND_KWARGS:
            b = ex._bound_ref(fc, BOUND_KWARGS[k.arg], k.arg, k.value, call)
            b.agent = owner
            ex.obs.bounds.append(b)
    if fname in BOUND_CALLS and call.args and id(call) not in ex._agent_bounds:
        ex.obs.bounds.append(ex._bound_ref(fc, BOUND_CALLS[fname], fname, call.args[0], call))
    # gates
    gate_kind = _gate_kind(ex, fc, call)
    if gate_kind:
        if not any(g.code_ref == ref for g in ex.obs.gates):
            # inside a tool's own body it gates that tool only; in a dispatcher / graph node it gates the tools that
            # function names (string constants equal to a declared tool), else whatever runs there (scope unknown)
            fn = ex._enclosing_fn.get(id(call))
            owner = (ex.node_tool.get(id(fn)) or ex._method_owner.get(id(fn))) if fn is not None else None
            named: list[str] = []
            if owner is None and fn is not None:
                named = sorted(
                    {
                        n.value
                        for n in ast.walk(fn)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in ex.obs.declared_tools
                    }
                )
            cov, why = (
                gate_coverage(fn, call)
                if owner and fn is not None
                else ("unverified", "gate outside a tool body: which tools it covers is not known")
            )
            ex.obs.gates.append(
                GateRef(
                    kind=gate_kind,
                    tools=[owner] if owner else (named or None),
                    code_ref=ref,
                    coverage=cov,
                    coverage_note=why,
                )
            )
    for k in call.keywords:
        hook = k.arg not in GATE_KWARGS and bool(GATE_HOOK_KWARG.match(k.arg or ""))
        if (k.arg in GATE_KWARGS or hook) and k.arg != "step_callback":
            ok, val = literal(k.value)
            if ok and val in (False, None, [], (), "NEVER"):
                continue
            if hook and ok and val is not True:
                continue  # a hook is a callable: a literal string/number under that name is configuration, not a gate
            where = _ref(fc.rel, k.value)
            if k.arg in _TOOL_SCOPED_GATES:
                # needs_approval / requires_approval belong to ONE tool: the decorated function or the
                # wrapped callable. They never gate the agent's other tools.
                scope: list[str] | None = (
                    [ex._decorator_tool[id(call)]]
                    if id(call) in ex._decorator_tool
                    else (ex._tools_from_expr(fc, call) if fname in TOOL_WRAPPERS else [])
                )
                # needs_approval=True: the framework asks before every call (config, all paths); a callable
                # decides per call at runtime, so its coverage is not known statically
                cov = "all_paths" if scope and ok and val is True else "unverified"
                ex.obs.gates.append(
                    GateRef(
                        kind=GATE_KWARGS[k.arg] if scope else "needs_approval(unscoped)",
                        tools=scope or None,
                        code_ref=where,
                        coverage=cov,
                        coverage_note="framework asks before every call"
                        if cov == "all_paths"
                        else f"{k.arg}= is not the literal True: the framework decides per call at runtime",
                    )
                )
                continue
            if k.arg == "human_input" and fname == "Task":
                # CrewAI Task(human_input=True): a human reviews the task's final answer AFTER its tools
                # ran. It approves no tool call, so it is not an approval gate.
                ex.obs.gates.append(GateRef(kind="output_review", tools=None, code_ref=where))
                continue
            # interrupt_before/after name graph nodes, not tools. On an agent constructor the gate belongs to THAT agent's
            # tools (create_react_agent(tools=[...], interrupt_before=["tools"]) pauses before its tool node); on a
            # graph compile() it applies to whatever those nodes run.
            scope, cov, note = None, "unverified", None
            if fname in ex.agent_ctors and (k.arg in ("interrupt_before", "interrupt_after") or hook):
                tl = _kw(call, "tools")
                owned = bound or (ex._tools_from_expr(fc, tl, in_tool_list=True) if tl is not None else [])
                if owned:
                    scope = list(owned)
                    if (
                        k.arg == "interrupt_before"
                        and isinstance(k.value, (ast.List, ast.Tuple))
                        and any(_str(e) == "tools" for e in k.value.elts)
                    ):
                        cov, note = "all_paths", "the tool node is paused before it runs"
            ex.obs.gates.append(
                GateRef(
                    kind=GATE_KWARGS.get(k.arg or "", "approval_call"),
                    tools=scope,
                    code_ref=where,
                    coverage=cov,
                    coverage_note=note,
                )
            )
    if fname in ex.agent_ctors and assigned_to in ex.agent_vars:
        tl = _kw(call, "tools")
        for el in tl.elts if isinstance(tl, (ast.List, ast.Tuple)) else []:
            if isinstance(el, ast.Call) and _name(el.func) == "create_handoff_tool":
                h = ex._static_str(fc, _kw(el, "agent_name") or (el.args[0] if el.args else None))
                if h and h not in ex.agent_vars[assigned_to].handoffs:
                    ex.agent_vars[assigned_to].handoffs.append(h)
    if fname == "set_default_openai_client" and call.args:
        ex._default_provider = ex._provider_of_expr(fc, call.args[0]) or ex._default_provider


def _agent_instances(
    ex: ProjectExtractor, fc: FileCtx, node: ast.AST, depth: int = 0
) -> tuple[list[_Instance], str | None]:
    """Every agent one constructor call builds: one per row of the loops / comprehensions around it,
    times one per call of the factory function it sits in (its parameters bound to that call's
    arguments). ``([], reason)`` when the number of agents is not known statically."""
    ev = ex.ev
    par = ev.parents(fc)
    loops: list[tuple[ast.AST, ast.AST, list[ast.expr], ast.AST]] = []  # innermost first
    fn: FuncDef | None = None
    child, cur = node, par.get(id(node))
    while cur is not None:
        if (
            isinstance(cur, (ast.For, ast.AsyncFor))
            and child is not cur.iter
            and child is not cur.target
            and not any(child is s for s in cur.orelse)
        ):
            loops.append((cur.target, cur.iter, [], cur))
        elif isinstance(cur, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            gens = cur.generators
            used = gens[: gens.index(child)] if isinstance(child, ast.comprehension) else gens
            loops += [(g.target, g.iter, list(g.ifs), cur) for g in reversed(used)]
        elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = cur
            break
        elif isinstance(cur, (ast.ClassDef, ast.Lambda)):
            break
        child, cur = cur, par.get(id(cur))
    fr = ev.frame_of(fc, node)
    envs: list[ProjectExtractor._Instance] = [({}, None, None, None)]
    if fn is not None and depth < 3:  # a factory: one agent per call site that varies its arguments
        params = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]} - {"self", "cls"}
        mentioned = {
            n.id for x in [node, *(it for _, it, _, _ in loops)] for n in ast.walk(x) if isinstance(n, ast.Name)
        }
        for a in ast.walk(fn):  # `kwargs = dict(name=spec.name, ...)` then `Agent(**kwargs)`: the local's sources count
            if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id in mentioned for t in a.targets):
                mentioned |= {x.id for x in ast.walk(a.value) if isinstance(x, ast.Name)}
        sites = ex._call_sites(fn) if params & mentioned else []
        if sites:
            envs = []
            for sfc, scall in sites:
                outer, why = ex._agent_instances(sfc, scall, depth + 1)
                if why is not None:
                    return [], f"{why} (through factory {fn.name}() called at {sfc.rel}:{scall.lineno})"
                sfr = ev.frame_of(sfc, scall)
                bound_self = (
                    isinstance(scall.func, ast.Attribute)
                    and bool(fn.args.args)
                    and fn.args.args[0].arg in ("self", "cls")
                )
                for oenv, ohint, _, _ in outer:
                    penv = ev.call_env(sfr, fn, subst(scall, oenv), bound_self)
                    envs.append(
                        (
                            penv,
                            ohint or ex._site_name(sfc, scall),
                            f"factory {fn.name}() called at {sfc.rel}:{scall.lineno}",
                            sfc,
                        )
                    )
    for target, it, conds, owner in reversed(loops):
        nxt: list[ProjectExtractor._Instance] = []
        for env, hint, how, site in envs:
            ev.touched.clear()
            rows = ev.rows(fr, subst(it, env))
            if rows is None and not {n.id for n in ast.walk(target) if isinstance(n, ast.Name)} & {
                n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            }:
                nxt.append((env, hint, how, site))  # every iteration builds the same agent
                continue
            if rows is None:
                return [], f"built in a loop over a value only known at runtime: `{ast.unparse(it)[:80]}`"
            src = ", ".join(sorted(ev.touched)) or f"{fc.rel}:{getattr(owner, 'lineno', '?')}"
            for el in rows:
                b = bind_target(target, el)
                if b is None:
                    return (
                        [],
                        f"loop target `{ast.unparse(target)}` does not match the rows of `{ast.unparse(it)[:60]}`",
                    )
                e2 = {**env, **b}
                if any((v := ev.value(fr, subst(c, e2))) is not UNK and not v for c in conds):
                    continue  # a comprehension filter that is statically false for this row
                nxt.append((e2, hint, f"loop over {src}" + (f"; {how}" if how else ""), site))
        envs = nxt
    return envs, None


def _call_sites(ex: ProjectExtractor, fn: FuncDef) -> list[tuple[FileCtx, ast.Call]]:
    """Where a project function / method is called (by name, resolved through imports when possible)."""
    if id(fn) in ex._sites:
        return ex._sites[id(fn)]
    unique = len(ex._fn_index.get(fn.name, [])) == 1
    out: list[tuple[FileCtx, ast.Call]] = []
    for fc in ex.files:
        if is_test_file(fc.rel):
            continue  # a test calling the factory does not add an agent to the project
        for n in ast.walk(fc.tree):
            if not isinstance(n, ast.Call) or _name(n.func) != fn.name or isinstance(n.func, ast.Subscript):
                continue
            if any(x is n for x in ast.walk(fn)):
                continue  # recursion is not another agent
            r = ex.symbols.resolve(fc, n.func)
            if (r is not None and r.node is fn) or (r is None and unique):
                out.append((fc, n))
    ex._sites[id(fn)] = out
    return out


def _site_name(ex: ProjectExtractor, fc: FileCtx, call: ast.Call) -> str | None:
    """The name a call's result is stored under: ``x = f()``, ``self.x = f()``, ``{"x": f()}``."""
    p = ex.ev.parents(fc).get(id(call))
    if isinstance(p, (ast.Assign, ast.AnnAssign)):
        t = p.targets[0] if isinstance(p, ast.Assign) else p.target
        return t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
    if isinstance(p, ast.Dict):
        for k, v in zip(p.keys, p.values):
            if v is call:
                return _str(k)
    if isinstance(p, ast.keyword) and p.arg not in (None, "tools", "agents"):
        return p.arg
    return None


def _tool_names_in(ex: ProjectExtractor, fr: Any, node: ast.AST) -> list[str]:
    """Names of declared tools held in a literal list or tuple anywhere in ``node`` (``pick(*spec.tools)``)."""
    for n in ast.walk(node):
        v = ex.ev.value(fr, n) if isinstance(n, ast.expr) else None
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            # tool names (the tools may be declared in a file read later): identifiers, never prose
            if all(re.fullmatch(r"[A-Za-z_][\w.-]{0,63}", x) for x in v):
                return list(v)
    return []


def _declared_agent(
    ex: ProjectExtractor,
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
    """One agent from a constructor call, with a row's / call site's bindings substituted."""
    ev = ex.ev
    fr = ev.frame_of(fc, call)
    c = subst(call, env) if env else call
    # CrewAI Agent(config=self.agents_config["x"]) / Agent(**spec): key -> (frame, value expression)
    cfg: dict[str, tuple[Frame, ast.expr]] = {}
    for node in [_kw(c, "config"), *(k.value for k in c.keywords if k.arg is None)]:
        if node is None:
            continue
        cfr, d = ev.reduce(fr, node)
        if isinstance(d, ast.Call) and _name(d.func) == "dict" and not d.args:  # kwargs = dict(name=spec.name, ...)
            d = subst(d, env) if env else d
            cfg.update({k.arg: (cfr, k.value) for k in d.keywords if k.arg})
        if isinstance(d, ast.Dict):
            cfg.update(
                {
                    k.value: (cfr, v)
                    for k, v in zip(d.keys, d.values)
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
            )

    def cfgv(k: str) -> object:
        got = cfg.get(k)
        return ev.value(*got) if got else UNK

    def sv(k: str) -> str | None:
        node = _kw(c, k)
        return (ex._static_str(fc, node) or ev.string(fr, node)) if node is not None else None

    def cv(k: str) -> str | None:
        v = cfgv(k)
        return " ".join(v.split()) if isinstance(v, str) and v.strip() else None

    positional = (
        (ex._static_str(fc, c.args[0]) or ev.string(fr, c.args[0])) if c.args and fname in _NAMED_FIRST_ARG else None
    )
    enclosing = ev.enclosing(fc, call, (ast.FunctionDef, ast.AsyncFunctionDef))
    # CrewAI @agent methods / build_x() helpers name the agent they return
    returned = (
        enclosing.name
        if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(ev.parents(fc).get(id(call)), ast.Return)
        and not _GENERIC_BUILDER.match(enclosing.name)
        else None
    )
    name = (
        sv("name") or positional or sv("role") or cv("name") or cv("role") or hint or assigned_to or returned or fname
    )
    name = " ".join(name.split()).strip("\"'") or fname  # YAML `role: >` keeps written quotes
    prompt = (
        sv("instructions")
        or sv("system_prompt")
        or sv("backstory")
        or sv("prompt")
        or sv("system_message")
        or sv("description")
        or cv("backstory")
        or cv("instructions")
        or cv("system_prompt")
        or cv("goal")
    )
    tools = list(bound or [])
    if env or not tools:
        node = ex._bound_node(c)
        if node is not None:
            rfr, r = ev.reduce(fr, node)
            tools = ex._tools_from_expr(rfr.fc, r, in_tool_list=True) or (
                ex._tools_from_expr(fc, node, in_tool_list=True) if r is not node else []
            )
            if site is not None and site is not rfr.fc:  # tools passed in by a call site in another file
                tools += [t for t in ex._tools_from_expr(site, r, in_tool_list=True) if t not in tools]
        if node is None and "tools" in cfg:  # tools listed in the config entry (names or objects)
            cfr, tnode = cfg["tools"]
            tools = ex._tools_from_expr(cfr.fc, tnode, in_tool_list=True)
            if not tools:  # tools=pick(*spec.tools): a list of tool names the code looks the tools up by
                tools = _tool_names_in(ex, cfr, tnode)
        for t in tools:
            if ref not in ex.obs.bound_tools.setdefault(t, []):
                ex.obs.bound_tools[t].append(ref)
    handoffs: list[str] = []
    ho = _kw(c, "handoffs") or _kw(c, "sub_agents") or _kw(c, "can_handoff_to")
    if ho is None and fname in ("create_supervisor", "create_swarm"):  # langgraph-supervisor / swarm
        ho = _kw(c, "agents") or (c.args[0] if c.args else None)
    if ho is not None:
        handoffs = [h for h in (ex._handoff_target(el) for el in _seq_elements(ev, fr, ho)) if h]
    managed = _kw(c, "managed_agents")  # smolagents: the manager calls them like tools
    delegates = [h for h in (ex._handoff_target(el) for el in _seq_elements(ev, fr, managed)) if h]
    deleg = _kw(c, "allow_delegation")
    allow = ev.value(fr, deleg) if deleg is not None else cfgv("allow_delegation")
    if env or cfg:  # bounds per agent: max_iter=c.get("max_iter", 6) / config max_iter
        ex._agent_bounds.add(id(call))
        for kw, kind in BOUND_KWARGS.items():
            node = _kw(c, kw)
            v = ev.value(fr, node) if node is not None else cfgv(kw)
            if node is None and v is UNK:
                continue
            orig = _kw(call, kw)
            assumed = None
            if not isinstance(v, (int, float)) and isinstance(node, ast.Attribute):  # config.settings.max_turns
                from agenticreq.ast.constants import setting_default

                got = setting_default(ex.obs.root, node.attr)
                if got is not None:
                    v, assumed = got
            ex.obs.bounds.append(
                BoundRef(
                    kind=kind,
                    name=kw,
                    value=float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None,
                    code_ref=_ref(fc.rel, orig if orig is not None else call),
                    agent=name,
                    assumed_from=assumed,
                )
            )
    return DeclaredAgent(
        name=name,
        tools=tools,
        system_prompt=prompt,
        handoffs=handoffs,
        kind=fname,
        delegates=delegates,
        var=assigned_to or (hint if how and how.startswith("factory") else None),
        allow_delegation=allow is True,
        code_ref=ref,
        built_from=how,
    )


def _bound_node(ex: ProjectExtractor, call: ast.Call) -> ast.expr | None:
    """The expression a call passes as its tools (``tools=``, ``bind_tools(x)``, create_react_agent's 2nd arg)."""
    fname = _name(call.func)
    if fname in BIND_CALLS:
        return call.args[0] if call.args else _kw(call, "tools")
    node = next((_kw(call, kw) for kw in TOOL_LIST_KWARGS if _kw(call, kw) is not None), None)
    if node is None and fname == "create_react_agent" and len(call.args) > 1:
        node = call.args[1]
    if (
        node is None
        and call.args
        and fname.startswith(("from_", "create_", "make_", "build_"))
        and "tool" in fname.lower()
    ):
        node = call.args[0]  # ReActAgent.from_tools(tools, ...), make_tool_agent(tools)
    return node


def _assigned_value(fc: FileCtx, name: str) -> ast.expr | None:
    for n in ast.walk(fc.tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value
    return None


def _default_of(ex: ProjectExtractor, fc: FileCtx, node: ast.AST, call: ast.Call | None) -> str | None:
    """A model id that is a parameter default or a CLI default, not a literal at the call:
    ``def main(model_id="o1"): LiteLLMModel(model_id=model_id)`` and
    ``parser.add_argument("--model-id", default="o1")`` ... ``args.model_id``."""
    if isinstance(node, ast.Name) and call is not None:
        fn = ex._enclosing_fn.get(id(call))
        if fn is not None:
            a = fn.args
            pos = [*a.posonlyargs, *a.args]
            for arg, default in zip(pos[len(pos) - len(a.defaults) :], a.defaults):
                if arg.arg == node.id:
                    return _str(default)
            for arg, kd in zip(a.kwonlyargs, a.kw_defaults):
                if arg.arg == node.id and kd is not None:
                    return _str(kd)
    if isinstance(node, ast.Attribute):  # args.model_id <- add_argument("--model-id", default=...)
        want = node.attr
        for n in ast.walk(fc.tree):
            if isinstance(n, ast.Call) and _name(n.func) == "add_argument":
                flags = [(_str(x) or "").lstrip("-").replace("-", "_") for x in n.args]
                if want in flags or _str(_kw(n, "dest")) == want:
                    return _str(_kw(n, "default"))
    return None


def _code_executor(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, fname: str, agent: str, ref: SourceRef) -> None:
    """smolagents CodeAgent(executor_type=..., additional_authorized_imports=[...]); AutoGen
    UserProxyAgent(code_execution_config={"use_docker": ...}) / CodeExecutorAgent(code_executor=...)."""
    executor, unrestricted = None, False
    if fname in ("CodeAgent", "CodeActAgent"):
        ok, et = literal(_kw(call, "executor_type"))
        executor = str(et) if ok and et else "local"
        ok, imports = literal(_kw(call, "additional_authorized_imports"))
        if ok and isinstance(imports, (list, tuple)):
            unrestricted = bool({"*", "os", "subprocess", "sys", "shutil", "socket"} & {str(x) for x in imports})
    elif fname == "UserProxyAgent":
        cfg = _kw(call, "code_execution_config")
        ok, val = literal(cfg)
        if ok and val is False:
            return  # code execution switched off
        use_docker = None
        if isinstance(cfg, ast.Dict):
            for k, v in zip(cfg.keys, cfg.values):
                if _str(k) == "use_docker":
                    use_docker = literal(v)[1]
        executor = "local" if use_docker is False else "docker"
        unrestricted = executor == "local"
    elif fname == "CodeExecutorAgent":
        exec_node = _kw(call, "code_executor") or (call.args[1] if len(call.args) > 1 else None)
        exname = (
            _name(exec_node.func)
            if isinstance(exec_node, ast.Call)
            else _name(exec_node)
            if exec_node is not None
            else ""
        )
        executor = "docker" if "Docker" in exname else "jupyter" if "Jupyter" in exname else "local"
        unrestricted = executor == "local"
    if executor is not None:
        ex.obs.code_executors.append(
            CodeExecRef(agent=agent, kind=fname, executor=executor, unrestricted=unrestricted, code_ref=ref)
        )


def _group_agent(
    ex: ProjectExtractor, fc: FileCtx, call: ast.Call, fname: str, assigned_to: str | None, ref: SourceRef
) -> None:
    """A team / workflow / hierarchical crew is a coordinator that hands work to each participant."""
    members = _kw(call, "participants") or _kw(call, "agents") or (call.args[0] if call.args else None)
    if fname == "Crew":
        proc = _kw(call, "process")
        if proc is None or "hierarchical" not in ast.unparse(proc):
            return  # a sequential crew passes outputs along a list: no delegation unless allow_delegation
        manager = _kw(call, "manager_agent")
        if isinstance(manager, ast.Name) and isinstance(members, (ast.List, ast.Tuple)):
            mgr = ex.agent_vars.get(manager.id)
            if mgr is not None:
                mgr.handoffs += [
                    h
                    for h in (ex._handoff_target(e) for e in members.elts)
                    if h and h not in mgr.handoffs and h != manager.id
                ]
                return
    if not isinstance(members, (ast.List, ast.Tuple)):
        return
    names = [h for h in (ex._handoff_target(e) for e in members.elts) if h]
    if names:
        ex.obs.declared_agents.append(
            DeclaredAgent(name=assigned_to or fname, kind="group_chat", handoffs=names, var=assigned_to, code_ref=ref)
        )


def _tool_body_delegation(ex: ProjectExtractor) -> None:
    """A tool whose body runs another agent (`await research_agent.run(q)`, `writer.invoke(...)`) is
    delegation: the agent holding the tool calls that agent and gets its answer back (PydanticAI agent
    delegation, and the same pattern in any framework)."""
    holders: dict[str, list[DeclaredAgent]] = {}
    for ag in ex.obs.declared_agents:
        for t in ag.tools:
            holders.setdefault(t, []).append(ag)
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            tool = ex.node_tool.get(id(n)) or ex._method_owner.get(id(n))
            if tool is None or not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for c in ast.walk(n):
                if not (
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr in _RUN_CALLS
                    and isinstance(c.func.value, ast.Name)
                ):
                    continue
                target = ex.agent_vars.get(c.func.value.id)
                if target is None:
                    continue
                for owner in holders.get(tool, []):
                    if owner is not target and target.var and target.var not in owner.delegates:
                        owner.delegates.append(target.var)


def _find_run_receivers(ex: ProjectExtractor) -> set[str]:
    """Names holding something whose call runs a whole agent loop (agent / graph / team / runner)."""
    names = set(ex.agent_vars) | {"Runner"}
    builders: set[str] = set()
    for defs in ex._fn_index.values():
        for _fc, fn in defs:
            if any(
                isinstance(r, ast.Return)
                and isinstance(r.value, ast.Call)
                and _name(r.value.func) in _RUN_BUILDERS | ex.agent_ctors
                for r in ast.walk(fn)
            ):
                builders.add(fn.name)
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            if (
                isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call)
                and _name(n.value.func) in _RUN_BUILDERS | ex.agent_ctors | builders
            ):
                names |= {_name(t) for t in n.targets if isinstance(t, (ast.Name, ast.Attribute))}
    return names | builders


def _awaits_run(ex: ProjectExtractor, e: ast.AST, depth: int = 0) -> bool:
    """The awaited expression runs an agent: directly (`graph.ainvoke(...)`) or through a project
    coroutine whose body does (`async def go(): ... runner.run_async(...)`; `wait_for(go(), 90)`)."""
    for c in ast.walk(e):
        if not isinstance(c, ast.Call):
            continue
        if ex._runs_agent(c):
            return True
        if depth < 2:
            for _fc, fn in ex._fn_index.get(_name(c.func), [])[:2]:
                if any(ex._awaits_run(x, depth + 1) for x in fn.body):
                    return True
    return False


def _runs_agent(ex: ProjectExtractor, call: ast.Call) -> bool:
    """`graph.ainvoke(...)`, `Runner.run(...)`, `crew.kickoff()`, `self.app.invoke(...)`, `build().invoke()`."""
    if _name(call.func) not in _RUN_CALLS or not isinstance(call.func, ast.Attribute):
        return False
    recv = call.func.value
    if isinstance(recv, ast.Call):
        return _name(recv.func) in ex._run_receivers | _RUN_BUILDERS
    return _name(recv) in ex._run_receivers
