"""Project-wide signals: secrets, models, env files, tracing, structured output, helper data flows.

Moved out of ``extractor.py`` unchanged; each function takes the extractor as ``ex``.
"""

from __future__ import annotations

import ast
import re
from agenticreq.ast.classify import is_pinned
from agenticreq.ast.classify import is_test_file
from agenticreq.ast.classify import looks_like_model
from agenticreq.ast.classify import provider_from
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.ast.symbols import FileCtx
from agenticreq.ast.symbols import FuncDef
from agenticreq.models.finding import SourceRef
from agenticreq.models.observed import ModelRef
from agenticreq.models.observed import SecretRef
from agenticreq.models.observed import UnresolvedModel
from pathlib import Path
from typing import Callable
from agenticreq.ast.extract_support import (
    _AUDIT_METHODS,
    _CALLBACK_KEYS,
    _CHECKPOINTERS,
    _COST_CALLS,
    _COST_KWARGS,
    _COUNTER_NAME,
    _DURATION_CALLS,
    _DURATION_KWARGS,
    _ENV_CALLS,
    _ENV_FILES,
    _HTTP_CALLS,
    _HTTP_MODULES,
    _KNOWN_TRACERS,
    _LIMIT_NAME,
    _LIVE_SECRET,
    _MAX_DEPTH,
    _NOT_A_SECRET,
    _NOT_STEPS,
    _PLACEHOLDER,
    _PROD_URL,
    _SECRET_NAME,
    _SETTINGS_BASES,
    _STRUCTURED_CALLS,
    _SCHEMA_KWARGS,
    _STRUCTURED_KWARGS,
    _VALIDATE_CALLS,
    _TOKEN_LIMIT_KWARGS,
    _TRACER_NAME,
    _TRACE_CALLS,
    _TRACE_ENV,
    _TRACE_PKGS,
    is_trace_kwarg,
    _host_controlled,
    _in_thread,
    _kw,
    _label,
    _ref,
    _str,
    _url_vars,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor


def _env_file_signals(ex: ProjectExtractor) -> None:
    """Tracing switched on by deployment config (.env / .env.example / langgraph.json), not in code."""
    import json as _json

    seen: set[Path] = set()
    for d in {ex.base, ex.root if ex.root.is_dir() else ex.base}:
        for name in _ENV_FILES:
            f = d / name
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*['\"]?([^'\"#\s]*)", line)
                if m and m.group(1) in _TRACE_ENV and m.group(2).lower() not in ("", "false", "0", "no"):
                    ex.obs.signals.setdefault("audit_hook", []).append(
                        SourceRef(file=str(f.relative_to(ex.base)), line=i)
                    )
        lg = d / "langgraph.json"
        if lg.is_file() and lg not in seen:
            seen.add(lg)
            try:
                cfg = _json.loads(lg.read_text(encoding="utf-8-sig"))
            except ValueError:
                continue
            if isinstance(cfg, dict) and cfg.get("env"):  # the deployment injects an env file
                ex.obs.signals.setdefault("env_read", []).append(SourceRef(file=str(lg.relative_to(ex.base)), line=1))


def _helper_flows(ex: ProjectExtractor) -> None:
    """Follow tools into the project functions they call (depth 2), and look at graph nodes and
    orchestrators: a URL that reaches an HTTP call through a helper, a broad except that swallows an
    error on the agent's path."""
    url_fns = ex._url_param_functions()
    tool_fns: list[tuple[FileCtx, FuncDef, str]] = []
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            if id(n) not in ex.node_tool:
                continue
            name = ex.node_tool[id(n)]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tool_fns.append((fc, n, name))
            elif isinstance(n, ast.ClassDef):  # its run method, and __init__ (tools often load data there)
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in (
                        "_run",
                        "_arun",
                        "forward",
                        "run",
                        "__call__",
                        "invoke",
                        "__init__",
                    ):
                        tool_fns.append((fc, m, name))
    # 1. URL through a helper: VisitTool.forward(url) -> self.browser.visit_page(url) -> requests.get(url)
    for _fc, fn, name in tool_fns:
        t = ex.obs.declared_tools.get(name)
        if t is None or "url_from_arg" in t.features:
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
        tainted = _url_vars(fn, params)
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and _name(c.func) in url_fns:
                positions = url_fns[_name(c.func)]
                if any(i < len(c.args) and _host_controlled(c.args[i], tainted) for i in positions):
                    t.features = sorted({*t.features, "url_from_arg", "http"})
                    break
    # 2. fail-open on the agent's path: helpers of tools (depth 2), graph nodes, orchestrators
    on_path: dict[int, tuple[FileCtx, FuncDef]] = {}
    frontier = [(fc, fn) for fc, fn, _ in tool_fns]
    for _ in range(2):
        nxt = []
        for _fc, fn in frontier:
            for c in ast.walk(fn):
                if isinstance(c, ast.Call):
                    for dfc, dfn in ex._fn_index.get(_name(c.func), [])[:3]:
                        if id(dfn) not in on_path and id(dfn) not in ex.node_tool:
                            on_path[id(dfn)] = (dfc, dfn)
                            nxt.append((dfc, dfn))
        frontier = nxt
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            if isinstance(n, ast.Call) and _name(n.func) == "add_node":  # LangGraph node functions
                for a in n.args[:2]:
                    for dfc, dfn in ex._fn_index.get(_name(a), [])[:1]:
                        on_path.setdefault(id(dfn), (dfc, dfn))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(c, ast.Call) and ex._runs_agent(c) for c in ast.walk(n)
            ):
                on_path.setdefault(id(n), (fc, n))  # an orchestrator that runs an agent
    tool_ids = {id(fn) for _, fn, _ in tool_fns}
    for fc, fn in on_path.values():
        if id(fn) in tool_ids or is_test_file(fc.rel):
            continue
        for h in ast.walk(fn):
            if isinstance(h, ast.ExceptHandler) and ex._fails_open(h):
                ex.obs.signals.setdefault("fail_open_path", []).append(_ref(fc.rel, h))


def _url_param_functions(ex: ProjectExtractor) -> dict[str, set[int]]:
    """function name -> positions of parameters that decide the host of an HTTP call inside it
    (directly, or through another such function), unless the function checks an allowlist."""
    out: dict[str, set[int]] = {}
    for _ in range(3):
        changed = False
        for name, defs in ex._fn_index.items():
            for _fc, fn in defs[:3]:
                pos = [a.arg for a in fn.args.args if a.arg not in ("self", "cls")]
                for i, p in enumerate(pos):
                    if i in out.get(name, set()):
                        continue
                    tainted = _url_vars(fn, {p})
                    hit = False
                    for c in ast.walk(fn):
                        if not isinstance(c, ast.Call):
                            continue
                        cn = _name(c.func)
                        owner = _name(c.func.value) if isinstance(c.func, ast.Attribute) else ""
                        if cn in _HTTP_CALLS and (owner in _HTTP_MODULES or cn == "urlopen"):
                            first = c.args[0] if c.args else _kw(c, "url")
                            hit = first is not None and _host_controlled(first, tainted)
                        elif cn in out and cn != name:
                            hit = any(j < len(c.args) and _host_controlled(c.args[j], tainted) for j in out[cn])
                        if hit:
                            break
                    if hit and ex._egress_guarded(fn, {p}) != "dominating":
                        out.setdefault(name, set()).add(i)
                        changed = True
        if not changed:
            break
    return out


def _static_str(ex: ProjectExtractor, fc: FileCtx, node: ast.AST | None, depth: int = 0) -> str | None:
    """A string known without running code: literals, constants (across imports), ``+`` and f-strings
    of those. Anything dynamic -> None."""
    if node is None or depth > _MAX_DEPTH:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, (ast.Name, ast.Attribute)):
        r = ex.symbols.resolve(fc, node)
        return ex._static_str(r.fc, r.node, depth + 1) if r is not None and r.kind == "const" else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = ex._static_str(fc, node.left, depth + 1), ex._static_str(fc, node.right, depth + 1)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            part = ex._static_str(fc, v.value if isinstance(v, ast.FormattedValue) else v, depth + 1)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _instance_field_default(ex: ProjectExtractor, fc: FileCtx, node: ast.AST) -> tuple[str, str] | None:
    """``settings.model`` where ``settings = Settings()`` and the class declares ``model: str = "gpt-4.1-..."``:
    (default, label). The deployment may override it (pydantic-settings reads the environment), so it is assumed."""
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
        return None
    inst = ex.symbols.resolve(fc, node.value)
    if inst is None or inst.kind != "const" or not isinstance(inst.node, ast.Call):
        return None
    cls = ex.symbols.resolve(inst.fc, inst.node.func)
    if cls is None or cls.kind != "class" or not isinstance(cls.node, ast.ClassDef):
        return None
    for st in cls.node.body:
        value = None
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == node.attr:
            value = st.value
        elif isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == node.attr for t in st.targets):
            value = st.value
        if value is not None:
            if isinstance(value, ast.Call) and _name(value.func) == "Field":
                value = _kw(value, "default") or (value.args[0] if value.args else None)
            got = ex._static_str(cls.fc, value)
            return (got, f"default of {cls.node.name}.{node.attr}") if got else None
    return None


def _model_from_node(
    ex: ProjectExtractor,
    fc: FileCtx,
    node: ast.AST,
    ref: SourceRef,
    provider: str | None = None,
    call: ast.Call | None = None,
) -> None:
    val = ex._static_str(fc, node) or ex._default_of(fc, node, call)
    assumed: str | None = None
    if not val:
        field = _instance_field_default(ex, fc, node)
        if field is not None:
            val, assumed = field
    if not val and call is not None:  # os.getenv("MODEL", "gpt-4o-2024-08-06"): the code's own default, assumed
        a = ex._env_assumed(fc, node)
        if a is not None and isinstance(a.value, str) and looks_like_model(a.value):
            val, assumed = a.value, a.label
    if not val:  # a value in a static config file: CFG["model"] of yaml.safe_load(open("models.yml"))
        val = ex.ev.string(ex.ev.frame_of(fc, node), node)
    if not val and isinstance(node, ast.Attribute):  # a settings object's default: settings.model
        from agenticreq.ast.constants import plain_field, setting_text

        val = setting_text(ex.obs.root, node.attr)
        assumed = (
            None if val is None or plain_field(ex.obs.root, node.attr) else f"default in code of settings.{node.attr}"
        )
    if not val and call is not None and (why := _runtime_model_source(ex, fc, node)):
        if not any(u.code_ref == ref for u in ex.obs.unresolved_models):
            ex.obs.unresolved_models.append(UnresolvedModel(code_ref=ref, reason=why))
    if val and looks_like_model(val) and not any(m.name == val and m.code_ref == ref for m in ex.obs.models):
        ex.obs.models.append(
            ModelRef(
                name=val,
                pinned=is_pinned(val),
                code_ref=ref,
                assumed_from=assumed,
                provider=provider or provider_from(model=val),
            )
        )


_CONFIG_BASES = {"settings", "config", "cfg", "conf", "args", "env", "environ", "os", "params", "options", "opts"}


def _runtime_model_source(ex: ProjectExtractor, fc: FileCtx, node: ast.AST, depth: int = 0) -> str | None:
    """Why a `model=` value cannot be read from the code (env var, config, helper function), else None. A bare
    name is left alone: it is usually a model *object* built elsewhere, which its own constructor reports."""
    if depth > 3:
        return None
    if isinstance(node, ast.Subscript):
        return f"`{ast.unparse(node)[:60]}` is looked up at run time"
    if isinstance(node, ast.Call):
        fn = _name(node.func) or ""
        text = ast.unparse(node)
        reads_env = "environ" in text or "getenv" in text
        # Model(...) is a constructor. get_model() / os.getenv() decide at run time; behind a variable only an env
        # read is certain (`m = make_model()` is most likely a model object, which its own constructor reports)
        if fn and not fn[:1].isupper() and (depth == 0 or reads_env):
            return f"`{ast.unparse(node)[:60]}` decides the model at run time"
        return None
    if isinstance(node, ast.Attribute):
        base = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        b = (base.id if isinstance(base, ast.Name) else "").lower()
        if b in _CONFIG_BASES or b.endswith(("settings", "config")):
            return f"`{ast.unparse(node)[:60]}` is read from configuration at run time"
        return None
    if isinstance(node, ast.Name):
        r = ex.symbols.resolve(fc, node)
        if r is not None and r.kind == "const" and isinstance(r.node, ast.expr):
            return _runtime_model_source(ex, r.fc, r.node, depth + 1)
    return None


def _provider_of_expr(ex: ProjectExtractor, fc: FileCtx, v: ast.AST, depth: int = 0) -> str | None:
    """Provider of a client / provider value: a call, a variable holding one, or a name like "groq"."""
    if depth > 3:
        return None
    if isinstance(v, ast.Call):
        return ex._provider_of(fc, v, _name(v.func))
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return provider_from(model=v.value + ":") or provider_from(hint=v.value)
    if isinstance(v, ast.Name):
        for n in ast.walk(fc.tree):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == v.id for t in n.targets):
                return ex._provider_of_expr(fc, n.value, depth + 1)
    return None


def _provider_of(ex: ProjectExtractor, fc: FileCtx, call: ast.Call, fname: str) -> str | None:
    """ChatGroq(...) / ChatOpenAI(base_url="https://api.groq.com/...") / base_url=GROQ_BASE_URL."""
    from urllib.parse import urlparse

    found = provider_from(cls=fname)
    if found:
        return found
    for k in ("provider", "openai_client", "client", "async_client"):
        # PydanticAI GroqModel(..., provider=GroqProvider()); agents SDK
        # OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(base_url=<groq>))
        v = _kw(call, k)
        if v is not None:
            found = ex._provider_of_expr(fc, v)
            if found:
                return found
    for k in ex._BASE_URL_KWARGS:
        v = _kw(call, k)
        if v is None:
            continue
        url = ex._static_str(fc, v)
        if url:
            found = provider_from(host=urlparse(url).hostname or url)
        if not found:  # env-var / constant NAME: os.environ["GROQ_BASE_URL"], GROQ_BASE_URL
            hint = " ".join(
                _name(x) if isinstance(x, (ast.Name, ast.Attribute)) else str(x.value)
                for x in ast.walk(v)
                if isinstance(x, (ast.Name, ast.Attribute))
                or (isinstance(x, ast.Constant) and isinstance(x.value, str))
            )
            found = provider_from(hint=hint)
        if found:
            return found
    return None


def _fold_str(node: ast.AST | None) -> str | None:
    """A string built only from literals: ``"sk-" + "live-" + "abc"`` (an assembled secret is still hard-coded)."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _fold_str(node.left), _fold_str(node.right)
        return a + b if a is not None and b is not None else None
    return None


def _secret_and_model_assign(ex: ProjectExtractor, fc: FileCtx, node: ast.Assign | ast.AnnAssign) -> None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    value = node.value
    if isinstance(value, ast.Call) and _name(value.func) == "Field":  # research_model: str = Field(default=...)
        value = _kw(value, "default") or (value.args[0] if value.args else None)
    val = _str(value)
    if val is None and isinstance(value, (ast.BinOp, ast.JoinedStr)):  # assembled from parts
        val = ex._static_str(fc, value) or _fold_str(value)
        if val is not None and _LIVE_SECRET.match(val):
            ex.obs.secrets.append(SecretRef(name="<literal>", code_ref=_ref(fc.rel, node), preview=val[:6] + "…"))
            return
    if val is None:
        return
    for t in targets:  # model ids configured under any *model* name, not only model=/llm= kwargs
        tname = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else "")
        ref = _ref(fc.rel, node)
        if (
            "model" in tname.lower()
            and looks_like_model(val)
            and not any(m.name == val and m.code_ref == ref for m in ex.obs.models)
        ):
            ex.obs.models.append(ModelRef(name=val, pinned=is_pinned(val), code_ref=ref))
    for t in targets:
        tname = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else "")
        if (
            _SECRET_NAME.search(tname)
            and len(val) >= 12
            and not _PLACEHOLDER.search(val)
            and " " not in val
            and not _LIVE_SECRET.match(val)
            and not _NOT_A_SECRET.search(val)
        ):
            ex.obs.secrets.append(SecretRef(name=tname, code_ref=_ref(fc.rel, node), preview=val[:4] + "…"))


#: objects that stop a run after a wall-clock time (AutoGen TimeoutTermination(seconds))
_DURATION_CONSTRUCTS = {"TimeoutTermination"}
_TRAIL_VERBS = ("save", "write", "dump", "log", "record", "persist", "store", "export", "archive")
_TRAIL_NOUNS = (
    "transcript",
    "message",
    "history",
    "conversation",
    "trace",
    "audit",
    "event",
    "interaction",
    "session",
    "run",
    "result",
    "output",
    "response",
    "step",
)
#: calls that put their argument somewhere durable (a file, a log, a store)
_PERSIST_CALLS = {
    "dump",
    "write",
    "writelines",
    "write_text",
    "info",
    "debug",
    "warning",
    "error",
    "log",
    "save_state",
    "save",
    "put_item",
    "insert_one",
    "insert",
    "execute",
    "add_message",
}
_TRAIL_ARG = re.compile(r"\b(messages?|transcript|chat_history|conversation|history|trace|events?)\b")


def _is_audit_trail_call(ex: ProjectExtractor, fc: FileCtx, node: ast.Call) -> bool:
    """The code stores what happened in a run (a transcript, the messages, a trace): by what the call does, not
    by the name of a tracing library. Either a project function named like `save_transcript` whose body (or its
    helpers) writes or logs, or a persisting call (json.dump, logger.info, f.write ...) handed the messages."""
    from agenticreq.ast import body_analysis

    fname = _name(node.func)
    low = fname.lower()
    if any(v in low for v in _TRAIL_VERBS) and any(n in low for n in _TRAIL_NOUNS):
        r = ex.symbols.resolve(fc, node.func)
        if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            effects = set(body_analysis._body_features(r.node)) | ex._callee_effects(r.fc, r.node)
            return bool(effects & {"fs_write", "logs"})
    if fname in _PERSIST_CALLS:
        text = " ".join(ast.unparse(a) for a in [*node.args, *(k.value for k in node.keywords)])[:400]
        return bool(_TRAIL_ARG.search(text))
    return False


def _signals(ex: ProjectExtractor, fc: FileCtx) -> None:
    """Project facts that no single tool owns: where config comes from, tracing, run-wide caps."""
    sig = ex.obs.signals

    def add(key: str, node: ast.AST) -> None:
        sig.setdefault(key, []).append(_ref(fc.rel, node))

    if fc.imports & _TRACE_PKGS:  # an import alone traces nothing: weak evidence, never a PASS by itself
        add("audit_import", fc.tree.body[0] if fc.tree.body else fc.tree)
    used = ex._use_index(fc)
    agents_sdk = "agents" in fc.imports
    for node in ast.walk(fc.tree):
        if (
            isinstance(node, ast.Compare)
            and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
        ):
            a, b = _label(node.left), _label(node.comparators[0])
            if (
                (_COUNTER_NAME.search(a) and _LIMIT_NAME.search(b))
                or (_COUNTER_NAME.search(b) and _LIMIT_NAME.search(a))
            ) and not (_NOT_STEPS.search(a) or _NOT_STEPS.search(b)):
                add("loop_counter", node)  # the project bounds its own loop
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Subscript) and _name(t.value) == "environ" and _str(t.slice) in _TRACE_ENV
                for t in node.targets
            )
            and (_str(node.value) or "").lower() not in ("", "false", "0")
        ):
            add("audit_hook", node)  # os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Subscript) and _str(t.slice) == "OPENAI_AGENTS_DISABLE_TRACING" for t in node.targets
            )
            and (_str(node.value) or "").lower() in ("1", "true", "yes")
        ):
            add("tracing_disabled", node)  # os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
        if isinstance(node, ast.Subscript) and _str(node.slice) == "configurable":
            add("env_read", node)  # LangGraph: config["configurable"] is injected per deployment
        if (
            isinstance(node, (ast.With, ast.AsyncWith))
            and any(
                isinstance(it.context_expr, ast.Call) and _name(it.context_expr.func) in _DURATION_CALLS
                for it in node.items
            )
            and any(isinstance(c, ast.Call) and ex._runs_agent(c) for b in node.body for c in ast.walk(b))
        ):
            add("duration_bound", node)  # async with asyncio.timeout(60): await graph.ainvoke(...)
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            add("env_read", node)
        elif isinstance(node, ast.ClassDef) and {_name(b) for b in node.bases} & _SETTINGS_BASES:
            add("env_read", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _PROD_URL.match(node.value):
            add("prod_endpoint", node)
        elif isinstance(node, ast.Dict):  # LangGraph / LangChain: config={"callbacks": [...], ...}
            items = {_str(k): v for k, v in zip(node.keys, node.values) if k is not None}
            cb = items.get("callbacks") or items.get("callback_manager")
            if any(is_trace_kwarg(k) and k not in _CALLBACK_KEYS for k in items) or (
                cb is not None and ex._traces(fc, cb)
            ):
                add("audit_hook", node)
            # a budget key in a dict is only a value: whether the run enforces it is not visible
            if set(items) & _COST_KWARGS and used(node):
                add("cost_bound_key", node)
            if set(items) & _DURATION_KWARGS:
                add("duration_bound_key", node)
        if not isinstance(node, ast.Call):
            continue
        fname = _name(node.func)
        kwargs = {k.arg for k in node.keywords if k.arg}
        if (
            fname in _ENV_CALLS
            or (fname == "config" and "decouple" in fc.imports)
            or fname in ("from_runnable_config", "ensure_config", "get_config")
        ):
            add("env_read", node)
        if (
            fname in ("setdefault", "update")
            and isinstance(node.func, ast.Attribute)
            and _name(node.func.value) == "environ"
            and any(_str(a) in _TRACE_ENV for a in node.args[:1])
        ):
            add("audit_hook", node)
        if agents_sdk and fname in ("trace", "custom_span", "agent_span", "gen_trace_id"):
            add("agents_trace", node)  # OpenAI Agents SDK spans (count only while tracing is enabled)
        if (
            fname == "set_tracing_disabled"
            and node.args
            and literal(node.args[0])[1] is True
            or (kwargs & {"tracing_disabled"} and literal(_kw(node, "tracing_disabled"))[1] is True)
        ):
            add("tracing_disabled", node)
        final = fname in ex.agent_ctors or fname in ("Task", "create_react_agent", "create_agent")
        if fname in _STRUCTURED_CALLS and (fname != "parse" or _kw(node, "response_format") is not None):
            add("structured_output_call", node)  # types one model call, not shown to be the final answer
            # the schema: with_structured_output(Plan), PydanticOutputParser(pydantic_object=Plan),
            # PydanticToolsParser(tools=[A, B]), Report.model_validate_json(raw). Frameworks send it as a tool.
            schema_args = [k.value for k in node.keywords if k.arg in _SCHEMA_KWARGS]
            if fname in _VALIDATE_CALLS:  # the receiver is the schema, the argument is the model's text
                schema_args += [node.func.value] if isinstance(node.func, ast.Attribute) else []
            elif fname != "parse":
                schema_args += node.args[:1]
            for arg in schema_args:
                for e in arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]:
                    if isinstance(e, (ast.Name, ast.Attribute)) and _name(e) not in ex.obs.output_types:
                        ex.obs.output_types.append(_name(e))
        for k in kwargs & _STRUCTURED_KWARGS:
            val = _kw(node, k)
            if val is None:
                continue
            ok, v = literal(val)
            if ok and v in (None, False, "text"):
                continue
            elts = val.elts if isinstance(val, (ast.List, ast.Tuple)) else [val]
            if any(isinstance(e, ast.Name) and e.id == "str" for e in elts):
                continue  # output_type=str / [str, DeferredToolRequests]: the model may answer in free text
            add("structured_output" if final else "structured_output_call", node)
            for e in elts:
                if isinstance(e, (ast.Name, ast.Attribute)) and _name(e) not in ex.obs.output_types:
                    ex.obs.output_types.append(_name(e))
        cb = next((k.value for k in node.keywords if k.arg in _CALLBACK_KEYS), None)
        if (
            fname in _TRACE_CALLS
            or any(is_trace_kwarg(k) and k not in _CALLBACK_KEYS for k in kwargs)
            or (cb is not None and ex._traces(fc, cb))
        ):
            add("audit_hook", node)
        if (fname in _COST_CALLS or kwargs & _COST_KWARGS) and not (
            fname == "UsageLimits" and not kwargs & _TOKEN_LIMIT_KWARGS
        ):
            # a budget object only caps spend once it is handed to the run: TokenBudget(...) built and
            # dropped caps nothing. A cost kwarg on an agent / a run call applies directly.
            builds_object = fname in _COST_CALLS or (fname[:1].isupper() and fname not in ex.agent_ctors)
            if not builds_object or used(node):
                add("cost_bound", node)
        if kwargs & _DURATION_KWARGS or fname == "alarm" or fname in _DURATION_CONSTRUCTS:
            add("duration_bound", node)
        if not is_test_file(fc.rel) and _is_audit_trail_call(ex, fc, node):
            add("audit_trail", node)
        elif fname == "wait_for" and node.args and not _in_thread(node.args[0]) and ex._awaits_run(node.args[0]):
            add("duration_bound", node)  # asyncio.wait_for(graph.ainvoke(...), 60): the RUN is bounded
            # (wait_for(to_thread(...)) / future.result(timeout=) return control but the worker thread
            # keeps running and spending: not a run-wide time limit)
        elif "timeout" in kwargs and (fname in ex.agent_ctors or ex._runs_agent(node)):
            add("duration_bound", node)  # a timeout on a scraper / retriever / HTTP call bounds nothing
        if "checkpointer" in kwargs:
            ok, v = literal(_kw(node, "checkpointer"))
            if not (ok and v in (None, False)):
                add("checkpointer", node)
        elif fname in _CHECKPOINTERS:
            add("checkpointer_ctor", node)  # built; not shown to be passed to compile(checkpointer=)


def _use_index(fc: FileCtx) -> Callable[[ast.AST], bool]:
    """`used(node)`: is this value handed on — passed to a call, returned, or assigned to a name
    (or attribute) that is later passed or returned?"""
    arg_ids: set[int] = set()
    names: set[str] = set()
    for n in ast.walk(fc.tree):
        roots: list[ast.AST] = []
        if isinstance(n, ast.Call):
            roots = [*n.args, *(k.value for k in n.keywords)]
        elif isinstance(n, (ast.Return, ast.Yield)) and n.value is not None:
            roots = [n.value]
        for r in roots:
            for x in ast.walk(r):
                arg_ids.add(id(x))
                if isinstance(x, (ast.Name, ast.Attribute)):
                    names.add(_name(x))
    assigned: dict[int, list[str]] = {}
    for n in ast.walk(fc.tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            assigned[id(n.value)] = [_name(t) for t in targets if isinstance(t, (ast.Name, ast.Attribute))]

    for n in ast.walk(fc.tree):  # config = {"callbacks": [TokenBudget(5)]}; graph.invoke({}, config)
        if (
            isinstance(n, (ast.Assign, ast.AnnAssign))
            and n.value is not None
            and any(t in names for t in assigned.get(id(n.value), []))
        ):
            arg_ids.update(id(x) for x in ast.walk(n.value))

    def used(node: ast.AST) -> bool:
        return id(node) in arg_ids or any(t in names for t in assigned.get(id(node), []))

    return used


def _traces(ex: ProjectExtractor, fc: FileCtx, value: ast.AST) -> bool:
    """Does a `callbacks=` value contain something that records tool / agent activity?
    A handler that only counts tokens (on_llm_end) is a budget, not an audit trail."""
    elts: list[ast.AST] = list(value.elts) if isinstance(value, (ast.List, ast.Tuple, ast.Set)) else [value]
    for el in elts:
        cls: str | None = None
        if isinstance(el, ast.Call):
            cls = _name(el.func)
        elif isinstance(el, ast.Name):  # handler = MyHandler(...); callbacks=[handler]
            cls = next(
                (
                    _name(n.value.func)
                    for n in ast.walk(fc.tree)
                    if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call)
                    and any(isinstance(t, ast.Name) and t.id == el.id for t in n.targets)
                ),
                None,
            )
        if cls is None:
            return True  # cannot tell statically: do not claim the hook is missing
        if cls in _KNOWN_TRACERS:
            return True
        local = next((f.classes[cls] for f in ex.files if cls in f.classes), None)
        if local is not None:
            methods = {m.name for m in local.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            bases = {_name(b) for b in local.bases}
            if methods & _AUDIT_METHODS or bases & _KNOWN_TRACERS:
                return True
            continue  # a project handler that never sees a tool / agent event
        if _TRACER_NAME.search(cls):
            return True
    return False


def _framework(ex: ProjectExtractor) -> None:
    from agenticreq.ast.framework import classify_execution_model, detect_framework

    from agenticreq.ast.framework import declared_dists, framework_imports

    imports = framework_imports(
        ((fc.path, fc.tree) for fc in ex.files), lambda mod, path: ex.symbols.find_module(mod, path) is not None
    )
    detect_framework(ex.obs, imports, ex.base, declared=declared_dists(ex.base, ex.symbols.project_root))
    classify_execution_model(ex.obs)
