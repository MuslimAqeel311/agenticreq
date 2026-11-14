"""`agenticreq mocks`: write mocks.yml from the code, so `agenticreq mock-run` needs no flags.

What it reads (the same extraction as audit / init) and what it writes:

* ``entry``  the function that runs the agent: a public module-level function that calls the agent
  (``.invoke(...)``, ``Runner.run(...)``, ``.kickoff(...)`` ...). A ``main()`` that parses the command line is
  skipped: the mock run calls the entry itself. Not found -> ``TODO`` with a comment saying how to fill it.
* ``args``   an input for each required parameter of the entry. A parameter used to pick an agent from a dict
  (``TOOLSETS[name]``) gets one of the dict's keys, preferring an agent whose tools need no outside service.
* ``order``  every tool found. ``tools`` one entry per tool: a read-only tool gets sample ``args``; a tool that
  writes, pays, sends or deletes gets ``patch:`` + ``returns:`` so its real function never runs; a tool listed
  under ``prohibited:`` in spec.yml gets ``call: false``.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml

from agenticreq.models.observed import ObservedAgent, ToolSchema

#: calls that run an agent (method or function name)
_RUN_CALLS = frozenset(
    {
        "invoke",
        "ainvoke",
        "run",
        "arun",
        "run_sync",
        "run_async",
        "kickoff",
        "kickoff_async",
        "stream",
        "astream",
        "chat",
        "achat",
        "initiate_chat",
        "generate_reply",
        "run_stream",
        "query",
        "aquery",
        "get_response",  # Semantic Kernel agents
        "invoke_async",  # Strands
        "stream_async",
        "forward",  # DSPy modules
        "acall",
    }
)
_NAME_HINTS = ("run", "ask", "agent", "chat", "answer", "invoke", "handle", "respond", "query", "solve")
#: parameters that hold a parsed command line, never a request
_CLI_PARAMS = frozenset({"args", "argv", "ns", "namespace", "opts", "options", "cli"})
_TEXT_PARAMS = frozenset(
    {
        "task",
        "request",
        "query",
        "question",
        "prompt",
        "message",
        "input",
        "text",
        "q",
        "user_input",
        "goal",
        "topic",
        "instruction",
        "msg",
        "content",
    }
)
#: modules a tool may use and still need no outside service
_OFFLINE = frozenset(
    {
        "json",
        "math",
        "re",
        "datetime",
        "statistics",
        "random",
        "decimal",
        "fractions",
        "itertools",
        "functools",
        "collections",
        "typing",
        "dataclasses",
        "numpy",
        "np",
        "pandas",
        "pd",
        "operator",
        "string",
        "textwrap",
        "uuid",
        "time",
        "enum",
        "pydantic",
    }
)
SAMPLE_TEXT = "What can you help me with?"


@dataclass
class EntryGuess:
    entry: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    file: str = ""


def _module_name(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _calls_agent(fn: ast.AST, agent_vars: frozenset[str] = frozenset()) -> bool:
    """The function runs an agent: a run call (invoke, kickoff, run_sync ...), a provider SDK call, or the agent
    object called directly (Strands ``agent(task)``, a DSPy module ``react(question=q)``)."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in _RUN_CALLS or _provider_sdk_call(f):
                return True
            if isinstance(f, ast.Name) and f.id in agent_vars:
                return True
    return False


def _provider_sdk_call(f: ast.expr) -> bool:
    """A plain SDK loop calls the model itself: ``client.chat.completions.create(...)``, ``client.messages.create``,
    ``client.responses.create``, ``client.models.generate_content``."""
    if not isinstance(f, ast.Attribute):
        return False
    owner = f.value.attr if isinstance(f.value, ast.Attribute) else ""
    return (f.attr == "create" and owner in ("completions", "messages", "responses")) or (
        f.attr.startswith("generate_content") and owner == "models"
    )


def _reads_command_line(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in ("parse_args", "argv"):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
            return True
    return False


def _required(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    pos = [*fn.args.posonlyargs, *fn.args.args]
    req = pos[: len(pos) - len(fn.args.defaults)]
    req += [a for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is None]
    return [a for a in req if a.arg not in ("self", "cls")]


def _ann(a: ast.arg) -> str:
    return ast.unparse(a.annotation) if a.annotation is not None else ""


def _score(fn: ast.FunctionDef | ast.AsyncFunctionDef, agent_vars: frozenset[str] = frozenset()) -> int | None:
    if fn.name.startswith("_") or not _calls_agent(fn, agent_vars) or _reads_command_line(fn):
        return None
    if any(_ann(a).endswith("State") for a in _required(fn)):
        return None  # a graph node (LangGraph `def agent(state: MessagesState)`): the graph calls it, not a run
    score = 5
    if any(h in fn.name.lower() for h in _NAME_HINTS):
        score += 2
    for a in _required(fn):
        t = _ann(a)
        if a.arg in _CLI_PARAMS:
            score -= 4
        elif t in ("", "str") and (a.arg in _TEXT_PARAMS or t == "str"):
            score += 1
        elif t not in ("", "str"):
            score -= 2  # a state dict / object the user has to build
    return score


def _module_dicts(tree: ast.Module) -> dict[str, dict[str, ast.expr]]:
    """Module-level ``NAME = {"key": value, ...}`` dicts with string keys."""
    out: dict[str, dict[str, ast.expr]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            keys = {
                k.value: v
                for k, v in zip(node.value.keys, node.value.values)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if keys:
                out[node.targets[0].id] = keys
    return out


def _dict_param(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    param: str,
    dicts: dict[str, dict[str, ast.expr]],
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    depth: int = 0,
) -> str | None:
    """The dict a parameter picks from (``TOOLSETS[name]``), directly or in a same-module helper it is passed to
    (``build_agent(name)``)."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in dicts
            and isinstance(node.slice, ast.Name)
            and node.slice.id == param
        ):
            return node.value.id
    if depth >= 2:
        return None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            callee = funcs[node.func.id]
            names = [a.arg for a in (*callee.args.posonlyargs, *callee.args.args)]
            for i, a in enumerate(node.args):
                if isinstance(a, ast.Name) and a.id == param and i < len(names):
                    hit = _dict_param(callee, names[i], dicts, funcs, depth + 1)
                    if hit:
                        return hit
    return None


def _call_targets(fn: ast.AST) -> set[str]:
    return {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _tool_modules_used(tree: ast.Module) -> dict[str, set[str]]:
    """{function name: modules it reads (``yf.Ticker`` -> yfinance)}, following same-module helpers once."""
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    imported: dict[str, str] = {}  # alias -> top-level module name
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update({(a.asname or a.name).split(".")[0]: a.name.split(".")[0] for a in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.update({a.asname or a.name: node.module.split(".")[0] for a in node.names})

    # module variables built from an imported client: engine = create_engine(URL), client = OpenAI()
    built: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            base = f
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in imported:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        built[t.id] = imported[base.id]

    def direct(fn: ast.AST) -> set[str]:
        out = {
            imported[n.value.id]
            for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in imported
        }
        # only the body: a decorator (@tool from crewai / langchain_core) is the framework, not a service the tool uses
        body = [n for st in getattr(fn, "body", []) for n in ast.walk(st)]
        for n in body:  # the module's engine / client, or text(...) from `from sqlalchemy import text`
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and built.get(n.id) in _SERVICE_MODULES:
                out.add(built[n.id])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and imported.get(n.func.id) in _SERVICE_MODULES:
                out.add(imported[n.func.id])
        return out

    out: dict[str, set[str]] = {}
    for name, fn in funcs.items():
        used = direct(fn)
        for helper in _call_targets(fn) & set(funcs):
            used |= direct(funcs[helper])
        out[name] = used
    return out


#: the database / HTTP client libraries whose names (a bare imported call, a module variable built from one) mean
#: the tool reaches a service; an attribute use (``yf.Ticker``) keeps the older, wider rule
_SERVICE_MODULES = frozenset(
    {
        "sqlalchemy",
        "oracledb",
        "cx_Oracle",
        "psycopg2",
        "psycopg",
        "asyncpg",
        "pymysql",
        "mysql",
        "pymongo",
        "motor",
        "redis",
        "pyodbc",
        "elasticsearch",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "boto3",
    }
)
#: standard-library modules that do reach the network
_STDLIB_NETWORK = frozenset({"urllib", "http", "socket", "smtplib", "ftplib", "imaplib", "poplib", "telnetlib", "ssl"})


def _local_or_stdlib(root: Path, module: str) -> bool:
    """A module of the project itself, or a standard-library one that stays on this machine."""
    if (root / f"{module}.py").is_file() or (root / module).is_dir():
        return True
    import sys

    names: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())
    return module in names and module not in _STDLIB_NETWORK


def outside_services(obs: ObservedAgent) -> dict[str, list[str]]:
    """{tool: modules it uses that are not plain computation (an API client, an HTTP library)}."""
    out: dict[str, list[str]] = {}
    root = Path(obs.root)
    cache: dict[Path, dict[str, set[str]]] = {}
    for name, t in obs.declared_tools.items():
        if not t.code_ref or not t.func_name:
            continue
        p = Path(t.code_ref.file)
        p = p if p.is_absolute() else root / p
        if p not in cache:
            try:
                cache[p] = _tool_modules_used(ast.parse(p.read_text(encoding="utf-8-sig")))
            except (OSError, SyntaxError, ValueError):
                cache[p] = {}
        mods = sorted(
            m for m in cache[p].get(t.func_name, set()) if m not in _OFFLINE and not _local_or_stdlib(root, m)
        )
        if mods:
            out[name] = mods
    return out


def _tools_of(value: ast.expr, lists: dict[str, list[str]]) -> list[str]:
    if isinstance(value, ast.Name):
        return lists.get(value.id, [])
    if isinstance(value, (ast.List, ast.Tuple)):
        return [e.id for e in value.elts if isinstance(e, ast.Name)]
    return []


def _lists(tree: ast.Module) -> dict[str, list[str]]:
    """Module-level ``NAME = [a, b, ...]`` lists of names (tool lists)."""
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            out[node.targets[0].id] = [e.id for e in node.value.elts if isinstance(e, ast.Name)]
    return out


def _pick_key(
    keys: dict[str, ast.expr], lists: dict[str, list[str]], online: dict[str, list[str]], tool_vars: dict[str, str]
) -> str:
    """The first key whose tools all run offline; else the first key."""
    for key, value in keys.items():
        tools = [tool_vars.get(v, v) for v in _tools_of(value, lists)]
        if tools and not any(t in online for t in tools):
            return key
    return next(iter(keys))


def _load_tree(p: Path) -> ast.Module | None:
    try:
        return ast.parse(p.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, ValueError):
        return None


def _known_lists(tree: ast.Module, root: Path, module: str) -> dict[str, list[str]]:
    """The tool lists of this module and of the sibling modules it imports from (``from .tools import X``)."""
    out = _lists(tree)
    base = root.joinpath(*module.split(".")[:-1])
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            sib = _load_tree(base / (node.module.replace(".", os.sep) + ".py"))
            if sib is not None:
                out = {**_lists(sib), **out}
    return out


def _no_entry_hint(obs: ObservedAgent) -> str:
    """Why no entry was written, and the function to add: the agent object the scan found and how its framework
    runs it (data/entry_calls.yml). A mock run needs a function to call; a module that only builds an agent has none."""
    from agenticreq.data import load_yaml

    agent = next((a for a in obs.declared_agents if a.var), None)
    if agent is None or agent.var is None:
        return (
            "no function that runs the agent was found: name the function that starts one run of your agent "
            "(it calls the model and returns the answer)"
        )
    calls = (load_yaml("entry_calls.yml") or {}).get("frameworks") or {}
    template = calls.get(obs.framework or "")
    call = template.format(agent=agent.var) if template else f"{agent.var}.run(task)"
    where = f" ({agent.code_ref})" if getattr(agent, "code_ref", None) else ""
    return (
        f"no function runs the agent `{agent.var}`{where}: add one, e.g. `def run(task): return {call}`, "
        "and name it here as module:run"
    )


def guess_entry(root: Path, obs: ObservedAgent) -> EntryGuess:
    from agenticreq.ast.symbols import iter_py_files

    best: tuple[int, int, str, ast.FunctionDef | ast.AsyncFunctionDef, ast.Module, Path] | None = None
    agent_vars = frozenset(a.var for a in obs.declared_agents if a.var)
    # a tool (even one that runs another agent, `ask_billing`) is called by the model, never the run's entry
    tool_funcs = {(t.code_ref.file, t.func_name) for t in obs.declared_tools.values() if t.code_ref and t.func_name}
    for i, p in enumerate(iter_py_files(root, include_tests=False)):
        tree = _load_tree(p)
        if tree is None:
            continue
        rel = p.relative_to(root).as_posix()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (rel, node.name) in tool_funcs:
                    continue
                sc = _score(node, agent_vars)
                if sc is None:
                    continue
                key = (-sc, len(p.relative_to(root).parts), str(p.relative_to(root)))
                if best is None or key < (best[0], best[1], best[2]):
                    best = (key[0], key[1], key[2], node, tree, p)
    if best is None:
        return EntryGuess(why=_no_entry_hint(obs))
    _, _, _, fn, tree, p = best
    module = _module_name(root, p)
    entry = f"{module}:{fn.name}"
    dicts = _module_dicts(tree)
    online = outside_services(obs)
    lists = _known_lists(tree, root, module)
    args: dict[str, Any] = {}
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for a in _required(fn):
        d = _dict_param(fn, a.arg, dicts, funcs)
        t = _ann(a)
        if "Path" in t or a.arg.endswith(("_dir", "_path", "workspace", "_root", "_folder")):
            args[a.arg] = MOCK_DIR  # a folder the run may write to, never null for a Path
        elif "None" in t or "Optional" in t:  # session: SQLiteSession | None -> null, never the task text
            args[a.arg] = None
        elif d:
            args[a.arg] = _pick_key(dicts[d], lists, online, obs.tool_vars)
        elif t in ("int",):
            args[a.arg] = 1
        elif t in ("float",):
            args[a.arg] = 1.0
        elif t in ("bool",):
            args[a.arg] = False
        elif t.startswith(("list", "List")):
            args[a.arg] = []
        elif t.startswith(("dict", "Dict")):
            args[a.arg] = {}
        elif t in ("", "str", "Any", "typing.Any"):
            args[a.arg] = SAMPLE_TEXT
        else:
            args[a.arg] = None  # an object (a session, a client): the agent builds its own
    return EntryGuess(
        entry=entry,
        args=args,
        why=f"{fn.name}() in {p.relative_to(root)} calls the agent",
        file=str(p.relative_to(root)),
    )


#: where a mock run's sample files go: agenticreq's own folder, never the project's
MOCK_DIR = ".agenticreq/mock_run"


def _schema(t: ToolSchema) -> dict[str, Any]:
    props = {p.name: {**({"type": p.type} if p.type else {}), **p.constraints} for p in t.params}
    return {"type": "object", "properties": props, "required": [p.name for p in t.params if p.required]}


def _patch_target(root: Path, t: ToolSchema) -> str | None:
    if not t.code_ref or not t.func_name:
        return None
    p = Path(t.code_ref.file)
    p = p if p.is_absolute() else root / p
    try:
        return f"{_module_name(root, p.resolve())}:{t.func_name}"
    except ValueError:
        return None


def _flow(v: Any) -> str:
    """A value as one line of YAML flow style."""
    return yaml.safe_dump(v, default_flow_style=True, sort_keys=False, width=10_000).strip().removesuffix("...").strip()


HEADER = """\
# mocks.yml - what `agenticreq mock-run` runs. Written by `agenticreq mocks`; edit it freely.
# A mock run executes your agent with a canned model (no real LLM, no API key) and these tools.
#
# entry:       the function that runs the agent (module:function or file.py:function)
# args:        the inputs the mock run calls the entry with
# python:      the interpreter; empty = ./.venv when it exists, else the one running agenticreq
# order:       the order the canned model asks for tools; a tool not listed is asked for after these. Under
#              runs:, a run's own order: is its whole path: tools it does not list are not asked for
# tools:       per tool: args = sample arguments; patch + returns = replace the real function, so it never
#              runs (used for tools that write, pay, send or delete); call: false = never ask for it
#              (a tool this run does not need: `delete_file: {call: false}`; not_needed is for spec.yml)
# runs:       optional list of named mock runs, one per task: {name, entry, args, order, tools: {t: {args, call}}}.
#              A run reaches one specialist agent; list a run for each (the top level is the default for all).
# final_text:  the answer the canned model gives once every tool was asked for
# human_input: what an input() approval prompt is answered during the run
# env:         non-secret settings the agent reads from its environment (live-looking keys are refused)
# allow_write: folders (relative to this one) the agent may write to during the run
"""


def render_mocks(
    root: Path, obs: ObservedAgent, prohibited: set[str] | None = None, effects: dict[str, str] | None = None
) -> str:
    """``effects``: tool_effects: from spec.yml, which come before what the code shows."""
    from agenticreq.ast.classify import effect_unsure, is_write_tool
    from agenticreq.jsonschema_util import example_arguments

    from agenticreq.names import norm, norm_set

    prohibited = norm_set(prohibited or set())
    declared = {norm(k): v for k, v in (effects or {}).items()}
    for t in obs.declared_tools.values():
        t.declared_effect = declared.get(norm(t.name), t.declared_effect)  # type: ignore[assignment]
    guess = guess_entry(root, obs)
    lines = [HEADER]
    for where, client, call in connections(root):
        lines.append(
            f"# {where} opens a {client} connection ({call}) when it runs: the mock run blocks it. Point it at a "
            "local stub under env: (its URL / DSN variable), or replace the function that opens it with patch:"
        )
    if guess.entry:
        lines.append(f"entry: {guess.entry}  # {guess.why}")
        lines.append(f"args: {_flow(guess.args)}")
    elif guess.why.startswith("no function runs the agent `"):  # the agent was found: the comment says what to add
        lines.append(f"entry: TODO  # {guess.why}")
        lines.append('args: TODO  # its inputs, e.g. {task: "What can you help me with?"}')
    else:
        lines.append(
            "entry: TODO  # the function that runs your agent, e.g. my_package.agent:run (module:function)"
            f" or agent.py:main; {guess.why}"
        )
        lines.append('args: TODO  # its inputs, e.g. {question: "What can you help me with?"}')
    lines.append("python:")
    names = list(obs.declared_tools)
    lines.append(f"order: {_flow(names)}")
    online = outside_services(obs)
    lines.append("tools:" if names else "tools: {}")
    for name in names:
        t = obs.declared_tools[name]
        key = name if name.replace("_", "").isalnum() else _flow(name)
        if norm(name) in prohibited:
            lines.append(f"  {key}: {{call: false}}  # prohibited in spec.yml: never asked for")
            continue
        target = _patch_target(root, t)
        if is_write_tool(t):
            if target:
                body = {"patch": target, "returns": f"mock result for {name}"}
                lines.append(f"  {key}: {_flow(body)}  # writes: the real function never runs")
            else:
                lines.append(f"  {key}: {{call: false}}  # writes, and no function to replace was found")
            continue
        note = (
            "  # cannot tell whether it writes: add patch + returns, or tool_effects: in spec.yml"
            if effect_unsure(t)
            else "  # read-only"
        )
        if name in online and target and set(online[name]) & _SERVICE_MODULES:
            # a database / HTTP client: the first run must reach nothing, so the function is replaced. Any other
            # library (an agent framework a tool delegates through, an SDK) keeps the comment: patching it would hide
            # the agent behind it
            stubbed: dict[str, Any] = {
                "args": example_arguments(_schema(t)),
                "patch": target,
                "returns": f"mock result for {name}",
            }
            unsure = "cannot tell whether it writes; " if effect_unsure(t) else ""
            lines.append(
                f"  {key}: {_flow(stubbed)}  # {unsure}uses {', '.join(online[name])} (network / database): replaced "
                "so the run reaches nothing; remove patch: and returns: to run the real function"
            )
            continue
        if name in online:
            note += f"; uses {', '.join(online[name])}, which may reach the network: add patch + returns so it does not"
        lines.append(f"  {key}: {_flow({'args': example_arguments(_schema(t))})}{note}")
    lines.append('final_text: "Done."')
    lines.append('human_input: "no"  # never approve a real action during a mock run')
    folders = write_folders(root)
    if folders:
        where = ", ".join(sorted({w for ws in folders.values() for w in ws})[:3])
        lines.append(
            f"allow_write: {_flow([f + '/' for f in folders])}  # the code writes here ({where}); a write "
            "anywhere else is blocked"
        )
    else:
        lines.append(
            "allow_write: []  # folders (relative to this one) the agent may write to during the run; a "
            "write anywhere else is blocked"
        )
    return "\n".join(lines) + "\n"


#: client libraries that open a connection to a database or a remote service, by top-level module
DB_CLIENTS = {
    "sqlalchemy": "SQLAlchemy",
    "oracledb": "Oracle",
    "cx_Oracle": "Oracle",
    "psycopg2": "PostgreSQL",
    "psycopg": "PostgreSQL",
    "asyncpg": "PostgreSQL",
    "pymysql": "MySQL",
    "mysql": "MySQL",
    "pymongo": "MongoDB",
    "motor": "MongoDB",
    "redis": "Redis",
    "pyodbc": "ODBC",
    "elasticsearch": "Elasticsearch",
}
_CONNECT_CALLS = {"connect", "create_engine", "create_async_engine", "MongoClient", "Redis", "from_url", "create_pool"}


def connections(root: Path) -> list[tuple[str, str, str]]:
    """(file:line, client, call) for every place the project opens a database / service connection
    (``create_engine(URL)``, ``oracledb.connect(...)``, ``MongoClient(...)``), in any function or at import."""
    from agenticreq.ast.symbols import iter_py_files

    out: list[tuple[str, str, str]] = []
    for p in iter_py_files(root, include_tests=False):
        tree = _load_tree(p)
        if tree is None:
            continue
        imported: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update({(a.asname or a.name).split(".")[0]: a.name.split(".")[0] for a in node.names})
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.update({a.asname or a.name: node.module.split(".")[0] for a in node.names})
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name not in _CONNECT_CALLS:
                continue
            base = f.value if isinstance(f, ast.Attribute) else f
            while isinstance(base, ast.Attribute):
                base = base.value
            mod = imported.get(base.id, "") if isinstance(base, ast.Name) else ""
            if mod in DB_CLIENTS:
                rel = p.relative_to(root).as_posix()
                out.append((f"{rel}:{node.lineno}", DB_CLIENTS[mod], ast.unparse(node)[:60]))
    return out


def write_folders(root: Path) -> dict[str, list[str]]:
    """Folders of this project the code clearly writes to (folder -> where), read from its write calls:
    ``open("reports/x.md", "w")``, ``Path(OUT) / name).write_text(...)``, ``os.makedirs("logs")``, a path built
    from a constant, a local, ``os.getenv("DIR", "workspace")`` or a project function that returns one."""
    import ast

    from agenticreq.ast.symbols import iter_py_files

    out: dict[str, list[str]] = {}
    for f in iter_py_files(root, include_tests=False):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError, ValueError):
            continue
        rel = f.relative_to(root).as_posix()
        consts = {
            t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
        }
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for fn in funcs.values():
            local = {
                t.id: n.value
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            for n in ast.walk(fn):
                target = _write_target(n)
                if target is None:
                    continue
                folder = _folder_of(target, {**consts, **local}, funcs, 0)
                if folder and not folder.startswith((".", "/")) and ":" not in folder:
                    out.setdefault(folder, []).append(f"{rel}:{getattr(n, 'lineno', '?')}")
    return dict(sorted(out.items()))


def _write_target(n: object) -> object | None:
    """The path expression a write call writes to, or None when ``n`` is not a write."""
    import ast

    if not isinstance(n, ast.Call):
        return None
    f = n.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
    if name == "open" and isinstance(f, ast.Name) and n.args:
        mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax+"):
            return n.args[0]
        return None
    if isinstance(f, ast.Attribute) and name in ("write_text", "write_bytes", "touch"):
        return f.value
    if isinstance(f, ast.Attribute) and name == "open":
        mode = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "mode"), None)
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax+"):
            return f.value
    if name in ("makedirs", "mkdir") and isinstance(f, ast.Attribute):
        return n.args[0] if name == "makedirs" and n.args else f.value if name == "mkdir" else None
    return None


def _folder_of(node: object, names: Mapping[str, object], funcs: Mapping[str, object], depth: int) -> str | None:
    """The first folder of a path expression, when the code spells it out."""
    import ast

    if depth > 4 or node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.replace("\\", "/").strip()
        if "/" in text:
            return text.split("/")[0] or None
        return text if text and "." not in text and "{" not in text else None
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return _folder_of(node.values[0], names, funcs, depth + 1) if "/" in str(node.values[0].value) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _folder_of(node.left, names, funcs, depth + 1)
    if isinstance(node, ast.Call):
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if fname in ("Path", "PurePath", "join", "abspath", "expanduser", "resolve") and node.args:
            return _folder_of(node.args[0], names, funcs, depth + 1)
        if fname in ("expanduser", "resolve", "absolute") and isinstance(f, ast.Attribute):
            return _folder_of(f.value, names, funcs, depth + 1)
        if fname in ("getenv", "get") and len(node.args) >= 2:  # os.getenv("DIR", "workspace")
            return _folder_of(node.args[1], names, funcs, depth + 1)
        fn = funcs.get(fname) if isinstance(f, ast.Name) else None
        if fn is not None:  # a project helper that returns the folder: workspace() -> Path("workspace")
            for r in ast.walk(fn):  # type: ignore[arg-type]
                if isinstance(r, ast.Return):
                    got = _folder_of(r.value, names, funcs, depth + 1)
                    if got:
                        return got
        return None
    if isinstance(node, ast.Name) and node.id in names:
        return _folder_of(names[node.id], names, funcs, depth + 1)
    return None


def detect_python(folder: Path, given: str | None = None) -> tuple[str, str]:
    """(interpreter, where it came from). See :mod:`agenticreq.pyenv` for the order and the sources."""
    from agenticreq.pyenv import choose

    pick = choose(folder, None, given)
    return pick.path, pick.source
