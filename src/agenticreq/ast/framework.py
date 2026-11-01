"""Framework detection and execution-model classification (the first step of every check)."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from agenticreq.models.observed import ObservedAgent

#: import name -> (framework label, distribution name)
FRAMEWORKS: dict[str, tuple[str, str]] = {
    "langgraph": ("langgraph", "langgraph"),
    "crewai": ("crewai", "crewai"),
    "agents": ("openai-agents", "openai-agents"),
    "smolagents": ("smolagents", "smolagents"),
    "pydantic_ai": ("pydantic-ai", "pydantic-ai-slim"),
    "langchain": ("langchain", "langchain"),
    "langchain_core": ("langchain", "langchain-core"),
    "autogen": ("autogen", "pyautogen"),
    "autogen_agentchat": ("autogen", "autogen-agentchat"),
    "autogen_core": ("autogen", "autogen-core"),
    "autogen_ext": ("autogen", "autogen-ext"),
    "ag2": ("autogen", "ag2"),  # the community fork keeps the autogen API under its own import name
    "agent_framework": (
        "microsoft-agent-framework",
        "agent-framework",
    ),  # Microsoft Agent Framework (AutoGen + SK successor)
    "langgraph_supervisor": ("langgraph", "langgraph-supervisor"),
    "langgraph_swarm": ("langgraph", "langgraph-swarm"),
    "crewai_tools": ("crewai", "crewai-tools"),
    "haystack_integrations": ("haystack", "haystack-ai"),
    "llama_index": ("llamaindex", "llama-index-core"),
    "google_adk": ("google-adk", "google-adk"),  # `google.adk` (the top-level `google` is a namespace)
    "agno": ("agno", "agno"),
    "camel": ("camel", "camel-ai"),
    "claude_agent_sdk": ("claude-agent-sdk", "claude-agent-sdk"),
    "dspy": ("dspy", "dspy"),
    "haystack": ("haystack", "haystack-ai"),
    "semantic_kernel": ("semantic-kernel", "semantic-kernel"),
    "strands": ("strands", "strands-agents"),
    "swarm": ("swarm", "swarm"),  # OpenAI's experimental Swarm (`from swarm import Swarm, Agent`)
    "browser_use": ("browser-use", "browser-use"),
    "needle": ("needle", "cactus-needle"),
    "mcp": ("mcp-server", "mcp"),
    "fastmcp": ("mcp-server", "fastmcp"),
    "anthropic": ("anthropic-sdk", "anthropic"),
    "openai": ("openai-sdk", "openai"),
}
#: more specific frameworks win over the generic ones they build on
_PRIORITY = [
    "needle",
    "browser-use",
    "langgraph",
    "crewai",
    "openai-agents",
    "smolagents",
    "pydantic-ai",
    "microsoft-agent-framework",
    "autogen",
    "llamaindex",
    "google-adk",
    "agno",
    "strands",
    "semantic-kernel",
    "claude-agent-sdk",
    "camel",
    "haystack",
    "dspy",
    "swarm",
    "langchain",
    "anthropic-sdk",
    "openai-sdk",
    "mcp-server",
]
SCHEMA_BASED = {
    "mcp-server",
    "needle",
    "browser-use",
    "langgraph",
    "crewai",
    "openai-agents",
    "pydantic-ai",
    "langchain",
    "anthropic-sdk",
    "openai-sdk",
    "autogen",
    "microsoft-agent-framework",
    "llamaindex",
    "google-adk",
    "agno",
    "strands",
    "semantic-kernel",
    "claude-agent-sdk",
    "camel",
    "haystack",
    "swarm",
}  # dspy puts its tools in the prompt: left out, so it never reads as schema-based


#: label -> every distribution that provides it (a framework ships under several names)
DISTS: dict[str, tuple[str, ...]] = {}
for _mod, (_label, _dist) in FRAMEWORKS.items():
    if _dist not in DISTS.get(_label, ()):
        DISTS[_label] = (*DISTS.get(_label, ()), _dist)
for _label, _extra in {
    "autogen": ("autogen",),
    "llamaindex": ("llama-index",),
    "haystack": ("farm-haystack",),
    "dspy": ("dspy-ai",),
    "pydantic-ai": ("pydantic-ai",),
}.items():
    DISTS[_label] = (*DISTS[_label], *(d for d in _extra if d not in DISTS[_label]))
#: dependency-name prefixes: llama-index-llms-openai, agent-framework-azure-ai, ...
_DIST_PREFIX = (("llama-index-", "llamaindex"), ("agent-framework-", "microsoft-agent-framework"))
#: an SDK or protocol library is not an agent framework: declaring it never names the framework
GENERIC = {"anthropic-sdk", "openai-sdk", "mcp-server"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_DIST_LABEL: dict[str, str] = {_norm(d): label for label, dists in DISTS.items() for d in dists}


def dist_label(dist: str) -> str | None:
    n = _norm(dist)
    if n in _DIST_LABEL:
        return _DIST_LABEL[n]
    return next((label for pre, label in _DIST_PREFIX if n.startswith(pre)), None)


# ----------------------------------------------------------------------------- import evidence
#: dotted module -> FRAMEWORKS key, for frameworks that live inside a namespace package (`google` is not one)
_DOTTED = {"google.adk": "google_adk"}
#: names common enough that a project's own package may carry them: a local module of that name is the
#: project's, not the framework's
_AMBIGUOUS = {"agents", "swarm", "camel", "strands", "needle"}
_OPENAI_AGENTS_NAMES = {
    "Agent",
    "Runner",
    "function_tool",
    "handoff",
    "Handoff",
    "ModelSettings",
    "RunContextWrapper",
    "RunConfig",
    "trace",
    "custom_span",
    "set_tracing_disabled",
    "set_default_openai_key",
    "set_default_openai_client",
    "set_default_openai_api",
    "OpenAIChatCompletionsModel",
    "OpenAIResponsesModel",
    "InputGuardrail",
    "OutputGuardrail",
    "input_guardrail",
    "output_guardrail",
    "GuardrailFunctionOutput",
    "WebSearchTool",
    "FileSearchTool",
    "ComputerTool",
    "CodeInterpreterTool",
    "ImageGenerationTool",
    "HostedMCPTool",
    "FunctionTool",
    "ItemHelpers",
    "SQLiteSession",
    "StopAtTools",
    "AgentOutputSchema",
    "RunHooks",
    "AgentHooks",
    "TResponseInputItem",
    "MaxTurnsExceeded",
    "enable_verbose_stdout_logging",
    "add_trace_processor",
}
_OPENAI_AGENTS_SUBMODULES = {
    "mcp",
    "tracing",
    "extensions",
    "models",
    "run",
    "tool",
    "items",
    "handoffs",
    "guardrail",
    "lifecycle",
    "memory",
    "realtime",
    "voice",
    "agent",
    "result",
    "exceptions",
    "run_context",
    "stream_events",
    "function_schema",
    "strict_schema",
    "usage",
    "model_settings",
    "agent_output",
    "computer",
    "tool_guardrails",
}


def _agents_is_openai(mod: str, names: tuple[str, ...], plain: bool) -> bool:
    """`agents` is also what projects call their own package. Without a local module to rule it out,
    only the openai-agents API (names or submodules) names the SDK."""
    parts = mod.split(".")
    if len(parts) > 1:
        return parts[1] in _OPENAI_AGENTS_SUBMODULES
    return plain or "*" in names or bool(set(names) & (_OPENAI_AGENTS_NAMES | _OPENAI_AGENTS_SUBMODULES))


def _dynamic_modules(node: ast.AST) -> list[str]:
    """`importlib.import_module("crewai")` / `__import__("crewai")` with a literal name."""
    if not isinstance(node, ast.Call) or not node.args:
        return []
    f = node.func
    called = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    arg = node.args[0]
    if called in ("import_module", "__import__") and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return [arg.value]
    return []


def framework_imports(files: Iterable[tuple[Path, ast.Module]], is_local: Callable[[str, Path], bool]) -> set[str]:
    """FRAMEWORKS keys the code imports: static `import`/`from`, `from google import adk`, and literal
    dynamic imports. Ambiguous names are dropped when the project has a module of that name."""
    out: set[str] = set()
    for path, tree in files:
        for node in ast.walk(tree):
            hits: list[tuple[str, tuple[str, ...], bool]] = []  # (module, imported names, plain `import x`)
            if isinstance(node, ast.Import):
                hits = [(a.name, (), True) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.level:
                    hits = [(node.module, tuple(a.name for a in node.names), False)]
            else:
                hits = [(m, (), True) for m in _dynamic_modules(node)]
            for mod, names, plain in hits:
                for dotted in (mod, *(f"{mod}.{n}" for n in names)):
                    for prefix, key in _DOTTED.items():
                        if dotted == prefix or dotted.startswith(prefix + "."):
                            out.add(key)
                top = mod.split(".")[0]
                if top not in FRAMEWORKS:
                    continue
                if top in _AMBIGUOUS:
                    if is_local(mod, path):
                        continue
                    if top == "agents" and not _agents_is_openai(mod, names, plain):
                        continue
                out.add(top)
    return out


# ----------------------------------------------------------------------------- declared dependencies
_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([=~<>!@;].*)?$")
_QUOTED_DEP = re.compile(r"""["']\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*((?:[=~<>!@;][^"']*)?)["']""")
_TOML_KEY = re.compile(
    r"""^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*=\s*(?:["']([^"']*)["']|\{[^}]*?version\s*=\s*["']([^"']*)["'])?"""
)
_YAML_DEP = re.compile(r"^\s*-\s*([A-Za-z0-9][\w.-]*)\s*([=<>!~].*)?$")
_MANIFESTS_TXT = ("requirements*.txt", "requirements*.in", "constraints*.txt")
_MANIFESTS_OTHER = ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "environment.yml", "environment.yaml")
_VERSION = re.compile(r"\d+(?:\.\w+)*")


def declared_dists(*roots: Path) -> dict[str, str | None]:
    """{normalised distribution: version spec or None} declared in requirements / pyproject / setup / Pipfile /
    conda files under the given roots. Lock files are ignored: they list every transitive dependency."""
    out: dict[str, str | None] = {}
    seen: set[Path] = set()
    for root in dict.fromkeys(roots):
        files: list[Path] = []
        for pat in _MANIFESTS_TXT:
            files += sorted(root.glob(pat))
        files += sorted((root / "requirements").glob("*.txt")) if (root / "requirements").is_dir() else []
        files += [root / n for n in _MANIFESTS_OTHER]
        for f in files:
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            try:
                text = f.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            is_req = f.suffix in (".txt", ".in")
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                if is_req:
                    m = _REQ_LINE.match(line.split(" #")[0])
                    if m and not line.lstrip().startswith("-"):
                        out.setdefault(_norm(m.group(1)), m.group(2) or None)
                    continue
                for q in _QUOTED_DEP.finditer(line):
                    out.setdefault(_norm(q.group(1)), q.group(2) or None)
                k = _TOML_KEY.match(line)
                if k and f.name in ("pyproject.toml", "Pipfile"):
                    out.setdefault(_norm(k.group(1)), k.group(2) or k.group(3))
                elif f.suffix in (".yml", ".yaml"):
                    y = _YAML_DEP.match(line)
                    if y:
                        out.setdefault(_norm(y.group(1)), y.group(2))
    return out


def _version(label: str, declared: dict[str, str | None], root: Path | None = None) -> tuple[str | None, str | None]:
    """The version the agent runs (see agenticreq.versions): its own environment or lock file first, an exact pin
    next, agenticreq's environment only when it agrees with the project, else the declared range."""
    from agenticreq.versions import agent_version

    return agent_version(root, DISTS.get(label, ()), declared)


def detect_framework(
    obs: ObservedAgent, imports: set[str], root: Path, *, declared: dict[str, str | None] | None = None
) -> None:
    """Framework = what the code imports (static-proven). A framework only declared as a dependency is
    reported too, but marked `dependency` (a guess): it never outranks an imported one and its built-in
    defaults are not assumed."""
    declared = declared if declared is not None else declared_dists(root)
    imported: list[str] = []
    for mod, (label, _dist) in FRAMEWORKS.items():
        if mod in imports and label not in imported:
            imported.append(label)
    dep_only: list[str] = []
    for dist in declared:
        fw = dist_label(dist)
        if fw and fw not in GENERIC and fw not in imported and fw not in dep_only:
            dep_only.append(fw)
    if any(f not in GENERIC for f in imported):
        ordered = [f for f in _PRIORITY if f in imported] + [f for f in _PRIORITY if f in dep_only]
    else:
        ordered = [f for f in _PRIORITY if f in imported or f in dep_only]
    obs.frameworks = ordered
    if ordered:
        obs.framework = ordered[0]
        obs.framework_source = "import" if ordered[0] in imported else "dependency"
        obs.framework_version, obs.framework_version_source = _version(ordered[0], declared, root)


def classify_execution_model(obs: ObservedAgent) -> None:
    code_agents = [a for a in obs.declared_agents if a.kind == "CodeAgent"]
    schema_agents = [a for a in obs.declared_agents if a.kind != "CodeAgent"]
    has_schema_binding = bool(obs.bound_tools) and not set(obs.bound_tools) <= set(obs.prompt_only_tools)
    if code_agents and not schema_agents and not has_schema_binding:
        a = code_agents[0]
        obs.execution_model = "prompt-only"
        obs.execution_model_evidence = (
            f"{a.kind}(...) sends tools as Python signatures inside the system prompt; the request carries tools: []"
        )
        obs.execution_model_ref = a.code_ref
    elif obs.framework in SCHEMA_BASED or schema_agents or has_schema_binding or obs.framework == "smolagents":
        obs.execution_model = "schema-based"
        obs.execution_model_evidence = f"{obs.framework or 'generic'}: tools bound with JSON schemas"
        if code_agents:
            obs.execution_model_evidence += f"; {len(code_agents)} CodeAgent(s) are prompt-only"
            obs.execution_model_ref = code_agents[0].code_ref
    else:
        obs.execution_model = "unknown"
        obs.execution_model_evidence = "no recognised agent framework or tool binding"


def refine_with_wire(obs: ObservedAgent) -> None:
    """Wire evidence: requests with tools: [] and a long system prompt naming declared tools = prompt-only."""
    if not obs.captures:
        return
    with_tools = [c for c in obs.captures if any(not t.name.startswith("hosted:") for t in c.tools)]
    if with_tools:
        if obs.execution_model == "unknown":
            obs.execution_model = "schema-based"
            obs.execution_model_evidence = f"{len(with_tools)} captured request(s) carry tool schemas"
        return
    names = set(obs.declared_tools)
    prompt_hits = [c for c in obs.captures if c.system_prompt and any(n in c.system_prompt for n in names)]
    if prompt_hits:
        c = prompt_hits[0]
        obs.execution_model = "prompt-only"
        obs.execution_model_evidence = (
            f"captured requests send tools: [] while the {len(c.system_prompt or '')}-char "
            "system prompt lists the tools as Python signatures"
        )
