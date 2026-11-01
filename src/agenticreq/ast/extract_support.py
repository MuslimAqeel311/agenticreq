"""Constants, patterns and small pure helpers shared by the extractor modules.

Moved out of ``extractor.py`` unchanged.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import TypeGuard

from agenticreq.ast.classify import BOUND_PARAMS
from agenticreq.ast.classify import canonical
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.models.finding import SourceRef


#: decorators that make a function a tool (``@tool``, ``@agent.tool``, ``@kernel_function`` ...)
TOOL_DECORATORS = {
    "tool",
    "function_tool",
    "tool_plain",
    "mcp_tool",
    "tool_def",
    "kernel_function",
    "register_for_llm",
    "ai_function",
    "llm_tool",
}
TOOL_BASE_CLASSES = {"BaseTool", "Tool", "StructuredTool", "Toolkit"}
#: callables that wrap a Python function into a tool: ``W(f)``, ``W(**opts)(f)``, ``W(func=f)``
TOOL_WRAPPERS = {
    "tool",
    "function_tool",
    "from_function",
    "from_defaults",
    "StructuredTool",
    "Tool",
    "FunctionTool",
    "kernel_function",
    "register_function",
    "ai_function",
    "llm_tool",
}
#: keyword names a wrapper uses for the wrapped callable
_FUNC_KWARGS = ("func", "fn", "function", "coroutine", "callable", "f")
#: any other keyword that is function-shaped by name (async_fn, run_fn, handler, tool_callable ...)
FUNC_KW_PATTERN = re.compile(
    r"^(a?fn|func[a-z_]*|function[a-z_]*|callable[a-z_]*|coroutine[a-z_]*|handler)$|"
    r"_(fn|func|function|callable|coroutine|handler)$",
    re.IGNORECASE,
)
AGENT_CTORS = {
    "Agent",
    "CodeAgent",
    "ToolCallingAgent",
    "MultiStepAgent",
    "create_react_agent",
    "ReActAgent",
    "AssistantAgent",
    "ConversableAgent",
    "FunctionAgent",
    "create_tool_calling_agent",
    "UserProxyAgent",
    "CodeExecutorAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "CodeActAgent",
    "AgentExecutor",
    "initialize_agent",
    "create_agent",
    "LlmAgent",
    "ReAct",
    "ChatAgent",
    "create_openai_tools_agent",
    "create_openai_functions_agent",
    "create_structured_chat_agent",
    "create_supervisor",
    "create_swarm",
    "OpenAIAgent",
    "ChatCompletionAgent",
    "ClaudeAgentOptions",
}
TOOL_LIST_KWARGS = {"tools", "functions", "allowed_tools", "plugins"}
#: keyword arguments through which a framework receives tools that exist only at run time
DYNAMIC_TOOL_KWARGS = {
    "mcp_servers",
    "mcps",
    "mcp_tools",
    "mcp_clients",
    "mcp_config",
    "toolsets",
    "toolset",
    "toolkit",
    "toolkits",
    "tool_registry",
    "tool_provider",
    "tool_providers",
    "tool_sources",
    "tool_servers",
    "workbench",
    "workbenches",
}
#: constructors / loaders (matched in lower case) that fetch tools from outside the code. Anything with "mcp" in
#: its name counts, except the server side (FastMCP ...), which defines tools instead of receiving them.
DYNAMIC_TOOL_CALLS = {
    "toolbox",
    "toolboxclient",
    "toolboxtoolset",
    "toolcollection",
    "from_hub",
    "get_tools",
    "aget_tools",
    "get_tools_async",
    "list_tools",
    "alist_tools",
    "toolspec",
    "to_tool_list",
}
MCP_SERVER_SIDE = {"fastmcp", "mcpserver", "create_sdk_mcp_server", "mcp_server", "run_mcp_server"}
#: calls whose first argument is a tool list (or a single tool) bound to a model / agent
BIND_CALLS = {
    "bind_tools",
    "bind_functions",
    "ToolNode",
    "ToolExecutor",
    "with_tools",
    "add_tools",
    "add_tool",
    "register_tools",
    "add_plugin",
    "add_plugins",
}
BOUND_KWARGS = {
    "recursion_limit": "recursion_limit",
    "max_iter": "max_iter",
    "max_iterations": "max_iterations",
    "max_turns": "max_turns",
    "max_steps": "max_steps",
    "max_retries": "max_retries",
    "max_execution_time": "max_execution_time",
    "max_rpm": "max_rpm",
    "timeout": "timeout",
    "max_tool_calls": "max_tool_calls",
    "request_limit": "request_limit",
    "tool_calls_limit": "tool_calls_limit",
    "max_consecutive_auto_reply": "max_turns",
    "max_tool_iterations": "max_iterations",
    "max_llm_calls": "max_steps",
    # other frameworks' spelling of the same bound: AutoGen GroupChat / MaxMessageTermination, Haystack Agent,
    # DSPy ReAct, Semantic Kernel termination strategy and function-choice behaviour, Agno
    "max_round": "max_turns",
    "max_messages": "max_turns",
    "max_agent_steps": "max_steps",
    "max_iters": "max_iterations",
    "maximum_iterations": "max_iterations",
    "maximum_auto_invoke_attempts": "max_iterations",
    "max_auto_invoke_attempts": "max_iterations",
    "tool_call_limit": "tool_calls_limit",
}
#: constructors whose first positional argument is a step bound: MaxMessageTermination(10)
BOUND_CALLS = {"MaxMessageTermination": "max_turns"}
GATE_CALLS = {
    "interrupt": "interrupt",
    "request_approval": "approval_call",
    "ask_human": "human_input",
    "get_human_approval": "approval_call",
    "require_approval": "approval_call",
    "confirm": "approval_call",
}
#: a call that asks a person, recognised by WHAT it says it does (ask_user_to_confirm, request_human_signoff,
#: get_approval, confirm_with_user ...), not only by the few exact names above
HUMAN_GATE_CALL = re.compile(
    r"^(ask|request|get|require|wait_for|obtain|seek|prompt|await)_[a-z_]*(human|user|approv[a-z]*|confirm[a-z_]*|consent|"
    r"permission|sign_?off)[a-z_]*$|^(confirm|approve|authori[sz]e)_[a-z_]+$|^human_[a-z_]+$|"
    r"^[a-z_]*_(approval|confirmation|consent)$",
    re.IGNORECASE,
)
GATE_KWARGS = {
    "interrupt_before": "interrupt_before",
    "interrupt_after": "interrupt_after",
    "human_input": "human_input",
    "needs_approval": "needs_approval",
    "input_guardrails": "input_guardrail",
    "output_guardrails": "output_guardrail",
    "requires_approval": "needs_approval",
    "require_approval": "needs_approval",
    "require_confirmation": "needs_approval",
    "tool_input_guardrails": "input_guardrail",
    "step_callback": "step_callback",
    "human_input_mode": "human_input",
    "requires_confirmation": "needs_approval",
}
#: any other keyword that names a before-tool hook or an approval policy (Google ADK before_tool_callback, pre_tool_hook,
#: tool_approval_policy, confirm_fn ...): a callable may block the call, but whether it does is not provable (unverified)
GATE_HOOK_KWARG = re.compile(
    r"^(before|pre)_?(tool|function|action)\w*$|^(tool|function|action)_?(approval|confirm\w*|gate|guard\w*)\w*$|"
    r"^(approval|confirm\w*|human_?approv\w*|hitl)\w*$",
    re.IGNORECASE,
)
#: keyword / dict keys that take tools OUT of the model's choice (Claude Agent SDK disallowed_tools, Semantic Kernel
#: excluded_functions ...): the code enforces the prohibition there
EXCLUDE_KEY = re.compile(
    r"^(excluded?|disallowed|blocked?|denied|forbidden|blacklist(ed)?|deny(_?list)?)_?(tools?|functions?|actions?)$|"
    r"^(tools?|functions?|actions?)_?(excluded?|blocked?|denied|blacklist|denylist)$|^(disallowed|denylist|blacklist)$",
    re.IGNORECASE,
)
#: approval kwargs that belong to one tool (on its decorator / wrapper), never to the whole agent
_TOOL_SCOPED_GATES = {"needs_approval", "requires_approval", "requires_confirmation", "require_confirmation"}
GATE_DECORATORS = {
    "input_guardrail": "input_guardrail",
    "output_guardrail": "output_guardrail",
    "tool_input_guardrail": "input_guardrail",
    "requires_approval": "needs_approval",
}
MODEL_KWARGS = {"model", "model_id", "model_name", "llm", "deployment_name", "azure_deployment"}

_LIVE_SECRET = re.compile(
    r"^(?:(?:Bearer|Token|Basic)\s+)?"  # also inside an Authorization header value
    r"(sk-(proj|ant|live|svcacct)-[\w-]{8,}|sk-[A-Za-z0-9]{32,}|sk_live_\w{8,}|rk_live_\w{8,}|AKIA[0-9A-Z]{16}|"
    r"ghp_\w{20,}|gho_\w{20,}|github_pat_\w{20,}|xox[abpr]-[\w-]{10,}|AIza[0-9A-Za-z_-]{30,}|glpat-[\w-]{16,}|"
    r"gsk_[A-Za-z0-9]{40,}|hf_[A-Za-z0-9]{30,}|xai-[A-Za-z0-9]{40,}|pplx-[A-Za-z0-9]{40,}|r8_[A-Za-z0-9]{30,}|"
    r"csk-[A-Za-z0-9]{30,}|fw_[A-Za-z0-9]{20,})$"
)
#: values that name where a secret goes, not the secret: token_url = "https://...", key_path = "/etc/..."
_NOT_A_SECRET = re.compile(
    r"^(https?|wss?|file|s3|gs)://|^[./~][\w./-]*$|^[A-Z][A-Z0-9_]*$|\.(json|pem|key|txt|ya?ml)$"
)
_SECRET_NAME = re.compile(r"(api_?key|secret|token|password|passwd)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"(xxx|your|placeholder|changeme|dummy|fake|example|test|<|\$\{|TODO|\.\.\.)", re.IGNORECASE)
_EXEC_CALLS = {
    "exec",
    "eval",
    "compile",
    "system",
    "popen",
    "run",
    "call",
    "check_output",
    "check_call",
    "Popen",
    "spawn",
    "execv",
    "execve",
}
_EXEC_MODULES = {"subprocess", "os", "pty"}
_HTTP_CALLS = {"get", "post", "put", "delete", "request", "urlopen", "fetch", "head", "patch"}
_HTTP_MODULES = {"requests", "httpx", "urllib", "request", "aiohttp", "session", "client"}
_MAX_DEPTH = 8
#: packages whose `tool` decorator builds an agent-side tool (vs an MCP server's)
_AGENT_TOOL_PKGS = {"langchain_core", "langchain", "crewai", "smolagents", "agents", "pydantic_ai", "strands", "needle"}
#: parameters browser-use's action registry injects (never sent to the model)
_BROWSER_USE_INJECTED = {
    "browser",
    "browser_context",
    "browser_session",
    "page",
    "page_extraction_llm",
    "available_file_paths",
    "has_sensitive_data",
    "sensitive_data",
    "context",
}
#: calls that mint a fresh identifier: a retry gets a new one, so the side effect repeats
_FRESH_ID_CALLS = {
    "uuid4",
    "uuid1",
    "token_hex",
    "token_urlsafe",
    "randint",
    "choices",
    "choice",
    "getrandbits",
    "random",
    "urandom",
    "ObjectId",
    "cuid",
    "nanoid",
}
#: calls that add a record to something outside the function
_RECORD_WRITES = {
    "insert",
    "insert_one",
    "insert_many",
    "add",
    "add_all",
    "append",
    "extend",
    "aput",
    "put",
    "create",
    "bulk_create",
    "send",
    "publish",
    "enqueue",
    "push",
    "rpush",
    "lpush",
    "xadd",
}


#: ---- Level-2 project signals (collected by ``_signals``)
#: keyword arguments that ask the model for a typed / schema-validated answer
_STRUCTURED_KWARGS = {
    "response_format",
    "output_type",
    "output_pydantic",
    "output_json",
    "result_type",
    "output_schema",
    "response_model",
    "text_format",
    "output_cls",
    "output_parser",
    "structured_output",
    "response_schema",
    "output_content_type",
}
_STRUCTURED_CALLS = {
    "with_structured_output",
    "PydanticOutputParser",
    "JsonOutputParser",
    "StructuredOutputParser",
    "JsonOutputToolsParser",
    "PydanticToolsParser",
    "LLMTextCompletionProgram",
    "parse",
    # validating the model's JSON text against a schema: Report.model_validate_json(raw)
    "model_validate_json",
    "validate_json",
    "parse_raw",
}
#: structured-output calls whose receiver is the schema (Report.model_validate_json(raw))
_VALIDATE_CALLS = {"model_validate_json", "validate_json", "parse_raw"}
#: keywords of a structured-output call that name the schema (with_structured_output(schema=Plan))
_SCHEMA_KWARGS = {"schema", "pydantic_object", "tools", "output_cls", "output_type", "response_format"}
#: tracing / observability packages: importing one is an audit hook for every LLM and tool call
_TRACE_PKGS = {
    "langsmith",
    "langfuse",
    "opentelemetry",
    "logfire",
    "phoenix",
    "openinference",
    "arize",
    "agentops",
    "monocle_apptrace",
    "langtrace_python_sdk",
    "mlflow",
    "weave",
    "wandb",
    "traceloop",
    "openlit",
    "helicone",
    "braintrust",
    "lmnr",
    "opik",
    "openllmetry",
    "literalai",
    "portkey_ai",
}
_TRACE_KWARGS = {
    "callbacks",
    "callback_manager",
    "step_callback",
    "task_callback",
    "tracer",
    "trace_processors",
    "tracing",
    "instrument",
    "event_handlers",
}
_TRACE_KWARG_PATTERN = re.compile(
    r"^(step|tool|agent|run|task|event|node|llm|model)_?(callbacks?|hooks?|handlers?|listeners?|observers?|processors?)$|"
    r"^trace_[a-z_]+$|^(tracers?|tracing|telemetry)$",
    re.IGNORECASE,
)


def is_trace_kwarg(name: str | None) -> bool:
    """A keyword that wires an audit / trace hook (step_callbacks=, on_event_hook=, tracer=, trace_processors= ...)."""
    if not name:
        return False
    return name in _TRACE_KWARGS or bool(_TRACE_KWARG_PATTERN.match(name))


_TRACE_CALLS = {
    "add_trace_processor",
    "set_trace_processors",
    "set_tracing_export_api_key",
    "instrument",
    "setup_monocle_telemetry",
    "register_otel",
    "instrument_openai",
    "instrument_pydantic_ai",
    "set_global_handler",
    "configure_tracing",
}
_CALLBACK_KEYS = {"callbacks", "callback_manager"}
#: agent constructors whose first positional argument is the agent's name (AutoGen)
_NAMED_FIRST_ARG = {"AssistantAgent", "UserProxyAgent", "ConversableAgent", "CodeExecutorAgent"}
#: multi-agent groups: every participant takes turns / is selected by a coordinator (AutoGen teams,
#: LlamaIndex AgentWorkflow, CrewAI hierarchical crews, AutoGen 0.2 GroupChat)
_GROUP_CTORS = {
    "RoundRobinGroupChat",
    "SelectorGroupChat",
    "Swarm",
    "MagenticOneGroupChat",
    "GroupChat",
    "AgentWorkflow",
    "Crew",
}
#: environment switches that turn tracing on when set in code / a .env file
_TRACE_ENV = {
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_TRACES_EXPORTER",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
}
_ENV_FILES = (".env", ".env.example", ".env.sample", ".env.template", "example.env")
#: a project's own loop guard: `if state.loop_step < cfg.max_loops`, `iterations >= max_researcher_iterations`
_COUNTER_NAME = re.compile(r"(step|iter|loop|turn|round|attempt|calls|count|hops|depth)", re.IGNORECASE)
_LIMIT_NAME = re.compile(r"(max|limit|budget|allowed|cap)", re.IGNORECASE)
#: a cap on tokens / money / time / size is not a cap on loop steps
_NOT_STEPS = re.compile(r"(token|cost|usd|price|spend|second|time|duration|byte|char|length|size)", re.IGNORECASE)
#: calls that hand a run to an agent / graph / team: timeouts around these bound the run
_RUN_BUILDERS = {
    "compile",
    "create_react_agent",
    "create_agent",
    "create_supervisor",
    "create_swarm",
    "AgentExecutor",
    "Crew",
    "AgentWorkflow",
    "RoundRobinGroupChat",
    "SelectorGroupChat",
    "Swarm",
    "GroupChatManager",
    "InMemoryRunner",
    "Runner",
}
_SANDBOX_EXECUTORS = {"docker", "e2b", "modal", "wasm", "blaxel", "azure", "remote"}
#: callback handlers that record runs: LangChain / Langfuse / Arize / MLflow / W&B / Opik tracers
_KNOWN_TRACERS = {
    "LangChainTracer",
    "ConsoleCallbackHandler",
    "FileCallbackHandler",
    "StdOutCallbackHandler",
    "LangfuseCallbackHandler",
    "CallbackHandler",
    "OpenInferenceTracer",
    "ArizeCallbackHandler",
    "MlflowCallbackHandler",
    "WandbTracer",
    "OpikTracer",
    "BaseTracer",
    "EvaluatorCallbackHandler",
    "LLMonitorCallbackHandler",
    "AgentOpsCallbackHandler",
}
#: handler methods that see tool / agent / chain events (an audit trail), not just token counts
_AUDIT_METHODS = {
    "on_tool_start",
    "on_tool_end",
    "on_tool_error",
    "on_agent_action",
    "on_agent_finish",
    "on_chain_start",
    "on_chain_end",
    "on_chain_error",
}
_TRACER_NAME = re.compile(r"(Tracer|Tracing|Langfuse|Observ|Audit|Monitor|Telemetry|Logger)", re.IGNORECASE)
#: keyword arguments that cap money / tokens for a whole run (max_tokens alone caps one response, not a run)
_COST_KWARGS = {
    "max_cost",
    "max_cost_usd",
    "max_budget",
    "max_budget_usd",
    "budget",
    "budget_usd",
    "cost_limit",
    "max_price",
    "token_budget",
    "total_tokens_limit",
    "request_tokens_limit",
    "response_tokens_limit",
    "input_tokens_limit",
    "output_tokens_limit",
    "max_total_tokens",
    "max_tokens_per_run",
    "max_spend",
    "max_total_token",
}
#: UsageLimits(request_limit=...) caps steps, not spend: it is a cost cap only with one of these
_TOKEN_LIMIT_KWARGS = {
    "total_tokens_limit",
    "request_tokens_limit",
    "response_tokens_limit",
    "input_tokens_limit",
    "output_tokens_limit",
}
_COST_CALLS = {"UsageLimits", "BudgetManager", "TokenBudget", "CostLimit", "TokenUsageTermination"}
#: keyword arguments that cap wall-clock time; `timeout=` only counts on an agent / a run call
_DURATION_KWARGS = {
    "max_execution_time",
    "max_time",
    "time_limit",
    "deadline",
    "max_duration",
    "max_duration_s",
    "execution_timeout",
    "run_timeout",
    "max_wall_time",
}
_RUN_CALLS = {
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
    "run_stream",
    "execute",
    "query",
    "aquery",
    "run_async",
    "run_live",
}
_DURATION_CALLS = {"wait_for", "timeout", "alarm", "timeout_after", "move_on_after", "fail_after"}
_CHECKPOINTERS = {
    "MemorySaver",
    "InMemorySaver",
    "SqliteSaver",
    "AsyncSqliteSaver",
    "PostgresSaver",
    "AsyncPostgresSaver",
    "RedisSaver",
    "AsyncRedisSaver",
    "MongoDBSaver",
    "DynamoDBSaver",
}
_ENV_CALLS = {
    "getenv",
    "load_dotenv",
    "dotenv_values",
    "find_dotenv",
    "get_secret",
    "get_secret_value",
    "SecretClient",
    "get_parameter",
}
_SETTINGS_BASES = {"BaseSettings", "Settings"}
_PROD_URL = re.compile(r"^https?://[^/\s]*\b(prod|production|live)\b[^\s]*$", re.IGNORECASE)
_LOG_CALLS = {"info", "warning", "error", "exception", "debug", "critical", "log"}
#: an except handler that mentions the failure is a reported error, not a swallowed one
_ERROR_WORDS = re.compile(r"error|fail|exception|could not|couldn't|unable|denied|invalid|refus", re.IGNORECASE)
_ERROR_TARGET = re.compile(r"(error|err$|exc|trace|fail)", re.IGNORECASE)
_RETRY_NAME = re.compile(r"(attempt|retr|tries|try_count|failures)", re.IGNORECASE)
#: a call named like a validator: is_allowed(url), validate_url(url), _check_host(host), ensure_safe(url)
_GUARD_NAME = re.compile(
    r"^(?!check_output$|check_call$)\w*(allow|whitelist|permit|valid|verif|check|safe|guard|"
    r"restrict|trusted)",
    re.IGNORECASE,
)
_ALLOWLIST_NAME = re.compile(r"(allow|whitelist|permit|trusted|approved|safe|domain|host|origin)", re.IGNORECASE)
#: a literal that names a host: "api.x.com", ".corp.com", "https://api.x.com/" (not "http", not "https://")
_HOSTISH = re.compile(r"^(https?://)?\.?[a-z0-9-]+(\.[a-z0-9-]+)+(:\d+)?(/.*)?$", re.IGNORECASE)
#: assigning to these attributes from inside a tool rewrites the agent's own instructions / capabilities
_SELF_ATTRS = {
    "system_prompt",
    "instructions",
    "system_message",
    "backstory",
    "guardrails",
    "tools",
    "prompt_template",
    "system_template",
}


#: a literal that fixes the host: "https://api.x.com/", "https://api.x.com", "api.x.com/v1"
_FIXED_HOST = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?[^/{}\s:@]+\.[^/{}\s:@]+(?::\d+)?(?:/|$)", re.IGNORECASE)
_URL_WRAPPERS = {"Request", "URL", "str", "quote", "strip", "rstrip", "lstrip", "geturl"}


def _in_thread(e: ast.AST) -> bool:
    """asyncio.to_thread(...) / loop.run_in_executor(...): the work runs on a thread no timeout can stop."""
    return any(isinstance(c, ast.Call) and _name(c.func) in ("to_thread", "run_in_executor") for c in ast.walk(e))


def _label(e: ast.AST) -> str:
    """A readable name for a value: `x`, `cfg.max_loops`, `config["configurable"]["max_loops"]`, `len(x)`."""
    if isinstance(e, ast.Subscript):
        key = _str(e.slice)
        return key if key else _label(e.value)
    if isinstance(e, ast.Call):
        return _name(e.func) + " " + " ".join(_label(a) for a in e.args[:1])
    return _name(e)


def _host_controlled(e: ast.AST, tainted: set[str]) -> bool:
    """Does a model-supplied value decide the HOST this expression points at?

    ``url``, ``u`` (= url), ``Request(url)``, ``f"https://{host}/x"`` -> yes.
    ``f"{BASE}/search?q={query}"``, ``"https://api.x.com/" + path`` -> no: the host is fixed, only
    the path / query varies, which is not an egress risk.
    """
    if isinstance(e, ast.Name):
        return e.id in tainted
    if isinstance(e, (ast.Attribute, ast.Subscript)):
        return _host_controlled(e.value, tainted)
    if isinstance(e, ast.Constant):
        return False
    if isinstance(e, ast.JoinedStr):
        if not e.values:
            return False
        head = e.values[0]
        if isinstance(head, ast.Constant):  # f"https://{host}/..." vs f"https://api.x.com/{p}"
            return not _FIXED_HOST.match(str(head.value)) and any(
                _host_controlled(v.value, tainted) for v in e.values[1:2] if isinstance(v, ast.FormattedValue)
            )
        return isinstance(head, ast.FormattedValue) and _host_controlled(head.value, tainted)
    if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Mod)):
        left = e.left
        if isinstance(left, ast.Constant):
            return not _FIXED_HOST.match(str(left.value)) and _host_controlled(e.right, tainted)
        return _host_controlled(left, tainted)
    if isinstance(e, ast.Call):
        name = _name(e.func)
        if name == "urljoin" and len(e.args) == 2:  # an absolute second argument replaces the host
            return _host_controlled(e.args[0], tainted) or _host_controlled(e.args[1], tainted)
        if name == "format" and isinstance(e.func, ast.Attribute) and isinstance(e.func.value, ast.Constant):
            tpl = str(e.func.value.value)
            return tpl.startswith("{") and bool(e.args) and _host_controlled(e.args[0], tainted)
        if name in _URL_WRAPPERS:
            target = e.args[0] if e.args else (e.func.value if isinstance(e.func, ast.Attribute) else None)
            return target is not None and _host_controlled(target, tainted)
    return False


def _url_vars(fn: ast.AST, params: set[str]) -> set[str]:
    """Parameters plus locals whose host they decide: ``u = url``, ``req = Request(url)``."""
    tainted = set(params)
    for _ in range(4):
        before = len(tainted)
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and _host_controlled(n.value, tainted):
                tainted |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        if len(tainted) == before:
            break
    return tainted


_SQL_SELECT = re.compile(r"^\s*select\b", re.IGNORECASE)
_SQL_BOUNDED = re.compile(r"\b(limit|top|fetch\s+first|rownum)\b|\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


def _sql_text(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(str(v.value) if isinstance(v, ast.Constant) else " ? " for v in node.values)
    return None


def _unbounded_read(call: ast.Call) -> bool:
    """A read whose result size nothing caps, whatever the tool is called:
    ``cur.execute("SELECT * FROM t")`` (no LIMIT / aggregate), ``session.query(X).all()``."""
    name = _name(call.func)
    if name in ("execute", "read_sql", "read_sql_query", "text", "sql") and call.args:
        sql = _sql_text(call.args[0])
        return bool(sql and _SQL_SELECT.match(sql) and not _SQL_BOUNDED.search(sql))
    if (
        name == "all"
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and _name(call.func.value.func) in ("query", "filter", "filter_by", "select", "scalars", "where")
    ):
        return True
    return False


#: request parameters that cap how many results an API returns (compared without case / separators)
_RESULT_CAPS = {canonical(p).replace("_", "") for p in BOUND_PARAMS} | {
    "hitsperpage",
    "rows",
    "size",
    "maxrecords",
    "maxhits",
    "maxresults",
    "pagesize",
    "perpage",
    "topk",
    "numresults",
    "resultsperpage",
    "maxitems",
    "take",
    "top",
    "srlimit",
    "rvlimit",
}


def _caps_results(node: ast.AST | None) -> bool:
    """A dict literal passed to a call that caps the result count: ``{"limit": 1}``, ``{"hitsPerPage": n}``."""
    return isinstance(node, ast.Dict) and any(
        isinstance(k, ast.Constant) and isinstance(k.value, str) and canonical(k.value).replace("_", "") in _RESULT_CAPS
        for k in node.keys
    )


#: helper names that say nothing about which agent they return (`build()`, `create_agent()`)
_GENERIC_BUILDER = re.compile(r"^_|^(build|create|make|get|new|init|setup|main|run|agent)(_agents?)?$")


def _ref(file: str, node: ast.AST) -> SourceRef:
    return SourceRef(file=file, line=getattr(node, "lineno", None), column=(getattr(node, "col_offset", 0) or 0) + 1)


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _str(node: ast.AST | None) -> str | None:
    ok, v = literal(node)
    return v if ok and isinstance(v, str) else None


class _Subst(ast.NodeTransformer):
    """Replace a comprehension variable with one concrete element (for ``[tool(f) for f in (a, b)]``)."""

    def __init__(self, var: str, value: ast.expr) -> None:
        self.var, self.value = var, value

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.var:
            return ast.copy_location(copy.deepcopy(self.value), node)
        return node


def _subst_all(node: ast.AST, env: dict[str, ast.expr]) -> ast.AST:
    out = copy.deepcopy(node)
    for var, value in env.items():
        out = _Subst(var, value).visit(out)
    return out


def _appends_to(node: ast.AST, var: str) -> TypeGuard[ast.Call]:
    """``var.append(x)`` / ``var.extend(xs)``"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("append", "extend")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == var
        and bool(node.args)
    )


def _merge_groups(groups: list[set[str]]) -> list[list[str]]:
    """Groups that share a tool are one context (a list later passed to bind_tools, ...)."""
    merged: list[set[str]] = []
    for g in groups:
        hit = [m for m in merged if m & g]
        merged = [m for m in merged if not (m & g)] + [set(g).union(*hit)]
    return sorted(sorted(m) for m in merged)
