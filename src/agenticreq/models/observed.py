"""ObservedAgent — what the code (AST) and the wire (capture / MCP / traces) say.

Runtime capture is only a *lower bound* on the tool inventory (spike: mean
visible/defined ratio 0.80), so declared (AST) and observed (wire) are kept
separate and diffed in both directions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import SourceRef

ExecutionModel = Literal["schema-based", "prompt-only", "unknown"]
Provider = Literal["openai_chat", "openai_responses", "anthropic", "gemini", "unknown"]
ToolSource = Literal["ast", "wire", "mcp", "trace"]


class ParamSchema(BaseModel):
    name: str
    type: str | None = None  # JSON-schema type name ("string", "integer", ...)
    description: str | None = None
    required: bool = True
    has_default: bool = False
    default: Any = None
    constraints: dict[str, Any] = Field(default_factory=dict)  # minimum / maximum / maxLength / enum


class ToolSchema(BaseModel):
    name: str
    description: str | None = None
    params: list[ParamSchema] = Field(default_factory=list)
    source: ToolSource = "ast"
    code_ref: SourceRef | None = None
    kind: str | None = None  # how it was declared: "@tool", "BaseTool", "Tool(...)", "wire:openai_chat"
    func_name: str | None = None
    features: list[str] = Field(default_factory=list)  # body facts: exec, subprocess, url_from_arg, ...
    body_bounds: dict[str, list[float]] = Field(default_factory=dict)  # arg -> numbers compared against
    #: caps found in the body or a helper it calls (ast/limits.py), as report text: "capped at 200 (counter, a.py:9)"
    limits: list[str] = Field(default_factory=list)
    #: a cap whose value is read at run time ("env MAX_ROWS", "settings.max_rows"): exists, value not known
    limit_runtime: str | None = None
    #: what the body does (ast/effects.py): read | write | unclear, why, and the line; None = no body was read
    effect: Literal["read", "write", "unclear"] | None = None
    effect_reason: str = ""
    effect_line: int | None = None
    #: tool_effects: in spec.yml, the person's declaration (read | write)
    declared_effect: Literal["read", "write"] | None = None
    #: what a framework's built-in tool can do (shell_execution, file_write): data/known_injected_tools.yml
    capability: str | None = None
    raw: dict[str, Any] | None = None

    def param(self, name: str) -> ParamSchema | None:
        return next((p for p in self.params if p.name == name), None)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ToolResult(BaseModel):
    call_id: str | None = None
    name: str | None = None
    content: str = ""


class CapturedCall(BaseModel):
    """One LLM HTTP exchange seen at the client boundary."""

    index: int
    run_id: str = "run-0"
    provider: Provider
    url: str
    client_lib: str = "httpx"  # httpx | httpx2
    model: str | None = None
    system_prompt: str | None = None
    system_prompt_sha256: str | None = None
    tools: list[ToolSchema] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)  # in the response
    tool_results: list[ToolResult] = Field(default_factory=list)  # in the request
    prior_tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant tool calls replayed in the request
    message_count: int = 0
    trace_id: str | None = None
    parent_id: str | None = None
    overhead_ms: float = 0.0
    #: True when tool_calls came from agenticreq's canned responder (discover), not a real model.
    #: Offered tools are real evidence; synthetic tool calls are never evidence of a violation.
    synthetic: bool = False
    timestamp: float | None = None  # unix seconds when the request left the client (rate checks)
    #: human-approval gates that fired after this call's response and before the run's next LLM call
    #: (discover records input(), LangGraph interrupt(), ... in capture/gates.py): the runtime proof of AS-205
    gates_after: list[str] = Field(default_factory=list)
    raw_request: dict[str, Any] | None = None


class ModelRef(BaseModel):
    name: str
    pinned: bool
    code_ref: SourceRef | None = None
    source: Literal["ast", "wire"] = "ast"
    provider: str | None = None  # when recognisable: client class, base_url host / env name, id prefix
    #: set when the id is not a literal but the default of an environment read ("default in code of env MODEL"):
    #: the deployment may override it, so it is an assumption, not a fact
    assumed_from: str | None = None


class UnresolvedModel(BaseModel):
    """A `model=` argument that is decided at run time (env var, config, helper): which model runs is unknown."""

    code_ref: SourceRef
    reason: str


class BoundRef(BaseModel):
    kind: str  # recursion_limit | max_iter | max_turns | max_steps | max_retries | timeout
    value: float | None
    code_ref: SourceRef | None = None
    #: the framework's own spelling (max_messages, max_round, max_agent_steps ...); `kind` is the common concept
    name: str | None = None
    #: set when the value is the default of an environment read, not a literal ("default in code of env MAX")
    assumed_from: str | None = None
    #: the agent whose constructor carries the bound; None = a run-level bound (graph.compile, Runner.run ...)
    #: that applies to every agent
    agent: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.kind

    def describe(self) -> str:
        v = "unreadable" if self.value is None else f"{self.value:g}"
        return f"{self.label}={v}" + (f" (assumed: {self.assumed_from})" if self.assumed_from else "")


def bounds_summary(bounds: list[BoundRef]) -> str:
    """One entry per distinct bound, with a count: `max_iter=5 x5, timeout=30`, never the same bound five times."""
    seen: dict[str, int] = {}
    for b in bounds:
        seen[b.describe()] = seen.get(b.describe(), 0) + 1
    return ", ".join(d + (f" x{n}" if n > 1 else "") for d, n in seen.items()) or "none"


class GateRef(BaseModel):
    """A human-approval / guardrail mechanism found in code."""

    kind: str  # interrupt | interrupt_before | human_input | needs_approval | input_guardrail | ...
    tools: list[str] | None = None  # None = applies to everything / unknown scope
    code_ref: SourceRef | None = None
    #: does the gate run on every path to the tool's action? all_paths = verified (framework-enforced
    #: config, or a call that dominates the tool body); partial = some paths skip it; unverified = not shown
    coverage: Literal["all_paths", "partial", "unverified"] = "unverified"
    coverage_note: str | None = None
    #: the gate waits for a person with no time limit (a bare input()): an approval timeout cannot apply
    waits_forever: bool = False

    @property
    def verified(self) -> bool:
        return self.coverage == "all_paths" and bool(self.tools)


class CodeExecRef(BaseModel):
    """An agent that executes model-written code: smolagents CodeAgent, AutoGen code executors."""

    agent: str
    kind: str  # CodeAgent | UserProxyAgent | CodeExecutorAgent ...
    executor: str  # local | docker | e2b | modal | wasm | jupyter | unknown
    unrestricted: bool  # additional_authorized_imports=["*"] / os / subprocess, or a local shell executor
    code_ref: SourceRef | None = None

    @property
    def sandboxed(self) -> bool:
        return self.executor in ("docker", "e2b", "modal", "wasm", "blaxel", "azure", "remote")


class SecretRef(BaseModel):
    name: str
    code_ref: SourceRef
    preview: str


class DeclaredAgent(BaseModel):
    name: str
    tools: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    handoffs: list[str] = Field(default_factory=list)
    #: agents this one calls like a tool and gets an answer back from: smolagents managed_agents,
    #: a tool body that runs another agent (PydanticAI agent delegation), agent.as_tool / AgentTool
    delegates: list[str] = Field(default_factory=list)
    kind: str | None = None  # constructor name: Agent, CodeAgent, ToolCallingAgent, create_react_agent ...
    var: str | None = None  # python variable the agent is assigned to
    allow_delegation: bool = False
    code_ref: SourceRef | None = None
    #: set when one constructor call builds several agents: how they were expanded
    #: ("loop over config/agents.yaml", "factory make_agent called at main.py:12")
    built_from: str | None = None


class AgentNode(BaseModel):
    id: str
    system_prompt_sha256: str
    name: str | None = None
    tools: list[str] = Field(default_factory=list)
    calls: int = 0
    #: code = a declared agent; framework = a call the framework made itself (graph/origin.py); unresolved = the
    #: project's own agent whose name the code does not show
    origin: Literal["code", "framework", "unresolved"] = "code"

    @property
    def identified(self) -> bool:
        return self.origin == "code"


class ObservedEdge(BaseModel):
    src: str
    dst: str
    via_tool: str
    confidence: Confidence
    evidence: str


class TraceSpan(BaseModel):
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    name: str = ""
    operation: str | None = None  # gen_ai.operation.name
    agent_name: str | None = None
    tool_name: str | None = None
    model: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[str] = Field(default_factory=list)
    start_time: float | None = None  # unix seconds


class TraceEvidence(BaseModel):
    """One way a trace id can follow a run, with the places the code shows it (ast/tracing.py)."""

    mechanism: str  # "OpenTelemetry", "correlation id", a framework's own mechanism
    name: str  # the id
    created: list[SourceRef] = Field(default_factory=list)
    used: list[SourceRef] = Field(default_factory=list)
    logged: list[SourceRef] = Field(default_factory=list)

    def text(self) -> str:
        def at(refs: list[SourceRef]) -> str:
            return ", ".join(str(r) for r in refs[:3])

        parts = [f"{self.mechanism} ({self.name})"]
        if self.created:
            parts.append(f"created at {at(self.created)}")
        if self.used:
            parts.append(f"used at {at(self.used)}")
        if self.logged:
            parts.append(f"in log records at {at(self.logged)}")
        return ", ".join(parts)


class ExtractionError(BaseModel):
    file: str
    line: int | None = None
    message: str


class UnresolvedTool(BaseModel):
    """A tool is constructed here, but its name or callable cannot be resolved statically."""

    code_ref: SourceRef
    reason: str
    #: "construction" = a tool built from a runtime value; "dynamic" = tools supplied by an MCP server,
    #: toolset or registry at run time (a capture or an `--mcp` list can resolve those)
    kind: str = "construction"


class UnresolvedAgent(BaseModel):
    """An agent constructor runs here, but how many agents it builds (and their names / tools) is not static."""

    code_ref: SourceRef
    reason: str
    name: str | None = None  # the placeholder node it is reported under


class ObservedAgent(BaseModel):
    root: str = "."
    files: list[str] = Field(default_factory=list)
    framework: str | None = None
    framework_version: str | None = None
    #: where framework_version came from: "runtime" (recorded inside the captured run),
    #: "installed" (analyzer's environment), "requirements" (pin in requirements/pyproject)
    framework_version_source: str | None = None
    frameworks: list[str] = Field(default_factory=list)
    #: how `framework` was found: "import" (the code imports it: static-proven), "runtime" (it was loaded in a
    #: captured run: observed), "dependency" (only declared in requirements/pyproject: a guess)
    framework_source: str | None = None
    execution_model: ExecutionModel = "unknown"
    execution_model_evidence: str = ""
    execution_model_ref: SourceRef | None = None

    # static (AST)
    declared_tools: dict[str, ToolSchema] = Field(default_factory=dict)
    bound_tools: dict[str, list[SourceRef]] = Field(default_factory=dict)  # tool -> where bound to an agent
    declared_agents: list[DeclaredAgent] = Field(default_factory=list)
    models: list[ModelRef] = Field(default_factory=list)
    bounds: list[BoundRef] = Field(default_factory=list)
    gates: list[GateRef] = Field(default_factory=list)
    secrets: list[SecretRef] = Field(default_factory=list)
    graph_reachable: dict[str, bool] | None = None  # from recognizers (LangGraph)
    prompt_only_tools: list[str] = Field(default_factory=list)  # tools of code agents (no wire schema)
    trace_evidence: list[TraceEvidence] = Field(default_factory=list)  # ast/tracing.py
    patched_tools: list[str] = Field(default_factory=list)  # tools mocks.yml replaced with `patch:` in the mock run
    mock_unsupported: list[str] = Field(default_factory=list)  # providers / transports the mock run cannot intercept
    direct_calls: dict[str, list[SourceRef]] = Field(default_factory=dict)  # tool fn called from Python
    tool_vars: dict[str, str] = Field(default_factory=dict)  # python identifier holding a tool -> tool name
    #: tools that are bound together (one list / bind_tools call / Controller class): one LLM context each
    tool_groups: list[list[str]] = Field(default_factory=list)
    #: project-level code facts for Level-2 checks -> where they were seen. Keys: env_read,
    #: structured_output, audit_hook, cost_bound, duration_bound, checkpointer, prod_endpoint
    signals: dict[str, list[SourceRef]] = Field(default_factory=dict)
    code_executors: list[CodeExecRef] = Field(default_factory=list)
    #: classes passed as structured output (output_cls / output_type / response_format ...): frameworks
    #: send them to the model as a tool, which is not an undeclared tool
    output_types: list[str] = Field(default_factory=list)

    def is_output_schema(self, name: str) -> bool:
        """A structured-output schema the framework sent to the model as a tool (with_structured_output(Plan))."""
        from agenticreq.ast.classify import canonical

        c = canonical(name)
        if c in {canonical(k) for k in self.declared_tools}:
            return False  # a tool in code by that name: it is a tool, whatever else shares the name
        return c in {canonical(t.rsplit(".", 1)[-1]) for t in self.output_types}

    def framework_side_tool(self, name: str) -> str | None:
        """Why a wire tool is the framework's own, not the user's: a structured-output schema or another
        agent called as a tool. None when it is neither."""
        from agenticreq.ast.classify import canonical

        c = canonical(name)
        if self.is_output_schema(name):
            return "structured-output schema sent as a tool"
        if c in {canonical(a.name) for a in self.declared_agents} | {
            canonical(a.var or "") for a in self.declared_agents
        }:
            return "another agent called as a tool (delegation, checked by Level 3)"
        return None

    def signal(self, key: str) -> list[SourceRef]:
        return self.signals.get(key, [])

    # dynamic (capture / MCP / traces)
    captures: list[CapturedCall] = Field(default_factory=list)
    wire_tools: dict[str, ToolSchema] = Field(default_factory=dict)
    mcp_tools: dict[str, ToolSchema] = Field(default_factory=dict)
    trace_spans: list[TraceSpan] = Field(default_factory=list)
    unsupported_calls: list[str] = Field(default_factory=list)

    # derived
    nodes: list[AgentNode] = Field(default_factory=list)
    edges: list[ObservedEdge] = Field(default_factory=list)
    extraction_errors: list[ExtractionError] = Field(default_factory=list)
    #: tools that exist in code but could not be seen (dynamic names / callables): absence claims are not proven
    unresolved_tools: list[UnresolvedTool] = Field(default_factory=list)
    #: tools the code takes out of the model's choice (excluded_functions=, disallowed_tools=, ...): enforcement, not exposure
    excluded_tools: list[str] = Field(default_factory=list)
    #: tools bound under a condition the scan could not evaluate (`[t for t in ALL if t.name not in BLOCKED]`)
    conditional_tools: list[str] = Field(default_factory=list)
    #: agent constructions that build an unknown number of agents (a loop over a runtime value): which
    #: agents exist and which tools each holds is not known, so per-agent checks are not proven
    unresolved_agents: list[UnresolvedAgent] = Field(default_factory=list)
    unresolved_models: list[UnresolvedModel] = Field(default_factory=list)
    #: no framework recognised: functions whose bodies do something unbounded (HTTP to a parameter URL,
    #: exec, an uncapped read). Whether the model can call them is unknown, so they are reported as such.
    loose_functions: list[ToolSchema] = Field(default_factory=list)

    def framework_tag(self) -> str:
        """'crewai 1.9.3 (runtime)' — attached to every version-sensitive finding."""
        if not self.framework:
            return "framework unrecognized"
        decl = " [declared, not imported]" if self.framework_source == "dependency" else ""
        if not self.framework_version:
            return f"{self.framework} (version unresolved){decl}"
        return f"{self.framework} {self.framework_version} ({self.framework_version_source or 'unknown source'}){decl}"

    @property
    def framework_confirmed(self) -> bool:
        """The code imports (or a run loaded) the framework: its built-in defaults really apply."""
        return bool(self.framework) and self.framework_source != "dependency"

    @property
    def has_runtime(self) -> bool:
        return bool(self.captures or self.trace_spans)

    def run_ids(self) -> list[str]:
        ids = {c.run_id for c in self.captures}
        ids |= {s.trace_id for s in self.trace_spans if s.trace_id}
        return sorted(ids)

    def observed_call_counts(self) -> dict[str, int]:
        """tool -> number of runs (captures/traces) in which the model invoked it."""
        per_run: dict[str, set[str]] = {}
        for c in self.captures:
            if c.synthetic:
                continue
            for tc in c.tool_calls:
                per_run.setdefault(tc.name, set()).add(c.run_id)
        for s in self.trace_spans:
            if s.tool_name and (s.operation in (None, "execute_tool")):
                per_run.setdefault(s.tool_name, set()).add(s.trace_id or "trace")
        return {k: len(v) for k, v in per_run.items()}

    def offered_tools(self) -> set[str]:
        """Tools the model was told about on the wire (or invoked in real traces)."""
        names = set(self.wire_tools) | set(self.mcp_tools)
        names |= set(self.observed_call_counts())
        return names

    def real_run_ids(self) -> list[str]:
        """Runs whose tool calls came from a real model (captures not synthetic, or traces)."""
        ids = {c.run_id for c in self.captures if not c.synthetic}
        ids |= {s.trace_id or "trace" for s in self.trace_spans}
        return sorted(ids)
