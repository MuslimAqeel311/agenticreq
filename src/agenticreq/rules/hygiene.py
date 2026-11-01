"""AS-219..227 — Level-2 engineering hygiene, checkable from code with no spec (risk_level defaults).

Each rule reads one Level-2 parameter from the effective spec (the user's override, else the
risk_level default) and looks for the code construct that enforces it. Absence of a construct is
`heuristic`: the enforcement may live outside the scanned code, and the finding says so.
"""

from __future__ import annotations

import re
from typing import Iterable

from agenticreq import thresholds as th
from agenticreq.ast.classify import (
    canonical,
    effect_unsure,
    is_high_impact_write,
    is_mcp_server_tool,
    is_test_file,
    is_write_tool,
)
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck, SourceRef
from agenticreq.models.observed import ObservedAgent, bounds_summary
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.conformance import code_tools
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")
H, U = Confidence.heuristic, Confidence.unknown

#: tool names that rewrite the agent's own instructions, tools or code
_SELF_MOD_NAME = re.compile(
    r"^(update|set|modify|edit|change|rewrite|patch|replace|overwrite|append_to)_?"
    r"(own_|my_|agent_|self_)?(system_?prompt|prompt|instructions?|system_message|guardrails?|agent_spec|"
    r"own_code|source_code|persona)$"
    r"|^(self_(modify|update|improve)|add_tool|register_tool|create_tool|install_tool|define_tool)$"
)
#: tools declared as classes (LangChain / CrewAI BaseTool, smolagents Tool): func_name is None, but the
#: features come from their _run / forward body
_CLASS_TOOL_KINDS = {"BaseTool", "smolagents.Tool"}
_DRY_RUN_PARAMS = {"dry_run", "dryrun", "preview", "simulate", "validate_only", "plan_only", "confirm"}


def _signal(obs: ObservedAgent, key: str) -> list[SourceRef]:
    """Signal sites outside test files (tests configure things the shipped agent does not)."""
    return [r for r in obs.signal(key) if not is_test_file(r.file)]


def _has_agent_loop(obs: ObservedAgent) -> bool:
    if obs.declared_agents:
        return True
    return bool(obs.declared_tools) and not all(is_mcp_server_tool(t) for t in obs.declared_tools.values())


def _spec_ref(ctx: CheckContext, key: str) -> SourceRef | None:
    """Where the effective value came from: the user's own key, else the risk_level that implied it."""
    if not ctx.loaded:
        return None
    return ctx.spec_ref(key) if key in ctx.loaded.raw else ctx.spec_ref("risk_level")


def _where(refs: list[SourceRef]) -> str:
    return ", ".join(str(r) for r in refs[:3]) + (f" (+{len(refs) - 3})" if len(refs) > 3 else "")


@rule("conformance", "AS-219", modes=_BOTH)
def fail_open(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or not ctx.effective().get("fail_closed"):
        return []
    strict = ctx.mode == "check" and ctx.risk_level in ("high", "critical")
    out = []
    clean = 0
    for t in code_tools(obs):
        if t.func_name is None and t.kind not in _CLASS_TOOL_KINDS:
            continue  # schema-only tool: its body is elsewhere (class tools carry their _run body's features)
        ctx.clause(extracted=True)
        if "fail_open" not in t.features:
            clean += 1
            continue
        out.append(
            make_finding(
                "AS-219",
                confidence=H(),
                code_ref=t.code_ref,
                severity="blocker" if strict else "warning",
                title=f"{t.name} swallows errors (fail-open)",
                detail=f"{t.name} catches a broad exception and neither re-raises nor reports it; "
                "the agent sees success for a step that failed (fail_closed="
                f"{ctx.effective().get('fail_closed')})",
                evidence=f"except Exception/bare except without raise or error result; features: {t.features}",
            )
        )
    for ref in _signal(obs, "fail_open_path"):
        out.append(
            make_finding(
                "AS-219",
                confidence=H(),
                code_ref=ref,
                severity="blocker" if strict else "warning",
                title="Error swallowed on the agent's execution path (fail-open)",
                detail="a broad except in a helper a tool calls, a graph node, or the code that runs "
                "the agent neither re-raises nor reports the error, so the run carries on as "
                "if that step had succeeded",
                evidence=f"except Exception/bare except without raise or error result at {ref}",
            )
        )
    if clean and not out:  # one aggregate pass about error handling, never a pass *for a tool*
        ctx.pass_(
            PassedCheck(
                rule_id="AS-219",
                subject="tool error handling",
                confidence=H(),
                note=f"no swallowed exception in {clean} tool bodies",
            )
        )
    return out


@rule("conformance", "AS-220", modes=_BOTH)
def self_modification(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or ctx.effective().get("self_modification") is True:
        return []
    out = []
    for t in code_tools(obs):
        by_body = "self_modify" in t.features
        by_name = bool(_SELF_MOD_NAME.match(canonical(t.name)))
        if not (by_body or by_name):
            continue
        ctx.clause(extracted=True)
        out.append(
            make_finding(
                "AS-220",
                confidence=H(),
                code_ref=t.code_ref,
                severity="blocker" if by_body else "warning",
                title=f"{t.name} lets the agent modify itself",
                detail=(
                    f"{t.name} "
                    + (
                        "rewrites the agent's instructions / tool list / source file in its body"
                        if by_body
                        else "is named like a tool that changes the agent's own "
                        "prompt or tools (name only; body not conclusive)"
                    )
                    + "; self_modification is false at every risk level"
                ),
                evidence=f"name={t.name!r}; features: {t.features or 'none'}",
            )
        )
    return out


def _bound_rule(
    ctx: CheckContext, rule_id: str, signal: str, p: th.Numeric, what: str, extra_kinds: tuple[str, ...] = ()
) -> list[Finding]:
    """A run-wide cost / time limit. With a spec only a limit the spec itself declares is a requirement (the
    the library proposes no number); audit has no spec, so it asks whether the code has this kind of limit at
    all. The value in code is read when the framework states it (a number); a budget construct alone proves a limit
    exists, not what it is."""
    obs = ctx.observed
    if not obs or not _has_agent_loop(obs):
        return []
    risk = ctx.risk_level
    spec_n = th.spec_value(ctx, p)
    sites = _signal(obs, signal) + [
        b.code_ref for b in obs.bounds if b.kind in extra_kinds and b.code_ref and not is_test_file(b.code_ref.file)
    ]
    read = [
        b.value
        for b in obs.bounds
        if b.kind in extra_kinds and b.value is not None and not (b.code_ref and is_test_file(b.code_ref.file))
    ]
    if ctx.mode == "check" and ctx.loaded is not None:
        if spec_n is None and not read:
            return []  # nothing the spec asks for (a TODO is AS-101's)
    ctx.clause(extracted=True)
    sref = _spec_ref(ctx, "bounds")
    if spec_n is None and read:  # a value in code and none in the spec: report it as found
        only = max(read)
        return th.emit(
            ctx,
            p,
            th.judge(p, risk, code=only),
            risk=risk,
            code=only,
            rule_id=rule_id,
            code_ref=sites[0] if sites else None,
        )
    if sites and spec_n is None:  # no number to hold the code to: a limit exists, its value is not asked for
        ctx.pass_(
            PassedCheck(rule_id=rule_id, subject=what, confidence=H(), note=f"found at {_where(sites)}", param=p.key)
        )
        return []
    if spec_n is not None and (sites or not _signal(obs, f"{signal}_key")):
        # the spec states a number: the code must show one at least as strict; a limit whose value cannot be read is
        # signed off under confirmed:, or a mock run checks it
        code = max(read) if read else None  # the loosest limit in code is what a run may use
        att = th.attestation(ctx, p)
        d = th.judge(p, risk, spec=spec_n, code=code, code_present=bool(sites), confirmed=att is not None)
        return th.emit(
            ctx,
            p,
            d,
            risk=risk,
            code=code,
            spec=spec_n,
            spec_ref=sref,
            code_ref=sites[0] if sites else None,
            rule_id=rule_id,
            attested=att,
            evidence=f"{p.key}: spec={th.fmt(spec_n)}; "
            + (f"a {what} exists at {_where(sites)}" if sites else f"no {what} in code"),
        )
    keys = _signal(obs, f"{signal}_key")
    if keys:  # a budget value in a dict: nothing shows the run enforces it
        return [
            make_finding(
                rule_id,
                confidence=U(),
                spec_ref=sref,
                param=p.key,
                title=f"{what} set but enforcement not verified",
                detail=f"a {p.name} value is set at {_where(keys)}, but agenticreq cannot see that the "
                "framework or the loop enforces it on every path",
                evidence=f"budget keys at {_where(keys)}; no framework budget construct found",
            )
        ]
    d = th.judge(p, risk, spec=spec_n)
    return [
        make_finding(
            rule_id,
            confidence=H(),
            spec_ref=sref,
            reason=d.reason,
            status_hint=d.line_status,
            param=p.key,
            detail=f"no {what} found in code. A loop bound alone does not cap spend or time.",
            evidence=f"searched: {signal} constructs; step bounds found: {bounds_summary(obs.bounds)}",
        )
    ]


@rule("conformance", "AS-221", modes=_BOTH)
def cost_bound(ctx: CheckContext) -> Iterable[Finding]:
    return _bound_rule(ctx, "AS-221", "cost_bound", th.MAX_COST, "run-wide cost / token budget")


@rule("conformance", "AS-222", modes=_BOTH)
def duration_bound(ctx: CheckContext) -> Iterable[Finding]:
    return _bound_rule(
        ctx, "AS-222", "duration_bound", th.MAX_DURATION, "run-wide time limit", extra_kinds=("max_execution_time",)
    )


@rule("conformance", "AS-223", modes=_BOTH)
def environment_separation(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    # framework-agnostic: with no recognised framework there is no "agent loop" to find, but config and
    # endpoints are still in the code
    if (
        not obs
        or not ctx.effective().get("environment_separation")
        or not (_has_agent_loop(obs) or not obs.framework and obs.files)
    ):
        return []
    out = []
    ctx.clause(extracted=True)
    for ref in _signal(obs, "prod_endpoint"):
        out.append(
            make_finding(
                "AS-223",
                confidence=H(),
                code_ref=ref,
                title="Production endpoint is hard-coded",
                detail="a production URL is a literal in code, so the same build cannot run against dev/staging",
                evidence=f"string literal at {ref}",
            )
        )
    if not _signal(obs, "env_read"):
        out.append(
            make_finding(
                "AS-223",
                confidence=H(),
                detail="no configuration is read from the environment (os.environ / getenv / "
                "dotenv / a Settings class); endpoints, keys and models are fixed in code",
                evidence=f"files scanned: {len(obs.files)}; env reads: 0",
            )
        )
    elif not out:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-223",
                subject="environment",
                confidence=H(),
                note=f"config read from the environment at {_where(_signal(obs, 'env_read'))}",
            )
        )
    return out


@rule("conformance", "AS-224", modes=_BOTH)
def structured_output(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or not obs.declared_agents and not obs.models:
        return []
    if not obs.framework and not obs.declared_agents:
        return []  # how the output is typed is framework-specific: listed as not checked (AS-904)
    required = bool(ctx.effective().get("structured_output"))
    sites = _signal(obs, "structured_output")
    if sites:
        ctx.clause(extracted=True)
        ctx.pass_(
            PassedCheck(
                rule_id="AS-224",
                subject="structured output",
                confidence=H(),
                note=f"agent output type at {_where(sites)}",
            )
        )
        return []
    if not required and ctx.mode != "audit":
        return []
    ctx.clause(extracted=True)
    calls = _signal(obs, "structured_output_call")
    if calls:  # one model call is typed; that it is the final answer on every path is not shown
        return [
            make_finding(
                "AS-224",
                confidence=U(),
                spec_ref=_spec_ref(ctx, "structured_output"),
                title="Structured output found, but not shown to type the final answer",
                detail=f"a schema-typed model call exists at {_where(calls)}, but agenticreq cannot "
                "verify that every path ends in it (no output type on the agent itself)",
                evidence=f"typed calls: {_where(calls)}",
            )
        ]
    return [
        make_finding(
            "AS-224",
            confidence=H(),
            severity="warning" if required else "info",
            spec_ref=_spec_ref(ctx, "structured_output"),
            detail="no typed / schema-validated output found (response_format, output_type, "
            "with_structured_output, output_pydantic and the like); "
            + (
                f"required at risk_level {ctx.risk_level}"
                if required
                else "not required at the medium default, required at high / critical"
            ),
            evidence="searched: structured-output kwargs and parser calls",
        )
    ]


@rule("conformance", "AS-225", modes=_BOTH)
def audit_hook(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    eff = ctx.effective()
    if not obs or not _has_agent_loop(obs) or not (eff.get("audit_log") or eff.get("trace_policy")):
        return []
    ctx.clause(extracted=True)
    sites = _signal(obs, "audit_hook")
    if not _signal(obs, "tracing_disabled"):
        sites += _signal(obs, "agents_trace")
        # the SDK exports its default traces to the OpenAI platform only: with a model from another provider
        # (Groq, a custom OpenAI-compatible client) nothing is recorded unless a processor is configured
        other_provider = any(m.provider for m in obs.models)
        if "openai-agents" in obs.frameworks and not sites and not other_provider:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-225",
                    subject="audit hook",
                    confidence=H(),
                    note="OpenAI Agents SDK traces every run by default and nothing disables it "
                    "(set_tracing_disabled / RunConfig(tracing_disabled=True))",
                )
            )
            return []
    logging_tools = [t.name for t in code_tools(obs) if "logs" in t.features]
    writes = [t for t in code_tools(obs) if is_write_tool(t)]
    unlogged_writes = [t.name for t in writes if "logs" not in t.features]
    if sites:
        ctx.pass_(PassedCheck(rule_id="AS-225", subject="audit hook", confidence=H(), note=f"found at {_where(sites)}"))
        return []
    imports = _signal(obs, "audit_import")
    trail = _signal(obs, "audit_trail")
    if trail and not imports:
        return [
            make_finding(
                "AS-225",
                confidence=U(),
                title="Run transcript is stored; per-tool-call audit not verified",
                detail=f"the code stores run messages / a transcript at {_where(trail)}, but agenticreq cannot "
                "tell whether it records every tool call with its arguments and result; no tracer or "
                "callback is attached",
                evidence=f"transcript / log writes: {_where(trail)}; tools that log: {logging_tools or 'none'}",
                remediation="Check that the stored transcript contains each tool call, its arguments and its "
                "result, or attach tracing (OpenTelemetry, callbacks=...) so it is guaranteed.",
            )
        ]
    if imports or (logging_tools and not unlogged_writes):
        # a tracing import, or a log call somewhere in each write tool, is not an audit trail of every tool call
        why = (
            f"a tracing package is imported at {_where(imports)} but never wired to the run"
            if imports
            else f"every write tool contains a log call ({logging_tools}), but not shown on every path"
        )
        return [
            make_finding(
                "AS-225",
                confidence=U(),
                title="Audit logging not verified on every tool call",
                detail=f"{why}; no callback / tracer is attached to the agent",
                evidence=f"tracing imports: {_where(imports) if imports else 'none'}; "
                f"tools that log: {logging_tools or 'none'}",
            )
        ]
    return [
        make_finding(
            "AS-225",
            confidence=H(),
            detail="no tracing / callback hook and "
            + (f"write tools {unlogged_writes} do not log" if unlogged_writes else "no tool logs its calls")
            + "; tool actions cannot be reconstructed afterwards",
            evidence=f"tracing imports/callbacks: none; tools that log: {logging_tools or 'none'}",
        )
    ]


@rule("conformance", "AS-226")
def checkpointing(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or not ctx.effective().get("checkpointing") or not _has_agent_loop(obs):
        return []
    ctx.clause(extracted=obs.framework == "langgraph")
    sites = _signal(obs, "checkpointer")
    if sites:
        ctx.pass_(
            PassedCheck(rule_id="AS-226", subject="checkpointer", confidence=H(), note=f"found at {_where(sites)}")
        )
        return []
    built = _signal(obs, "checkpointer_ctor")
    if built:
        return [
            make_finding(
                "AS-226",
                confidence=U(),
                spec_ref=_spec_ref(ctx, "checkpointing"),
                title="Checkpointer built but not shown to be used",
                detail=f"a saver is constructed at {_where(built)}, but no compile(checkpointer=...) "
                "hands it to the graph",
                evidence=f"checkpointer constructors: {_where(built)}",
            )
        ]
    if obs.framework != "langgraph":
        return [
            make_finding(
                "AS-226",
                confidence=U(),
                severity="info",
                spec_ref=_spec_ref(ctx, "checkpointing"),
                title="Checkpointing cannot be verified for this framework",
                detail=f"checkpointing is required at risk_level {ctx.risk_level}; agenticreq only "
                f"recognises LangGraph checkpointers ({obs.framework_tag()})",
                evidence="checkpointer constructs: none recognised",
            )
        ]
    return [
        make_finding(
            "AS-226",
            confidence=H(),
            spec_ref=_spec_ref(ctx, "checkpointing"),
            detail=f"risk_level {ctx.risk_level} requires checkpointing; no compile(checkpointer=...) or *Saver found",
            evidence=f"{obs.framework_tag()}; checkpointers found: none",
        )
    ]


@rule("conformance", "AS-227")
def dry_run(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    want = ctx.effective().get("dry_run_required")
    if not obs or not want:
        return []
    only = {canonical(x) for x in want} if isinstance(want, list) else None
    prohibited = {canonical(t) for t in (ctx.spec.prohibited_tools() if ctx.spec else set())}
    approval = {canonical(t) for t in (ctx.spec.approval_tools() if ctx.spec else {})}
    out = []
    for t in code_tools(obs):
        c = canonical(t.name)
        if c in prohibited or not is_write_tool(t) or (only is not None and c not in only):
            continue
        if only is None and not (is_high_impact_write(t) or c in approval):
            continue  # a dry run matters for writes that cannot be undone, not for a cache entry or a calendar slot
        ctx.clause(extracted=True)
        if any(canonical(p.name) in _DRY_RUN_PARAMS for p in t.params):
            if "dry_run_branch" in t.features:
                ctx.pass_(
                    PassedCheck(
                        rule_id="AS-227",
                        subject=f"dry-run parameter of {t.name}",
                        confidence=H(),
                        note="dry-run parameter present and the body branches on it",
                    )
                )
            else:  # a parameter the body never tests changes nothing
                out.append(
                    make_finding(
                        "AS-227",
                        confidence=U(),
                        code_ref=t.code_ref,
                        title=f"Dry-run parameter of {t.name} not shown to be honoured",
                        detail=f"{t.name} takes a dry-run parameter, but no branch in its body tests it",
                        evidence=f"params: {[p.name for p in t.params]}; features: {t.features}",
                    )
                )
            continue
        unsure = effect_unsure(t)  # the name says write; the body cannot be followed: a mock run decides
        out.append(
            make_finding(
                "AS-227",
                confidence=U() if unsure else H(),
                code_ref=t.code_ref,
                spec_ref=_spec_ref(ctx, "dry_run_required"),
                reason=f"Cannot tell whether {t.name} writes: it {t.effect_reason} (line {t.effect_line})"
                ". Declare it under tool_effects: in spec.yml, or run: agenticreq mock-run"
                if unsure
                else None,
                status_hint="CHECK" if unsure else None,
                title=f"Cannot tell whether {t.name} needs a dry-run mode"
                if unsure
                else f"Write tool {t.name} has no dry-run mode",
                detail=f"dry_run_required is set (risk_level {ctx.risk_level}) but {t.name} takes no "
                f"{sorted(_DRY_RUN_PARAMS)} parameter",
                evidence=f"params: {[p.name for p in t.params]}",
            )
        )
    return out


@rule("conformance", "AS-215", modes=_BOTH)
def code_executing_agents(ctx: CheckContext) -> Iterable[Finding]:
    """An agent that runs model-written code (smolagents CodeAgent, AutoGen code executors). The tools
    are not where the danger is: the model's own code is. Sandboxed executors pass; a local executor with
    unrestricted imports (`*`, os, subprocess) or a local shell is a finding."""
    obs = ctx.observed
    if not obs:
        return []
    out = []
    for x in obs.code_executors:
        ctx.clause(extracted=True)
        if x.sandboxed:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-215",
                    subject=f"{x.agent} code executor",
                    confidence=H(),
                    note=f"{x.kind} runs generated code in executor={x.executor}",
                )
            )
            continue
        if not x.unrestricted:
            continue  # local, restricted interpreter (smolagents' default import allowlist): not flagged
        strict = ctx.risk_level in ("high", "critical") or ctx.effective().get("sandbox_policy") == "required"
        out.append(
            make_finding(
                "AS-215",
                confidence=H(),
                code_ref=x.code_ref,
                severity="blocker" if strict else "warning",
                title=f"{x.agent} runs model-written code locally without restriction",
                detail=f"{x.kind} executes code the model writes, in executor={x.executor} with unrestricted imports / "
                f"shell; sandbox_policy={ctx.effective().get('sandbox_policy')!r}",
                evidence=f"{x.kind}: executor={x.executor}, unrestricted={x.unrestricted}",
            )
        )
    return out
