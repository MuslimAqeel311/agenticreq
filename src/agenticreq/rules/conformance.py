"""AS-2xx — conformance: declared (AST) vs observed (wire), and spec vs code.

Static reachability is primary for safety claims; traces can prove a violation but
never prove safety ("violation found" vs "no violation observed").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agenticreq.ast.classify import (
    action_verb,
    canonical,
    claims_read_only,
    has_idempotency,
    is_handoff_tool,
    is_high_impact_write,
    is_mcp_server_tool,
    is_test_file,
    is_unbounded,
    effect_unsure,
    is_write_tool,
    model_key,
    pin_judge,
    near_miss,
    offers_snapshots,
    verb_class,
)
from agenticreq import thresholds as th
from agenticreq.data import InjectedTools
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck, SourceRef
from agenticreq.models.observed import (
    CapturedCall,
    ModelRef,
    ObservedAgent,
    ParamSchema,
    ToolSchema,
    UnresolvedTool,
    bounds_summary,
)
from agenticreq.models.spec import AgentSpec
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")
P, H, U = Confidence.proven, Confidence.heuristic, Confidence.unknown


def open_unresolved(obs: ObservedAgent) -> list[UnresolvedTool]:
    """Unresolved tool sources that still matter. Tools an MCP server / toolset supplies stop mattering once a
    capture or an `--mcp` list has named what the agent was really offered."""
    if obs.mcp_tools or obs.wire_tools:
        return [u for u in obs.unresolved_tools if u.kind != "dynamic"]
    return list(obs.unresolved_tools)


def absence_confidence(obs: ObservedAgent) -> Confidence:
    """Confidence of a claim that something is NOT in the code. Absence is inferred from how much of the code
    was extracted, never read directly from config, wire or AST: heuristic at best, unknown when tool
    constructions could not be resolved (the missing thing may be one of them)."""
    return U() if open_unresolved(obs) or obs.extraction_errors else H()


def _unresolved_note(obs: ObservedAgent) -> str:
    if not obs.unresolved_tools:
        return ""
    return (
        f"; {len(obs.unresolved_tools)} tool construction(s) could not be resolved statically "
        f"({', '.join(str(u.code_ref) for u in obs.unresolved_tools[:3])})"
    )


def _declared(obs: ObservedAgent) -> dict[str, ToolSchema]:
    return {canonical(k): v for k, v in obs.declared_tools.items()}


def _bound_ref(obs: ObservedAgent, name: str) -> SourceRef | None:
    for k, refs in obs.bound_tools.items():
        if canonical(k) == canonical(name) and refs:
            return refs[0]
    return None


def _is_bound(obs: ObservedAgent, name: str) -> bool:
    return any(canonical(k) == canonical(name) for k in obs.bound_tools)


def _counts(obs: ObservedAgent) -> dict[str, int]:
    return {canonical(k): v for k, v in obs.observed_call_counts().items()}


def _tool_ref(obs: ObservedAgent, name: str) -> SourceRef | None:
    t = _declared(obs).get(canonical(name))
    return t.code_ref if t else None


def _own_scope(ctx: CheckContext) -> tuple[set[str] | None, set[str]]:
    """(tools in scope for the spec's own agent, tools that are in scope ONLY because an unconnected agent holds them).
    The first is None for a single agent (all tools)."""
    from agenticreq.graph.analysis import build, own_node

    if not ctx.observed:
        return None, set()
    g = build(ctx.observed, ctx.spec)
    if len(g.nodes) < 2:
        return None, set()
    me = own_node(g, ctx.spec)
    if not me:
        return None, set()  # several entry agents and the spec names none: every agent is checked
    mine = {canonical(t) for t in g.nodes.get(me, set())}
    # an agent with no delegation path to or from the spec's own agent is still part of the scanned code: its
    # tools count too (reachable delegates stay AS-304's job, so a prohibited tool there is not reported twice)
    linked = (
        {me}
        | {e.dst for p in g.paths_from(me) for e in p}
        | {n for n in g.nodes if me in {e.dst for p in g.paths_from(n) for e in p}}
    )
    extra: set[str] = set()
    for n in g.nodes:
        if n not in linked:
            extra |= {canonical(t) for t in g.nodes[n]}
    return mine | extra, extra - mine


def own_tools(ctx: CheckContext) -> set[str] | None:
    """Canonical tools of the spec's own agent in a multi-agent graph; None = single agent (all tools)."""
    return _own_scope(ctx)[0]


def code_tools(obs: ObservedAgent) -> list[ToolSchema]:
    """Tools in code that matter: bound to an agent or offered on the wire."""
    offered = {canonical(n) for n in obs.offered_tools()}
    return [
        t
        for t in obs.declared_tools.values()
        if _is_bound(obs, t.name) or canonical(t.name) in offered or not obs.bound_tools
    ]


# =============================================================== declared ⟷ observed diff
@rule("conformance", "AS-201", modes=_BOTH)
def never_reached(ctx: CheckContext) -> Iterable[Finding]:
    """declared_tools ∖ observed_tools → unknown (never PASS)."""
    obs = ctx.observed
    if not obs or not obs.has_runtime:
        return []
    offered = {canonical(n) for n in obs.offered_tools()}
    out = []
    for name, t in sorted(obs.declared_tools.items()):
        ctx.clause(extracted=canonical(name) in offered)
        if canonical(name) in offered:
            continue
        reasons = []
        if not _is_bound(obs, name):
            reasons.append("not bound to any agent in code")
        if obs.graph_reachable is not None and obs.graph_reachable.get(name) is False:
            reasons.append("its graph node is unreachable from START (static)")
        elif _is_bound(obs, name):
            reasons.append("bound on a branch / phase the run did not reach")
        if canonical(name) in {canonical(k) for k in obs.direct_calls}:
            reasons.append("called directly from Python, not via the model")
        if name in obs.prompt_only_tools:  # it never travels as a schema: another run would show the same
            out.append(
                make_finding(
                    "AS-201",
                    confidence=Confidence.attestation(),
                    severity="attestation",
                    code_ref=t.code_ref,
                    title=f"{name} never reached the model boundary",
                    detail=f"{name} is used by a code agent: it reaches the model as prompt text, not as a schema",
                    evidence=f"wire tools seen: {sorted(obs.offered_tools())[:12]}",
                    reason=f"Prompt-only: {name} reaches the model as prompt text (a code agent), so no run can "
                    "show it. Confirm it under confirmed: in spec.yml.",
                    status_hint="SIGN-OFF",
                )
            )
            continue
        out.append(
            make_finding(
                "AS-201",
                confidence=U(),
                code_ref=t.code_ref,
                title=f"{name} never reached the model boundary",
                detail=f"{name} is declared in code but was not offered to the model in "
                f"{len(obs.run_ids())} run(s): {'; '.join(reasons) or 'no reason recorded'}. Waits for a mock run; not a PASS.",
                evidence=f"wire tools seen: {sorted(obs.offered_tools())[:12]}",
                reason=_never_reached_reason(obs, name),
                status_hint="CHECK",
            )
        )
    return out


def _agents_that_ran(obs: ObservedAgent, tools_of: dict[str, set[str]]) -> set[str]:
    """Agents a captured model call came from: the call's tools (schemas aside) are a subset of the agent's, and
    no other agent's tool set fits it more closely (ties: all of them)."""
    ran: set[str] = set()
    for call in obs.captures:
        offered = {canonical(t.name) for t in call.tools if not obs.is_output_schema(t.name)}
        if not offered:
            continue
        fits = {a: len(ts - offered) for a, ts in tools_of.items() if offered <= ts}
        if fits:
            best = min(fits.values())
            ran |= {a for a, n in fits.items() if n == best}
    return ran


def _enum_values(schema: object) -> set[str]:
    """Every enum / const string anywhere in a JSON schema."""
    out: set[str] = set()
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k == "enum" and isinstance(v, list):
                out |= {x for x in v if isinstance(x, str)}
            elif k == "const" and isinstance(v, str):
                out.add(v)
            else:
                out |= _enum_values(v)
    elif isinstance(schema, list):
        for v in schema:
            out |= _enum_values(v)
    return out


def _routing_schemas(obs: ObservedAgent, agents: list[str]) -> list[str]:
    """Structured-output schemas seen in the run whose choices name one of ``agents`` (RouteDecision.next, a plan
    step's agent): in a mock run the canned model fills them, so they, not the task text, pick the route."""
    wanted = {canonical(a) for a in agents}
    return sorted(
        t.name
        for t in obs.wire_tools.values()
        if obs.is_output_schema(t.name) and wanted & {canonical(v) for v in _enum_values(t.raw)}
    )


def _then_run_text(text: str) -> str:
    return f"{text}. Declare it under tool_effects: in spec.yml, or run: agenticreq mock-run"


def _never_reached_reason(obs: ObservedAgent, name: str) -> str:
    """Why a tool was never offered, in the step that fixes it. Not offered means no model call carried it: the
    agent holding it never ran (the run order in mocks.yml only orders tools that were offered)."""
    c = canonical(name)
    tools_of = {a.name: {canonical(t) for t in a.tools} for a in obs.declared_agents}
    holders = [a for a, ts in tools_of.items() if c in ts]
    if holders:
        ran = _agents_that_ran(obs, tools_of)
        idle = [a for a in holders if a not in ran]
        if idle:
            who = ", ".join(idle)
            them = "them" if len(idle) > 1 else "it"
            routes = _routing_schemas(obs, idle)
            if routes:  # the canned model fills the schema that picks the next agent: its args decide the route
                return (
                    f"Never offered to the model: the run never reached {who}. Add a run under runs: in mocks.yml "
                    f"whose {' or '.join(routes)} args send the run to {them}, then run: agenticreq mock-run"
                )
            return (
                f"Never offered to the model: the run never reached {who}. Add a run under runs: in mocks.yml "
                f"with a task that reaches {them} (and order: its handoff first, if the model picks the route), "
                "then run: agenticreq mock-run"
            )
        return (
            f"Never offered to the model: {', '.join(holders)} ran without it (the code leaves it out at run "
            "time; check the settings under env: in mocks.yml), then run: agenticreq mock-run"
        )
    # the code does not name the agent (tools picked at run time, e.g. TOOLSETS[name]): say only what is known
    where = (
        "the part of the agent that holds it"
        if _is_bound(obs, name)
        else "the agent given it (the code does not show which)"
    )
    return (
        f"Never offered to the model: the run did not reach {where}. Add a run under runs: in mocks.yml with an "
        "input that reaches it, then run: agenticreq mock-run"
    )


@rule("conformance", "AS-234", modes=_BOTH)
def same_normalized_name(ctx: CheckContext) -> Iterable[Finding]:
    """Two different names in the code that normalize to the same one (save_file and Save File): every comparison
    with the spec and the wire matches both, so the report cannot tell them apart."""
    from agenticreq.names import collisions

    obs = ctx.observed
    if not obs:
        return []
    out = []
    for kind, names in (
        ("tool", list(obs.declared_tools)),
        ("agent", [a.name for a in obs.declared_agents if a.kind != "group_chat"]),
    ):
        for group in collisions(names):
            out.append(
                make_finding(
                    "AS-234",
                    confidence=P(),
                    param="tool_inventory_report" if kind == "tool" else "agent_registry",
                    code_ref=_tool_ref(obs, group[0]) if kind == "tool" else None,
                    title=f"two {kind}s normalize to the same name: {', '.join(group)}",
                    detail=f"two {kind}s normalize to the same name: {', '.join(group)}",
                    evidence=", ".join(group),
                )
            )
    return out


@rule("conformance", "AS-202", modes=_BOTH)
def undeclared_observed(ctx: CheckContext) -> Iterable[Finding]:
    """observed_tools ∖ declared_tools ∖ injected → finding."""
    obs = ctx.observed
    if not obs or not (obs.has_runtime or obs.mcp_tools):
        return []
    injected = InjectedTools(ctx.injected_allowlist)
    declared = set(_declared(obs))
    out = []
    for name in sorted(obs.offered_tools()):
        if name.startswith("hosted:"):
            continue
        if canonical(name) in declared:
            continue
        fw = injected.framework_of(name)
        if fw:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-202",
                    subject=name,
                    confidence=P(),
                    note=f"added by the framework ({obs.framework or fw}): a handoff or helper "
                    f"tool it makes itself, listed in known_injected_tools.yml [{obs.framework_tag()}]",
                )
            )
            continue
        why = obs.framework_side_tool(name)
        if why:
            # a schema is read from the with_structured_output(...) call; an agent-as-tool is matched by name
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-202", subject=name, confidence=P() if obs.is_output_schema(name) else H(), note=why
                )
            )
            continue
        src = "MCP server" if name in obs.mcp_tools else "the wire"
        out.append(
            make_finding(  # offered: read from the wire; "not declared": inferred from extraction coverage
                "AS-202",
                confidence=absence_confidence(obs),
                title=f"{name} offered to the model but not declared in code",
                detail=f"{src} carried tool {name!r}, and no tool declaration agenticreq could resolve in the scanned code "
                f"defines it; it is not a known framework-injected tool{_unresolved_note(obs)}",
                evidence=f"{obs.framework_tag()} | {_wire_evidence(obs, name)}",
            )
        )
    return out


def _wire_evidence(obs: ObservedAgent, name: str) -> str:
    for c in obs.captures:
        if any(t.name == name for t in c.tools):
            return f"request #{c.index} ({c.provider}, run {c.run_id}) tools[] includes {name}"
    return f"{name} in MCP tools/list" if name in obs.mcp_tools else name


def _by_capability(obs: ObservedAgent, name: str) -> str:
    """``shell_execution`` in the spec is the tool that has that ability here (Claude Agent SDK ``Bash``)."""
    from agenticreq.data import load_yaml

    words = load_yaml("known_injected_tools.yml").get("capability_names") or {}
    wanted = words.get(canonical(name))
    tool = next((t.name for t in obs.declared_tools.values() if wanted and t.capability == wanted), None)
    return tool or name


# =============================================================== spec vs code
@rule("conformance", "AS-203", "AS-204")
def prohibited(ctx: CheckContext) -> Iterable[Finding]:
    s, obs = ctx.spec, ctx.observed
    if not s or not obs:
        return []
    out = []
    counts = _counts(obs)
    real_runs = len(obs.real_run_ids())
    mine, _unlinked = _own_scope(ctx)
    excluded = {canonical(x) for x in obs.excluded_tools}
    conditional = {canonical(x) for x in obs.conditional_tools}
    offered = {canonical(n) for n in obs.offered_tools()}
    offered_anywhere = set(offered)
    wire_runs = capture_runs(obs)
    declared = _declared(obs)
    if mine is not None:  # multi-agent: only this agent's tools; delegates are AS-304's job
        offered &= mine
        declared = {k: v for k, v in declared.items() if k in mine}
    for i, p in enumerate(s.prohibited or []):
        if p.detector is not None:
            out += _detector_hits(ctx, obs, i, p.action or p.tool or "?", p.detector.type, p.detector.params, real_runs)
        if not p.tool:
            continue
        p = p.model_copy(update={"tool": _by_capability(obs, p.tool)})
        assert p.tool is not None
        key = canonical(p.tool)
        sref = ctx.spec_ref("prohibited", i)
        ctx.clause(extracted=True)
        # runtime proof of absence: offered to no agent (whatever the scope) in any run, and never called
        never_offered = (
            bool(wire_runs)
            and key not in offered_anywhere
            and not counts.get(key, 0)
            and not near_miss(p.tool, set(obs.declared_tools) | obs.offered_tools())
        )
        # violation found (observed)
        n = counts.get(key, 0)
        if n:
            out.append(
                make_finding(
                    "AS-204",
                    confidence=Confidence.observed(n, max(real_runs, n)),
                    spec_ref=sref,
                    code_ref=_tool_ref(obs, p.tool),
                    detail=f"violation found: the model invoked prohibited tool {p.tool} "
                    f"in {n} of {max(real_runs, n)} real run(s)",
                    evidence=f"tool call {p.tool} in traces/captures; reason: {p.reason}",
                )
            )
        # static reachability (primary)
        if key in offered:
            out.append(
                make_finding(
                    "AS-203",
                    confidence=P(),
                    spec_ref=sref,
                    code_ref=_tool_ref(obs, p.tool),
                    detail=f"{p.tool} is prohibited ({p.reason or 'no reason'}) but the model was "
                    "offered it on the wire",
                    evidence=_wire_evidence(obs, p.tool),
                )
            )
        elif key in excluded and key in declared:
            # the code itself takes the tool out of the model's choice (excluded_functions= / disallowed_tools=)
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-203",
                    subject=p.tool,
                    confidence=H(),
                    note="declared, but excluded from the model's tool choice in code",
                )
            )
        elif key in declared and _is_bound(obs, p.tool):
            reach = obs.graph_reachable.get(declared[key].name) if obs.graph_reachable else None
            note = " (graph node currently unreachable from START — one edge away from live)" if reach is False else ""
            # (an agent with no link to the spec's agent still counts, by design: tests/test_audit_fixes.py F2)
            if key in conditional:  # bound under a filter the scan could not evaluate
                why = "is bound only under a condition the scan could not evaluate"
                out.append(
                    make_finding(
                        "AS-203",
                        confidence=H(),
                        severity="warning",
                        spec_ref=sref,
                        code_ref=_bound_ref(obs, p.tool) or declared[key].code_ref,
                        title=f"Prohibited tool {p.tool} {why}",
                        detail=f"{p.tool} is prohibited ({p.reason or 'no reason'}) and {why}; "
                        "it may not be reachable, but the code does not prove that",
                        evidence=f"declared {declared[key].kind} at {declared[key].code_ref}",
                    )
                )
            else:
                out.append(
                    make_finding(
                        "AS-203",
                        confidence=P(),
                        spec_ref=sref,
                        code_ref=_bound_ref(obs, p.tool) or declared[key].code_ref,
                        detail=f"{p.tool} is prohibited ({p.reason or 'no reason'}) but is bound to an "
                        f"agent/tool node in code{note}",
                        evidence=f"declared {declared[key].kind} at {declared[key].code_ref}; bound at "
                        f"{_bound_ref(obs, p.tool)}",
                    )
                )
        elif key in declared and is_mcp_server_tool(declared[key]):
            out.append(
                make_finding(
                    "AS-203",
                    confidence=P(),
                    spec_ref=sref,
                    code_ref=declared[key].code_ref,
                    title=f"Prohibited tool {p.tool} is exposed by the MCP server",
                    detail=f"{p.tool} is prohibited ({p.reason or 'no reason'}) but this MCP server "
                    "offers it to every client that connects",
                    evidence=f"declared {declared[key].kind} at {declared[key].code_ref}",
                )
            )
        elif key in declared and mine is None:
            out.append(
                make_finding(
                    "AS-203",
                    confidence=H(),
                    severity="warning",
                    spec_ref=sref,
                    code_ref=declared[key].code_ref,
                    title=f"Prohibited tool {p.tool} exists in code (no static binding found)",
                    detail=f"{p.tool} is declared but no binding was found; dynamic registration "
                    "cannot be ruled out statically",
                    evidence=f"declared {declared[key].kind} at {declared[key].code_ref}",
                )
            )
        elif never_offered:  # every captured run shows the model's tool list, and the tool is never in it
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-203",
                    subject=p.tool,
                    confidence=Confidence.observed(0, len(wire_runs)),
                    note=f"never offered to the model in {len(wire_runs)} of {len(wire_runs)} run(s), "
                    "and not bound in the scanned code",
                )
            )
        elif obs.extraction_errors or open_unresolved(obs) or obs.execution_model == "prompt-only":
            out.append(
                make_finding(
                    "AS-203",
                    confidence=U(),
                    spec_ref=sref,
                    title=f"Cannot rule out prohibited tool {p.tool}",
                    detail="extraction was incomplete, so absence of the tool is not proof" + _unresolved_note(obs),
                    evidence=f"extraction errors: {len(obs.extraction_errors)}; unresolved tools: "
                    f"{len(open_unresolved(obs))}; execution model: {obs.execution_model}",
                )
            )
        elif near_miss(p.tool, set(obs.declared_tools) | obs.offered_tools()):
            pass  # absent only by spelling: the look-alike is reported by AS-217, and this is no PASS
        else:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-203",
                    subject=p.tool,
                    confidence=absence_confidence(obs),
                    note="not declared, bound or offered anywhere in scanned code/wire",
                )
            )
        if not n and real_runs:
            ctx.pass_(PassedCheck(rule_id="AS-204", subject=p.tool, confidence=Confidence.observed(0, real_runs)))
    return out


def _detector_hits(
    ctx: CheckContext, obs: ObservedAgent, i: int, what: str, dtype: str, params: dict[str, object], real_runs: int
) -> list[Finding]:
    from agenticreq import detectors

    sref = ctx.spec_ref("prohibited", i)
    det = detectors.get(dtype)
    if det is None:
        return [
            make_finding(
                "AS-902",
                confidence=U(),
                spec_ref=sref,
                title=f"Unrecognised detector {dtype!r}",
                detail=f"prohibited[{i}] uses detector {dtype!r}, which is not registered",
                evidence=f"detector: {dtype}",
            )
        ]
    ctx.clause(extracted=real_runs > 0)
    if not real_runs:
        return []  # no real evidence: the clause stays unverified (AS-903 / AS-905 say why)
    hits = det.detect(dict(params), obs)
    if not hits:
        ctx.pass_(PassedCheck(rule_id="AS-204", subject=what, confidence=Confidence.observed(0, real_runs)))
        return []
    runs = {h.split(":", 1)[0] for h in hits}
    return [
        make_finding(
            "AS-204",
            confidence=Confidence.observed(len(runs), max(real_runs, len(runs))),
            spec_ref=sref,
            title=f"Violation found: prohibited action {what!r}",
            detail=f"detector {dtype} matched in {len(runs)} of {max(real_runs, len(runs))} real run(s)",
            evidence="; ".join(hits[:3]),
        )
    ]


def capture_runs(obs: ObservedAgent) -> dict[str, list[CapturedCall]]:
    """Captured calls per run, in order. Synthetic runs (discover) count: which tools the framework offers, and
    what the agent's code does when the model asks for a tool, are the code's behaviour, not the mock's."""
    runs: dict[str, list[CapturedCall]] = {}
    for c in sorted(obs.captures, key=lambda c: (c.run_id, c.index)):
        runs.setdefault(c.run_id, []).append(c)
    return runs


@dataclass
class ApprovalEvidence:
    held: list[str]  # runs where every request for the tool met an approval before its result
    ran_ungated: list[str]  # runs where the tool's result reached the model with no approval recorded
    kinds: list[str]  # gate kinds seen


def approval_evidence(obs: ObservedAgent, tool: str) -> ApprovalEvidence:
    """For each time the model asked for ``tool``: did an approval gate fire before its result came back?
    A request whose result never came back counts as held only when a gate fired (an interrupt that stopped the
    run); with no gate and no result the run says nothing either way."""
    key = canonical(tool)
    ev = ApprovalEvidence([], [], [])
    for run_id, calls in capture_runs(obs).items():
        verdicts = []
        for i, c in enumerate(calls):
            for tc in c.tool_calls:
                if canonical(tc.name) != key:
                    continue
                j = next(
                    (
                        j
                        for j in range(i + 1, len(calls))
                        if any(
                            (r.call_id and r.call_id == tc.id)
                            or (not r.call_id and r.name and canonical(r.name) == key)
                            for r in calls[j].tool_results
                        )
                    ),
                    None,
                )
                gates = [g for x in calls[i : j if j is not None else len(calls)] for g in x.gates_after]
                ev.kinds += [g for g in gates if g not in ev.kinds]
                verdicts.append("held" if gates else "ungated" if j is not None else "silent")
        if "ungated" in verdicts:
            ev.ran_ungated.append(run_id)
        elif "held" in verdicts:
            ev.held.append(run_id)
    return ev


@rule("conformance", "AS-205")
def approval_gates(ctx: CheckContext) -> Iterable[Finding]:
    s, obs = ctx.spec, ctx.observed
    if not s or not obs:
        return []
    out = []
    declared = _declared(obs)
    for i, a in enumerate(s.approval_required or []):
        key = canonical(a.tool)
        ctx.clause(extracted=key in declared)
        if key not in declared:
            continue  # AS-217 reports spec tools missing from code
        rt = approval_evidence(obs, a.tool)
        scoped = [g for g in obs.gates if g.tools and key in {canonical(t) for t in g.tools}]
        global_ = [
            g
            for g in obs.gates
            if g.tools is None
            and g.kind in ("interrupt", "interrupt_before", "human_input", "needs_approval", "approval_call")
        ]
        # a run with no approval says something when a gate exists, in the code or seen firing in another run; with
        # neither, the static gap stays
        if rt.ran_ungated and (scoped or global_ or rt.held):  # its result reached the model and no known gate fired
            n, total = len(rt.ran_ungated), len(rt.ran_ungated) + len(rt.held)
            out.append(
                make_finding(
                    "AS-205",
                    confidence=U(),
                    spec_ref=ctx.spec_ref("approval_required", i),
                    code_ref=declared[key].code_ref,
                    title=f"{a.tool} ran without a recorded approval in {n} of {total} run(s)",
                    detail=f"{a.tool} requires approval by {a.approver}; when the model asked for it, its "
                    "result came back with no approval recorded (agenticreq records input(), "
                    "LangGraph interrupt() and OpenAI Agents needs_approval). If your gate is "
                    "another one, sign it off under confirmed: in spec.yml",
                    evidence=f"runs: {', '.join(rt.ran_ungated[:5])}",
                )
            )
            continue
        if rt.held:  # seen at run time: an approval before every result, in every run that asked for the tool
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-205",
                    subject=a.tool,
                    confidence=Confidence.observed(0, len(rt.held)),
                    note=f"approval ({', '.join(rt.kinds)}) fired before {a.tool} ran, every time the "
                    "model asked for it",
                )
            )
            continue
        verified = [g for g in scoped if g.verified]
        # PASS only for a gate verified on every path to the tool's action; a gate on some paths, or one
        # whose scope is unknown, is UNKNOWN: it may be bypassed, and a PASS would say it cannot be
        if verified:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-205",
                    subject=a.tool,
                    confidence=H(),
                    note=f"{verified[0].kind} at {verified[0].code_ref}: {verified[0].coverage_note}",
                )
            )
        elif scoped or global_:
            g = scoped[0] if scoped else global_[0]
            why = g.coverage_note if scoped else "a global gate exists but which tools it covers is not known"
            out.append(
                make_finding(
                    "AS-205",
                    confidence=U(),
                    spec_ref=ctx.spec_ref("approval_required", i),
                    code_ref=g.code_ref or declared[key].code_ref,
                    title=f"Approval gate for {a.tool} not verified on every path",
                    detail=f"{a.tool} requires approval by {a.approver}; a {g.kind} gate exists but "
                    f"path coverage was not verified: {why}",
                    evidence=f"gates for {a.tool}: "
                    f"{[(x.kind, x.coverage, str(x.code_ref)) for x in (scoped or global_)]}",
                )
            )
        else:
            out.append(
                make_finding(
                    "AS-205",
                    confidence=H(),
                    spec_ref=ctx.spec_ref("approval_required", i),
                    code_ref=declared[key].code_ref,
                    detail=f"{a.tool} requires approval by {a.approver}, but no interrupt/HITL/"
                    "needs_approval/guardrail gate was found for it in code",
                    evidence=f"gates found: {[g.kind for g in obs.gates] or 'none'}",
                )
            )
    return out


@rule("conformance", "AS-206")
def role_boundary(ctx: CheckContext) -> Iterable[Finding]:
    s, obs = ctx.spec, ctx.observed
    if not s or not obs or not s.role_boundary or not s.role_boundary.allowed_tools:
        return []
    allowed = {canonical(t) for t in s.role_boundary.allowed_tools}
    prohibited = {canonical(t) for t in s.prohibited_tools()}
    injected = InjectedTools(ctx.injected_allowlist)
    out = []
    names = {t.name for t in code_tools(obs)} | obs.offered_tools()
    mine = own_tools(ctx)
    if mine is not None:
        names = {n for n in names if canonical(n) in mine}
    extra: list[str] = []
    for name in sorted(names):
        c = canonical(name)
        if (
            name.startswith("hosted:")
            or c in prohibited
            or injected.framework_of(name)
            or is_handoff_tool(name)
            or obs.framework_side_tool(name)
        ):
            continue
        ctx.clause(extracted=True)
        if c in allowed:
            ctx.pass_(PassedCheck(rule_id="AS-206", subject=name, confidence=P()))
            continue
        extra.append(name)
    if extra:  # one line: every tool the code has that the spec does not list
        out.append(
            make_finding(
                "AS-206",
                confidence=P(),
                spec_ref=ctx.spec_ref("role_boundary", "allowed_tools"),
                code_ref=_bound_ref(obs, extra[0]) or _tool_ref(obs, extra[0]),
                param="role_boundary",
                title=code_only(extra),
                detail=f"{code_only(extra)} (role_boundary.allowed_tools)",
                evidence=f"allowed_tools: {sorted(s.role_boundary.allowed_tools)}",
            )
        )
    return out


def code_only(names: list[str]) -> str:
    """A name the code has that the spec does not list: "the code has a, b; they are not in your spec"."""
    return f"the code has {', '.join(names)}; {'it is' if len(names) == 1 else 'they are'} not in your spec"


def spec_only(kind: str, names: list[str]) -> str:
    """A name in the spec that the code does not have: "spec lists a, b; the code has no such tools"."""
    return f"spec lists {', '.join(names)}; the code has no such {kind}{'s' if len(names) > 1 else ''}"


_LIMIT_MAP = {
    "max": ("maximum", "exclusiveMaximum"),
    "min": ("minimum", "exclusiveMinimum"),
    "max_length": ("maxLength",),
    "enum": ("enum",),
    "pattern": ("pattern",),
}


@rule("conformance", "AS-207", "AS-217")
def limits(ctx: CheckContext) -> Iterable[Finding]:
    s, obs = ctx.spec, ctx.observed
    if not s or not obs:
        return []
    out = []
    declared = _declared(obs)
    wire = {canonical(k): v for k, v in {**obs.mcp_tools, **obs.wire_tools}.items()}
    strict = ctx.risk_level in ("high", "critical")
    for i, lim in enumerate(s.limits or []):
        key = canonical(lim.tool)
        sref = ctx.spec_ref("limits", i)

        def _param(t: ToolSchema | None) -> ParamSchema | None:
            if t is None:
                return None
            return t.param(lim.arg) or next((x for x in t.params if canonical(x.name) == canonical(lim.arg)), None)

        code_t, wire_t = declared.get(key), wire.get(key)
        if code_t is None and wire_t is None:
            continue  # AS-217
        # constraints are read where the model sees them (wire) when the wire carries the parameter; whether
        # the parameter exists at all is a fact about the code declaration, not about what the framework sent
        p = _param(wire_t) or _param(code_t)
        tool = wire_t if _param(wire_t) is not None else (code_t or wire_t)
        assert tool is not None
        ctx.clause(extracted=p is not None)
        if p is None:
            where = "code" if code_t is not None else "the schema sent to the model"
            src_t = code_t or wire_t
            assert src_t is not None
            out.append(
                make_finding(
                    "AS-217",
                    confidence=P(),
                    spec_ref=sref,
                    code_ref=src_t.code_ref,
                    title=f"Limit names an argument {lim.tool} does not have: {lim.tool}.{lim.arg}",
                    detail=f"{lim.tool} has no parameter {lim.arg!r} in {where}",
                    evidence=f"params ({where}): {[x.name for x in src_t.params]}",
                )
            )
            continue
        for k, v in lim.limit.model_dump(exclude_none=True).items():
            schema_keys = _LIMIT_MAP.get(k, ())
            have = {sk: p.constraints[sk] for sk in schema_keys if sk in p.constraints}
            ok = False
            if k == "max":
                ok = any(isinstance(x, (int, float)) and x <= v for x in have.values())
            elif k == "min":
                ok = any(isinstance(x, (int, float)) and x >= v for x in have.values())
            elif k == "max_length":
                ok = any(isinstance(x, int) and x <= v for x in have.values())
            elif k == "enum":
                ok = "enum" in have and set(have["enum"]) <= set(v)
            elif k == "pattern":
                ok = have.get("pattern") == v
            src = "wire" if _param(wire_t) is not None else "code"
            if ok:
                ctx.pass_(
                    PassedCheck(
                        rule_id="AS-207",
                        subject=f"{lim.tool}.{lim.arg}.{k}",
                        confidence=P(),
                        note=f"{src} schema has {have}",
                    )
                )
                continue
            code_t = declared.get(key)
            body = code_t.body_bounds.get(p.name, []) if code_t else []
            in_body = k in ("max", "min") and any(abs(b - float(v)) < 1e-9 for b in body)
            numbers = [x for x in have.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
            both = (
                f"{lim.tool}.{lim.arg}: code allows {th.fmt(numbers[0])}; spec.yml says {th.fmt(v)}."
                if numbers and isinstance(v, (int, float))
                else None
            )
            out.append(
                make_finding(
                    "AS-207",
                    confidence=P(),
                    spec_ref=sref,
                    code_ref=tool.code_ref,
                    severity="blocker" if strict and not in_body else "warning",
                    reason=both,
                    detail=f"spec limits {lim.tool}.{lim.arg} {k} {th.fmt(v) if isinstance(v, (int, float)) else v}; the {src} schema has "
                    f"{have or 'no such constraint'}"
                    + (
                        " — the function body compares against it (a code pattern), but the model never sees it"
                        if in_body
                        else ""
                    ),
                    evidence=f"{src} {lim.tool}.{p.name}: type={p.type} constraints={p.constraints}",
                )
            )
    # spec references to tools that do not exist anywhere
    names = set(obs.declared_tools) | obs.offered_tools()
    known = set(declared) | {canonical(n) for n in names}
    prohibited = {canonical(t) for t in s.prohibited_tools()}
    if not obs.declared_tools:
        return out
    missing: dict[str, list[str]] = {}
    for name in sorted(s.spec_tools()):
        if canonical(name) in known:
            continue
        near = near_miss(name, names)
        if canonical(name) in prohibited:
            # an absent prohibited tool is the goal — unless the code has a look-alike the prohibition misses
            if near is None:
                continue
            i = next(i for i, x in enumerate(s.prohibited or []) if x.tool and canonical(x.tool) == canonical(name))
            out.append(
                make_finding(
                    "AS-217",
                    confidence=H(),
                    severity="blocker" if strict else "warning",
                    spec_ref=ctx.spec_ref("prohibited", i),
                    code_ref=_tool_ref(obs, near),
                    title=f"Prohibition {name!r} matches no tool, but {near!r} exists",
                    detail=f"prohibited names {name!r}; code has {near!r}. If they are the same tool, "
                    "the prohibition protects nothing and it is reachable",
                    evidence=f"spec: {name}; code: {near}",
                )
            )
            continue
        missing.setdefault(_spec_field_of(s, name), []).append(name + (f" (did you mean {near}?)" if near else ""))
    for where, listed in missing.items():  # one line per spec field: every name it lists that the code lacks
        out.append(
            make_finding(
                "AS-217",
                confidence=absence_confidence(obs),
                param=where,
                spec_ref=ctx.spec_ref(where.split("[")[0].split(".")[0]),
                title=spec_only("tool", listed),
                detail=f"{spec_only('tool', listed)} (in code or on the wire){_unresolved_note(obs)}",
                evidence=f"declared: {sorted(obs.declared_tools)}",
            )
        )
    return out


def _spec_field_of(s: AgentSpec, name: str) -> str:
    """Where the spec lists a tool name: role_boundary.allowed_tools, approval_required[].tool, limits[].tool."""
    c = canonical(name)
    if s.role_boundary and any(canonical(t) == c for t in s.role_boundary.allowed_tools):
        return "role_boundary.allowed_tools"
    if any(canonical(a.tool) == c for a in s.approval_required or []):
        return "approval_required[].tool"
    if any(canonical(lim.tool) == c for lim in s.limits or []):
        return "limits[].tool"
    return "prohibited[].tool"


# =============================================================== Tier 2 (audit = zero-config)
_STEP_KINDS = {
    "recursion_limit",
    "max_iter",
    "max_iterations",
    "max_turns",
    "max_steps",
    "request_limit",
}


def effective_step_bound(obs: ObservedAgent) -> tuple[str, float | None, str] | None:
    """(kind, value, where) of the loop bound this agent runs under: explicit in code, else the framework's
    finite default (data/framework_defaults.yml). None = nothing bounds the loop."""
    steps = [b for b in obs.bounds if b.kind in _STEP_KINDS]
    if steps:
        return steps[0].kind, steps[0].value, f"code {steps[0].code_ref}"
    from agenticreq.data import load_yaml

    dflt = ((load_yaml("framework_defaults.yml") or {}).get("step_bounds") or {}).get(
        obs.framework if obs.framework_confirmed else ""
    )  # a declared-only dependency sets no default
    if dflt:
        return dflt["kind"], float(dflt["value"]), f"framework default ({dflt['source']}, {obs.framework_tag()})"
    return None


def _agents_of_run(obs: ObservedAgent) -> list[str]:
    """The named agents of a multi-agent run (a team group is not one); empty for a single agent."""
    names = [a.name for a in obs.declared_agents if a.kind != "group_chat"]
    return names if len(names) > 1 else []


@rule("conformance", "AS-208", "AS-209", modes=_BOTH)
def step_bounds(ctx: CheckContext) -> Iterable[Finding]:
    """bounds.max_steps. A step bound passes when it exists, and when it is at least as strict as the spec's max_steps
    if the spec has one (agenticreq/thresholds.py). The library proposes no number."""
    obs = ctx.observed
    if not obs or not obs.declared_tools:
        return []
    if not obs.declared_agents and all(is_mcp_server_tool(t) for t in obs.declared_tools.values()):
        return []  # a tool server only: there is no agent loop here to bound
    p, risk = th.MAX_STEPS, ctx.risk_level
    spec_max = th.spec_value(ctx, p)
    steps = [b for b in obs.bounds if b.kind in _STEP_KINDS]
    ctx.clause(extracted=True)
    counters = [r for r in obs.signal("loop_counter") if not is_test_file(r.file)]
    if not steps and counters:  # `if state.loop_step < cfg.max_loops`: a comparison, not a verified bound
        return [
            make_finding(
                "AS-208",
                confidence=U(),
                spec_ref=ctx.spec_ref("bounds") if ctx.loaded else None,
                title="Loop bound not verified",
                param=p.key,
                detail=f"the code compares a step/iteration counter with a limit at {counters[0]}"
                + (f" (+{len(counters) - 1})" if len(counters) > 1 else "")
                + ", but agenticreq cannot verify that it stops every path of the agent loop",
                evidence=f"loop counters: {[str(r) for r in counters[:3]]}; framework bounds: none",
            )
        ]
    sref = ctx.spec_ref("bounds") if ctx.loaded else None
    if not steps:
        dflt = effective_step_bound(obs)  # no explicit bound, so this is the framework default (or None)
        if dflt and dflt[1] is not None:
            kind, value, where = dflt
            d = th.judge(p, risk, spec=spec_max, code=value)
            if d.status == "FAIL":
                return [
                    make_finding(
                        "AS-209",
                        confidence=H(),
                        spec_ref=sref,
                        reason=f"No bound in code: {where} allows {value:g} steps; spec.yml says {th.fmt(spec_max)}.",
                        status_hint=d.line_status,
                        param=p.key,
                        title=f"Framework default {kind}={value:g} exceeds the spec bound",
                        detail=f"no explicit bound in code; {where}: {kind}={value:g}, spec.yml "
                        f"max_steps={th.fmt(spec_max)}",
                        evidence=f"framework_defaults.yml: {obs.framework}",
                    )
                ]
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-208",
                    subject=kind,
                    confidence=H(),
                    param=p.key,
                    note=f"no explicit bound; {where}: {kind}={value:g}",
                )
            )
            return []
        d = th.judge(p, risk, spec=spec_max)  # nothing sets a step bound (a framework default that means "no limit")
        return [
            make_finding(
                "AS-208",
                confidence=H(),
                spec_ref=sref,
                reason=d.reason,
                status_hint=d.line_status,
                param=p.key,
                detail="no recursion_limit / max_iter / max_turns / max_steps found",
                evidence=f"bounds found: {bounds_summary(obs.bounds)}",
            )
        ]
    out: list[Finding] = []
    for b in steps:
        if b.value is None:
            out.append(
                make_finding(
                    "AS-209",
                    confidence=U(),
                    code_ref=b.code_ref,
                    param=p.key,
                    agent=b.agent,
                    title=f"{b.label} is not a literal",
                    detail=f"{b.label} value could not be read statically",
                    evidence=f"{b.label} at {b.code_ref}",
                )
            )
            continue
        # a value that is the default of an environment read is assumed: the bound exists, the deployment may retune it
        conf = H() if b.assumed_from else P()
        assumed = f" (assumed: {b.assumed_from}; the deployment may override it)" if b.assumed_from else ""
        d = th.judge(p, risk, spec=spec_max, code=b.value)
        if d.status == "FAIL":
            out.append(
                make_finding(
                    "AS-209",
                    confidence=conf,
                    code_ref=b.code_ref,
                    spec_ref=ctx.spec_ref("bounds"),
                    reason=d.reason,
                    status_hint=d.line_status,
                    param=p.key,
                    agent=b.agent,
                    detail=f"{b.label}={int(b.value)} in code{assumed}, spec.yml max_steps={th.fmt(spec_max)}",
                    evidence=b.describe(),
                )
            )
        elif b.assumed_from:  # an assumed value is a guess about the deployment: never a PASS
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-209",
                    subject=b.kind,
                    confidence=conf,
                    param=p.key,
                    agent=b.agent,
                    note=f"{b.label}={b.value:g}{assumed}",
                )
            )
        else:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-209",
                    subject=th.fmt(b.value),
                    confidence=conf,
                    param=p.key,
                    agent=b.agent,
                    note=str(b.code_ref) if b.code_ref else "",
                )
            )
    # several agents: one without a bound of its own, and no run-level bound over all of them, has none
    agents = _agents_of_run(obs)
    if agents and all(b.agent for b in steps):
        d = th.judge(p, risk, spec=spec_max)
        for name in agents:
            if not any(b.agent == name for b in steps):
                out.append(
                    make_finding(
                        "AS-208",
                        confidence=H(),
                        spec_ref=sref,
                        reason=d.reason,
                        status_hint=d.line_status,
                        param=p.key,
                        agent=name,
                        detail=f"{name} sets no recursion_limit / max_iter / max_turns / max_steps",
                        evidence=f"bounds found: {bounds_summary(obs.bounds)}",
                    )
                )
    return out


@rule("conformance", "AS-209", modes=("check",))
def static_bounds(ctx: CheckContext) -> Iterable[Finding]:
    """bounds.max_retries and bounds.max_tool_calls. A number in code is reported (PASS);
    when spec.yml states one, the number in code must be at least as strict. A number nowhere is not newly required:
    these two never had a risk_level requirement."""
    obs = ctx.observed
    if not obs or not ctx.loaded or not obs.declared_tools:
        return []
    if not obs.declared_agents and all(is_mcp_server_tool(t) for t in obs.declared_tools.values()):
        return []
    risk = ctx.risk_level
    out: list[Finding] = []
    for p, kinds in ((th.MAX_RETRIES, ("max_retries",)), (th.MAX_TOOL_CALLS, ("max_tool_calls", "tool_calls_limit"))):
        spec_n = th.spec_value(ctx, p)
        if th.spec_todo(ctx, p):
            continue  # AS-101 reports a TODO
        found = [b for b in obs.bounds if b.kind in kinds and b.value is not None]
        if spec_n is None and not found:
            continue  # written nowhere, and never asked for
        ctx.clause(extracted=True)
        sref = ctx.spec_ref(*p.key.split(".")) if spec_n is not None else None
        if not found:
            out += th.emit(
                ctx,
                p,
                th.judge(p, risk, spec=spec_n),
                risk=risk,
                spec=spec_n,
                spec_ref=sref,
                evidence=f"bounds found: {bounds_summary(obs.bounds)}",
            )
            continue
        for b in found:
            assert b.value is not None
            out += th.emit(
                ctx,
                p,
                th.judge(p, risk, spec=spec_n, code=b.value),
                risk=risk,
                code=b.value,
                spec=spec_n,
                code_ref=b.code_ref,
                spec_ref=sref,
                agent=b.agent,
                evidence=b.describe(),
            )
    return out


@rule("conformance", "AS-210", modes=_BOTH)
def model_pinning(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or ctx.effective().get("version_pinning") is False:
        return []
    # one finding per distinct model: "openai:gpt-4o" in code and "gpt-4o" on the wire are the same model,
    # and ten call sites of one alias are one problem. Test files configure models too; they don't ship.
    groups: dict[str, list[ModelRef]] = {}
    for m in obs.models:
        if m.code_ref is not None and is_test_file(m.code_ref.file):
            continue
        groups.setdefault(model_key(m.name), []).append(m)
    out = []
    # a model chosen at run time (env var, config, helper) is a clause we could not read: UNKNOWN, and it stays
    # in the coverage denominator so the gap cannot hide itself
    for u in obs.unresolved_models:
        if is_test_file(u.code_ref.file):
            continue
        ctx.clause(extracted=False)
        out.append(
            make_finding(
                "AS-210",
                confidence=U(),
                code_ref=u.code_ref,
                title="Model is chosen at run time, so its pinning cannot be checked",
                detail=f"{u.reason}: which model version runs is not in the code, so whether it is "
                "pinned cannot be checked",
                evidence=f"model argument at {u.code_ref}",
                remediation="Pin the model in code (a dated model id), or record the exact model id "
                "in the spec and keep regression_triggers: [model_change].",
            )
        )
    stated = {model_key(x): x for x in (ctx.spec.pinned_models if ctx.spec else None) or []}
    for key, refs in groups.items():
        if stated and key not in stated:  # the owner said which ids run; the code uses another
            first = next((m for m in refs if m.code_ref is not None), refs[0])
            ctx.clause(extracted=True)
            out.append(
                make_finding(
                    "AS-210",
                    confidence=P(),
                    code_ref=first.code_ref,
                    spec_ref=ctx.spec_ref("pinned_models"),
                    param="version_pinning",
                    status_hint="GAP",
                    title=f"Model {first.name!r} is not in pinned_models",
                    detail=f"the code uses {first.name} ({first.code_ref}); spec.yml pinned_models lists {', '.join(sorted(stated.values()))}",
                    reason=f"The code uses {first.name} ({first.code_ref}); spec.yml pinned_models lists "
                    f"{', '.join(sorted(stated.values()))}.",
                    evidence=f"model={first.name!r}",
                )
            )
            continue
        if key in stated:  # the owner's statement decides what the id cannot show
            first = next((m for m in refs if m.code_ref is not None), refs[0])
            ctx.clause(extracted=True)
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-210",
                    subject=first.name,
                    confidence=P(),
                    note=f"listed in pinned_models, {first.code_ref}",
                )
            )
            continue
        no_snapshots = next((m.provider for m in refs if not offers_snapshots(m.provider)), None)
        if no_snapshots and not all(m.pinned for m in refs):
            # pinning is impossible here: say what that means instead of advising the impossible
            ctx.clause(extracted=False)
            first = next((m for m in refs if m.code_ref is not None), refs[0])
            out.append(
                make_finding(
                    "AS-210",
                    confidence=U(),
                    severity="info",
                    code_ref=first.code_ref,
                    title=f"Provider {no_snapshots} offers no model snapshots: {first.name!r} can change without a "
                    "code change",
                    detail=f"{first.name} is served by {no_snapshots}, which publishes no dated / immutable model "
                    "versions, so version pinning cannot be enforced in code",
                    evidence=f"provider={no_snapshots} (data/provider_pinning.yml); model={first.name!r}",
                    remediation="Record the exact model id in the spec, keep regression_triggers: [model_change], and "
                    "re-run your evals whenever the provider announces a model change or deprecation.",
                    fixable=False,
                )
            )
            continue
        ctx.clause(extracted=True)
        assumed_from = next((m.assumed_from for m in refs if m.assumed_from), None)
        conf = H() if assumed_from else P()
        if all(m.pinned for m in refs):
            at = next((m.code_ref for m in refs if m.code_ref is not None), None)
            rule = pin_judge(refs[0].name)[1]
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-210",
                    subject=refs[0].name,
                    confidence=conf,
                    note=f"assumed: {assumed_from}; confirm the deployed value"
                    if assumed_from
                    else f"{rule}{f', {at}' if at else ''}",
                )
            )
            continue
        bad = [m for m in refs if not m.pinned]
        first = next((m for m in bad if m.code_ref is not None), bad[0])
        state, rule = pin_judge(first.name)
        if state == "undecidable":  # never "not pinned": the id does not say
            out.append(
                make_finding(
                    "AS-210",
                    confidence=Confidence.attestation(),
                    severity="attestation",
                    code_ref=first.code_ref,
                    param="version_pinning",
                    status_hint="SIGN-OFF",
                    title=f"Is {first.name!r} a fixed version?",
                    detail=f"{first.name}: {rule}, so the id alone does not say whether it is a fixed version",
                    reason=f"Is this model id a fixed version? {first.name} ({rule}, {first.code_ref}). Confirm under "
                    "confirmed: in spec.yml, or list it under pinned_models:.",
                    evidence=f"model={first.name!r}; rule: {rule}",
                )
            )
            continue
        sites = [str(m.code_ref) for m in bad if m.code_ref is not None]
        sources = sorted({m.source for m in bad})
        out.append(
            make_finding(
                "AS-210",
                confidence=conf,
                code_ref=first.code_ref,
                title=f"Model {first.name!r} is not pinned" + (f" (assumed: {assumed_from})" if assumed_from else ""),
                reason=f"Model {first.name!r} is not pinned ({rule}, {first.code_ref})."
                if first.code_ref is not None
                else None,
                detail=f"{first.name} is an alias; the provider can change the model behind it"
                + (f" ({len(sites)} call sites)" if len(sites) > 1 else ""),
                evidence=f"{'+'.join(sources)}: model={first.name!r}"
                + (f"; sites: {', '.join(sites[:6])}" if len(sites) > 1 else ""),
            )
        )
    return out


@rule("conformance", "AS-232", modes=("check",))
def declared_effects(ctx: CheckContext) -> Iterable[Finding]:
    """tool_effects: in spec.yml says a tool reads, and its body writes: the line that writes is the evidence."""
    s, obs = ctx.spec, ctx.observed
    if not s or not obs or not s.tool_effects:
        return []
    out = []
    for t in code_tools(obs):
        if t.declared_effect is None:
            continue
        ctx.clause(extracted=True)
        where = f"{t.code_ref.file}:{t.effect_line}" if t.code_ref and t.effect_line else str(t.code_ref or "")
        if t.declared_effect == "read" and t.effect == "write":
            out.append(
                make_finding(
                    "AS-232",
                    confidence=P(),
                    code_ref=t.code_ref,
                    param="tool_effects",
                    spec_ref=ctx.spec_ref("tool_effects", t.name),
                    title=f"{t.name} is declared read, but it writes",
                    detail=f"tool_effects: says {t.name} only reads; its body {t.effect_reason} at {where}",
                    evidence=f"{t.effect_reason} at {where}",
                    reason=f"spec.yml declares {t.name} read, but it {t.effect_reason} at {where}.",
                )
            )
        elif t.effect == t.declared_effect:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-232",
                    subject=t.name,
                    confidence=P(),
                    param="tool_effects",
                    note=f"declared {t.declared_effect}, and the body agrees ({t.effect_reason})",
                )
            )
    return out


@rule("conformance", "AS-211", "AS-212", "AS-213", "AS-215", "AS-216", modes=_BOTH)
def tool_shape(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    eff = ctx.effective()
    s = ctx.spec
    approval = {canonical(t) for t in (s.approval_tools() if s else {})}
    # what the spec itself says acts on the world: a tool behind an approval is a write, whatever its name says
    # (names are an open vocabulary; the human-written spec is not). A numeric limit says so only when the name
    # does not already say the tool reads: read_file.max_chars caps a read, process_refund.amount caps an action
    limited = {canonical(lim.tool) for lim in (s.limits or [] if s else [])}
    declared_actions = approval | {c for c in limited if verb_class(c) != "read"}
    prohibited = {canonical(t) for t in (s.prohibited_tools() if s else set())}
    gated = {canonical(t) for g in obs.gates if g.verified for t in (g.tools or [])}
    partly = {canonical(t): g for g in obs.gates if not g.verified for t in (g.tools or [])}
    global_gate = any(
        g.tools is None
        and g.kind in ("interrupt", "interrupt_before", "human_input", "needs_approval", "approval_call")
        for g in obs.gates
    )
    unbounded_policy = eff.get("unbounded_tool_policy", "warn")
    out = []
    for t in obs.loose_functions:  # no framework: what each function does, not whether the model can call it
        what = (
            "executes dynamic code / shell"
            if "exec" in t.features
            else "fetches a URL taken from its arguments"
            if "url_from_arg" in t.features
            else "reads without a cap"
        )
        rid = "AS-215" if "exec" in t.features else "AS-216" if "url_from_arg" in t.features else "AS-213"
        out.append(
            make_finding(
                rid,
                confidence=H(),
                severity="warning",
                code_ref=t.code_ref,
                title=f"{t.name} {what}",
                detail=f"no agent framework was recognised, so whether the model can call {t.name} is "
                f"unknown; if it can, it {what} with no bound in the code",
                evidence=f"function {t.name} at {t.code_ref}; features: {t.features}",
            )
        )
    for t in code_tools(obs):
        c = canonical(t.name)
        if c in prohibited:
            continue
        if is_write_tool(t) or (c in declared_actions and not (c not in approval and claims_read_only(t))):
            ctx.clause(extracted=True)
            # AS-211 needs evidence beyond the name (validation R2: name-only was 4/4 false positives).
            # It fires when the spec explicitly demands idempotency keys, or when the body shows a
            # side effect that a retry repeats (fresh id, record append, remote write).
            explicit = bool(s and s.write_safety is not None and s.write_safety.idempotency_key)
            body = "non_idempotent" in t.features
            if (eff.get("write_safety") or {}).get("idempotency_key") and not has_idempotency(t):
                if explicit or body:
                    why = (
                        "its body repeats the side effect on a retry (fresh id / record append / remote write)"
                        if body
                        else "the spec requires an idempotency key on every write (write_safety)"
                    )
                    unsure = effect_unsure(t)  # the name says write; the body cannot be followed
                    out.append(
                        make_finding(
                            "AS-211",
                            confidence=U() if unsure else H(),
                            code_ref=t.code_ref,
                            spec_ref=ctx.spec_ref("write_safety") if explicit else None,
                            reason=_then_run_text(
                                f"Cannot tell whether {t.name} writes: it {t.effect_reason} (line {t.effect_line})"
                            )
                            if unsure
                            else None,
                            status_hint="CHECK" if unsure else None,
                            title=f"Cannot tell whether {t.name} needs an idempotency key"
                            if unsure
                            else f"Write tool {t.name} has no idempotency key",
                            detail=f"{t.name} takes no idempotency key and {why}",
                            evidence=f"params: {[p.name for p in t.params]}; body features: {t.features or 'none'}",
                        )
                    )
                else:
                    ctx.clause(extracted=False)  # write by name only, body shows no repeatable effect
            # zero-config audit: only irreversible / external writes need a gate (a timer or a note does not);
            # with a spec at high/critical risk every write does, as before
            high = is_high_impact_write(t) or c in approval
            # irreversible / external writes need a gate at every risk level, with or without a spec (a
            # medium spec must not be quieter than no spec); at high / critical every write does
            needs_gate = high or ctx.risk_level in ("high", "critical")
            if is_mcp_server_tool(t) and not _is_bound(obs, t.name):
                needs_gate = False  # an MCP server provides the tool; the client confirms, via AS-218 annotations
            if (
                needs_gate
                and c not in approval
                and c not in gated
                and (c in partly or not global_gate)
                and effect_unsure(t)
            ):
                # the name says write, the body calls something agenticreq cannot follow: a mock run decides
                out.append(
                    make_finding(
                        "AS-212",
                        confidence=U(),
                        code_ref=t.code_ref,
                        title=f"Cannot tell whether {t.name} writes",
                        detail=f"{t.name} {t.effect_reason}; its name suggests a write that would run "
                        "without approval. Declare it under tool_effects: in spec.yml, or let "
                        "a mock run show it",
                        evidence=f"line {t.effect_line}; gates found: {[g.kind for g in obs.gates] or 'none'}",
                        reason=_then_run_text(
                            f"Cannot tell whether {t.name} writes: it {t.effect_reason} (line {t.effect_line})"
                        ),
                        status_hint="CHECK",
                    )
                )
            elif needs_gate and c not in approval and c not in gated and (c in partly or not global_gate):
                part = partly.get(c)
                out.append(
                    make_finding(
                        "AS-212",
                        confidence=H(),
                        code_ref=t.code_ref,
                        title=f"Write tool {t.name} runs without approval"
                        if part is None
                        else f"Write tool {t.name} can run without approval on some paths",
                        detail=(f"{t.name} {t.effect_reason} (line {t.effect_line}): " if t.effect == "write" else "")
                        + f"{t.name} looks like a {'high-impact ' if high else ''}write "
                        + (
                            "and nothing gates it (no approval_required entry, no interrupt/HITL in code)"
                            if part is None
                            else f"and its {part.kind} gate does not cover every path: {part.coverage_note}"
                        ),
                        evidence=f"gates found: {[g.kind for g in obs.gates] or 'none'}; "
                        f"body features: {t.features or 'none'}",
                    )
                )
        elif (
            ctx.risk_level in ("high", "critical")
            and t.params
            and t.effect != "read"
            and t.declared_effect is None
            and (effect_unsure(t) or verb_class(t.name) == "unknown")
            and not claims_read_only(t)
            and t.kind != "agent-as-tool"
            and not is_handoff_tool(c)
            and not is_mcp_server_tool(t)
        ):
            # the body cannot be followed (or there is no body and the verb says nothing): say so instead of
            # silently treating the tool as a read
            out.append(
                make_finding(
                    "AS-212",
                    confidence=U(),
                    code_ref=t.code_ref,
                    title=f"Cannot tell whether {t.name} changes anything",
                    reason=_then_run_text(
                        f"Cannot tell whether {t.name} changes anything: it {t.effect_reason} (line {t.effect_line})"
                    )
                    if effect_unsure(t)
                    else None,
                    status_hint="CHECK" if effect_unsure(t) else None,
                    detail=f"the verb {action_verb(c)!r} is in neither the read nor the write "
                    f"vocabulary, so the name does not show whether {t.name} acts. If it does, "
                    "list it under approval_required or limits in the spec (or add the verb to "
                    "data/action_verbs.yml) so it is checked",
                    evidence="vocabulary: data/action_verbs.yml",
                )
            )
        if (
            unbounded_policy != "allow"
            and t.limit_runtime
            and not t.limits[1:]
            and is_unbounded(
                t.model_copy(update={"features": [f for f in t.features if f != "bounded_output_runtime"]})
            )
        ):
            # the body caps it, with a number read at run time: the cap exists, its value is not known here
            out.append(
                make_finding(
                    "AS-213",
                    confidence=U(),
                    code_ref=t.code_ref,
                    param="output_size_limits",
                    title=f"{t.name}: limit set from {t.limit_runtime}",
                    detail=f"{t.name} {t.limits[0]}; the value is read at run time",
                    evidence=f"limits: {t.limits}",
                    status_hint="CHECK",
                    reason=f"{t.name}: limit set from {t.limit_runtime}, so its value is not in the "
                    f"code. Set it under env: in mocks.yml, then run: agenticreq mock-run",
                )
            )
        elif (
            unbounded_policy != "allow"
            and t.limits
            and is_unbounded(t.model_copy(update={"features": [f for f in t.features if f != "bounded_output"]}))
        ):
            # it would be unbounded by its name or body, but the body caps it: say how, and where
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-213",
                    subject=t.name,
                    confidence=P(),
                    param="output_size_limits",
                    note=t.limits[0],
                    reason=f"{t.name} {t.limits[0]}.",
                )
            )
        elif unbounded_policy != "allow" and is_unbounded(t):
            # a name like search_* is a guess: only a body that reads without a cap is evidence strong enough to block
            hard = "unbounded_read" in t.features
            out.append(
                make_finding(
                    "AS-213",
                    confidence=H(),
                    code_ref=t.code_ref,
                    severity="blocker" if unbounded_policy == "forbid" and hard else "warning",
                    title=f"{t.name} can return unbounded output",
                    detail=f"{t.name} reads a collection but has no limit/max_results/top_k param",
                    evidence=f"params: {[p.name for p in t.params]}",
                )
            )
        if "exec" in t.features:
            out.append(
                make_finding(
                    "AS-215",
                    confidence=H(),
                    code_ref=t.code_ref,
                    title=f"{t.name} executes dynamic code / shell",
                    detail=f"{t.name} calls exec/eval/subprocess/os.system; sandbox_policy="
                    f"{eff.get('sandbox_policy')!r}",
                    evidence=f"features: {t.features}",
                )
            )
        egress = eff.get("egress_policy") or {}
        if "url_from_arg" in t.features and egress.get("default", "deny") == "deny":
            ctx.clause(extracted=True)
            if "egress_guard" in t.features:  # a top-level check runs before any statement that fetches
                ctx.pass_(
                    PassedCheck(
                        rule_id="AS-216",
                        subject=f"egress guard in {t.name}",
                        confidence=H(),
                        note="the URL is checked against an allowlist before every HTTP call",
                    )
                )
                continue
            if "egress_guard_partial" in t.features:  # a check exists; that it runs before every fetch is not shown
                out.append(
                    make_finding(
                        "AS-216",
                        confidence=U(),
                        code_ref=t.code_ref,
                        title=f"Egress guard in {t.name} not verified on every path",
                        detail=f"{t.name} checks the URL against an allowlist, but not in a position "
                        "that runs before every HTTP call (inside a branch / try / loop, or "
                        "after a fetch)",
                        evidence=f"features: {t.features}",
                    )
                )
                continue
            allow = egress.get("allow", [])
            out.append(
                make_finding(
                    "AS-216",
                    confidence=H(),
                    code_ref=t.code_ref,
                    title=f"{t.name} fetches a model-controlled URL",
                    detail=f"{t.name} passes a parameter into an HTTP call and checks no allowlist "
                    "(host set, URL prefix, regex or validator) before it"
                    + (f"; the spec allows only {allow} but the code does not enforce it" if allow else ""),
                    evidence=f"features: {t.features}",
                )
            )
    return out


@rule("conformance", "AS-214", modes=_BOTH)
def secrets(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    return [
        make_finding(
            "AS-214",
            confidence=P(),
            code_ref=sr.code_ref,
            detail=f"string literal assigned to {sr.name} looks like a credential",
            evidence=f"{sr.name} = {sr.preview!r} (redacted)",
        )
        for sr in obs.secrets
    ]


@rule("conformance", "AS-218", modes=_BOTH)
def mcp_destructive_annotations(ctx: CheckContext) -> Iterable[Finding]:
    """MCP servers tell clients which tools change things (ToolAnnotations). The spec's defaults are the
    safe ones (readOnlyHint=false, destructiveHint=true), so an unannotated tool is treated as destructive.
    The risk is a destructive-looking tool that *claims* otherwise: clients may then run it without asking."""
    obs = ctx.observed
    if not obs:
        return []
    from agenticreq.ast.classify import _HIGH_IMPACT_VERBS, action_verb

    out = []
    for t in obs.declared_tools.values():
        if not is_mcp_server_tool(t) or action_verb(t.name) not in _HIGH_IMPACT_VERBS:
            continue
        ctx.clause(extracted="mcp:annotations=dynamic" not in t.features)
        ro = "mcp:readOnlyHint=true" in t.features
        dfalse = "mcp:destructiveHint=false" in t.features
        if not (ro or dfalse):
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-218",
                    subject=t.name,
                    confidence=P(),
                    note="no read-only / non-destructive claim (MCP defaults treat it as destructive)",
                )
            )
            continue
        out.append(
            make_finding(
                "AS-218",
                confidence=P(),
                code_ref=t.code_ref,
                title=f"MCP tool {t.name} looks destructive but is annotated as safe",
                detail=f"{t.name} is named like a destructive action ({action_verb(t.name)}) but declares "
                f"{'readOnlyHint=true' if ro else 'destructiveHint=false'}: clients may run it without asking",
                evidence=f"annotations: {[f for f in t.features if f.startswith('mcp:')]}",
            )
        )
    return out
