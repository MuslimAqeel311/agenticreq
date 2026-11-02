"""AS-4xx — taint: untrusted source → LLM → write sink.

Static (AS-401): an agent that holds both an untrusted source and an un-gated write
tool can be steered by that source — `heuristic`. Untrusted content reaches an agent's context
three ways, and all three are followed:

1. the agent holds the source tool itself;
2. Python code calls the source directly and the agent built in that file reads the result
   (a LangGraph prefetch node) — invisible at the wire by construction;
3. another agent holding the source is connected by an edge. Direction matters: a delegate's
   answer returns to the delegator (CrewAI delegation, agents-as-tools: dst → src), while a handoff
   moves the conversation forward and never returns (``transfer_to_*``, ``handoffs=``: src → dst).
Observed (AS-402): in a real (non-synthetic) run, a source's result entered the
context and a write tool was called afterwards — "violation found", `observed (n/N)`.
"""

from __future__ import annotations

from typing import Any, Iterable

from agenticreq.ast.classify import canonical, effect_unsure, is_mcp_server_tool, is_write_tool, looks_untrusted_source
from agenticreq.graph.analysis import build
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding
from agenticreq.models.observed import ObservedAgent, ToolSchema
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")


def _agent_tools(obs: ObservedAgent) -> list[ToolSchema]:
    """Tools an agent in this code can call: MCP-server tools that no agent here binds are provided to
    clients elsewhere, and have no LLM context in this code."""
    bound = {canonical(k) for k in obs.bound_tools}
    return [t for t in obs.declared_tools.values() if not is_mcp_server_tool(t) or canonical(t.name) in bound] + list(
        obs.wire_tools.values()
    )


def _sources(ctx: CheckContext, obs: ObservedAgent) -> set[str]:
    tools = _agent_tools(obs)
    if ctx.spec and ctx.spec.untrusted_inputs is not None:
        declared = {canonical(s) for s in ctx.spec.untrusted_sources()}
        out = set()
        for t in tools:
            c = canonical(t.name)
            if c in declared or any(d in c for d in declared):
                out.add(c)
        return out
    return {canonical(t.name) for t in tools if looks_untrusted_source(t)}


def _sinks(ctx: CheckContext, obs: ObservedAgent) -> dict[str, ToolSchema]:
    gated = {canonical(t) for g in obs.gates if g.verified for t in (g.tools or [])}  # all paths only
    approval = {canonical(t) for t in (ctx.spec.approval_tools() if ctx.spec else {})}
    out = {}
    for t in _agent_tools(obs):
        c = canonical(t.name)
        if is_write_tool(t) and c not in gated:
            out[c] = t
        elif c in approval and c not in gated:
            out[c] = t
    return out


def _groups(ctx: CheckContext, obs: ObservedAgent) -> dict[str, set[str]]:
    """Which tools share a context window (one agent node)."""
    g = build(obs, ctx.spec)
    # a manager with no tools of its own still carries context between its delegates, so it stays a node
    linked = {n for e in g.edges for n in (e.src, e.dst)}
    groups = {n: {canonical(t) for t in tools} for n, tools in g.nodes.items() if tools or n in linked}
    if any(groups.values()):
        return groups
    # no agent graph: tools bound together in code are one context; everything else shares one
    every = {canonical(t.name) for t in _agent_tools(obs)}
    out = {
        f"tools[{', '.join(grp[:3])}{', ...' if len(grp) > 3 else ''}]": {canonical(t) for t in grp}
        for grp in obs.tool_groups
        if len(obs.tool_groups) > 1
    }
    rest = every - set().union(*out.values()) if out else every
    if rest:
        out["agent"] = rest
    return out


@rule("taint", "AS-401", modes=_BOTH)
def static_taint(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    sources, sinks = _sources(ctx, obs), _sinks(ctx, obs)
    global_gate = any(
        g.tools is None
        and g.kind in ("interrupt", "interrupt_before", "human_input", "needs_approval", "approval_call")
        for g in obs.gates
    )
    if not sinks or not sources and not _zero_trust(ctx):
        return []
    out = []
    strict = ctx.risk_level in ("high", "critical")
    conf = Confidence.heuristic()
    # one finding per (source, sink): the same untrusted text reaching the same tool through several agent orders
    # (cycles, coordinator hops) is one problem, reported with its shortest path
    shortest: dict[tuple[str, str], tuple[Any, ...]] = {}
    for item in _flows(ctx, obs, sources, set(sinks)):
        key = (item[1], item[2])
        if key not in shortest or len(item[3]) < len(shortest[key][3]):
            shortest[key] = item
    for node, src, sink, path, how in sorted(shortest.values(), key=lambda i: (i[1], i[2])):
        ctx.clause(extracted=True)
        if global_gate:  # a gate exists, but not shown to cover this sink: unknown, never a PASS
            out.append(
                make_finding(
                    "AS-401",
                    confidence=Confidence.unknown(),
                    code_ref=sinks[sink].code_ref,
                    path=path,
                    title=f"{' → '.join(path)}: gate coverage not verified",
                    detail=f"agent {node} {how} and can call write tool {sink}; a global approval gate exists but "
                    "agenticreq could not verify that it covers this tool on every path",
                    evidence=f"source {src}; sink {sink}; gates: {[(g.kind, g.coverage) for g in obs.gates]}",
                )
            )
            continue
        partial = next(
            (g for g in obs.gates if not g.verified and g.tools and sink in {canonical(t) for t in g.tools}), None
        )
        unsure = effect_unsure(sinks[sink]) and canonical(sinks[sink].name) not in {
            canonical(t) for t in (ctx.spec.approval_tools() if ctx.spec else {})
        }
        out.append(
            make_finding(  # a sink that writes by its name only (its body cannot be followed): a mock run decides
                "AS-401",
                confidence=Confidence.unknown() if unsure else conf,
                severity="blocker" if strict else "warning",
                code_ref=sinks[sink].code_ref,
                path=path,
                title=f"{' → '.join(path)} has no gate"
                if partial is None
                else f"{' → '.join(path)} can bypass its gate",
                detail=f"agent {node} {how} and can call write tool {sink} with no approval gate in between"
                if partial is None
                else f"agent {node} {how} and can call write tool {sink}; its {partial.kind} gate does not cover "
                f"every path: {partial.coverage_note}",
                evidence=f"source {src}; sink {sink} ({sinks[sink].kind}); gates: {[g.kind for g in obs.gates] or 'none'}",
            )
        )
    return out


def _zero_trust(ctx: CheckContext) -> bool:
    """cross_agent_trust: none — every other agent's output is untrusted input (multi-agent only)."""
    return bool(ctx.spec and ctx.spec.is_multi_agent and ctx.effective().get("cross_agent_trust") == "none")


def _is_handoff_edge(via: str) -> bool:
    c = canonical(via)
    return c.startswith(("transfer_to_", "transfer_back_to_", "handoff_to_")) or via == "handoffs="


def _flows(
    ctx: CheckContext, obs: ObservedAgent, sources: set[str], sinks: set[str]
) -> list[tuple[str, str, str, list[str], str]]:
    """(agent, source, sink, path, how the source got in) for every static untrusted → sink route."""
    groups = _groups(ctx, obs)
    # which sources reach each agent's context, and how
    reach: dict[str, dict[str, tuple[list[str], str]]] = {n: {} for n in groups}
    for node, tools in groups.items():
        for src in sources & tools:
            reach[node][src] = ([src, "LLM"], f"reads untrusted input via {src}")
    direct = {canonical(k): refs for k, refs in obs.direct_calls.items()}
    for src in sources & set(direct):
        files = {r.file for r in direct[src]}
        for node, tools in groups.items():
            sink_files = {r.file for k, refs in obs.bound_tools.items() if canonical(k) in tools for r in refs}
            if files & sink_files and src not in reach[node]:
                where = sorted(files & sink_files)[0]
                reach[node][src] = (
                    [src, "python", "LLM"],
                    f"receives {src}'s result from Python code in {where} (never visible at the wire)",
                )
    g = build(obs, ctx.spec)
    neighbours: dict[str, set[str]] = {n: set() for n in groups}  # node -> nodes whose context flows into it
    for e in g.edges:
        if e.src in neighbours and e.dst in neighbours:
            if _is_handoff_edge(e.via):
                neighbours[e.dst].add(e.src)  # target inherits the history
                if not groups[e.src]:  # a coordinator without tools of its own (hierarchical manager, group chat)
                    neighbours[e.src].add(e.dst)  # also collects what its members produce
            else:
                neighbours[e.src].add(e.dst)  # delegator receives the delegate's answer
                neighbours[e.dst].add(e.src)  # and the delegate receives the task the delegator wrote
    if _zero_trust(ctx):  # a sibling's output is itself an untrusted source, whatever it read or holds
        for e in sorted(g.edges, key=lambda e: (e.src, e.dst)):
            # a handoff moves the conversation forward (src -> dst); a delegate called as a tool both gets the
            # delegator's request (src -> dst) and returns its answer (dst -> src)
            pairs = [(e.src, e.dst)] if _is_handoff_edge(e.via) else [(e.dst, e.src), (e.src, e.dst)]
            for sender, receiver in pairs:
                if receiver in groups:  # the sender needs no tools: its words alone can steer the receiver
                    reach[receiver].setdefault(
                        f"agent:{sender}",
                        ([sender, receiver, "LLM"], f"receives agent {sender}'s text (cross_agent_trust: none)"),
                    )
    changed = True
    while changed:  # propagate along edges to a fixpoint
        changed = False
        for node in groups:
            for nb in sorted(neighbours[node]):
                for src, (path, _) in list(reach[nb].items()):
                    if src not in reach[node] and node not in path:  # never re-enter an agent already on the path
                        reach[node][src] = (
                            [*path[:-1], *([] if path[-2:-1] == [nb] else [nb]), node, "LLM"]
                            if path[-1] == "LLM"
                            else [*path, node],
                            f"gets {src}'s output from agent {nb} (delegation result or handed-off history)",
                        )
                        changed = True
    out = []
    for node, tools in groups.items():
        for src, (path, how) in sorted(reach[node].items()):
            for sink in sorted(sinks & tools):
                out.append((node, src, sink, [*path, sink], how))
    return out


@rule("taint", "AS-402", modes=_BOTH)
def observed_taint(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    sources, sinks = _sources(ctx, obs), set(_sinks(ctx, obs))
    if not sources or not sinks:
        return []
    flows: dict[tuple[str, str], set[str]] = {}
    # captures (real models only)
    by_id: dict[str, str] = {}
    for c in sorted(obs.captures, key=lambda c: (c.run_id, c.index)):
        if c.synthetic:
            continue
        for tc in c.prior_tool_calls:
            if tc.id:
                by_id[tc.id] = canonical(tc.name)
        tainted = {by_id.get(r.call_id or "", canonical(r.name or "")) for r in c.tool_results} & sources
        for tc in c.tool_calls:
            for src in tainted:
                if canonical(tc.name) in sinks:
                    flows.setdefault((src, canonical(tc.name)), set()).add(c.run_id)
    # traces: source tool span before sink tool span in the same trace
    per_trace: dict[str, list[str]] = {}
    for s in obs.trace_spans:
        if s.tool_name:
            per_trace.setdefault(s.trace_id or "trace", []).append(canonical(s.tool_name))
    for tid, seq in per_trace.items():
        for i, a in enumerate(seq):
            if a in sources:
                for b in seq[i + 1 :]:
                    if b in sinks:
                        flows.setdefault((a, b), set()).add(tid)
    total = len(obs.real_run_ids())
    return [
        make_finding(
            "AS-402",
            confidence=Confidence.observed(len(runs), max(total, len(runs))),
            path=[src, "LLM", sink],
            title=f"Violation found: {src} → LLM → {sink}",
            detail=f"in {len(runs)} of {max(total, len(runs))} real run(s), output of untrusted source "
            f"{src} was in context when the model called {sink}",
            evidence=f"runs: {sorted(runs)[:5]}",
        )
        for (src, sink), runs in sorted(flows.items())
    ]


@rule("taint", "AS-403")
def undeclared_sources(ctx: CheckContext) -> Iterable[Finding]:
    obs, s = ctx.observed, ctx.spec
    if not obs or not s or s.untrusted_inputs is None:
        return []
    declared = {canonical(x) for x in s.untrusted_sources()}
    out = []
    for t in obs.declared_tools.values():
        c = canonical(t.name)
        if looks_untrusted_source(t) and c not in declared and not any(d in c for d in declared):
            out.append(
                make_finding(
                    "AS-403",
                    confidence=Confidence.heuristic(),
                    code_ref=t.code_ref,
                    spec_ref=ctx.spec_ref("untrusted_inputs"),
                    title=f"{t.name} looks like an untrusted source",
                    detail=f"{t.name} reads external content but untrusted_inputs does not list it",
                    evidence=f"untrusted_inputs: {sorted(declared)}",
                )
            )
    return out
