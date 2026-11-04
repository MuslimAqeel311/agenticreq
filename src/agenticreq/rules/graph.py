"""AS-3xx — delegation / multi-agent integrity.

Observed edges are `heuristic` (no parent/child id on the wire); a path is only as
strong as its weakest edge. Every finding prints the exact path: A → B → C.
"""

from __future__ import annotations

from typing import Iterable

from agenticreq import thresholds as th
from agenticreq.ast.classify import canonical
from agenticreq.data import InjectedTools
from agenticreq.graph.analysis import DelegationGraph, Edge, build, path_confidence, path_names
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck
from agenticreq.models.observed import ObservedAgent
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")


def _graph(ctx: CheckContext) -> DelegationGraph | None:
    if not ctx.observed:
        return None
    g = build(ctx.observed, ctx.spec)
    if len(g.nodes) < 2 and not g.edges:
        return None
    return g


def _ev(ctx: CheckContext, evidence: str) -> str:
    """Delegation is inferred from framework handoff conventions, so it is version-sensitive."""
    tag = ctx.observed.framework_tag() if ctx.observed else "framework unrecognized"
    return f"{tag} | {evidence}"


def _arrow(names: list[str]) -> str:
    return " → ".join(names)


@rule("graph", "AS-301", "AS-308", "AS-233")
def undeclared_edges(ctx: CheckContext) -> Iterable[Finding]:
    g, s = _graph(ctx), ctx.spec
    if not g or not s:
        return []
    out = []
    declared = {(canonical(e.from_), canonical(e.to)) for e in s.declared_delegation or []}
    from agenticreq.graph.origin import looks_generated

    for e in g.edges:
        if e.src in g.groups:
            continue  # a team's coordinator -> member edge is the team itself, not a delegation to declare
        if looks_generated(e.src) or looks_generated(e.dst):
            continue  # an agent the code does not name: never a name to declare (AS-910 asks what it is)
        ctx.clause(extracted=True)
        if (canonical(e.src), canonical(e.dst)) in declared:
            ctx.pass_(PassedCheck(rule_id="AS-301", subject=f"{e.src} → {e.dst}", confidence=e.confidence))
            continue
        out.append(
            make_finding(
                "AS-301",
                confidence=e.confidence,
                path=[e.src, e.dst],
                spec_ref=ctx.spec_ref("declared_delegation"),
                title=f"Undeclared delegation {e.src} → {e.dst}",
                detail=f"{e.src} delegates to {e.dst} via {e.via}, which declared_delegation does not list",
                evidence=_ev(ctx, e.evidence),
            )
        )
    if s.agent_registry:
        out += _registry_vs_code(ctx, g)
    return out


def _registry_vs_code(ctx: CheckContext, g: DelegationGraph) -> list[Finding]:
    """Both directions, each its own line: an agent the code has that agent_registry does not list (AS-308), and
    an agent_registry entry the code does not have (AS-233)."""
    from agenticreq.rules.conformance import code_only, spec_only

    s, obs = ctx.spec, ctx.observed
    assert s is not None and s.agent_registry
    reg = {canonical(a.id) for a in s.agent_registry}
    placeholders = {u.name for u in obs.unresolved_agents} if obs else set()
    in_code = {a.name for a in obs.declared_agents} if obs else set()
    out = []
    from agenticreq.graph.origin import looks_generated

    for n in g.nodes:
        if n in placeholders or looks_generated(n):
            continue  # an agent built at runtime, or a generated id: never a name to register (AS-909 / AS-910)
        if canonical(n) not in reg and n not in g.groups:  # a team group is not an LLM agent
            # an agent the code declares by name is proven; one only seen in a run is a pattern
            conf = Confidence.proven() if n in in_code else Confidence.heuristic()
            out.append(
                make_finding(
                    "AS-308",
                    confidence=conf,
                    spec_ref=ctx.spec_ref("agent_registry"),
                    param="agent_registry",
                    agent=n,
                    title=code_only([n]),
                    detail=f"{code_only([n])} (agent_registry; prompt sha {g.hashes.get(n, '?')})",
                    evidence=_ev(ctx, f"tools: {sorted(g.nodes[n])}"),
                )
            )
    if obs is None or obs.unresolved_agents:
        return out  # agents built at runtime may be the missing ones: AS-909 says so
    from agenticreq.graph.analysis import unidentified

    framework = {n.id for n in unidentified(obs, s, "framework")}
    made = [a.id for a in s.agent_registry if a.id not in g.backed and a.id in framework]
    if made:  # registered because an older report asked for it: a call the framework makes itself, not an agent
        out.append(
            make_finding(
                "AS-233",
                confidence=Confidence.proven(),
                spec_ref=ctx.spec_ref("agent_registry"),
                param="agent_registry[].id",
                title=f"spec lists {', '.join(made)}; the framework makes {'that call' if len(made) == 1 else 'those calls'} "
                "itself, it is not an agent: remove it from agent_registry",
                detail=f"spec lists {', '.join(made)} in agent_registry; it is a call the framework "
                "makes itself (listed in _derived.injected_by_framework), not an agent of "
                "this project: remove it from agent_registry",
                evidence=f"framework calls: {sorted(framework)}",
            )
        )
    missing = [a.id for a in s.agent_registry if a.id not in g.backed and a.id not in framework]
    if missing:
        out.append(
            make_finding(
                "AS-233",
                confidence=Confidence.proven(),
                spec_ref=ctx.spec_ref("agent_registry"),
                param="agent_registry[].id",
                title=spec_only("agent", missing),
                detail=spec_only("agent", missing) + " (agent_registry)",
                evidence=f"agents in code: {sorted(in_code)}",
            )
        )
    return out


@rule("graph", "AS-302", "AS-303", modes=_BOTH)
def depth_and_cycles(ctx: CheckContext) -> Iterable[Finding]:
    g = _graph(ctx)
    if not g:
        return []
    out = []
    eff = ctx.effective()
    # the depth the code allows is the longest delegation chain; the number to hold it to is the spec's, if it has one
    p, risk = th.MAX_DEPTH, ctx.risk_level
    max_depth = th.spec_value(ctx, p)
    d, path = g.depth()
    ctx.clause(extracted=True)
    if path:
        dec = th.judge(p, risk, spec=max_depth, code=d)
        if dec.status == "FAIL":
            names = path_names(path)
            out.append(
                make_finding(
                    "AS-302",
                    confidence=path_confidence(path),
                    path=names,
                    spec_ref=ctx.spec_ref("max_delegation_depth"),
                    reason=dec.reason,
                    status_hint=dec.line_status,
                    param=p.key,
                    detail=f"{_arrow(names)} is {d} hops deep; spec.yml max_delegation_depth={th.fmt(max_depth)}",
                    evidence=_ev(ctx, "; ".join(e.evidence for e in path)),
                )
            )
        else:
            conf = path_confidence(path)
            names = path_names(path)
            agents = ctx.observed.declared_agents if ctx.observed else []
            first = next((a.code_ref for a in agents if canonical(a.name) == canonical(names[0])), None)
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-302",
                    subject=th.fmt(d),
                    confidence=conf,
                    param=p.key,
                    note=f"{_arrow(names)}{f', {first}' if first else ''}",
                )
            )
    if eff.get("no_cycles", True):
        from agenticreq.rules.conformance import effective_step_bound

        bound = effective_step_bound(ctx.observed) if ctx.observed else None
        # one finding per group of agents that can loop among themselves (a strongly connected
        # component), not one per simple cycle: a hub with 5 specialists has dozens of those
        groups: list[tuple[set[str], list[list[Edge]]]] = []
        for cyc in g.cycles():
            members = {e.src for e in cyc} | {e.dst for e in cyc}
            hit = [grp for grp in groups if grp[0] & members]
            merged_members = set(members).union(*(grp[0] for grp in hit))
            merged_cycles = [cyc, *(c for grp in hit for c in grp[1])]
            groups = [grp for grp in groups if grp not in hit] + [(merged_members, merged_cycles)]
        for members, cycles in groups:
            shortest = min(cycles, key=len)
            names = path_names(shortest)
            # a loop the runtime cuts off after N turns is still a loop, but not an unbounded one
            note = (
                f"; bounded by {bound[0]}={bound[1]:g} ({bound[2]})"
                if bound and bound[1]
                else "; nothing bounds the loop"
            )
            out.append(
                make_finding(
                    "AS-303",
                    confidence=path_confidence(shortest),
                    path=names,
                    severity="warning" if bound and bound[1] else None,
                    title=f"Delegation cycle among {len(members)} agents",
                    detail=f"{len(cycles)} cycle(s) among {sorted(members)}, e.g. {_arrow(names)}{note}",
                    evidence=_ev(ctx, "; ".join(e.evidence for e in shortest)),
                )
            )
    return out


@rule("graph", "AS-304")
def transitive_prohibition(ctx: CheckContext) -> Iterable[Finding]:
    g, s = _graph(ctx), ctx.spec
    if not g or not s or ctx.effective().get("transitive_prohibition") is False:
        return []
    out = []
    # prohibitions per registry agent; the top-level `prohibited` applies to the spec's own agent / roots
    per_agent: dict[str, set[str]] = {}
    for a in s.agent_registry or []:
        per_agent[a.id] = {canonical(t) for t in a.prohibited}
    top = {canonical(t) for t in s.prohibited_tools()}
    starts = [s.agent_id] if s.agent_id and s.agent_id in g.nodes else g.roots()
    for st in starts:
        per_agent[st] = per_agent.get(st, set()) | top
    for start, banned in per_agent.items():
        if not banned or start not in g.nodes:
            continue
        # in a group chat every member's words reach every other member: a teammate's tools are reachable
        teams = [e for e in g.edges if e.dst == start and e.src in g.groups]
        # printed as `writer -> team -> researcher`: the shared conversation, not a writer->researcher edge
        paths = g.paths_from(start) + [
            [Edge(start, t.src, "group_chat (shared conversation)", t.confidence, t.evidence), *p]
            for t in teams
            for p in g.paths_from(t.src)
            if p[0].dst != start
        ]
        for p in paths:
            end = p[-1].dst
            hits = sorted(t for t in g.nodes.get(end, set()) if canonical(t) in banned)
            for tool in hits:
                names = path_names(p)
                ctx.clause(extracted=True)
                out.append(
                    make_finding(
                        "AS-304",
                        confidence=path_confidence(p),
                        path=names,
                        spec_ref=ctx.spec_ref("prohibited"),
                        title=f"{_arrow(names)} can reach {tool}",
                        detail=f"{tool} is prohibited for {start}, but {start} delegates (transitively) to {end}, "
                        f"which holds it. Path: {_arrow(names)} can reach {tool}",
                        evidence=_ev(ctx, "; ".join(e.evidence for e in p)),
                    )
                )
    return out


@rule("graph", "AS-203", "AS-206")
def registry_authority(ctx: CheckContext) -> Iterable[Finding]:
    """Each agent_registry entry is that agent's own contract: its `prohibited` tools must not be held by
    it directly (AS-304 covers reaching them through a delegate), and when it lists `tools`, it holds
    nothing else."""
    g, s = _graph(ctx), ctx.spec
    if not g or not s:
        return []
    injected = InjectedTools(ctx.injected_allowlist)
    out = []
    for i, a in enumerate(s.agent_registry or []):
        held = g.nodes.get(a.id)
        if held is None or a.id not in g.backed:
            continue  # not matched to a node in code: AS-308 / AS-114 / AS-909 report registry mismatches
        banned = {canonical(t) for t in a.prohibited}
        allowed = {canonical(t) for t in a.tools}
        for tool in sorted(held):
            c = canonical(tool)
            ctx.clause(extracted=True)
            if c in banned:
                out.append(
                    make_finding(
                        "AS-203",
                        confidence=Confidence.proven(),
                        spec_ref=ctx.spec_ref("agent_registry", i),
                        title=f"Agent {a.id} holds its prohibited tool {tool}",
                        detail=f"agent_registry prohibits {tool} for {a.id}, and {a.id} is given it directly in code",
                        evidence=f"{a.id} tools in code: {sorted(held)}; registry prohibited: {sorted(a.prohibited)}",
                    )
                )
            elif allowed and c not in allowed and not injected.framework_of(tool):
                out.append(
                    make_finding(
                        "AS-206",
                        confidence=Confidence.proven(),
                        spec_ref=ctx.spec_ref("agent_registry", i),
                        title=f"Agent {a.id} holds {tool}, which its registry entry does not list",
                        detail=f"agent_registry lists {sorted(a.tools)} for {a.id}; code also gives it {tool}",
                        evidence=f"{a.id} tools in code: {sorted(held)}",
                    )
                )
    return out


@rule("graph", "AS-303")
def declared_cycles(ctx: CheckContext) -> Iterable[Finding]:
    """A cycle written into declared_delegation is a design decision to allow loops; with no_cycles it is
    a contradiction in the spec itself, whatever the code does."""
    s = ctx.spec
    if not s or not s.declared_delegation or not ctx.effective().get("no_cycles", True):
        return []
    nxt: dict[str, set[str]] = {}
    for e in s.declared_delegation:
        nxt.setdefault(e.from_, set()).add(e.to)

    def find(start: str) -> list[str] | None:
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for n in sorted(nxt.get(node, ())):
                if n == start:
                    return [*path, n]
                if n not in path:
                    stack.append((n, [*path, n]))
        return None

    seen: set[frozenset[str]] = set()
    out = []
    for start in sorted(nxt):
        cyc = find(start)
        if cyc and frozenset(cyc) not in seen:
            seen.add(frozenset(cyc))
            out.append(
                make_finding(
                    "AS-303",
                    confidence=Confidence.proven(),
                    path=cyc,
                    spec_ref=ctx.spec_ref("declared_delegation"),
                    title=f"declared_delegation contains a cycle: {_arrow(cyc)}",
                    detail=f"the spec itself declares {_arrow(cyc)} while no_cycles is true",
                    evidence=f"declared_delegation: {[(e.from_, e.to) for e in s.declared_delegation]}",
                )
            )
    return out


@rule("graph", "AS-305")
def authority(ctx: CheckContext) -> Iterable[Finding]:
    g = _graph(ctx)
    if not g or ctx.effective().get("authority_inheritance", "subset") != "subset":
        return []
    from agenticreq.ast.classify import canonical, is_write_tool, verb_class

    injected = InjectedTools(ctx.injected_allowlist)
    known = {canonical(k): v for k, v in (ctx.observed.declared_tools if ctx.observed else {}).items()}
    out = []
    for e in g.edges:
        if e.src in g.groups:
            continue  # a team coordinator holds no tools by design; it routes, it does not grant authority
        extra = sorted(t for t in g.nodes.get(e.dst, set()) - g.nodes.get(e.src, set()) if not injected.framework_of(t))
        # authority is the power to ACT: a specialist that only reads more than its caller escalates nothing
        extra = [
            t
            for t in extra
            if canonical(t) not in known
            or is_write_tool(known[canonical(t)])
            or verb_class(known[canonical(t)].name) == "unknown"
        ]  # only a positively-read tool escalates nothing
        if extra:
            out.append(
                make_finding(
                    "AS-305",
                    confidence=e.confidence,
                    path=[e.src, e.dst],
                    title=f"{e.dst} holds tools {e.src} does not",
                    detail=f"authority_inheritance=subset, but {e.dst} has {extra} which {e.src} lacks",
                    evidence=_ev(ctx, e.evidence),
                )
            )
    return out


def one_trace_across_agents(obs: ObservedAgent) -> tuple[str, int] | None:
    """(trace id, agents) when one id spans two or more agents in spans or in the model requests of a run."""
    by_trace: dict[str, set[str]] = {}
    for sp in obs.trace_spans:
        if sp.trace_id and sp.agent_name:
            by_trace.setdefault(sp.trace_id, set()).add(sp.agent_name)
    for c in obs.captures:
        if c.trace_id and c.system_prompt_sha256:
            by_trace.setdefault(c.trace_id, set()).add(c.system_prompt_sha256)
    best = max(sorted(by_trace), key=lambda t: len(by_trace[t]), default=None)
    return (best, len(by_trace[best])) if best is not None and len(by_trace[best]) >= 2 else None


@rule("graph", "AS-306", "AS-307", modes=_BOTH)
def multi_agent_hygiene(ctx: CheckContext) -> Iterable[Finding]:
    g = _graph(ctx)
    if not g or not g.edges:
        return []
    if len([n for n in g.nodes if n not in g.groups]) < 2:
        return []  # a coordinator over ONE agent: nothing to budget or trace across
    out = []
    s = ctx.spec
    obs = ctx.observed
    assert obs is not None
    from agenticreq.data import load_yaml
    from agenticreq.rules.conformance import effective_step_bound

    fw = load_yaml("framework_defaults.yml") or {}
    bound = effective_step_bound(obs)
    run_level = bool(bound and bound[1]) and obs.framework in (fw.get("run_level_bound") or [])
    names = (fw.get("whole_run") or {}).get(obs.framework or "") or []
    whole = next((b for b in obs.bounds if b.value and not b.agent and (b.name in names or b.kind in names)), None)
    if whole is not None:
        bound, run_level = (whole.label, whole.value, f"code {whole.code_ref}"), True
    if not (s and s.aggregate_budget) and obs.framework in (fw.get("no_whole_run") or []) and not run_level:
        out.append(
            make_finding(
                "AS-306",
                confidence=Confidence.attestation(),
                severity="attestation",
                param="aggregate_budget",
                status_hint="SIGN-OFF",
                spec_ref=ctx.spec_ref() if s else None,
                title="aggregate_budget: nothing in this framework limits a whole run",
                detail=f"{len(g.nodes)} agents, and {obs.framework} has no setting that covers a whole run",
                reason="What limits a whole run across the agents? This framework has no such setting. Confirm "
                "under confirmed: in spec.yml.",
                evidence=f"nodes: {sorted(g.nodes)}",
            )
        )
    elif not (s and s.aggregate_budget) and run_level and bound:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-306",
                subject=bound[0],
                confidence=Confidence.heuristic(),
                note=f"{bound[0]}={bound[1]:g} applies to the whole run in {obs.framework_tag()} "
                f"({bound[2]}); no aggregate_budget in the spec",
            )
        )
    elif not (s and s.aggregate_budget):
        out.append(
            make_finding(
                "AS-306",
                confidence=Confidence.proven() if s else Confidence.heuristic(),
                spec_ref=ctx.spec_ref() if s else None,
                detail=f"{len(g.nodes)} agents / {len(g.edges)} delegation edge(s) but no aggregate_budget; "
                "per-agent bounds do not bound the whole system",
                evidence=f"nodes: {sorted(g.nodes)}",
            )
        )
    if s and s.aggregate_budget:  # the numbers the spec states for the whole system (no number, no check)
        risk = ctx.risk_level
        for p in (th.AGG_STEPS, th.AGG_COST, th.AGG_TOOLS):
            spec_n = th.spec_value(ctx, p)
            if spec_n is None:
                continue  # unset, or a TODO (AS-101)
            code = float(bound[1]) if p is th.AGG_STEPS and run_level and bound and bound[1] else None
            att = th.attestation(ctx, p)
            out += th.emit(
                ctx,
                p,
                th.judge(p, risk, spec=spec_n, code=code, confirmed=att is not None),
                risk=risk,
                code=code,
                spec=spec_n,
                spec_ref=ctx.spec_ref("aggregate_budget", p.key.split(".")[1]),
                attested=att,
                code_ref=whole.code_ref if whole is not None else None,
                where=bound[2] if bound and whole is None else None,
                evidence=f"{p.key}: spec={th.fmt(spec_n)}; run-level bound: "
                f"{bound[0] + '=' + th.fmt(bound[1]) if bound else 'none'}",
            )
    runtime = one_trace_across_agents(obs)
    traced = {sp.trace_id for sp in obs.trace_spans if sp.trace_id}
    agents_in_trace = {sp.agent_name for sp in obs.trace_spans if sp.agent_name}
    builtin = (fw.get("builtin_tracing") or {}).get(obs.framework if obs.framework_confirmed else "")
    disabled = bool(obs.signal("tracing_disabled"))
    found = [e.text() for e in obs.trace_evidence] + ([builtin] if builtin and not disabled else [])
    if runtime:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-307",
                subject=f"one trace id across {runtime[1]} agents",
                confidence=Confidence.observed(0, 1),
                note=f"trace {runtime[0][:16]}",
            )
        )
    elif found:  # a mechanism exists; that it reaches every agent and log line is not provable from code
        out.append(
            make_finding(
                "AS-307",
                confidence=Confidence.attestation(),
                severity="attestation",
                param="end_to_end_trace_id",
                status_hint="SIGN-OFF",
                title="A trace id exists; that it reaches every agent is not shown",
                detail="; ".join(found),
                reason=f"Found: {'; '.join(found)}. Does this one id reach every agent and each log line? Confirm "
                "under confirmed: in spec.yml.",
                evidence="; ".join(found),
            )
        )
    elif ctx.effective().get("end_to_end_trace_id", True):
        out.append(
            make_finding(
                "AS-307",
                confidence=Confidence.heuristic(),
                detail="no trace carries spans from more than one agent under one trace id; "
                "delegation edges stay a code-pattern guess (inferred from request order)",
                evidence=f"trace ids: {len(traced)}; agents in traces: {sorted(agents_in_trace)}",
            )
        )
    return out
