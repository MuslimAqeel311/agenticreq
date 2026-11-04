"""AS-9xx — meta: execution model, extraction failures, missing observations."""

from __future__ import annotations

import re
from typing import Iterable

from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding
from agenticreq.models.observed import ObservedAgent, UnresolvedAgent
from agenticreq.rules.base import CheckContext, Mode, rule
from agenticreq.rules.registry import make_finding

_BOTH: tuple[Mode, ...] = ("check", "audit")


_RUNTIME_VALUE = re.compile(r"`([^`]+)`")


def _pool_reason(obs: ObservedAgent, u: UnresolvedAgent) -> str:
    """A pool of agents whose size only the run knows. A run that saw them says how many."""
    value = _RUNTIME_VALUE.search(u.reason)
    seen = len({c.system_prompt_sha256 for c in obs.captures if c.system_prompt_sha256})
    run = f"; a run saw {seen}" if seen else ""
    return (
        f"1+ agents built at {u.code_ref} from runtime value `{value.group(1) if value else 'unresolved'}`{run}. "
        "Write the names under agent_registry, and confirm them under confirmed: in spec.yml."
    )


def _prompt_only(obs: ObservedAgent) -> str:
    tools = ", ".join(sorted(obs.prompt_only_tools)) or "its tools"
    return (
        f"Prompt-only: a code agent gets {tools} as prompt text and calls them from code it writes, so neither the "
        "code nor a run can show how they are used. Confirm it under confirmed: in spec.yml."
    )


@rule("meta", "AS-901", modes=_BOTH)
def execution_model(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    if obs.execution_model == "prompt-only":
        return [
            make_finding(
                "AS-901",
                confidence=Confidence.unknown(),
                code_ref=obs.execution_model_ref,
                detail="prompt-only / code agent: tools are Python signatures in the system prompt and "
                "results land in unattributed execution logs. Tool checks wait for a mock run, never PASS.",
                evidence=f"{obs.framework_tag()} | {obs.execution_model_evidence or 'CodeAgent'}",
                reason=_prompt_only(obs),
                status_hint="SIGN-OFF",
            )
        ]
    out = []
    if obs.prompt_only_tools:
        out.append(
            make_finding(
                "AS-901",
                confidence=Confidence.unknown(),
                code_ref=obs.execution_model_ref,
                detail=f"tools {sorted(obs.prompt_only_tools)} are used by a code agent; "
                "their invocation cannot be verified at the boundary",
                evidence=f"{obs.framework_tag()} | {obs.execution_model_evidence}",
                reason=_prompt_only(obs),
                status_hint="SIGN-OFF",
            )
        )
    if obs.execution_model == "unknown":
        out.append(
            make_finding(
                "AS-901",
                confidence=Confidence.unknown(),
                severity="info",
                title="Execution model could not be determined",
                detail="no agent framework, tool binding or captured tool schema was found",
                evidence=f"files scanned: {len(obs.files)}",
            )
        )
    return out


@rule("meta", "AS-902", modes=_BOTH)
def extraction_errors(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    from agenticreq.models.finding import SourceRef

    out = [
        make_finding(
            "AS-902",
            confidence=Confidence.unknown(),
            code_ref=SourceRef(file=e.file, line=e.line),
            detail=e.message,
            evidence=e.message[:200],
        )
        for e in obs.extraction_errors
    ]
    # a tool that exists but cannot be seen is reported, never silently left out
    out += [
        make_finding(
            "AS-902",
            confidence=Confidence.unknown(),
            code_ref=u.code_ref,
            title="Tool built at runtime could not be resolved",
            detail=f"{u.reason}: its name, schema and permissions are not known statically, so role "
            "boundary, prohibitions and limits are not read for it",
            evidence=str(u.code_ref),
        )
        for u in obs.unresolved_tools
    ]
    return out


@rule("meta", "AS-909", modes=_BOTH)
def unresolved_agents(ctx: CheckContext) -> Iterable[Finding]:
    """An agent constructor that builds an unknown number of agents is reported, never counted as one."""
    obs = ctx.observed
    if not obs:
        return []
    out = [
        make_finding(
            "AS-909",
            confidence=Confidence.unknown(),
            code_ref=u.code_ref,
            detail=f"{u.reason}: how many agents this builds, their names and which tools each one "
            f"holds are not known statically (reported as one placeholder node, {u.name!r}). "
            "Per-agent role boundary, agent_registry and delegation checks wait for a mock run for them",
            evidence=str(u.code_ref),
            reason=_pool_reason(obs, u),
            status_hint="SIGN-OFF",
        )
        for u in obs.unresolved_agents
    ]
    if out and ctx.spec and ctx.spec.agent_registry:
        from agenticreq.graph.analysis import resolve_registry

        matched = set(resolve_registry(ctx.spec, obs).values())
        missing = [a.id for a in ctx.spec.agent_registry if a.id not in matched]
        if missing:
            out.append(
                make_finding(
                    "AS-909",
                    confidence=Confidence.unknown(),
                    spec_ref=ctx.spec_ref("agent_registry"),
                    title="agent_registry entries not matched to code",
                    detail=f"{missing} match no agent found in code; with agents built at runtime they may exist, so "
                    "their tools and delegation wait for a mock run, not passing",
                    evidence=f"unresolved agent constructions: {len(obs.unresolved_agents)}",
                    reason=f"{', '.join(missing)} match no agent in the code, and agents are built at run time. Confirm "
                    "them under confirmed: in spec.yml.",
                    status_hint="SIGN-OFF",
                )
            )
    return out


@rule("meta", "AS-910", modes=_BOTH)
def runtime_agents(ctx: CheckContext) -> Iterable[Finding]:
    """An agent seen at run time that the code does not name, and that the framework did not create: a person says
    what it is under confirmed: (runtime_agents). Never a gap asking to register a generated id."""
    from agenticreq.signoff import assess

    obs = ctx.observed
    if not obs:
        return []
    from agenticreq.graph.analysis import unidentified

    unnamed = unidentified(obs, ctx.spec, "unresolved")
    if not unnamed:
        return []
    what = "; ".join(
        f"prompt {n.system_prompt_sha256[:6]}, {n.calls} call(s), tools: {', '.join(n.tools) or 'none'}"
        for n in unnamed
    )
    many = len(unnamed) > 1
    head = (
        f"{len(unnamed)} agents created at runtime could not be identified"
        if many
        else "an agent created at runtime could not be identified"
    )
    a = next((x for x in (ctx.spec.confirmed if ctx.spec else []) if x.clause == RUNTIME_AGENTS), None)
    st = assess(a, ctx.today, obs.root) if a else None
    if a is not None and st is not None and st.state == "signed":
        return [
            make_finding(
                "AS-910",
                confidence=Confidence.attestation(),
                param=RUNTIME_AGENTS,
                title=f"{head}; signed by {a.by}",
                detail=f"{head} ({what}); signed by {a.by}",
                reason=f"Signed by {a.by}" + (f" on {a.date}." if a.date else "."),
                status_hint="SIGNED",
                evidence=a.evidence or f"confirmed: {RUNTIME_AGENTS}",
            )
        ]
    return [
        make_finding(
            "AS-910",
            confidence=Confidence.attestation(),
            param=RUNTIME_AGENTS,
            title=head,
            detail=f"{head} ({what})",
            evidence=what,
            status_hint="SIGN-OFF",
            reason=f"{head[0].upper()}{head[1:]} ({what}); confirm what {'they are' if many else 'it is'} "
            f"under confirmed: ({RUNTIME_AGENTS}).",
        )
    ]


RUNTIME_AGENTS = "runtime_agents"


@rule("meta", "AS-911", modes=("check",))
def sub_field_sign_off(ctx: CheckContext) -> Iterable[Finding]:
    """A sub-field the user wrote that the code cannot show: its own line with a plain question, never folded into
    its parent's PASS. A person confirms it under confirmed: with the sub-field's own name."""
    from agenticreq.subfields import question, rule_of, written

    if not ctx.spec:
        return []
    return [
        make_finding(
            "AS-911",
            confidence=Confidence.attestation(),
            param=path,
            status_hint="SIGN-OFF",
            spec_ref=ctx.spec_ref(*path.replace("[]", "").split(".")),
            title=f"{path}: {question(path)}",
            detail=f"{path}: the code cannot show this; a person confirms it",
            reason=f"{question(path)} Confirm under confirmed: ({path}).",
            evidence=path,
        )
        for path in written(ctx.spec)
        if not rule_of(path)
    ]


@rule("meta", "AS-903", modes=_BOTH)
def no_observation(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or obs.has_runtime or obs.mcp_tools:
        return []
    return [
        make_finding(
            "AS-903",
            confidence=Confidence.unknown(),
            detail="only static (AST) evidence: never_reached, undeclared-tool and schema-fidelity checks did not run",
            evidence=f"captures=0 traces=0; declared tools={len(obs.declared_tools)}",
        )
    ]


@rule("meta", "AS-907", modes=("audit",))
def needs_spec(ctx: CheckContext) -> Iterable[Finding]:
    """`audit` has no spec: the business rules (Level 1) and intended delegation (Level 3) are unknown —
    never implied to pass."""
    from agenticreq.params import NEEDS_SPEC

    if ctx.spec is not None:
        return []
    obs = ctx.observed
    multi = bool(obs and (len(obs.declared_agents) > 1 or obs.edges or obs.unresolved_agents))
    fields = [f for f in NEEDS_SPEC if f != "declared_delegation" or multi]
    return [
        make_finding(
            "AS-907",
            confidence=Confidence.unknown(),
            detail="audit checked only engineering defaults (Level 2, risk_level=medium). It cannot know "
            "what your business forbids, the money thresholds, who must approve, or which "
            "sub-agents were intended. Needs a spec: "
            + ", ".join(fields)
            + ". Run `agenticreq init <path>`, fill the TODOs, then `agenticreq check <path>`.",
            evidence="no --spec supplied (audit mode); unchecked: " + ", ".join(fields),
        )
    ]


@rule("meta", "AS-904", modes=_BOTH)
def framework(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs or obs.framework:
        return []
    from agenticreq.params import FRAMEWORK_SPECIFIC

    return [
        make_finding(
            "AS-904",
            confidence=Confidence.unknown(),
            detail="no known agent framework import found, so the constructs that enforce these are "
            "not recognisable and they wait for a mock run (not passing): "
            + ", ".join(FRAMEWORK_SPECIFIC)
            + ". Framework-agnostic checks still ran (secrets, model pinning, environment "
            "separation, unbounded calls).",
            evidence=f"{len(obs.files)} python files scanned; frameworks: none recognised",
        )
    ]


@rule("meta", "AS-908", modes=_BOTH)
def framework_version(ctx: CheckContext) -> Iterable[Finding]:
    """The code reading is validated per framework version; outside that range it may silently miss constructs."""
    obs = ctx.observed
    if not obs or not obs.framework or not obs.framework_version:
        return []
    from agenticreq.data import load_yaml
    from agenticreq.versions import allows_any, parse

    row = ((load_yaml("supported_versions.yml") or {}).get("frameworks") or {}).get(obs.framework)
    if not row:
        return []
    lo, below, validated = str(row["min"]), str(row["below"]), [str(v) for v in row.get("validated") or []]
    src = obs.framework_version_source or "an unrecorded source"
    if src.endswith("(range)"):  # only a requirement like `crewai>=0.100`: what pip installs can be validated
        if allows_any(obs.framework_version, lo, below, validated) is not False:
            return []
        why = "allows no version in"
    else:
        v = parse(obs.framework_version)
        if not v or parse(lo) <= v < parse(below):
            return []
        why = "is outside"
    return [
        make_finding(
            "AS-908",
            confidence=Confidence.unknown(),
            detail=f"{row['dist']} {obs.framework_version} ({src}) {why} {row['dist']} "
            f">={lo},<{below}, the range agenticreq's code reading was validated "
            "on. Findings for this framework can miss constructs, so treat a clean result as "
            "incomplete.",
            evidence=f"framework={obs.framework}; version={obs.framework_version} from {src}; "
            f"validated: {', '.join(validated)}",
        )
    ]


@rule("meta", "AS-906", modes=_BOTH)
def unsupported_wire(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    out = []
    for url in sorted(set(obs.unsupported_calls)):
        out.append(
            make_finding(
                "AS-906",
                confidence=Confidence.unknown(),
                detail=f"captured LLM-looking request to {url} in an unrecognised wire format",
                evidence=url,
            )
        )
    hosted = sorted({t for t in obs.wire_tools if t.startswith("hosted:")})
    if hosted:
        out.append(
            make_finding(
                "AS-906",
                confidence=Confidence.unknown(),
                title="Provider-hosted tools are outside scope",
                detail=f"{hosted} execute at the provider; their calls and results are not verified",
                evidence=", ".join(hosted),
            )
        )
    return out
