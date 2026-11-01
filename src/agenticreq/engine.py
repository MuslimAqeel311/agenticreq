"""Orchestration: load → extract → run rules → apply waivers → Report.

The CLI is a thin layer over these functions; they are also the Python API.
"""  #

from __future__ import annotations

from datetime import date
from pathlib import Path

from agenticreq.loader.yaml_loader import LoadedSpec, load_spec
from agenticreq.models.finding import Finding, Report
from agenticreq.models.spec import Waiver
from agenticreq.models.observed import ObservedAgent
from agenticreq.rules.base import CheckContext, Group, Mode, run_rules

# importing registers the rule implementations
from agenticreq.rules import validate as _validate  # noqa: F401


def _load_rule_modules() -> None:
    from agenticreq.rules import attest, conformance, fidelity, graph, hygiene, meta, numeric, runtime, taint  # noqa: F401
    from agenticreq.traces import coverage  # noqa: F401


def _apply_waivers(
    ctx: CheckContext, findings: list[Finding], extra: list[Waiver] | None = None
) -> tuple[list[Finding], list[dict[str, str]]]:
    waivers = [*(ctx.spec.waivers if ctx.spec else []), *(extra or [])]
    if not waivers:
        return findings, []
    kept: list[Finding] = []
    waived: list[dict[str, str]] = []
    from agenticreq.models.verdict import params_of_finding

    for f in findings:
        params = params_of_finding(f, ctx.mode)  # a waiver names a rule id, or the parameter the result is on
        w = next(
            (
                w
                for w in waivers
                if (w.rule_id == f.rule_id or w.rule_id in params)
                and (w.expires is None or w.expires >= ctx.today)
                and (not w.subject or w.subject in f.title or w.subject in f.detail)
            ),
            None,
        )
        if w is None or f.rule_id.startswith("AS-1"):  # spec errors cannot be waived
            kept.append(f)
        else:
            waived.append(
                {
                    "rule_id": f.rule_id,
                    "detail": f.detail,
                    "reason": w.reason,
                    "approved_by": w.approved_by or "",
                    "param": params[0] if params else "",
                }
            )
    return kept, waived


def known_names(observed: ObservedAgent | None = None) -> set[str]:
    """Every name a waiver or a confirmed: entry may use: a rule id, a parameter, a number's key or its short name,
    a sub-field, a tool or an agent of the code."""
    from agenticreq.models.verdict import _UNLISTED
    from agenticreq.params import PARAMS
    from agenticreq.rules.registry import REGISTRY
    from agenticreq.subfields import all_subfields
    from agenticreq.thresholds import BY_KEY

    names = {m.rule_id for m in REGISTRY.all()} | {p.name for p in PARAMS} | set(all_subfields())
    names |= {n for p in BY_KEY.values() for n in p.attest_names} | set(_UNLISTED.values()) | {"runtime_agents"}
    if observed is not None:
        names |= set(observed.declared_tools) | {a.name for a in observed.declared_agents}
    return names


def _unmatched_references(ctx: CheckContext) -> list[Finding]:
    """A waiver or a confirmed: entry that names nothing is an error, never silently ignored."""
    from agenticreq.models.confidence import Confidence
    from agenticreq.rules.registry import make_finding

    s = ctx.spec
    if s is None or ctx.mode == "validate":
        return []
    known = known_names(ctx.observed)
    out = []
    for i, w in enumerate(s.waivers):
        if w.rule_id not in known:
            out.append(
                make_finding(
                    "AS-119",
                    confidence=Confidence.proven(),
                    spec_ref=ctx.spec_ref("waivers", i),
                    param="waivers",
                    title=f"waiver matches no rule: {w.rule_id}",
                    detail=f"waiver matches no rule: {w.rule_id} (write a rule id or a parameter name)",
                    evidence=f"waivers[{i}].rule_id: {w.rule_id}",
                )
            )
    for i, a in enumerate(s.confirmed):
        if a.clause not in known:
            out.append(
                make_finding(
                    "AS-119",
                    confidence=Confidence.proven(),
                    spec_ref=ctx.spec_ref("confirmed", i),
                    param="confirmed",
                    title=f"confirmed: entry matches no parameter: {a.clause}",
                    detail=f"confirmed: entry matches no parameter: {a.clause} (write the parameter "
                    "name the report shows, e.g. approval_required[].timeout)",
                    evidence=f"confirmed[{i}].clause: {a.clause}",
                )
            )
    return out


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for f in findings:
        key = (f.rule_id, f.detail, str(f.location()))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _collapse(findings: list[Finding]) -> list[Finding]:
    """Repeated instances of one spec defect are one finding with a count and the list (six prohibited
    entries without a reason are one AS-109). Code findings stay separate: each is a different place."""
    groups: dict[tuple[str, str], list[Finding]] = {}
    order: list[tuple[str, str] | int] = []
    for i, f in enumerate(findings):
        if not f.rule_id.startswith("AS-1"):
            order.append(i)
            continue
        key = (f.rule_id, f.title)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(f)
    out: list[Finding] = []
    for k in order:
        if isinstance(k, int):
            out.append(findings[k])
            continue
        fs = groups[k]
        if len(fs) == 1:
            out.append(fs[0])
            continue
        first = fs[0]
        lines = [f"  - {f.detail}" + (f"  ({f.spec_ref})" if f.spec_ref else "") for f in fs]
        out.append(
            first.model_copy(
                update={
                    "title": f"{first.title} ({len(fs)} entries)",
                    "detail": f"{len(fs)} entries:\n" + "\n".join(lines),
                    "evidence": "; ".join(f.evidence for f in fs)[:600],
                }
            )
        )
    return out


def run(
    *,
    mode: Mode,
    loaded: LoadedSpec | None = None,
    observed: ObservedAgent | None = None,
    groups: list[Group] | None = None,
    today: date | None = None,
    waivers: list[Waiver] | None = None,
) -> Report:
    """`waivers`: extra waivers applied on top of the spec's (``audit --spec`` passes only these).

    A spec with the wrong shape stops here: one SPEC INVALID result listing every problem (each with the
    shape to write), and no rule runs. Shape errors never become BLOCKERS: a blocker means the agent is
    unsafe, and nothing about the agent has been checked yet."""
    _load_rule_modules()
    if loaded is not None and loaded.status == "invalid" and mode != "audit":
        raw_id = loaded.raw.get("agent_id") if isinstance(loaded.raw, dict) else None
        return Report(
            agent_id=raw_id if isinstance(raw_id, str) else None,
            command=mode,
            spec_status="invalid",
            spec_issues=sorted(
                loaded.issues,
                key=lambda i: (i.kind != "invalid", i.spec_ref.line if i.spec_ref and i.spec_ref.line else 0),
            ),
        )
    if observed is not None and loaded is not None and loaded.spec is not None and loaded.spec.tool_effects:
        from agenticreq.ast.classify import canonical

        declared = {canonical(k): v for k, v in loaded.spec.tool_effects.items()}
        for t in observed.declared_tools.values():  # the person's tool_effects: on the tools they name
            t.declared_effect = declared.get(canonical(t.name), t.declared_effect)
    ctx = CheckContext(loaded=loaded, observed=observed, mode=mode, today=today or date.today())
    findings = _collapse(_dedupe(run_rules(ctx, groups))) + _unmatched_references(ctx)
    findings, waived = _apply_waivers(ctx, findings, waivers)
    report = Report(
        agent_id=(ctx.spec.agent_id if ctx.spec else None),
        command=mode,
        findings=findings,
        passed=ctx.passed,
        spec_status=loaded.status if loaded is not None and mode != "audit" else None,  # type: ignore[arg-type]
        spec_issues=list(loaded.issues) if loaded is not None and mode != "audit" else [],
    )
    report.coverage.total_clauses = ctx.clauses_total
    report.coverage.extracted_clauses = ctx.clauses_extracted
    if waived:
        report.metadata["waived"] = waived
    if ctx.spec is not None and ctx.spec.confirmed and mode != "audit":
        from agenticreq.signoff import assess

        root = observed.root if observed is not None else None
        report.metadata["confirmed"] = {  # a sign-off line for any parameter is settled by its confirmed: entry
            a.clause: {"state": st.state, "by": a.by, "reason": a.reason or st.message}
            for a in ctx.spec.confirmed
            for st in [assess(a, ctx.today, root)]
        }
    if loaded is not None and mode != "audit":
        from agenticreq.not_needed import entries

        nn = entries(loaded.not_needed, ctx.spec, ctx.today, observed.root if observed is not None else None)
        if nn:  # each one replaces every result for its parameter (verdict.results)
            report.metadata["not_needed"] = [e.as_dict() for e in nn]
    if observed is not None:
        report.metadata["framework"] = observed.framework
        report.metadata["execution_model"] = (
            observed.execution_model if observed.execution_model != "unknown" else "not recognised"
        )
        report.metadata["analysis"] = {
            "files": len(observed.files),
            "parse_errors": len(observed.extraction_errors),
            "tools": len(observed.declared_tools),
            "agents": len(observed.declared_agents),
            "framework": observed.framework,
            "captures": len(observed.captures),
        }
        # the agents the code declares, with their tools: the report shows its results per agent
        report.metadata["agents"] = [{"name": a.name, "tools": list(a.tools)} for a in observed.declared_agents]
        from agenticreq.rules.conformance import open_unresolved

        from agenticreq.graph.analysis import unidentified

        agents = len(unidentified(observed, ctx.spec, "unresolved")) + len(observed.unresolved_agents)
        tools = len(open_unresolved(observed))
        if agents or tools:  # printed at the top of the report; tool-list results cannot pass (verdict.results)
            report.metadata["unresolved"] = {"agents": agents, "tools": tools}
    if mode == "audit":
        from agenticreq.params import audit_scope

        report.metadata["audit_scope"] = audit_scope(observed.framework if observed else None)
    if mode == "check" and ctx.spec is not None:
        report.metadata["level4"] = level4(ctx)
    return report


def level4(ctx: CheckContext) -> dict[str, object]:
    """Level 4 — derived, never written: enforceability, trace_coverage."""
    from agenticreq.derived import enforceability, trace_coverage

    spec = ctx.spec
    assert spec is not None
    enf = enforceability(spec, ctx.observed, ctx.passed)
    return {
        "enforceability": {k: v for k, v in enf.items() if k != "rows"},
        "model_only_clauses": [r["clause"] for r in enf["rows"] if not r["code"]],
        "trace_coverage": trace_coverage(spec, ctx.observed),
    }


def validate_spec(path: str | Path, today: date | None = None) -> Report:
    return run(mode="validate", loaded=load_spec(path), groups=["validate"], today=today)
