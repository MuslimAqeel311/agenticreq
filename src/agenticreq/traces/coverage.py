"""trace_coverage (Tier 4, derived): which spec scenarios the traces actually exercised.

Uncovered scenarios are `insufficient_evidence` (AS-905) — never PASS. A covered
scenario only says the situation occurred; whether it was handled correctly is a
separate check (violations are proven by traces, safety is not).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from agenticreq.ast.classify import canonical
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding
from agenticreq.models.observed import TraceSpan
from agenticreq.models.spec import AgentSpec
from agenticreq.rules.base import CheckContext, rule
from agenticreq.rules.registry import make_finding


@dataclass
class Scenario:
    id: str
    clause: str
    description: str


def derive_scenarios(spec: AgentSpec) -> list[Scenario]:
    out: list[Scenario] = []
    for p in spec.prohibited or []:
        if p.tool:
            out.append(Scenario(f"prohibited:{p.tool}", "prohibited", f"model attempts {p.tool} and is blocked"))
    for a in spec.approval_required or []:
        out.append(Scenario(f"approval:{a.tool}:granted", "approval_required", f"{a.tool} approved by {a.approver}"))
        out.append(
            Scenario(
                f"approval:{a.tool}:timeout",
                "approval_required",
                f"{a.tool} approval times out → {a.on_timeout or 'safe_stop'}",
            )
        )
    for lim in spec.limits or []:
        out.append(Scenario(f"limit:{lim.tool}.{lim.arg}", "limits", f"{lim.tool}.{lim.arg} at/over its limit"))
    for e in spec.escalation or []:
        out.append(Scenario(f"escalation:{e.metric}", "escalation", f"{e.metric} {e.op} {e.threshold} → {e.to}"))
    for u in spec.untrusted_inputs or []:
        out.append(
            Scenario(f"untrusted:{u.source}", "untrusted_inputs", f"untrusted {u.source} content reaches the model")
        )
    return out


def _args(s: TraceSpan) -> dict[str, object]:
    raw = s.attributes.get("gen_ai.tool.call.arguments")
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def covered(sc: Scenario, spans: list[TraceSpan], spec: AgentSpec) -> int:
    """Number of traces covering the scenario."""
    hits: set[str] = set()
    for s in spans:
        tid = s.trace_id or "trace"
        tag = s.attributes.get("agenticreq.scenario")
        if tag == sc.id:
            hits.add(tid)
            continue
        kind, _, rest = sc.id.partition(":")
        tool = canonical(s.tool_name or "")
        evs = " ".join(s.events).lower()
        if kind == "prohibited" and tool == canonical(rest):
            hits.add(tid)
        elif kind == "approval":
            t, _, outcome = rest.rpartition(":")
            if (
                tool == canonical(t)
                and "approval" in evs
                and (outcome in evs or outcome == "granted" and "approv" in evs)
            ):
                hits.add(tid)
        elif kind == "limit":
            t, _, arg = rest.partition(".")
            lim = next((x.limit for x in spec.limits or [] if x.tool == t and x.arg == arg), None)
            v = _args(s).get(arg)
            if tool == canonical(t) and lim and isinstance(v, (int, float)) and lim.max is not None and v >= lim.max:
                hits.add(tid)
        elif kind == "escalation" and "escalat" in evs:
            hits.add(tid)
        elif kind == "untrusted" and canonical(rest) in tool:
            hits.add(tid)
    return len(hits)


def coverage_matrix(spec: AgentSpec, spans: list[TraceSpan]) -> list[tuple[Scenario, int]]:
    return [(sc, covered(sc, spans, spec)) for sc in derive_scenarios(spec)]


@rule("meta", "AS-905")
def trace_coverage(ctx: CheckContext) -> Iterable[Finding]:
    s, obs = ctx.spec, ctx.observed
    if not s or not obs or not obs.trace_spans:
        return []
    total = len({sp.trace_id for sp in obs.trace_spans}) or 1
    out = []
    for sc, n in coverage_matrix(s, obs.trace_spans):
        ctx.clause(extracted=n > 0)
        if n == 0:
            out.append(
                make_finding(
                    "AS-905",
                    confidence=Confidence.insufficient(),
                    spec_ref=ctx.spec_ref(sc.clause),
                    title=f"Scenario not covered: {sc.id}",
                    detail=f"{sc.description} — none of {total} trace(s) exercised it; "
                    "the runs did not cover it, not PASS",
                    evidence=f"{total} traces, {len(obs.trace_spans)} spans",
                )
            )
    return out
