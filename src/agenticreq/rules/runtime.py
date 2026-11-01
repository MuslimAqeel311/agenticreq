"""AS-228..231 — Level-2 clauses that only a run can show: stop_conditions, progress_guard, bounds,
rate_limits. Evidence is REAL runs only (traces, or captures answered by a real model); discover's
canned responder drives the loop synthetically and proves nothing about the agent's behaviour.

Traces prove violations, not safety: a clean run is "no violation observed (0/N)", never a PASS of the
clause in general.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from agenticreq import thresholds as th
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck
from agenticreq.models.observed import ObservedAgent
from agenticreq.rules.base import CheckContext, rule
from agenticreq.rules.registry import make_finding

_LLM_OPS = {"chat", "text_completion", "generate_content", "invoke_model"}


@dataclass
class Run:
    id: str
    llm_calls: int = 0
    tools: list[tuple[str, str | None]] = field(default_factory=list)  # (name, canonical args or None)
    times: list[float] = field(default_factory=list)
    ended_mid_action: bool | None = None  # last model response still asked for tools (captures only)


def _args(a: Any) -> str | None:
    if a in (None, ""):
        return None
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return str(a)
    try:
        return json.dumps(a, sort_keys=True, default=str)
    except TypeError:
        return str(a)


def real_runs(obs: ObservedAgent) -> list[Run]:
    runs: dict[str, Run] = {}
    for c in sorted((c for c in obs.captures if not c.synthetic), key=lambda c: (c.run_id, c.index)):
        r = runs.setdefault(c.run_id, Run(c.run_id))
        r.llm_calls += 1
        r.tools += [(tc.name, _args(tc.arguments)) for tc in c.tool_calls]
        if c.timestamp is not None:
            r.times.append(c.timestamp)
        r.ended_mid_action = bool(c.tool_calls)  # the last one seen wins
    spans = sorted(obs.trace_spans, key=lambda s: (s.trace_id or "", s.start_time or 0.0))
    for s in spans:
        r = runs.setdefault(f"trace:{s.trace_id or 'trace'}", Run(f"trace:{s.trace_id or 'trace'}"))
        if s.operation in _LLM_OPS:
            r.llm_calls += 1
            if s.start_time is not None:
                r.times.append(s.start_time)
        elif s.tool_name and s.operation in (None, "execute_tool"):
            a = s.attributes.get("gen_ai.tool.call.arguments") or s.attributes.get("input.value")
            r.tools.append((s.tool_name, _args(a)))
    return list(runs.values())


def _longest_repeat(seq: list[tuple[str, str | None]]) -> tuple[int, tuple[str, str | None] | None]:
    best, cur, item, prev = 0, 0, None, None
    for x in seq:
        cur = cur + 1 if x == prev and x[1] is not None else 1
        prev = x
        if cur > best:
            best, item = cur, x
    return best, item


def _obs(n: int, total: int) -> Confidence:
    return Confidence.observed(n, max(total, n))


@rule("conformance", "AS-228")
def progress_guard(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    p, risk = th.NO_PROGRESS, ctx.risk_level
    if not obs or not ctx.loaded or th.spec_todo(ctx, p):
        return []  # a TODO is reported by AS-101
    limit = th.spec_value(ctx, p)
    runs = real_runs(obs)
    if limit is None or not runs:
        # no number in spec.yml, or no real run to measure it against: code cannot read this limit, so it is
        # signed off under confirmed:, or a mock run checks it
        att = th.attestation(ctx, p)
        from agenticreq.ast.constants import find

        hit = find(obs.root, "no", "progress")  # MAX_NO_PROGRESS_STEPS = 5: the code names the limit it sets
        code = hit[0] if hit else None
        return th.emit(
            ctx,
            p,
            th.judge(p, risk, spec=limit, code=code, confirmed=att is not None),
            risk=risk,
            spec=limit,
            code=code,
            spec_ref=ctx.spec_ref("progress_guard"),
            attested=att,
            evidence=f"{p.key}: spec={th.fmt(limit) if limit is not None else 'none'}; real runs: {len(runs)}",
        )
    bad = []
    for r in runs:
        n, item = _longest_repeat(r.tools)
        if item is not None and n > limit:
            bad.append((r, n, item))
    ctx.clause(extracted=True)
    if not bad:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-228",
                subject="progress_guard",
                confidence=_obs(0, len(runs)),
                param=p.key,
                note=f"no run repeated an identical tool call more than {th.fmt(limit)} times",
            )
        )
        return []
    r, n, (name, args) = max(bad, key=lambda b: b[1])
    return [
        make_finding(
            "AS-228",
            confidence=_obs(len(bad), len(runs)),
            spec_ref=ctx.spec_ref("progress_guard") if ctx.loaded else None,
            param=p.key,
            title=f"Violation found: {name} called {n} times in a row with the same arguments",
            detail=f"in {len(bad)} of {len(runs)} real run(s) the agent repeated an identical tool "
            f"call more than max_no_progress_steps={th.fmt(limit)} times (worst: run {r.id})",
            evidence=f"{name}({(args or '')[:120]}) x{n}",
        )
    ]


@rule("conformance", "AS-229")
def observed_bounds(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    if not obs:
        return []
    runs = real_runs(obs)
    out = []
    # measured against the number spec.yml states
    for p, measure in ((th.MAX_STEPS, lambda r: r.llm_calls), (th.MAX_TOOL_CALLS, lambda r: len(r.tools))):
        key, limit = p.name, th.spec_value(ctx, p)
        if not runs or not limit:
            continue
        ctx.clause(extracted=True)
        over = [(r, measure(r)) for r in runs if measure(r) > limit]
        if not over:
            ctx.pass_(
                PassedCheck(
                    rule_id="AS-229",
                    subject=key,
                    confidence=_obs(0, len(runs)),
                    param=p.key,
                    note=f"every real run stayed within {key}={th.fmt(limit)}",
                )
            )
            continue
        worst = max(over, key=lambda x: x[1])
        out.append(
            make_finding(
                "AS-229",
                confidence=_obs(len(over), len(runs)),
                spec_ref=ctx.spec_ref("bounds") if ctx.loaded else None,
                param=p.key,
                title=f"Violation found: a run used {worst[1]} {key.removeprefix('max_')} (bound {th.fmt(limit)})",
                detail=f"{len(over)} of {len(runs)} real run(s) exceeded bounds.{key}={th.fmt(limit)}; "
                "the code does not enforce the declared bound",
                evidence=f"run {worst[0].id}: {worst[1]}",
            )
        )
    return out


@rule("conformance", "AS-230")
def stop_conditions(ctx: CheckContext) -> Iterable[Finding]:
    """A run must end on a declared stop condition. From captures: the last model response either
    answers (goal_done) or the run hit its step bound (budget_exhausted). A run whose model was still
    asking for tools when it ended stopped for a reason nobody declared or surfaced."""
    obs = ctx.observed
    if not obs:
        return []
    runs = [r for r in real_runs(obs) if r.ended_mid_action is not None]
    if not runs:
        return []
    eff = ctx.effective()
    declared = set(eff.get("stop_conditions") or [])
    from agenticreq.rules.conformance import effective_step_bound

    max_steps = th.spec_value(ctx, th.MAX_STEPS)
    if max_steps is None:  # no number in spec.yml: the bound the code runs under
        code_bound = effective_step_bound(obs)
        max_steps = code_bound[1] if code_bound else None
    ctx.clause(extracted=True)
    cut = [
        r
        for r in runs
        if r.ended_mid_action and not (max_steps and r.llm_calls >= max_steps and "budget_exhausted" in declared)
    ]
    if not cut:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-230",
                subject="stop_conditions",
                confidence=_obs(0, len(runs)),
                note="every real run ended with an answer or at its declared budget",
            )
        )
        return []
    return [
        make_finding(
            "AS-230",
            confidence=_obs(len(cut), len(runs)),
            spec_ref=ctx.spec_ref("stop_conditions") if ctx.loaded else None,
            detail=f"{len(cut)} of {len(runs)} real run(s) ended while the model was still calling "
            f"tools, before its step bound ({max_steps}); declared stop_conditions: "
            f"{sorted(declared)}. If an error stopped it, nothing reported it as one.",
            evidence=f"runs: {[r.id for r in cut][:5]}",
        )
    ]


@rule("conformance", "AS-231")
def observed_rate(ctx: CheckContext) -> Iterable[Finding]:
    obs = ctx.observed
    p, risk = th.RATE_LIMITS, ctx.risk_level
    rpm = th.spec_value(ctx, p)
    if not obs or not rpm:
        return []
    runs = [r for r in real_runs(obs) if len(r.times) >= 2]
    if not runs:  # no timestamps to measure a rate from: signed off under confirmed:, or a mock run checks it
        att = th.attestation(ctx, p)
        return th.emit(
            ctx,
            p,
            th.judge(p, risk, spec=rpm, confirmed=att is not None),
            risk=risk,
            spec=rpm,
            spec_ref=ctx.spec_ref("rate_limits") if ctx.loaded else None,
            attested=att,
            evidence=f"rate_limits: spec={th.fmt(rpm)}/min; no run with timestamps",
        )
    ctx.clause(extracted=True)
    worst = 0
    for r in runs:
        ts = sorted(r.times)
        j = 0
        for i, t in enumerate(ts):
            while t - ts[j] >= 60.0:
                j += 1
            worst = max(worst, i - j + 1)
    if worst <= rpm:
        ctx.pass_(
            PassedCheck(
                rule_id="AS-231",
                subject="rate_limits",
                confidence=_obs(0, len(runs)),
                param=p.key,
                note=f"peak {worst} LLM requests in 60 s <= {th.fmt(rpm)}",
            )
        )
        return []
    return [
        make_finding(
            "AS-231",
            confidence=_obs(1, len(runs)),
            param=p.key,
            spec_ref=ctx.spec_ref("rate_limits") if ctx.loaded else None,
            detail=f"a real run sent {worst} LLM requests within 60 s; rate_limits allows {th.fmt(rpm)}/min",
            evidence=f"peak window: {worst} requests",
        )
    ]
