"""Numbers only a person or a runtime can confirm: output size, confidence threshold, approval timeout.

The library cannot read these from code. A number written in spec.yml waits for a mock run or a sign-off under
``confirmed:``; a number written nowhere is a gap. None of these is a risk_level requirement, so a number the spec
leaves out is only reported where a gate is expected (confidence_threshold at critical); the message names no number.
"""

from __future__ import annotations

from typing import Iterable

from agenticreq import thresholds as th
from agenticreq.ast.classify import canonical
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding
from agenticreq.models.observed import GateRef
from agenticreq.rules.base import CheckContext, rule
from agenticreq.rules.registry import make_finding


def _named(ctx: CheckContext, *words: str, seconds: bool = False) -> float | None:
    """The value of the one project constant named after this limit (ast/constants.py), else None."""
    from agenticreq.ast.constants import find

    hit = find(ctx.observed.root, *words, seconds=seconds) if ctx.observed is not None else None
    return hit[0] if hit else None


def _seconds(t: str) -> float | None:
    """ "4h" -> 14400."""
    import re

    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", t)
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)] if m else None


def _waits_forever(ctx: CheckContext, tool: str) -> GateRef | None:
    """The approval gate for ``tool`` that waits for a person with no time limit, if there is one."""
    obs = ctx.observed
    if obs is None:
        return None
    return next(
        (g for g in obs.gates if g.waits_forever and g.tools and canonical(tool) in {canonical(t) for t in g.tools}),
        None,
    )


@rule("conformance", "AS-501", "AS-205")
def unreadable_numbers(ctx: CheckContext) -> Iterable[Finding]:
    if not ctx.loaded or not ctx.spec:
        return []
    risk = ctx.risk_level
    out: list[Finding] = []
    for p in (th.CONFIDENCE, th.MAX_CHARS):
        if th.spec_todo(ctx, p):
            continue  # AS-101 reports a TODO
        spec_n = th.spec_value(ctx, p)
        # a number the spec leaves out is only asked for where a confidence gate is expected (critical risk)
        asked = p is th.CONFIDENCE and risk == "critical"
        if spec_n is None and not asked:
            continue
        att = th.attestation(ctx, p)
        code = _named(ctx, "max", "chars") if p is th.MAX_CHARS else None  # MAX_CHARS = 6000 in the code
        out += th.emit(
            ctx,
            p,
            th.judge(p, risk, spec=spec_n, code=code, confirmed=att is not None),
            risk=risk,
            spec=spec_n,
            code=code,
            spec_ref=ctx.spec_ref(*p.key.split(".")),
            attested=att,
        )
    p = th.APPROVAL_TIMEOUT
    if not th.spec_todo(ctx, p):
        att = th.attestation(ctx, p)
        for i, a in enumerate(ctx.spec.approval_required or []):
            t = th.timeout_of(a)
            if t is None:
                continue
            forever = _waits_forever(ctx, a.tool)
            if forever is not None and att is None:  # the approval call blocks until someone answers: no timeout
                out.append(
                    make_finding(
                        "AS-205",
                        confidence=Confidence.proven(),
                        param=p.key,
                        status_hint="GAP",
                        spec_ref=ctx.spec_ref("approval_required", i, "timeout"),
                        code_ref=forever.code_ref,
                        title=f"{a.tool}: the approval call can wait forever",
                        detail=f"spec.yml says timeout {a.timeout} for {a.tool}; the approval call at {forever.code_ref} "
                        "waits for a person with no time limit",
                        reason=f"spec.yml says timeout {a.timeout}; the approval call can wait forever "
                        f"({a.tool}, {forever.code_ref}).",
                        evidence=f"approval_required[{i}] ({a.tool}): timeout {a.timeout}",
                    )
                )
                continue
            secs, code = _seconds(t), _named(ctx, "approval", "timeout", seconds=True)
            if secs is not None and code is not None:  # APPROVAL_TIMEOUT_S = 4 * 3600: compared in seconds
                out += th.emit(
                    ctx,
                    p,
                    th.judge(p, risk, spec=secs, code=code, confirmed=att is not None),
                    risk=risk,
                    spec=secs,
                    code=code,
                    spec_ref=ctx.spec_ref("approval_required", i, "timeout"),
                    attested=att,
                    evidence=f"approval_required[{i}] ({a.tool}): timeout {t}",
                )
                continue
            out += th.emit(
                ctx,
                p,
                th.judge(p, risk, spec=t, confirmed=att is not None),
                risk=risk,
                spec=t,
                spec_ref=ctx.spec_ref("approval_required", i, "timeout"),
                attested=att,
                evidence=f"approval_required[{i}] ({a.tool}): timeout {t}",
            )
    return out
