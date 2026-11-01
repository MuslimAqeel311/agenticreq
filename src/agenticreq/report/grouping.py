"""Group findings by bucket and by quality attribute."""

from __future__ import annotations

from agenticreq.models.finding import Bucket, Finding, QualityAttribute
from agenticreq.rules.registry import REGISTRY

BUCKET_ORDER: tuple[Bucket, ...] = (
    Bucket.BLOCKERS,
    Bucket.WARNINGS,
    Bucket.UNKNOWNS,
    Bucket.ATTESTATIONS,
)


#: checks that fire on most agents and say nothing specific about this code (environment separation, free-text output,
#: no audit hook, no checkpointer, no runtime capture). They are reported together as GENERAL NOTES, never as the
#: findings to fix first, so the specific ones are not buried under them.
GENERAL_NOTE_RULES = frozenset({"AS-223", "AS-224", "AS-225", "AS-226", "AS-903"})


def is_general_note(f: Finding) -> bool:
    return f.rule_id in GENERAL_NOTE_RULES and f.bucket in (Bucket.WARNINGS, Bucket.UNKNOWNS)


def attribute_of(f: Finding) -> QualityAttribute:
    if f.rule_id in REGISTRY:
        return REGISTRY.get(f.rule_id).attribute
    return QualityAttribute.ACCOUNTABILITY


def group_by_attribute(findings: list[Finding]) -> dict[QualityAttribute, list[Finding]]:
    out: dict[QualityAttribute, list[Finding]] = {}
    for qa in QualityAttribute:
        hits = [f for f in findings if attribute_of(f) is qa]
        if hits:
            out[qa] = sorted(hits, key=lambda f: (f.rule_id, f.title))
    return out


def ordered_buckets(findings: list[Finding]) -> list[tuple[Bucket, list[Finding]]]:
    return [(b, [f for f in findings if f.bucket is b]) for b in BUCKET_ORDER]
