"""JSON renderer — stable, machine-readable, no score.

``results`` holds the per-agent parameter lines the console shows. Each line, finding and passed check carries
``group`` (gap | sign_off | mock_run | passed), ``status`` (GAP | SIGN-OFF | SIGNED | CHECK | PASS) and ``reason``;
``result`` and ``counts`` summarise the lines."""

from __future__ import annotations

import json
from typing import Any

from agenticreq import __version__
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.models.finding import Finding, Report
from agenticreq.models.verdict import FailOn, Verdict, finding_verdict, overall, passed_verdict, results, totals
from agenticreq.report.grouping import attribute_of, is_general_note

#: how a result was decided, in plain words (the internal confidence levels are never printed)
HOW_SURE: dict[ConfidenceLevel, str] = {
    ConfidenceLevel.PROVEN: "read from the code",
    ConfidenceLevel.OBSERVED: "seen in runs",
    ConfidenceLevel.HEURISTIC: "code pattern only",
    ConfidenceLevel.UNKNOWN: "needs a mock run",
    ConfidenceLevel.INSUFFICIENT_EVIDENCE: "the mock runs did not cover it",
    ConfidenceLevel.ATTESTATION: "needs a person to confirm",
}
#: a rule's severity, in the words of the groups
SEVERITY: dict[str, str] = {"blocker": "blocker", "warning": "warning", "info": "info", "attestation": "sign-off"}


def _verdict_fields(v: Verdict) -> dict[str, Any]:
    return {"group": v.group, "status": v.status, "reason": v.reason}


def _finding(f: Finding) -> dict[str, Any]:
    d: dict[str, Any] = {**_verdict_fields(finding_verdict(f))}
    raw = f.model_dump(mode="json", exclude={"confidence", "severity", "reason"})
    d.update(raw)
    d["severity"] = SEVERITY.get(f.severity, f.severity)
    d["how_sure"] = HOW_SURE[f.confidence.level]
    if f.confidence.n is not None and f.confidence.total is not None:
        d["runs"] = {"violated": f.confidence.n, "total": f.confidence.total}
    d["quality_attribute"] = attribute_of(f).value
    d["general_note"] = is_general_note(f)  # fires on most agents; not specific to this code
    return d


def report_to_dict(report: Report, fail_on: FailOn = "gaps") -> dict[str, Any]:
    lines = results(report)
    return {
        "tool": {"name": "agenticreq", "version": __version__},
        "command": report.command,
        "agent_id": report.agent_id,
        "result": overall(report, lines, fail_on),
        "fail_on": fail_on,
        "counts": totals(lines),
        "results": [
            {
                "agent": ln.agent,
                "param": ln.param,
                "group": ln.group,
                "status": ln.status,
                "reason": ln.reason,
                **({"found": ln.support} if ln.support else {}),
                **({"open": True} if ln.open else {}),
                "rule_ids": list(ln.rule_ids),
                "locations": list(ln.locations),
            }
            for ln in lines
        ],
        "coverage": {
            "extracted_clauses": report.coverage.extracted_clauses,
            "total_clauses": report.coverage.total_clauses,
            "note": "all counts are over the extracted portion only",
        },
        "general_notes": sum(1 for f in report.findings if is_general_note(f)),
        "findings": [_finding(f) for f in report.findings],
        "passed": [
            {
                **_verdict_fields(passed_verdict(p)),
                "rule_id": p.rule_id,
                "subject": p.subject,
                "how_sure": HOW_SURE[p.confidence.level],
                "note": p.note,
                **({"param": p.param} if p.param else {}),
                **({"agent": p.agent} if p.agent else {}),
            }
            for p in report.passed
        ],
        "nothing_checked": report.nothing_checked,
        "metadata": report.metadata,
        "spec": None
        if report.spec_status is None
        else {
            "status": report.spec_status,
            "issues": [{**i.model_dump(mode="json"), "count": i.count} for i in report.spec_issues],
        },
    }


def render_json(report: Report, indent: int = 2, fail_on: FailOn = "gaps") -> str:
    return json.dumps(report_to_dict(report, fail_on), indent=indent, default=str)
