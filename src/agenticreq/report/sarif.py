"""SARIF 2.1.0 renderer, for GitHub code-scanning PR annotations.

Levels follow the four result groups (models/verdict.py): a spec vs code gap -> "error", a mock-run check or an open
sign-off -> "note", a signed sign-off -> "none". The exit code (--fail-on) is what fails the job."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agenticreq import __version__
from agenticreq.models.finding import Finding, Report, SourceRef
from agenticreq.models.verdict import FailOn, finding_verdict, overall, results, totals
from agenticreq.report.json_report import HOW_SURE, SEVERITY
from agenticreq.report.grouping import attribute_of
from agenticreq.rules.registry import REGISTRY

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

#: a rule's default level, before any result is known (driver.rules[].defaultConfiguration)
_LEVEL = {"blocker": "error", "warning": "warning", "info": "note", "attestation": "note"}
#: a result's level: what the user has to do about it
_STATUS_LEVEL = {
    "GAP": "error",
    "CHECK": "note",
    "SIGN-OFF": "note",
    "SIGNED": "none",
    "NOT NEEDED": "none",
    "PASS": "none",
}


def _uri(ref: SourceRef, base: Path | None, repo_root: Path | None) -> str:
    """Repo-relative, forward-slash URI: GitHub places a PR annotation only when the path matches a file
    in the repository. Code refs are relative to the scanned folder (`base`); spec refs to the cwd."""
    p = Path(ref.file)
    if base is not None and not p.is_absolute():
        p = base / p
    try:
        rel = os.path.relpath(p.resolve(), (repo_root or Path.cwd()).resolve())
    except ValueError:  # another drive on Windows: keep what we have
        return ref.file.replace("\\", "/")
    return rel.replace("\\", "/")


def _location(ref: SourceRef, role: str, base: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    region: dict[str, Any] = {}
    if ref.line is not None:
        region["startLine"] = max(ref.line, 1)
    if ref.column is not None:
        region["startColumn"] = max(ref.column, 1)
    phys: dict[str, Any] = {"artifactLocation": {"uri": _uri(ref, base, repo_root)}}
    if region:
        phys["region"] = region
    return {"physicalLocation": phys, "message": {"text": role}}


def _fingerprint(f: Finding) -> str:
    key = "|".join([f.rule_id, f.title, str(f.location() or ""), ">".join(f.path or [])])
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _result(f: Finding, code_root: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    v = finding_verdict(f)
    text = f"{v.status} {v.reason} — {f.detail} [{HOW_SURE[f.confidence.level]}]"
    if f.path:
        text += " Path: " + " → ".join(f.path)
    locations: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []
    if f.code_ref:
        locations.append(_location(f.code_ref, "agent code", code_root, repo_root))
    if f.spec_ref:
        (related if locations else locations).append(_location(f.spec_ref, "spec", None, repo_root))
    res: dict[str, Any] = {
        "ruleId": f.rule_id,
        "level": _STATUS_LEVEL[v.status],
        "message": {"text": text},
        "partialFingerprints": {"agenticreq/v1": _fingerprint(f)},
        "properties": {
            "group": v.group,
            "status": v.status,
            "reason": v.reason,
            "howSure": HOW_SURE[f.confidence.level],
            "severity": SEVERITY.get(f.severity, f.severity),
            "qualityAttribute": attribute_of(f).value,
            "evidence": f.evidence,
            "fixable": f.fixable,
        },
    }
    if locations:
        res["locations"] = locations
    if related:
        for i, r in enumerate(related):
            r["id"] = i
        res["relatedLocations"] = related
    if f.path:
        res["properties"]["path"] = f.path
    if f.remediation:
        res["properties"]["remediation"] = f.remediation
    return res


def _spec_invalid(report: Report) -> list[Finding]:
    """An invalid spec is one SARIF result (AS-100) on the spec file, listing every field to fix."""
    from agenticreq.models.confidence import Confidence
    from agenticreq.rules.registry import make_finding

    bad = [i for i in report.spec_issues if i.kind == "invalid"]
    if report.spec_status != "invalid" or not bad:
        return []
    lines = [f"{i.field}: {i.message}" for i in report.spec_issues]
    return [
        make_finding(
            "AS-100",
            confidence=Confidence.proven(),
            spec_ref=bad[0].spec_ref,
            detail="spec has the wrong shape; nothing about the agent was checked:\n" + "\n".join(lines),
            evidence="; ".join(i.field for i in report.spec_issues),
        )
    ]


def report_to_sarif(
    report: Report, code_root: Path | None = None, repo_root: Path | None = None, fail_on: FailOn = "gaps"
) -> dict[str, Any]:
    lines = results(report)
    report = report.model_copy(update={"findings": [*_spec_invalid(report), *report.findings]})
    used = {f.rule_id for f in report.findings}
    rules = []
    for meta in REGISTRY.all():
        if meta.rule_id not in used:
            continue
        rules.append(
            {
                "id": meta.rule_id,
                "name": meta.name,
                "shortDescription": {"text": meta.title},
                "help": {"text": meta.remediation},
                "defaultConfiguration": {"level": _LEVEL[meta.severity]},
                "properties": {"qualityAttribute": meta.attribute.value, "fixable": meta.fixable},
            }
        )
    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "agenticreq",
                        "version": __version__,
                        "informationUri": "https://github.com/agenticreq/agenticreq",
                        "rules": rules,
                    }
                },
                "results": [_result(f, code_root, repo_root) for f in report.findings],
                "properties": {
                    "agentId": report.agent_id,
                    "result": overall(report, lines, fail_on),
                    "counts": totals(lines),
                    "coverage": {
                        "extracted": report.coverage.extracted_clauses,
                        "total": report.coverage.total_clauses,
                    },
                },
            }
        ],
    }


def render_sarif(
    report: Report, code_root: Path | None = None, repo_root: Path | None = None, fail_on: FailOn = "gaps"
) -> str:
    """`code_root`: the folder that was scanned (code refs are relative to it); `repo_root`: what the URIs
    are made relative to (default: the current directory, i.e. the repository root in CI)."""
    return json.dumps(report_to_sarif(report, code_root, repo_root, fail_on), indent=2)
