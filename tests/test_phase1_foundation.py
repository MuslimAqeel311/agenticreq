"""Phase 1 acceptance: hand-built findings render in console, JSON and SARIF; four buckets; no score."""

import json

import pytest

from agenticreq.models import Bucket, Confidence, Finding, PassedCheck, Report, SourceRef
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.report import render_console, render_json, render_sarif
from agenticreq.rules.registry import REGISTRY, make_finding


def _findings() -> list[Finding]:
    return [
        make_finding(
            "AS-203",
            confidence=Confidence.proven(),
            detail="issue_refund is bound to the agent",
            evidence="llm.bind_tools([..., issue_refund])",
            code_ref=SourceRef(file="agent.py", line=42),
            spec_ref=SourceRef(file="spec.yml", line=12),
        ),
        make_finding(
            "AS-603",
            confidence=Confidence.proven(),
            detail="code says warehouse is optional (default 'KHI'); the model is told it is required",
            evidence='{"required": ["sku", "warehouse"]}',
            code_ref=SourceRef(file="tools.py", line=8),
        ),
        make_finding(
            "AS-201",
            confidence=Confidence.unknown(),
            detail="issue_refund never reached",
            evidence="0/24 captured requests",
            code_ref=SourceRef(file="tools.py", line=30),
        ),
        make_finding(
            "AS-501",
            confidence=Confidence.attestation(),
            detail="data_rules.retention",
            evidence="spec.yml data_rules.retention: 30d",
            spec_ref=SourceRef(file="spec.yml", line=40),
        ),
        make_finding(
            "AS-304",
            confidence=Confidence.heuristic(),
            detail="A → B → C can reach pay_invoice",
            evidence="transfer_to_b, transfer_to_c",
            path=["A", "B", "C"],
        ),
        make_finding(
            "AS-204",
            confidence=Confidence.observed(2, 24),
            detail="pay_invoice called",
            evidence="tool_use pay_invoice {...}",
        ),
    ]


def test_confidence_rules() -> None:
    assert str(Confidence.observed(3, 5)) == "observed (3/5)"
    with pytest.raises(ValueError):
        Confidence(level=ConfidenceLevel.OBSERVED)
    with pytest.raises(ValueError):
        Confidence(level=ConfidenceLevel.PROVEN, n=1, total=2)
    with pytest.raises(ValueError):
        Confidence.observed(6, 5)


def test_unknown_and_insufficient_never_pass() -> None:
    for c in (Confidence.unknown(), Confidence.insufficient()):
        with pytest.raises(ValueError):
            PassedCheck(rule_id="AS-201", subject="x", confidence=c)
    p = PassedCheck(rule_id="AS-204", subject="pay_invoice", confidence=Confidence.observed(0, 24))
    assert p.wording.startswith("no violation observed")


def test_finding_requires_evidence_and_valid_id() -> None:
    with pytest.raises(ValueError):
        Finding(
            rule_id="AS-203", severity="blocker", confidence=Confidence.proven(), title="t", detail="d", evidence=""
        )
    with pytest.raises(ValueError):
        Finding(rule_id="X-1", severity="blocker", confidence=Confidence.proven(), title="t", detail="d", evidence="e")


def test_buckets() -> None:
    r = Report(agent_id="a", findings=_findings())
    b = r.by_bucket()
    assert [f.rule_id for f in b[Bucket.BLOCKERS]] == ["AS-203", "AS-304", "AS-204"]
    assert [f.rule_id for f in b[Bucket.WARNINGS]] == ["AS-603"]
    assert [f.rule_id for f in b[Bucket.UNKNOWNS]] == ["AS-201"]
    assert [f.rule_id for f in b[Bucket.ATTESTATIONS]] == ["AS-501"]
    assert r.exit_code == 1


def test_console_render() -> None:
    report = Report(agent_id="po-agent", findings=_findings())
    out = render_console(report)
    assert "Result: FAIL" in out and "BLOCKERS" not in out  # the four groups, never the internal buckets
    for title in ("SPEC VS CODE GAPS", "HUMAN SIGN-OFF", "CHECK WITH A MOCK RUN"):
        assert title in out, title
    out = render_console(report, verbose=True)  # --verbose adds the rule ids
    assert "AS-203" in out and "BLOCKERS" not in out
    assert "score" not in out.lower()


def test_json_render() -> None:
    d = json.loads(render_json(Report(agent_id="po-agent", findings=_findings())))
    assert {f["group"] for f in d["findings"]} == {"gap", "sign_off", "mock_run"}
    assert "score" not in json.dumps(d).lower()
    f = d["findings"][0]
    assert f["rule_id"] == "AS-203" and f["how_sure"] == "read from the code" and f["code_ref"]["line"] == 42
    assert f["group"] == "gap" and f["status"] == "GAP"


def test_sarif_render() -> None:
    s = json.loads(render_sarif(Report(agent_id="po-agent", findings=_findings())))
    assert s["version"] == "2.1.0"
    run = s["runs"][0]
    assert {r["id"] for r in run["tool"]["driver"]["rules"]} == {f.rule_id for f in _findings()}
    levels = {r["ruleId"]: r["level"] for r in run["results"]}
    # levels follow the groups: a gap -> error, an open sign-off or a mock-run check -> note
    assert levels["AS-203"] == "error" and levels["AS-603"] == "error" and levels["AS-501"] == "note"
    assert levels["AS-201"] == "note"
    res = next(r for r in run["results"] if r["ruleId"] == "AS-203")
    assert res["locations"][0]["physicalLocation"]["region"]["startLine"] == 42
    assert res["relatedLocations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "spec.yml"


def test_registry_complete() -> None:
    ids = [m.rule_id for m in REGISTRY.all()]
    for ns in ("AS-1", "AS-2", "AS-3", "AS-4", "AS-5", "AS-6", "AS-9"):
        assert any(i.startswith(ns) for i in ids)
