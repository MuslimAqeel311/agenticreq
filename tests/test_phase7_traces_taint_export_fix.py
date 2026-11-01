"""Phase 7 acceptance: OTel traces, trace_coverage, taint, OPA/Cedar export (evaluated), fix, MCP."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.export import to_cedar, to_rego, to_schema_constraints
from agenticreq.fix import apply as apply_patches
from agenticreq.fix import plan
from agenticreq.ast import extract
from agenticreq.loader import load_spec
from agenticreq.mcp import list_tools_stdio
from agenticreq.models import Bucket
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.observe import build_observed
from agenticreq.traces import load_traces
from agenticreq.traces.coverage import coverage_matrix

FIX = Path(__file__).parent / "fixtures"
AP = FIX / "agents_misc" / "email_payer"
TRACES = FIX / "traces" / "ap_agent.otlp.json"
ROOT = Path(__file__).parent.parent
OPA = shutil.which("opa") or (str(ROOT / ".tools" / "opa") if (ROOT / ".tools" / "opa").exists() else None)


def _check(with_traces: bool = True):
    spans = load_traces(TRACES) if with_traces else None
    obs = build_observed(AP, spans=spans)
    return run(mode="check", loaded=load_spec(AP / "spec.yml"), observed=obs), obs


# ------------------------------------------------------------------ traces
def test_otlp_reader() -> None:
    spans = load_traces(TRACES)
    assert len(spans) == 7
    assert {s.tool_name for s in spans if s.tool_name} == {
        "read_inbox",
        "pay_invoice",
        "search_vendors",
        "issue_refund",
    }
    assert spans[2].parent_span_id == "a1"


def test_trace_proves_violation_not_safety() -> None:
    r, _ = _check()
    f = next(f for f in r.findings if f.rule_id == "AS-204")
    assert f.confidence.level is ConfidenceLevel.OBSERVED and str(f.confidence) == "observed (1/3)"
    assert f.detail.startswith("violation found")


def test_no_violation_observed_wording(tmp_path: Path) -> None:
    spans = [s for s in load_traces(TRACES) if s.tool_name != "issue_refund"]
    obs = build_observed(AP, spans=spans)
    r = run(mode="check", loaded=load_spec(AP / "spec.yml"), observed=obs)
    p = next(p for p in r.passed if p.rule_id == "AS-204")
    assert p.wording.startswith("no violation observed")
    assert not any(f.rule_id == "AS-204" for f in r.findings)


def test_trace_coverage_matrix() -> None:
    spec = load_spec(AP / "spec.yml").spec
    assert spec is not None
    m = {sc.id: n for sc, n in coverage_matrix(spec, load_traces(TRACES))}
    assert m["approval:pay_invoice:granted"] == 1
    assert m["approval:pay_invoice:timeout"] == 0
    assert m["limit:pay_invoice.amount"] == 1  # amount 9000 >= 5000 seen
    assert m["prohibited:issue_refund"] == 1
    r, _ = _check()
    unc = [f for f in r.findings if f.rule_id == "AS-905"]
    assert {f.title for f in unc} >= {"Scenario not covered: approval:pay_invoice:timeout"}
    assert all(f.bucket is Bucket.UNKNOWNS for f in unc)  # insufficient_evidence never passes


# ------------------------------------------------------------------ taint
def test_static_taint_untrusted_to_write() -> None:
    r, _ = _check(with_traces=False)
    f = next(f for f in r.findings if f.rule_id == "AS-401")
    assert f.path == ["read_inbox", "LLM", "pay_invoice"] and f.severity == "blocker"  # high risk


def test_observed_taint_from_traces() -> None:
    r, _ = _check()
    f = next(f for f in r.findings if f.rule_id == "AS-402")
    assert f.title == "Violation found: read_inbox → LLM → pay_invoice"
    assert str(f.confidence) == "observed (1/3)"


def test_taint_negative_with_gate(tmp_path: Path) -> None:
    src = (
        (AP / "agent.py")
        .read_text(encoding="utf-8")
        .replace('return "paid"', 'if not interrupt("approve?"):\n        return "denied"\n    return "paid"')
        .replace(
            "from langchain_core.tools import tool",
            "from langchain_core.tools import tool\nfrom langgraph.types import interrupt",
        )
    )
    (tmp_path / "agent.py").write_text(src)
    obs = build_observed(tmp_path)
    r = run(mode="check", loaded=load_spec(AP / "spec.yml"), observed=obs)
    assert not any(f.rule_id == "AS-401" for f in r.findings)
    assert not any(f.rule_id == "AS-205" for f in r.findings)  # approval gate found for pay_invoice


def test_limit_in_body_not_schema() -> None:
    r, _ = _check(with_traces=False)
    f = next(f for f in r.findings if f.rule_id == "AS-207")
    assert "function body compares against it" in f.detail and f.severity == "warning"


# ------------------------------------------------------------------ export
def _spec():
    s = load_spec(FIX / "specs" / "good.yml").spec
    assert s is not None
    return s


@pytest.mark.skipif(OPA is None, reason="opa binary not available")
@pytest.mark.parametrize(
    "inp,allowed,denies",
    [
        ({"tool": "get_inventory", "args": {"max_results": 10}}, True, 0),
        ({"tool": "get_inventory", "args": {"max_results": 500}}, False, 1),
        ({"tool": "issue_refund", "args": {}}, False, 2),  # prohibited + outside role_boundary
        ({"tool": "pay_invoice", "args": {"amount": 100}}, False, 0),  # needs approval
        (
            {"tool": "pay_invoice", "args": {"amount": 100}, "approval": {"granted": True, "approver": "finance-lead"}},
            True,
            0,
        ),
        (
            {
                "tool": "pay_invoice",
                "args": {"amount": 9000},
                "approval": {"granted": True, "approver": "finance-lead"},
            },
            False,
            1,
        ),
    ],
)
def test_rego_evaluates(tmp_path: Path, inp: dict, allowed: bool, denies: int) -> None:
    pol = tmp_path / "p.rego"
    pol.write_text(to_rego(_spec()))
    subprocess.run([OPA, "check", str(pol)], check=True, capture_output=True)
    (tmp_path / "in.json").write_text(json.dumps(inp))
    out = subprocess.run(
        [
            OPA,
            "eval",
            "-f",
            "json",
            "-d",
            str(pol),
            "-i",
            str(tmp_path / "in.json"),
            "data.agenticreq.procurement_po_agent",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = json.loads(out.stdout)["result"][0]["expressions"][0]["value"]
    assert res["allow"] is allowed
    assert len(res["deny"]) == denies


cedarpy = pytest.importorskip("cedarpy")


@pytest.mark.parametrize(
    "tool,ctx,allowed",
    [
        ("get_inventory", {"args": {"max_results": 10}}, True),
        ("get_inventory", {"args": {"max_results": 500}}, False),
        ("issue_refund", {"args": {}}, False),
        ("pay_invoice", {"args": {"amount": 100}}, False),
        ("pay_invoice", {"args": {"amount": 100}, "approval": {"granted": True, "approver": "finance-lead"}}, True),
        ("pay_invoice", {"args": {"amount": 9000}, "approval": {"granted": True, "approver": "finance-lead"}}, False),
    ],
)
def test_cedar_evaluates(tool: str, ctx: dict, allowed: bool) -> None:
    req = {
        "principal": 'Agent::"procurement-po-agent"',
        "action": 'Action::"call_tool"',
        "resource": f'Tool::"{tool}"',
        "context": ctx,
    }
    res = cedarpy.is_authorized(req, to_cedar(_spec()), [])
    assert (res.decision == cedarpy.Decision.Allow) is allowed, res.diagnostics


def test_schema_constraints() -> None:
    d = to_schema_constraints(_spec())
    assert d["tools"]["pay_invoice"]["properties"]["amount"] == {"maximum": 5000.0}


# ------------------------------------------------------------------ fix
def test_fix_dry_run_then_apply(tmp_path: Path) -> None:
    src = FIX / "agents" / "langgraph_po"
    shutil.copytree(src, tmp_path / "a")
    before = (tmp_path / "a" / "agent.py").read_text(encoding="utf-8")
    r = CliRunner().invoke(app, ["fix", str(tmp_path / "a")])
    assert r.exit_code == 0 and '+MODEL = "gpt-4o-2024-08-06"' in r.stdout
    assert (tmp_path / "a" / "agent.py").read_text(encoding="utf-8") == before  # dry-run wrote nothing
    patches = plan(extract(tmp_path / "a"), tmp_path / "a", 15)
    # no idempotency patch (AS-211): the fixture's order / payment tools are stubs whose bodies write nothing
    assert {p.rule_id for p in patches} == {"AS-209", "AS-210"}
    apply_patches(patches)
    after = extract(tmp_path / "a")
    assert all(m.pinned for m in after.models)
    report = run(mode="audit", observed=after)
    assert not {"AS-209", "AS-210", "AS-211"} & {f.rule_id for f in report.findings}


def test_fix_adds_step_cap(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from crewai import Agent\nagent = Agent(role='r', goal='g', backstory='b', tools=[])\n"
    )
    patches = plan(extract(tmp_path), tmp_path, 15)
    assert [p.rule_id for p in patches] == ["AS-208"]
    apply_patches(patches)
    assert "max_iter=15" in (tmp_path / "a.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ MCP
def test_mcp_tools_list_paginated(tmp_path: Path) -> None:
    tools = list_tools_stdio([sys.executable, str(FIX / "mcp" / "fake_server.py")])
    assert [t.name for t in tools] == ["get_inventory", "delete_records"]
    (tmp_path / "a.py").write_text(
        "from langchain_core.tools import tool\n@tool\ndef get_inventory(sku: str) -> str:\n    '''Look up stock.'''\n"
    )
    obs = build_observed(tmp_path, mcp_tools=tools)
    r = run(mode="audit", observed=obs)
    assert any(f.rule_id == "AS-202" and "delete_records" in f.title for f in r.findings)
