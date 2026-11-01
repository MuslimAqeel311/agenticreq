"""Phase 4 acceptance: declared vs observed diff, injected allowlist, schema fidelity, execution model."""

import importlib.util
from pathlib import Path

import pytest
from conftest import chat_call

from agenticreq.ast import extract
from agenticreq.engine import run
from agenticreq.models import Bucket
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.observe import build_observed
from agenticreq.rules.fidelity import compare_tool

AGENT_SRC = '''
from langchain_core.tools import tool

@tool
def get_inventory(sku: str, warehouse: str = "KHI", max_results: int = 50) -> str:
    """Look up stock.

    Args:
        sku: Stock keeping unit.
        warehouse: Warehouse code.
        max_results: Maximum rows to return.
    """

@tool
def issue_refund(order_id: str) -> str:
    """Refund an order."""

TOOLS = [get_inventory, issue_refund]
llm = ChatOpenAI(model="gpt-4o-2024-08-06").bind_tools(TOOLS)
'''

GI_WIRE_FAITHFUL = {
    "name": "get_inventory",
    "description": "Look up stock.",
    "parameters": {
        "type": "object",
        "required": ["sku"],
        "properties": {
            "sku": {"type": "string", "description": "Stock keeping unit."},
            "warehouse": {"type": "string", "description": "Warehouse code.", "default": "KHI"},
            "max_results": {"type": "integer", "description": "Maximum rows to return.", "default": 50},
        },
    },
}
GI_WIRE_CREWAI_LIKE = {
    "name": "get_inventory",
    "description": "Look up stock.",
    "parameters": {
        "type": "object",
        "required": ["sku", "warehouse", "max_results"],
        "properties": {"sku": {"type": "string"}, "warehouse": {"type": "string"}, "max_results": {"type": "integer"}},
    },
}


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    (tmp_path / "agent.py").write_text(AGENT_SRC)
    return tmp_path


def _ids(report, bucket=None):
    fs = report.findings if bucket is None else report.by_bucket()[bucket]
    return [f.rule_id for f in fs]


def test_never_reached_is_unknown_not_pass(agent_dir: Path) -> None:
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_FAITHFUL])])
    r = run(mode="audit", observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-201")
    assert "issue_refund" in f.title and f.confidence.level is ConfidenceLevel.UNKNOWN
    assert f.bucket is Bucket.UNKNOWNS
    assert not any(p.subject == "issue_refund" for p in r.passed)


def test_reached_tool_not_reported(agent_dir: Path) -> None:
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_FAITHFUL])])
    r = run(mode="audit", observed=obs)
    assert not any(f.rule_id == "AS-201" and "get_inventory" in f.title for f in r.findings)


def test_undeclared_observed_tool(agent_dir: Path) -> None:
    sneaky = {"name": "wire_money", "description": "x", "parameters": {"type": "object", "properties": {}}}
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_FAITHFUL, sneaky])])
    r = run(mode="audit", observed=obs)
    assert any(f.rule_id == "AS-202" and "wire_money" in f.title for f in r.findings)


@pytest.mark.parametrize(
    "injected", ["delegate_work_to_coworker", "Delegate work to coworker", "transfer_to_policy_analyst", "final_answer"]
)
def test_injected_tools_allowlisted(agent_dir: Path, injected: str) -> None:
    t = {"name": injected, "description": "x", "parameters": {"type": "object", "properties": {}}}
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_FAITHFUL, t])])
    r = run(mode="audit", observed=obs)
    assert not any(f.rule_id == "AS-202" for f in r.findings)
    assert any(p.rule_id == "AS-202" and p.subject == injected for p in r.passed)


def test_faithful_schema_passes(agent_dir: Path) -> None:
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_FAITHFUL])])
    r = run(mode="audit", observed=obs)
    assert not [i for i in _ids(r) if i.startswith("AS-6")]
    assert any(p.rule_id == "AS-608" and p.subject == "get_inventory" for p in r.passed)


def test_crewai_like_drift_detected(agent_dir: Path) -> None:
    obs = build_observed(agent_dir, captures=[chat_call(0, [GI_WIRE_CREWAI_LIKE])])
    issues, verbatim, total = compare_tool(obs.declared_tools["get_inventory"], obs.wire_tools["get_inventory"])
    kinds = sorted(r for r, _, _ in issues)
    assert kinds == ["AS-602", "AS-602", "AS-602", "AS-603", "AS-603", "AS-605", "AS-605"]
    assert (verbatim, total) == (5, 12)
    r = run(mode="audit", observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-603" and "warehouse" in f.title)
    assert f.detail == (
        "code says warehouse is optional (default 'KHI') with description 'Warehouse code.'; "
        "the model is told it is required with no description"
    )


def test_codeagent_unknown(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from smolagents import CodeAgent, tool\n@tool\ndef t(x: str) -> str:\n"
        "    '''T.\n\n    Args:\n        x: X.\n    '''\nagent = CodeAgent(tools=[t], model=m)\n"
    )
    obs = build_observed(tmp_path)
    assert obs.execution_model == "prompt-only"
    r = run(mode="audit", observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-901")
    assert f.bucket is Bucket.UNKNOWNS
    assert not any(p.subject == "t" for p in r.passed)


def test_schema_based_classification(agent_dir: Path) -> None:
    assert extract(agent_dir).execution_model == "schema-based"


def test_synthetic_calls_never_count_as_violations(agent_dir: Path) -> None:
    obs = build_observed(
        agent_dir,
        captures=[chat_call(0, [GI_WIRE_FAITHFUL], synthetic=True, tool_calls=[("get_inventory", {"sku": "x"})])],
    )
    assert obs.observed_call_counts() == {}
    assert obs.real_run_ids() == []


# ------------------------------------------------------------------ integration on real frameworks
def _discover_and_observe(d: Path):
    from agenticreq.capture import discover

    r = discover(str(d / "agent.py"), mock_tools=str(d / "mocks.yml"), timeout=180)
    return r, build_observed(d, captures=r.calls, runtime_versions=r.versions)


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_langgraph_never_reached_and_drift(fix: Path) -> None:
    _, obs = _discover_and_observe(fix / "agents" / "langgraph_po")
    r = run(mode="audit", observed=obs)
    never = [f for f in r.findings if f.rule_id == "AS-201"]
    assert [f.title.split()[0] for f in never] == ["issue_refund"]
    assert "unreachable" in never[0].detail
    assert {f.title.split(":")[0] for f in r.findings if f.rule_id == "AS-602"} == {"get_inventory"}


@pytest.mark.skipif(importlib.util.find_spec("agents") is None, reason="openai-agents not installed")
def test_openai_agents_optional_made_required(fix: Path) -> None:
    _, obs = _discover_and_observe(fix / "agents" / "oai_agents_po")
    r = run(mode="audit", observed=obs)
    # strict mode keeps the default but marks the param required: a contradictory contract (AS-604)
    assert sorted(f.title for f in r.findings if f.rule_id == "AS-604") == [
        "get_inventory: contradictory field contract (warehouse)",
        "lookup_clause: contradictory field contract (max_results)",
    ]
    assert not any(f.rule_id == "AS-603" for f in r.findings)  # superseded, not double-reported
    assert not any(f.rule_id == "AS-202" for f in r.findings)  # transfer_to_* allowlisted


@pytest.mark.skipif(importlib.util.find_spec("smolagents") is None, reason="smolagents not installed")
def test_smolagents_defaults_dropped_and_codeagent_unknown(fix: Path) -> None:
    _, obs = _discover_and_observe(fix / "agents" / "smol_tools")
    r = run(mode="audit", observed=obs)
    assert [f.title for f in r.findings if f.rule_id == "AS-605"] == [
        "get_inventory: default dropped on the wire (warehouse)"
    ]
    _, obs = _discover_and_observe(fix / "agents" / "smol_code")
    r = run(mode="audit", observed=obs)
    assert obs.execution_model == "prompt-only"
    assert "AS-901" in _ids(r, Bucket.UNKNOWNS)
    assert not [p for p in r.passed if p.subject == "get_inventory"]
