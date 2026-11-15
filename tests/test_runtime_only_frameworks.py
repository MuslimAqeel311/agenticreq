"""A framework that is only installed or loaded next to the agent (seen in the mock run, not imported by the scanned
code) is never the project's own framework: another framework's project must not be credited with the OpenAI Agents
SDK's default tracing, and a real OpenAI Agents project still is.

Found by running the same LangGraph project with a mock run in two agent environments: the one with every framework
installed loaded the OpenAI Agents SDK, and its `audit_log` gap turned into a pass."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agenticreq.engine import run
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

LOADED = {"module:agents": "0.24.0", "openai-agents": "0.24.0"}

LANGGRAPH = '''
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def issue_refund(order_id: str) -> str:
    """Refund a customer order."""
    open("refunds.log", "w").write(order_id)
    return "refunded"


agent = create_react_agent("openai:gpt-4o-2024-08-06", tools=[issue_refund])
'''

OPENAI_AGENTS = '''
from agents import Agent, function_tool


@function_tool
def issue_refund(order_id: str) -> str:
    """Refund a customer order."""
    open("refunds.log", "w").write(order_id)
    return "refunded"


agent = Agent(name="a", model="gpt-4o-2024-08-06", tools=[issue_refund])
'''


def _audit_log(project: Path, versions: dict[str, str] | None) -> tuple[str, str]:
    obs = build_observed(project, runtime_versions=versions)
    report = run(mode="audit", observed=obs, today=date(2026, 11, 15))
    lines = [ln for ln in results(report) if ln.param == "audit_log"]
    return lines[0].status, lines[0].reason


def test_a_framework_only_loaded_in_the_run_is_recorded_as_runtime_only(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(LANGGRAPH, encoding="utf-8")
    obs = build_observed(tmp_path, runtime_versions=LOADED)
    assert "openai-agents" in obs.frameworks and obs.runtime_only_frameworks == ["openai-agents"]
    assert obs.framework == "langgraph"


def test_a_langgraph_project_is_not_credited_with_openai_agents_tracing(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(LANGGRAPH, encoding="utf-8")
    status, reason = _audit_log(tmp_path, None)
    assert status == "GAP" and "No audit / trace hook" in reason
    # the same project, in an environment where the OpenAI Agents SDK got loaded: the same result (R1)
    assert _audit_log(tmp_path, LOADED) == (status, reason)


def test_an_openai_agents_project_still_gets_its_default_tracing(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(OPENAI_AGENTS, encoding="utf-8")
    for versions in (None, LOADED):
        obs = build_observed(tmp_path, runtime_versions=versions)
        assert "openai-agents" in obs.frameworks and obs.runtime_only_frameworks == []
        status, reason = _audit_log(tmp_path, versions)
        assert status != "GAP", reason
        assert "OpenAI Agents SDK traces every run by default" in reason
