"""Agents and tools created at runtime: a generated id is never a name to register. A call the framework makes
itself is listed in _derived.injected_by_framework (0 gaps); the project's own unnamed agent is a sign-off; a named
agent the code lacks is still a gap; a tool built at runtime keeps the tool-list rules from passing."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import chat_call

from agenticreq.engine import run
from agenticreq.graph.origin import looks_generated
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results, unresolved_line
from agenticreq.observe import build_observed
from agenticreq.scaffold import derived

SPEC = """
agent_id: a
risk_level: medium
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
{extra}
"""
CODE = """
    from agents import Agent, function_tool

    @function_tool
    def lookup_order(order_id: str) -> str:
        '''Read an order.'''
        return order_id[:50]

    billing = Agent(name="billing", instructions="You handle billing questions for the shop, politely.",
                    tools=[lookup_order])
    triage = Agent(name="triage", instructions="You route each customer to the right specialist agent.",
                   handoffs=[billing])
"""
REG = "agent_registry: [{id: triage}, {id: billing}]\ndeclared_delegation: [{from: triage, to: billing}]"


def W(name: str) -> dict[str, Any]:
    return {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}


def _run(tmp: Path, calls: list[Any], extra: str = REG, files: dict[str, str] | None = None) -> Any:
    (tmp / "agent.py").write_text(textwrap.dedent(CODE), encoding="utf-8")
    for name, text in (files or {}).items():
        (tmp / name).write_text(text, encoding="utf-8")
    obs = build_observed(tmp, captures=calls)
    report = run(mode="check", loaded=load_spec_text(SPEC.format(extra=extra)), observed=obs, today=date(2026, 10, 30))
    return obs, report


AGENT_PARAMS = {
    "agent_registry",
    "agent_registry[].id",
    "declared_delegation",
    "runtime_agents",
    "authority_inheritance",
}


def _agent_gaps(lines: list[Any]) -> list[tuple[str, str]]:
    """Gaps about agents or delegation: none may come from a framework call or an unnamed agent."""
    return [(ln.param, ln.reason) for ln in lines if ln.status == "GAP" and ln.param in AGENT_PARAMS]


def _own_calls() -> list[Any]:
    return [
        chat_call(0, [W("transfer_to_billing")], system="You route each customer to the right specialist agent."),
        chat_call(1, [W("lookup_order")], system="You handle billing questions for the shop, politely."),
    ]


@pytest.mark.parametrize(
    "name,generated",
    [
        ("a5119f", True),
        ("3f2a9c7d", True),
        ("Agent#2", True),
        ("agent_3", True),
        ("0b8e2a4c-1d2e-4f5a-9b8c-7d6e5f4a3b2c", True),
        ("billing", False),
        ("Fact Checker", False),
        ("", True),
    ],
)
def test_looks_generated(name: str, generated: bool) -> None:
    assert looks_generated(name) is generated


def test_a_framework_created_agent_is_listed_and_never_a_gap(tmp_path: Path) -> None:
    framework = chat_call(
        2,
        [],
        system="Format your final answer according to the following OpenAPI schema: "
        '{"type": "json_schema", "json_schema": {"name": "Verdict"}}',
    )
    obs, report = _run(tmp_path, [*_own_calls(), framework])
    [node] = [n for n in obs.nodes if n.origin != "code"]
    assert node.origin == "framework"
    lines = results(report)
    assert not _agent_gaps(lines), _agent_gaps(lines)
    assert not any(node.id in ln.reason for ln in lines)
    assert any(node.id in x for x in derived(obs)["injected_by_framework"])  # type: ignore[union-attr]
    assert unresolved_line(report) == ""


def test_an_unnamed_project_agent_is_a_sign_off_not_a_gap(tmp_path: Path) -> None:
    prompt = "You are a meticulous senior reviewer who checks every refund decision twice."
    unnamed = chat_call(2, [], system=prompt + " Today is Monday.")
    obs, report = _run(tmp_path, [*_own_calls(), unnamed], files={"agents.yaml": f"reviewer:\n  backstory: {prompt}\n"})
    assert [n.origin for n in obs.nodes if n.origin != "code"] == ["unresolved"]
    lines = results(report)
    [ln] = [ln for ln in lines if ln.param == "runtime_agents"]
    assert ln.status == "SIGN-OFF" and "an agent created at runtime could not be identified" in ln.reason.lower()
    assert "confirmed:" in ln.reason
    assert not _agent_gaps(lines), _agent_gaps(lines)
    assert unresolved_line(report) == "1 agent could not be identified; see _derived.unresolved"


def test_the_unnamed_agent_confirmed_under_confirmed_is_signed(tmp_path: Path) -> None:
    prompt = "You are a meticulous senior reviewer who checks every refund decision twice."
    extra = REG + (
        "\nconfirmed:\n  - {clause: runtime_agents, by: lead@x.com, date: 2026-10-01, expires: 2027-10-01,"
        ' evidence: "the reviewer built from agents.yaml"}'
    )
    _, report = _run(
        tmp_path,
        [*_own_calls(), chat_call(2, [], system=prompt)],
        extra,
        files={"agents.yaml": f"reviewer:\n  backstory: {prompt}\n"},
    )
    [ln] = [ln for ln in results(report) if ln.param == "runtime_agents"]
    assert ln.status == "SIGNED"


def test_a_missing_named_agent_is_still_a_gap(tmp_path: Path) -> None:
    _, report = _run(tmp_path, _own_calls(), REG.replace("{id: billing}]", "{id: billing}, {id: refunds}]"))
    gaps = [ln for ln in results(report) if ln.status == "GAP"]
    assert any("spec lists refunds; the code has no such agent" in ln.reason.lower() for ln in gaps), gaps


RUNTIME_TOOL = """
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    def read_order(order_id: str) -> str:
        '''Read an order.'''
        return order_id[:50]

    def make(cfg):
        return StructuredTool.from_function(func=cfg["fn"], name=cfg["name"], description=cfg["desc"])

    TOOLS = [StructuredTool.from_function(func=read_order, name="read_order", description="Read an order."),
             make({"fn": read_order, "name": "dynamic", "desc": "d"})]
    agent = create_react_agent("openai:gpt-4o-2024-08-06", TOOLS)
"""


def test_a_runtime_built_tool_keeps_the_tool_list_rules_from_passing(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(textwrap.dedent(RUNTIME_TOOL), encoding="utf-8")
    obs = build_observed(tmp_path)
    if not obs.unresolved_tools:
        pytest.skip("this tool construction resolves statically")
    report = run(
        mode="check",
        loaded=load_spec_text(SPEC.format(extra="role_boundary: {allowed_tools: [read_order]}")),
        observed=obs,
        today=date(2026, 10, 30),
    )
    lines = results(report)
    [rb] = [ln for ln in lines if ln.param == "role_boundary"]
    assert rb.status != "PASS" and "built at runtime could not be identified" in rb.reason, rb
    assert unresolved_line(report).startswith(f"{len(obs.unresolved_tools)} tool")
