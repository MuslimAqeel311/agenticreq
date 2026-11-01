"""Names in spec.yml are the user's requirement: init never writes agent or tool names into the spec (only into
_derived), and check compares both directions, each on its own line."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from agenticreq.scaffold import FROM_CODE_WARNING, NAMES_HINT, code_agents, render_init

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
PROJECTS = sorted(
    [
        *(FIX / "agents").iterdir(),
        *(FIX / "agents_crewai").iterdir(),
        FIX / "e2e" / "langgraph_team",
        FIX / "agents_misc" / "finance_like",
    ]
)

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


@pytest.mark.parametrize("project", PROJECTS, ids=[p.name for p in PROJECTS])
def test_init_writes_todo_for_names_and_lists_them_in_derived(project: Path) -> None:
    obs = build_observed(project)
    text = render_init(obs)
    assert re.search(r"^role_boundary:\s+TODO", text, re.M) and re.search(r"^agent_registry:\s+TODO", text, re.M)
    assert text.count(NAMES_HINT) == 2
    names = [*obs.declared_tools, *code_agents(obs)]
    for line in text.splitlines():  # no line the user can uncomment to copy the code's names
        if re.match(r"#\s*(role_boundary|agent_registry|declared_delegation)\s*:", line):
            assert not any(n in line for n in names), line
    derived = text.split("\nagent_id:")[0]
    assert "what the code has today; not your requirement" in derived
    for n in names:
        assert f'"{n}"' in derived, n  # exactly as the code spells it
    assert load_spec_text(text).status != "invalid"


def _check(tmp: Path, code: str, extra: str) -> list[tuple[str, str, str]]:
    (tmp / "agent.py").write_text(textwrap.dedent(code), encoding="utf-8")
    report = run(mode="check", loaded=load_spec_text(SPEC.format(extra=extra)), observed=build_observed(tmp))
    return [(ln.param, ln.status, ln.reason) for ln in results(report)]


TOOLS = """
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_order(order_id: str) -> str:
        '''Read an order.'''
        return order_id[:100]

    @tool
    def get_customer(customer_id: str) -> str:
        '''Read a customer.'''
        return customer_id[:100]

    agent = create_react_agent("openai:gpt-4o-2024-08-06", [get_order, get_customer])
"""


def test_both_directions_for_tools(tmp_path: Path) -> None:
    lines = _check(tmp_path, TOOLS, "role_boundary: {allowed_tools: [get_order, refund_order]}")
    assert (
        "role_boundary.allowed_tools",
        "GAP",
        "Spec lists refund_order; the code has no such tool.",
    ) in lines or any(
        p == "role_boundary.allowed_tools" and "spec lists refund_order; the code has no such tool" in r.lower()
        for p, _, r in lines
    ), lines
    assert any(
        p == "role_boundary" and s == "GAP" and "the code has get_customer; it is not in your spec" in r.lower()
        for p, s, r in lines
    ), lines


AGENTS = """
    from agents import Agent
    billing = Agent(name="billing", instructions="b")
    triage = Agent(name="triage", instructions="t", handoffs=[billing])
"""


def test_both_directions_for_agents(tmp_path: Path) -> None:
    extra = "agent_registry: [{id: triage}, {id: ghost}]\ndeclared_delegation: [{from: triage, to: billing}]"
    lines = _check(tmp_path, AGENTS, extra)
    reasons = " | ".join(r.lower() for _, _, r in lines)
    assert "spec lists ghost; the code has no such agent" in reasons, lines
    assert "the code has billing; it is not in your spec" in reasons, lines
    gaps = [(p, r) for p, s, r in lines if s == "GAP" and ("ghost" in r or "has billing" in r)]
    assert len(gaps) == 2  # each direction its own line


def test_a_matching_spec_reports_no_name_mismatch(tmp_path: Path) -> None:
    lines = _check(
        tmp_path,
        AGENTS,
        "agent_registry: [{id: triage}, {id: billing}]\ndeclared_delegation: [{from: triage, to: billing}]",
    )
    assert not any("not in your spec" in r or "no such agent" in r for _, _, r in lines), lines


def test_init_from_code_fills_names_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "agent.py").write_text(textwrap.dedent(AGENTS), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["init", ".", "--from-code", "--quiet"], terminal_width=200)
    assert FROM_CODE_WARNING in " ".join(r.output.split())
    text = (tmp_path / "spec.yml").read_text(encoding="utf-8")
    assert re.search(r'^agent_registry: \[\{id: "billing"', text, re.M) and "role_boundary:" in text
    assert load_spec_text(text).status != "invalid"


CREW = Path(r"C:\Users\muslim.aqeel\Downloads\agentic_intel_crew\agentic_intel_crew")


@pytest.mark.skipif(not CREW.is_dir(), reason="the real agentic_intel_crew project is not on this machine")
def test_real_crew_derived_agents_equal_what_check_finds() -> None:
    obs = build_observed(CREW)
    text = render_init(obs)
    m = re.search(r"^  declared_agents: (\[.*\])$", text, re.M)
    assert m
    import json

    derived = json.loads(m.group(1))
    report = run(mode="check", loaded=load_spec_text(SPEC.format(extra="")), observed=obs)
    found = [a["name"] for a in report.metadata["agents"]]  # type: ignore[union-attr]
    assert len(derived) == len(set(found)) and set(derived) == set(found), (derived, found)
