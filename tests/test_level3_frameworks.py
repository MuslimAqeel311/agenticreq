"""Level 3 (multi-agent) on every framework's real delegation idiom, not only CrewAI.

Each case wires a coordinator ("triage") to a specialist ("billing") that holds a prohibited write tool,
the way that framework actually does it. The spec prohibits the tool for triage and declares no
delegation, so: AS-301 (undeclared edge) and AS-304 (prohibited tool reachable through delegation)
must fire; declaring the edge must clear AS-301.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.observe import build_observed

CASES = {
    "openai_handoffs": """
        from agents import Agent, function_tool
        @function_tool
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        billing = Agent(name="billing", instructions="b", tools=[issue_refund])
        triage = Agent(name="triage", instructions="t", handoffs=[billing])
    """,
    "openai_agent_as_tool": """
        from agents import Agent, function_tool
        @function_tool
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        billing = Agent(name="billing", instructions="b", tools=[issue_refund])
        triage = Agent(name="triage", instructions="t",
                       tools=[billing.as_tool(tool_name="billing_agent", tool_description="Billing.")])
    """,
    "adk_sub_agents": """
        from google.adk.agents import LlmAgent
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        billing = LlmAgent(name="billing", model="gemini-2.0-flash-001", tools=[issue_refund])
        triage = LlmAgent(name="triage", model="gemini-2.0-flash-001", sub_agents=[billing])
    """,
    "adk_agent_tool": """
        from google.adk.agents import LlmAgent
        from google.adk.tools.agent_tool import AgentTool
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        billing = LlmAgent(name="billing", model="gemini-2.0-flash-001", tools=[issue_refund])
        triage = LlmAgent(name="triage", model="gemini-2.0-flash-001", tools=[AgentTool(agent=billing)])
    """,
    "autogen_round_robin": """
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        triage = AssistantAgent("triage", model_client=client)
        billing = AssistantAgent("billing", model_client=client, tools=[issue_refund])
        team = RoundRobinGroupChat([triage, billing], max_turns=6)
    """,
    "autogen_swarm_handoffs": """
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import Swarm
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        triage = AssistantAgent("triage", model_client=client, handoffs=["billing"])
        billing = AssistantAgent("billing", model_client=client, tools=[issue_refund])
        team = Swarm([triage, billing])
    """,
    "llamaindex_workflow": """
        from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
        from llama_index.core.tools import FunctionTool
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        refund_tool = FunctionTool.from_defaults(fn=issue_refund)
        triage = FunctionAgent(name="triage", description="t", tools=[], llm=llm, can_handoff_to=["billing"])
        billing = FunctionAgent(name="billing", description="b", tools=[refund_tool], llm=llm)
        workflow = AgentWorkflow(agents=[triage, billing], root_agent="triage")
    """,
    "crewai_hierarchical": """
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import tool
        @tool("issue_refund")
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        triage = Agent(role="triage", goal="g", backstory="b", allow_delegation=False)
        billing = Agent(role="billing", goal="g", backstory="b", tools=[issue_refund])
        t = Task(description="d", expected_output="e")
        crew = Crew(agents=[billing], tasks=[t], process=Process.hierarchical, manager_agent=triage)
    """,
}

SPEC = """
agent_id: triage
risk_level: high
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: [{tool: issue_refund, reason: finance only}]
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
topology: supervisor
agent_registry: [{id: triage}, {id: billing}]
declared_delegation: DECL
"""


def _check(tmp: Path, src: str, decl: str) -> list:  # type: ignore[type-arg]
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    loaded = load_spec_text(SPEC.replace("DECL", decl))
    return run(mode="check", loaded=loaded, observed=build_observed(tmp)).findings


@pytest.mark.parametrize("fw", sorted(CASES))
def test_delegation_and_transitive_prohibition(tmp_path: Path, fw: str) -> None:
    fs = _check(tmp_path, CASES[fw], "[]")
    ids = {f.rule_id for f in fs}
    assert "AS-304" in ids, f"{fw}: prohibited issue_refund reachable via billing not reported: {sorted(ids)}"
    hit = next(f for f in fs if f.rule_id == "AS-304")
    assert hit.path and hit.path[0] in ("triage", "team", "workflow") and "billing" in hit.path
    if fw not in ("autogen_round_robin",):  # a round-robin team declares no direct triage -> billing edge
        assert "AS-301" in ids, f"{fw}: undeclared delegation not reported: {sorted(ids)}"


@pytest.mark.parametrize("fw", sorted(set(CASES) - {"autogen_round_robin"}))
def test_declared_edge_clears_as301(tmp_path: Path, fw: str) -> None:
    fs = _check(tmp_path, CASES[fw], "[{from: triage, to: billing}]")
    undeclared = [f.title for f in fs if f.rule_id == "AS-301" and "triage → billing" in f.title]
    assert not undeclared, f"{fw}: {undeclared}"


def test_swarm_cycle(tmp_path: Path) -> None:
    src = CASES["autogen_swarm_handoffs"].replace(
        'billing = AssistantAgent("billing", model_client=client, tools=[issue_refund])',
        'billing = AssistantAgent("billing", model_client=client, tools=[issue_refund], handoffs=["triage"])',
    )
    assert "AS-303" in {f.rule_id for f in _check(tmp_path, src, "[]")}


def test_team_coordinator_is_not_an_unregistered_agent_or_authority_escalation(tmp_path: Path) -> None:
    fs = _check(tmp_path, CASES["autogen_round_robin"], "[]")
    assert not [f for f in fs if f.rule_id == "AS-308" and "team" in f.title]
    assert not [f for f in fs if f.rule_id == "AS-305" and f.path and f.path[0] == "team"]


def test_autogen_initiate_chat_edge(tmp_path: Path) -> None:
    src = """
        from autogen import AssistantAgent, UserProxyAgent
        assistant = AssistantAgent("billing", llm_config=cfg)
        triage = UserProxyAgent("triage", code_execution_config=False)
        triage.initiate_chat(assistant, message="refund order 7")
    """
    titles = [f.title for f in _check(tmp_path, src, "[]") if f.rule_id == "AS-301"]
    assert any("triage → billing" in t for t in titles)


# ------------------------------------------------------------------ observed (wire) handoffs per framework
WIRE = {
    # framework: (code, handoff tool name, its arguments)
    "adk": (
        """
        from google.adk.agents import LlmAgent
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        billing = LlmAgent(name="billing", model="gemini-2.0-flash-001", instruction="You are billing.",
                           tools=[issue_refund])
        triage = LlmAgent(name="triage", model="gemini-2.0-flash-001", instruction="You are triage.",
                          sub_agents=[billing])
    """,
        "transfer_to_agent",
        {"agent_name": "billing"},
    ),
    "llamaindex": (
        """
        from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
        triage = FunctionAgent(name="triage", description="t", system_prompt="You are triage.", llm=llm,
                               can_handoff_to=["billing"])
        billing = FunctionAgent(name="billing", description="b", system_prompt="You are billing.", llm=llm)
        wf = AgentWorkflow(agents=[triage, billing], root_agent="triage")
    """,
        "handoff",
        {"to_agent": "billing", "reason": "refund"},
    ),
    "autogen_swarm": (
        """
        from autogen_agentchat.agents import AssistantAgent
        triage = AssistantAgent("triage", model_client=c, system_message="You are triage.", handoffs=["billing"])
        billing = AssistantAgent("billing", model_client=c, system_message="You are billing.")
    """,
        "transfer_to_billing",
        {},
    ),
}


@pytest.mark.parametrize("fw", sorted(WIRE))
def test_observed_handoff_edges(tmp_path: Path, fw: str) -> None:
    from agenticreq.models.observed import CapturedCall, ToolCall, ToolSchema

    src, tool, args = WIRE[fw]
    (tmp_path / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    handoff = ToolSchema(name=tool, source="wire")
    calls = [
        CapturedCall(
            index=0,
            provider="openai_chat",
            url="u",
            system_prompt="You are triage.",
            system_prompt_sha256="a" * 64,
            tools=[handoff],
            tool_calls=[ToolCall(name=tool, arguments=args)],
        ),
        CapturedCall(
            index=1, provider="openai_chat", url="u", system_prompt="You are billing.", system_prompt_sha256="b" * 64
        ),
    ]
    obs = build_observed(tmp_path, captures=calls)
    assert any(e.src == "triage" and e.dst == "billing" for e in obs.edges), [(e.src, e.dst) for e in obs.edges]
    r = run(mode="audit", observed=obs)
    assert not [f for f in r.findings if f.rule_id == "AS-202" and tool in f.title], "framework handoff tool flagged"
