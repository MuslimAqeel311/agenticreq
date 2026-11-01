"""Run 4 (17 real agents in 8 frameworks): each proven gap, as the minimal snippet that proved it."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.finding import Finding
from agenticreq.models.observed import CapturedCall, ToolSchema
from agenticreq.observe import build_observed

SPEC = """
agent_id: {aid}
risk_level: medium
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: []
approval_required: {approval}
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
{extra}
"""


def check(
    tmp: Path,
    src: str,
    aid: str = "a",
    approval: str = "[]",
    extra: str = "",
    captures: list[CapturedCall] | None = None,
) -> list[Finding]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    spec = load_spec_text(SPEC.format(aid=aid, approval=approval, extra=extra))
    return run(mode="check", loaded=spec, observed=build_observed(tmp, captures=captures)).findings


def audit(tmp: Path, src: str) -> list[Finding]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    return run(mode="audit", observed=build_observed(tmp)).findings


def ids(fs: list[Finding]) -> set[str]:
    return {f.rule_id for f in fs}


ADK = """
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
def save_note(topic: str, idempotency_key: str, dry_run: bool = True) -> str:
    '''Save a note.'''
    return topic
root_agent = LlmAgent(name="analyst", model="gemini-2.0-flash-001", instruction="i",
                      tools=[FunctionTool(save_note{gate})])
"""
APPROVE = "[{tool: save_note, approver: lead, timeout: 1h, on_timeout: safe_stop}]"


# ---- G1: Google ADK is a framework; FunctionTool(require_confirmation=True) is an approval gate
@pytest.mark.parametrize("gate,gated", [(", require_confirmation=True", True), ("", False)])
def test_adk_framework_and_confirmation_gate(tmp_path: Path, gate: str, gated: bool) -> None:
    fs = check(tmp_path, ADK.format(gate=gate), approval=APPROVE)
    assert "AS-904" not in ids(fs)
    assert ("AS-205" in ids(fs)) is not gated
    assert build_observed(tmp_path).framework == "google-adk"


def test_adk_default_step_bound_and_nested_run_timeout(tmp_path: Path) -> None:
    src = ADK.format(gate=", require_confirmation=True") + textwrap.dedent("""
        import asyncio
        from google.adk.runners import Runner
        async def main():
            runner = Runner(agent=root_agent, app_name="x", session_service=s)
            async def go():
                return [e async for e in runner.run_async(user_id="u", session_id="s", new_message=m)]
            return await asyncio.wait_for(go(), 90)
    """)
    fs = audit(tmp_path, src)
    assert "AS-208" not in ids(fs)  # RunConfig.max_llm_calls defaults to 500 (framework default)
    assert "AS-222" not in ids(fs)  # wait_for(go(), 90) and go() runs the agent


# ---- G2: a budget inside a config dict that is passed by variable
@pytest.mark.parametrize("call", ["graph.invoke({}, config)", "graph.invoke({}, config=config)"])
def test_budget_in_config_variable(tmp_path: Path, call: str) -> None:
    src = f"""
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    @tool
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    graph = create_react_agent("gpt-4o-2024-08-06", tools=[lookup])
    config = {{"recursion_limit": 5, "callbacks": [TokenBudget(max_total_tokens=5000)]}}
    {call}
    """
    assert "AS-221" not in ids(audit(tmp_path, src))


# ---- G3 (design, now consistent): a timeout around a worker thread is not a run-wide time limit
@pytest.mark.parametrize(
    "line",
    [
        "pool.submit(agent.run, q).result(timeout=90)",
        "asyncio.run(asyncio.wait_for(asyncio.to_thread(agent.run, q), 90))",
        "asyncio.run(asyncio.wait_for(asyncio.to_thread(lambda: agent.run(q)), 90))",
    ],
)
def test_thread_timeouts_do_not_bound_the_run(tmp_path: Path, line: str) -> None:
    src = (
        "import asyncio, concurrent.futures\nfrom smolagents import ToolCallingAgent, tool\n@tool\n"
        "def lookup(q: str) -> str:\n    '''Look.\n\n    Args:\n        q: query.\n    '''\n    return q\n"
        "agent = ToolCallingAgent(tools=[lookup], model=m, max_steps=5)\npool = concurrent.futures.ThreadPoolExecutor()\n"
        + line
        + "\n"
    )
    assert "AS-222" in ids(audit(tmp_path, src))


# ---- G4: chained comparison is a body bound for min and max
def test_chained_comparison_body_bound(tmp_path: Path) -> None:
    src = (
        "from langchain_core.tools import tool\n@tool\ndef q(limit: int) -> str:\n    '''Q.'''\n"
        "    if not 1 <= limit <= 10:\n        raise ValueError('limit')\n    return str(limit)\n"
    )
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    assert build_observed(tmp_path).declared_tools["q"].body_bounds["limit"] == [1.0, 10.0]


# ---- G5: an agent exposed as a tool is not a write tool, whatever its name
def test_agent_as_tool_named_like_a_write(tmp_path: Path) -> None:
    src = """
    from agents import Agent
    w = Agent(name="writer", instructions="w")
    triage = Agent(name="triage", instructions="t", tools=[w.as_tool(tool_name="write_note", tool_description="d")])
    """
    assert not ids(audit(tmp_path, src)) & {"AS-212", "AS-227", "AS-211"}


# ---- G6: AutoGen structured output and token budget
def test_autogen_output_type_and_token_termination(tmp_path: Path) -> None:
    src = """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TokenUsageTermination
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    a = AssistantAgent("a", model_client=c, tools=[lookup], output_content_type=Report)
    team = RoundRobinGroupChat([a], termination_condition=TokenUsageTermination(max_total_token=20000))
    """
    assert not ids(audit(tmp_path, src)) & {"AS-221", "AS-224"}


# ---- G7: delegation edges for create_supervisor, managed_agents, a tool that runs another agent
G7 = {
    "langgraph_supervisor": """
        from langgraph.prebuilt import create_react_agent
        from langgraph_supervisor import create_supervisor
        r = create_react_agent(model, tools=[], name="researcher")
        w = create_react_agent(model, tools=[], name="writer")
        workflow = create_supervisor([r, w], model=model)
    """,
    "smolagents_managed": """
        from smolagents import ToolCallingAgent
        r = ToolCallingAgent(tools=[], model=m, name="researcher", description="d")
        manager = ToolCallingAgent(tools=[], model=m, name="manager", managed_agents=[r])
    """,
    "pydantic_ai_delegation": """
        from pydantic_ai import Agent
        researcher = Agent("openai:gpt-4o-2024-08-06", name="researcher")
        supervisor = Agent("openai:gpt-4o-2024-08-06", name="supervisor")
        @supervisor.tool_plain
        async def ask_researcher(q: str) -> str:
            '''Ask the researcher.'''
            return (await researcher.run(q)).output
    """,
}


@pytest.mark.parametrize("fw", sorted(G7))
def test_delegation_edges_for_every_idiom(tmp_path: Path, fw: str) -> None:
    fs = check(tmp_path, G7[fw], extra="topology: supervisor\ndeclared_delegation: []")
    assert "AS-301" in ids(fs), fw


# ---- G8: discover reads the child's output as UTF-8 on every OS
def test_discover_survives_non_cp1252_output(tmp_path: Path) -> None:
    pytest.importorskip("httpx")
    from agenticreq.capture import discover

    (tmp_path / "agent.py").write_text("print('\\u2550' * 40)\n", encoding="utf-8")
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    r = discover(str(tmp_path / "agent.py"), mock_tools=str(tmp_path / "mocks.yml"))
    assert r.runs == 1 and not r.run_errors


# ---- G9: a `limit` parameter the body never applies bounds nothing
def test_unused_limit_param_is_not_a_bound(tmp_path: Path) -> None:
    src = (
        "from langchain_core.tools import tool\n@tool\ndef top_stories(limit: int = 5) -> str:\n    '''Top.'''\n"
        "    return str(db.execute('SELECT * FROM stories').fetchall())\n"
    )
    assert "AS-213" in ids(audit(tmp_path, src))


# ---- G10: the team path is printed through the team, not as a direct edge
def test_group_chat_path_goes_through_the_team(tmp_path: Path) -> None:
    src = """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    def read_db(q: str) -> str:
        '''Read.'''
        return q
    def save_note(t: str) -> str:
        '''Save.'''
        return t
    researcher = AssistantAgent("researcher", model_client=c, tools=[read_db])
    writer = AssistantAgent("writer", model_client=c, tools=[save_note])
    team = RoundRobinGroupChat([researcher, writer])
    """
    extra = (
        "topology: peer\nagent_registry: [{id: researcher, tools: [read_db], prohibited: [save_note]}, "
        "{id: writer, tools: [save_note], prohibited: [read_db]}]"
    )
    paths = [f.path for f in check(tmp_path, src, aid="researcher", extra=extra) if f.rule_id == "AS-304"]
    assert paths and all(p[1] == "team" for p in paths), paths


# ---- G11: output_type that allows str is free text
@pytest.mark.parametrize("ot,structured", [("[str, DeferredToolRequests]", False), ("str", False), ("Report", True)])
def test_output_type_with_str_is_free_text(tmp_path: Path, ot: str, structured: bool) -> None:
    src = f"from pydantic_ai import Agent\nagent = Agent('openai:gpt-4o-2024-08-06', output_type={ot})\n"
    assert ("AS-224" in ids(audit(tmp_path, src))) is not structured


# ---- G12: a delegate called as a tool is steered by its delegator's text (cross_agent_trust: none)
def test_zero_trust_through_as_tool(tmp_path: Path) -> None:
    src = """
    from agents import Agent, function_tool
    @function_tool
    def save_note(text: str) -> str:
        '''Save.'''
        with open("notes.txt", "a") as fh:  # writes: the body, not the name, makes it a write
            fh.write(text)
        return text
    writer = Agent(name="writer", instructions="w", tools=[save_note])
    triage = Agent(name="triage", instructions="t", tools=[writer.as_tool(tool_name="ask_writer", tool_description="d")])
    """
    extra = (
        "topology: supervisor\ncross_agent_trust: none\nagent_registry: [{id: triage}, {id: writer}]\n"
        "declared_delegation: [{from: triage, to: writer}]"
    )
    assert "AS-401" in ids(check(tmp_path, src, aid="triage", extra=extra))


# ---- G14: a token cap is not a loop counter
def test_token_cap_is_not_a_step_bound(tmp_path: Path) -> None:
    src = """
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import FunctionTool
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    agent = FunctionAgent(tools=[FunctionTool.from_defaults(fn=lookup)], llm=llm)
    class Budget:
        def check(self):
            if self.total_llm_token_count > self.max_total_tokens:
                raise RuntimeError("over budget")
    """
    assert "AS-208" in ids(audit(tmp_path, src))


# ---- G15: init does not suggest team -> member edges the registry cannot hold
def test_init_group_chat_suggestions_are_consistent(tmp_path: Path) -> None:
    from agenticreq.scaffold import render_init

    (tmp_path / "agent.py").write_text(
        textwrap.dedent("""
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        researcher = AssistantAgent("researcher", model_client=c)
        writer = AssistantAgent("writer", model_client=c)
        team = RoundRobinGroupChat([researcher, writer])
    """),
        encoding="utf-8",
    )
    text = render_init(build_observed(tmp_path))
    assert "{from: team" not in text
    assert "read from the code: peer" in text


# ---- G16: with captures, structured-output tools and agents-as-tools are not undeclared tools,
# and a manager whose prompt embeds its sub-agents' descriptions is still attributed to the manager
def test_capture_mode_framework_tools_and_attribution(tmp_path: Path) -> None:
    src = """
    from smolagents import ToolCallingAgent, tool
    @tool
    def query_rates(q: str) -> str:
        '''Rates.

        Args:
            q: query.
        '''
        return q
    researcher = ToolCallingAgent(tools=[query_rates], model=m, name="researcher", description="Finds rates.")
    manager = ToolCallingAgent(tools=[], model=m, name="manager", managed_agents=[researcher])
    """
    calls = [
        CapturedCall(
            index=0,
            provider="openai_chat",
            url="u",
            system_prompt="You manage. researcher: Finds rates.",
            system_prompt_sha256="a" * 64,
            tools=[ToolSchema(name="researcher", source="wire"), ToolSchema(name="final_answer", source="wire")],
        ),
        CapturedCall(
            index=1,
            provider="openai_chat",
            url="u",
            system_prompt="You are researcher. Finds rates.",
            system_prompt_sha256="b" * 64,
            tools=[ToolSchema(name="query_rates", source="wire"), ToolSchema(name="final_answer", source="wire")],
        ),
    ]
    extra = (
        "topology: supervisor\nagent_registry: [{id: manager}, {id: researcher, tools: [query_rates]}]\n"
        "declared_delegation: [{from: manager, to: researcher}]"
    )
    fs = check(tmp_path, src, aid="manager", extra=extra, captures=calls)
    assert not ids(fs) & {"AS-202", "AS-308"}
    assert not [f for f in fs if f.rule_id == "AS-206" and "researcher holds" in f.title]
    assert {n.id for n in build_observed(tmp_path, captures=calls).nodes} == {"manager", "researcher"}


# ---- LangChain @tool(args_schema=Model): the model's Field bounds are the tool schema (no false AS-207)
def test_langchain_decorator_args_schema_limits(tmp_path: Path) -> None:
    src = """
    from pydantic import BaseModel, Field
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    class RefundArgs(BaseModel):
        amount: float = Field(ge=0, le=100)
    @tool(args_schema=RefundArgs)
    def issue_refund(amount: float) -> str:
        '''Refund.'''
        return "ok"
    agent = create_react_agent("m", [issue_refund])
    """
    (tmp_path / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    spec_text = SPEC.format(aid="a", approval="[]", extra="").replace(
        "limits: []", "limits: [{tool: issue_refund, arg: amount, limit: {min: 0, max: 100}}]"
    )
    fs = run(mode="check", loaded=load_spec_text(spec_text), observed=build_observed(tmp_path)).findings
    assert "AS-207" not in ids(fs)
