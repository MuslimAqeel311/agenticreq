"""M7: mocks finds the entry (or says which function to add), stubs tools that reach a database / the network,
names connections the run will block, and the canned model takes each handoff once per run."""

from __future__ import annotations

from pathlib import Path

import yaml

from agenticreq.capture.responder import CannedResponder, MockConfig
from agenticreq.mockgen import connections, render_mocks
from agenticreq.models.observed import CapturedCall, ToolSchema
from agenticreq.observe import build_observed

REPRO = Path(__file__).parent / "fixtures" / "repro"


def _mocks(project: Path) -> tuple[str, dict]:
    text = render_mocks(project.resolve(), build_observed(project))
    return text, yaml.safe_load(text)


def test_m7_a_no_entry_names_the_agent_and_the_function_to_add() -> None:
    text, data = _mocks(REPRO / "h1_strands")
    assert data["entry"] == "TODO"
    assert "no function runs the agent `agent` (agent.py:26): add one, e.g. `def run(task): return agent(task)`" in text


def test_m7_a_a_function_calling_the_agent_object_is_the_entry(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from strands import Agent, tool\n\n@tool\ndef ping(x: str) -> str:\n    '''Ping.'''\n    return x\n\n"
        "agent = Agent(model='us.anthropic.claude-sonnet-4-20250514-v1:0', tools=[ping])\n\n"
        "def answer(question: str) -> str:\n    return str(agent(question))\n",
        encoding="utf-8",
    )
    _, data = _mocks(tmp_path)
    assert data["entry"] == "agent:answer"


def test_m7_c_database_and_http_tools_are_stubbed_and_connections_named() -> None:
    _, data = _mocks(REPRO / "m7_db")
    assert data["tools"]["count_orders"]["patch"] == "agent:count_orders"
    assert data["tools"]["fetch_rate"]["patch"] == "agent:fetch_rate"
    assert connections((REPRO / "m7_db").resolve()) == [
        ("agent.py:6", "SQLAlchemy", "create_engine('postgresql://user@db.internal/prod')")
    ]


def test_m7_c_local_and_stdlib_modules_are_not_the_network(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def clean(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text(
        "from agents import Agent, function_tool\nfrom helpers import clean\nfrom pathlib import Path\n\n"
        "@function_tool\ndef read_note(name: str) -> str:\n    '''Read a note.'''\n"
        "    return clean(Path(name).name)\n\nagent = Agent(name='a', tools=[read_note])\n",
        encoding="utf-8",
    )
    _, data = _mocks(tmp_path)
    assert "patch" not in data["tools"]["read_note"]


def test_m7_c_a_framework_decorator_is_not_a_service(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from crewai.tools import tool\nfrom crewai import Agent\n\n"
        "@tool('lookup')\ndef lookup(q: str) -> str:\n    '''Look up.'''\n    return q\n\n"
        "a = Agent(role='r', goal='g', backstory='b', tools=[lookup])\n",
        encoding="utf-8",
    )
    _, data = _mocks(tmp_path)
    assert "patch" not in data["tools"]["lookup"]


def test_m7_a_a_graph_node_is_not_the_entry() -> None:
    _, data = _mocks(Path(__file__).parent / "fixtures" / "agents" / "langgraph_po")
    assert data["entry"] != "agent:agent"


def _call(system: str, tools: list[str]) -> CapturedCall:
    return CapturedCall(
        run_id="r",
        index=0,
        provider="openai_chat",
        url="https://api.openai.com/v1/chat/completions",
        model="m",
        system_prompt_sha256=system,
        tools=[ToolSchema(name=t, source="wire") for t in tools],
    )


def test_m7_d_each_handoff_is_taken_once_per_run() -> None:
    r = CannedResponder(MockConfig())
    body = {"messages": [{"role": "user", "content": "task"}]}
    first = r.plan(_call("triage", ["transfer_to_billing"]), body)
    assert [t.name for t in first] == ["transfer_to_billing"]
    # billing's own conversation offers the way back and to itself: neither is taken again
    again = r.plan(_call("billing", ["transfer_to_billing", "transfer_back_to_triage", "lookup"]), body)
    assert [t.name for t in again] == ["transfer_back_to_triage"]
    third = r.plan(_call("triage-2", ["transfer_to_billing", "transfer_back_to_triage", "lookup"]), body)
    assert [t.name for t in third] == ["lookup"]
