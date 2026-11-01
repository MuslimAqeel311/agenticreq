"""`agenticreq fix` patches only what runs an agent loop, and never test code."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenticreq.ast import extract
from agenticreq.fix import plan

HEAD = """
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

@tool
def get_rate(target: str) -> str:
    '''Rate.'''
    return target

def build():
    return StateGraph(dict).compile()
"""


@pytest.mark.parametrize(
    "line,patched",
    [
        ("graph = create_react_agent('gpt-4o-2024-08-06', tools=[get_rate])\ngraph.invoke({'messages': []})", True),
        ("app = StateGraph(dict).compile()\napp.invoke({})", True),
        ("build().invoke({})", True),
        ("get_rate.invoke({'target': 'EUR'})", False),  # a tool call, not an agent loop
        ("llm = ChatOpenAI(model='gpt-4o-2024-08-06')\nllm.invoke('hi')", False),  # one model call
        ("graph = create_react_agent('m', tools=[get_rate])\ngraph.invoke({}, config={'recursion_limit': 5})", False),
    ],
)
def test_recursion_limit_only_on_graph_runs(tmp_path: Path, line: str, patched: bool) -> None:
    (tmp_path / "agent.py").write_text(textwrap.dedent(HEAD) + line + "\n", encoding="utf-8")
    ps = [p for p in plan(extract(tmp_path), tmp_path, 25) if p.rule_id == "AS-208"]
    assert bool(ps) is patched, [p.description for p in ps]


def test_test_files_are_never_patched(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "agent.py").write_text(textwrap.dedent(HEAD), encoding="utf-8")
    (tmp_path / "tests" / "test_agent.py").write_text(
        "from agent import build\nbuild().invoke({})\nm = 'gpt-4o'\n", encoding="utf-8"
    )
    files = {p.file.name for p in plan(extract(tmp_path), tmp_path, 25)}
    assert "test_agent.py" not in files


@pytest.mark.parametrize(
    "src,patched",
    [
        ("from crewai import Agent\na = Agent(role='r', goal='g', backstory='b', tools=[])\n", True),
        ("from agents import Agent\na = Agent(name='n', instructions='i', model='gpt-4o-2024-08-06')\n", False),
    ],
)
def test_max_iter_only_on_crewai_agents(tmp_path: Path, src: str, patched: bool) -> None:
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    ps = [p for p in plan(extract(tmp_path), tmp_path, 15) if "max_iter" in p.description]
    assert bool(ps) is patched
