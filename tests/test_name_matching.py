"""One name normalization everywhere: a framework's display name ("Web Search") and the name sent to the model
(web_search) are one tool. A tool is never both offered and never offered; a genuinely unused tool still shows;
two code names that normalize to the same name are a gap."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from conftest import chat_call
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.models.verdict import results
from agenticreq.names import collisions, norm, not_in
from agenticreq.observe import build_observed

runner = CliRunner()


@pytest.mark.parametrize(
    "a,b",
    [
        ("Web Search", "web_search"),
        ("search.web", "search_web"),
        ("Search-Web", "search_web"),
        ("__Fetch--Page__", "fetch_page"),
        ("Save  File", "save_file"),
    ],
)
def test_norm(a: str, b: str) -> None:
    assert norm(a) == b


def test_not_in_keeps_the_original_spelling() -> None:
    assert not_in(["Web Search", "Save File", "Unused Tool"], ["web_search", "save_file"]) == ["Unused Tool"]
    assert collisions(["save_file", "Save File", "web_search"]) == [["Save File", "save_file"]]


def W(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Do it.",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    }


CREWAI = """
    from crewai import Agent, Crew, Task
    from crewai.tools import BaseTool

    class WebSearchTool(BaseTool):
        name: str = "Web Search"
        description: str = "Search the web."
        def _run(self, query: str) -> str:
            return query[:100]

    class SaveFileTool(BaseTool):
        name: str = "Save File"
        description: str = "Save a file."
        def _run(self, path: str, text: str) -> str:
            with open(path, "w") as fh:
                fh.write(text)
            return "ok"

    class UnusedTool(BaseTool):
        name: str = "Unused Tool"
        description: str = "Never used."
        def _run(self, q: str) -> str:
            return q[:10]

    researcher = Agent(role="Researcher", goal="g", backstory="b", tools=[WebSearchTool(), SaveFileTool()])
    writer = Agent(role="Writer", goal="g", backstory="b", tools=[UnusedTool()])
    crew = Crew(agents=[researcher, writer], tasks=[Task(description="d", expected_output="e", agent=researcher)])
"""
LANGCHAIN = """
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool("search.web")
    def search_web(q: str) -> str:
        '''Search the web.'''
        return q[:100]

    @tool("unused.tool")
    def unused_tool(q: str) -> str:
        '''Never used.'''
        return q[:10]

    llm = ChatOpenAI(model="gpt-4o-2024-08-06").bind_tools([search_web, unused_tool])
"""
OPENAI_AGENTS = """
    from agents import Agent, function_tool

    @function_tool(name_override="get-weather")
    def get_weather(city: str) -> str:
        '''Weather for a city.'''
        return city[:50]

    @function_tool(name_override="unused-tool")
    def unused(q: str) -> str:
        '''Never used.'''
        return q[:10]

    agent = Agent(name="weather", instructions="w", tools=[get_weather, unused])
"""
CASES = [
    ("crewai", CREWAI, ["web_search", "save_file"], "Unused Tool"),
    ("langchain", LANGCHAIN, ["search_web"], "unused.tool"),
    ("openai_agents", OPENAI_AGENTS, ["get_weather"], "unused-tool"),
]


def _report(tmp: Path, src: str, wire: list[str]) -> Any:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    obs = build_observed(tmp, captures=[chat_call(0, [W(n) for n in wire])])
    return obs, run(mode="audit", observed=obs)


@pytest.mark.parametrize("fw,src,wire,unused", CASES, ids=[c[0] for c in CASES])
def test_display_and_wire_names_are_one_tool(tmp_path: Path, fw: str, src: str, wire: list[str], unused: str) -> None:
    obs, report = _report(tmp_path, src, wire)
    never = [f.title.split(" never reached")[0] for f in report.findings if f.rule_id == "AS-201"]
    assert never == [unused], (fw, never, sorted(obs.declared_tools))  # the unused one still shows, only it
    assert not any(f.rule_id == "AS-202" for f in report.findings)  # nor "offered but not declared"
    offered = obs.offered_tools()
    assert not any(norm(n) in {norm(o) for o in offered} for n in never)  # never in two opposite lists
    for ln in results(report):  # names shown as the code spells them
        if ln.param == unused:
            assert ln.status == "CHECK"


def test_every_tool_offered_means_the_never_offered_list_is_empty(tmp_path: Path) -> None:
    obs, report = _report(
        tmp_path,
        CREWAI.replace("[UnusedTool()]", "[]").replace(
            '    class UnusedTool(BaseTool):\n        name: str = "Unused Tool"\n        description: str = "Never used."\n'
            "        def _run(self, q: str) -> str:\n            return q[:10]\n",
            "",
        ),
        ["web_search", "save_file"],
    )
    assert not [f for f in report.findings if f.rule_id == "AS-201"], sorted(obs.declared_tools)


def test_two_code_names_that_normalize_to_the_same_name_are_a_gap(tmp_path: Path) -> None:
    src = textwrap.dedent(CREWAI) + textwrap.dedent("""
        from langchain_core.tools import tool

        @tool
        def save_file(path: str) -> str:
            '''Also save.'''
            return path[:10]
    """)
    _, report = _report(tmp_path, src, ["web_search", "save_file"])
    lines = [ln for ln in results(report) if "normalize to the same name" in ln.reason]
    assert len(lines) == 1 and lines[0].status == "GAP", [(ln.param, ln.reason) for ln in results(report)]
    assert "Two tools normalize to the same name: Save File, save_file." == lines[0].reason


MOCK_AGENT = (
    "import httpx\n"
    "def main():\n"
    "    tools = [{'type': 'function', 'function': {'name': n, 'description': 'd', 'parameters': "
    "{'type': 'object', 'properties': {}}}} for n in ('web_search', 'save_file')]\n"
    "    httpx.post('https://api.openai.com/v1/chat/completions', headers={'Authorization': 'Bearer x'},\n"
    "               json={'model': 'gpt-4o-2024-08-06', 'messages': [{'role': 'user', 'content': 'hi'}],"
    " 'tools': tools})\n"
)


def test_mock_run_summary_never_lists_an_offered_tool_as_never_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "agent.py").write_text(MOCK_AGENT, encoding="utf-8")
    (tmp_path / "mocks.yml").write_text(
        "entry: agent.py:main\norder: [Web Search, Save File, Unused Tool]\ntools: {}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["mock-run", "--quiet"], terminal_width=250)
    out = r.output
    assert "tools offered to the model: web_search, save_file" in out, out
    never = [ln for ln in out.splitlines() if ln.startswith("never offered to the model:")]
    assert len(never) == 1 and never[0].startswith("never offered to the model: Unused Tool:"), out
    assert "Web Search" not in never[0] and "Save File" not in never[0]
