"""One constructor call, several agents: loops over tables and config files, factories, CrewAI config=.

Each case builds a "research" agent holding `search` and a "writer" agent holding nothing, the way
real projects do it in that framework, and checks that both agents (with the right tools) come out.
Agents whose count only exists at runtime are reported (AS-909), never counted as one agent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.observe import build_observed

TOOL = '''
from crewai.tools import tool
@tool("search")
def search(q: str, max_results: int = 5) -> str:
    """Search."""
    return q
@tool("publish")
def publish(text: str) -> str:
    """Publish."""
    return text
'''

CASES: dict[str, dict[str, str]] = {
    # the crewai_research_agent shape: helper loads YAML, loop builds agents, a method picks tools by key
    "crewai_yaml_loop": {
        "config/agents.yaml": """
            research: {role: research, backstory: finds things, max_iter: 8}
            writer: {role: writer, backstory: writes, max_iter: 4}
        """,
        "crew.py": """
            from pathlib import Path
            import yaml
            from crewai import Agent
            from tools import search
            CONFIG_DIR = Path(__file__).parent / "config"
            def _load(name):
                with open(CONFIG_DIR / name, encoding="utf-8") as fh:
                    return yaml.safe_load(fh)
            class MyCrew:
                def __init__(self):
                    self.agents_cfg = _load("agents.yaml")
                def _tools_for(self, key):
                    return {"research": [search], "writer": []}[key]
                def build(self):
                    out = {}
                    for key, cfg in self.agents_cfg.items():
                        out[key] = Agent(role=cfg["role"], goal="g", backstory=cfg["backstory"],
                                         tools=self._tools_for(key), max_iter=cfg.get("max_iter", 6))
                    return out
        """,
    },
    "crewai_crewbase_default_config": {
        "config/agents.yaml": """
            researcher: {role: research, goal: g, backstory: b}
            reporting_analyst: {role: writer, goal: g, backstory: b}
        """,
        "crew.py": """
            from crewai import Agent, Crew
            from crewai.project import CrewBase, agent, crew
            from tools import search
            @CrewBase
            class Research:
                @agent
                def researcher(self) -> Agent:
                    return Agent(config=self.agents_config["researcher"], tools=[search])
                @agent
                def reporting_analyst(self) -> Agent:
                    return Agent(config=self.agents_config["reporting_analyst"])
        """,
    },
    "crewai_crewbase_explicit_path": {
        "cfg/a.yaml": """
            researcher: {role: research, goal: g, backstory: b}
            writer: {role: writer, goal: g, backstory: b}
        """,
        "crew.py": """
            from crewai import Agent
            from crewai.project import CrewBase, agent
            from tools import search
            @CrewBase
            class Research:
                agents_config = "cfg/a.yaml"
                @agent
                def researcher(self) -> Agent:
                    return Agent(config=self.agents_config["researcher"], tools=[search])
                @agent
                def writer(self) -> Agent:
                    return Agent(config=self.agents_config["writer"], tools=[])
        """,
    },
    "openai_dict_comprehension": {
        "agent.py": """
            from agents import Agent, function_tool
            @function_tool
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            @function_tool
            def publish(text: str) -> str:
                '''Publish.'''
                return text
            SPECS = {"research": [search], "writer": []}
            AGENTS = {n: Agent(name=n, instructions=f"You are {n}", tools=t) for n, t in SPECS.items()}
        """,
    },
    "langgraph_factory": {
        "agent.py": """
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
            @tool
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            @tool
            def publish(text: str) -> str:
                '''Publish.'''
                return text
            def make_agent(agent_name, agent_tools):
                return create_react_agent("openai:gpt-4o", agent_tools, name=agent_name)
            research = make_agent("research", [search])
            writer = make_agent(agent_name="writer", agent_tools=[])
        """,
    },
    "autogen_json_loop": {
        "agents.json": '[{"name": "research", "tools": ["search"]}, {"name": "writer", "tools": []}]',
        "agent.py": """
            import json, os
            from autogen_agentchat.agents import AssistantAgent
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            def publish(text: str) -> str:
                '''Publish.'''
                return text
            TOOLS = {"search": search, "publish": publish}
            HERE = os.path.dirname(__file__)
            with open(os.path.join(HERE, "agents.json")) as f:
                SPECS = json.load(f)
            team = [AssistantAgent(s["name"], model_client=None, tools=[TOOLS[t] for t in s["tools"]]) for s in SPECS]
        """,
    },
    "smolagents_tuple_loop": {
        "agent.py": """
            from smolagents import ToolCallingAgent, tool
            @tool
            def search(q: str, max_results: int = 5) -> str:
                '''Search.

                Args:
                    q: query
                    max_results: cap
                '''
                return q
            @tool
            def publish(text: str) -> str:
                '''Publish.

                Args:
                    text: body
                '''
                return text
            agents = []
            for agent_name, agent_tools in [("research", [search]), ("writer", [])]:
                agents.append(ToolCallingAgent(tools=agent_tools, model=None, name=agent_name))
        """,
    },
    "adk_yaml_tool_names": {
        "agents.yaml": """
            - {name: research, instruction: find, tools: [search]}
            - {name: writer, instruction: write, tools: []}
        """,
        "agent.py": """
            from pathlib import Path
            import yaml
            from google.adk.agents import LlmAgent
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            def publish(text: str) -> str:
                '''Publish.'''
                return text
            REGISTRY = {"search": search, "publish": publish}
            SPECS = yaml.safe_load((Path(__file__).parent / "agents.yaml").read_text())
            agents = [LlmAgent(name=s["name"], model="gemini-2.0-flash-001", instruction=s["instruction"],
                               tools=[REGISTRY[t] for t in s["tools"]]) for s in SPECS]
        """,
    },
    "pydantic_ai_method_factory": {
        "agent.py": """
            from pydantic_ai import Agent
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            def publish(text: str) -> str:
                '''Publish.'''
                return text
            class Builder:
                def make(self, label: str, fns: list):
                    return Agent("openai:gpt-4o", name=label, tools=fns)
            b = Builder()
            research = b.make("research", [search])
            writer = b.make("writer", [])
        """,
    },
}


def _write(tmp: Path, files: dict[str, str]) -> None:
    (tmp / "tools.py").write_text(textwrap.dedent(TOOL), encoding="utf-8")
    for rel, text in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_agent_is_expanded_with_its_own_tools(tmp_path: Path, case: str) -> None:
    _write(tmp_path, CASES[case])
    obs = build_observed(tmp_path)
    agents = {a.name: sorted(a.tools) for a in obs.declared_agents}
    assert agents == {"research": ["search"], "writer": []}, f"{case}: {agents}"
    assert not obs.unresolved_agents, obs.unresolved_agents


def test_bounds_are_read_per_agent_from_config(tmp_path: Path) -> None:
    _write(tmp_path, CASES["crewai_yaml_loop"])
    obs = build_observed(tmp_path)
    assert sorted(b.value for b in obs.bounds if b.kind == "max_iter") == [4.0, 8.0]
    assert all("agents.yaml" in (a.built_from or "") for a in obs.declared_agents)


def test_comprehension_filter_drops_disabled_rows(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent
        CFG = {"research": {"on": True}, "legacy": {"on": False}, "writer": {}}
        live = [Agent(name=k, instructions="x") for k, c in CFG.items() if c.get("on", True)]
    """
        },
    )
    assert sorted(a.name for a in build_observed(tmp_path).declared_agents) == ["research", "writer"]


def test_rows_the_code_does_not_name_stay_distinct(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent, function_tool
        @function_tool
        def search(q: str) -> str:
            '''Search.'''
            return q
        @function_tool
        def publish(text: str) -> str:
            '''Publish.'''
            return text
        workers = [Agent(instructions="x", tools=[t]) for t in (search, publish)]
    """
        },
    )
    agents = {a.name: a.tools for a in build_observed(tmp_path).declared_agents}
    assert agents == {"Agent#1": ["search"], "Agent#2": ["publish"]}


def test_factory_agents_keep_their_variable_for_delegation(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from google.adk.agents import LlmAgent
        def issue_refund(order_id: str) -> str:
            '''Refund.'''
            return order_id
        def specialist(n, t):
            return LlmAgent(name=n, model="gemini-2.0-flash-001", tools=t)
        billing = specialist("billing", [issue_refund])
        triage = LlmAgent(name="triage", model="gemini-2.0-flash-001", sub_agents=[billing])
    """
        },
    )
    spec = SPEC.replace("REG", "[{id: triage}, {id: billing}]")
    fs = run(mode="check", loaded=load_spec_text(spec), observed=build_observed(tmp_path)).findings
    hit = [f for f in fs if f.rule_id == "AS-304"]
    assert hit and hit[0].path == ["triage", "billing"], [f.title for f in fs]


SPEC = """
agent_id: triage
risk_level: medium
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: [{tool: issue_refund, reason: finance only}]
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
agent_registry: REG
declared_delegation: []
"""


def test_registry_checks_now_see_each_config_built_agent(tmp_path: Path) -> None:
    """The writer holds publish in code but its registry entry lists no such tool: AS-206 must fire,
    which it cannot while the loop is read as one anonymous agent."""
    files = dict(CASES["crewai_yaml_loop"])
    files["crew.py"] = (
        files["crew.py"]
        .replace('"writer": []', '"writer": [publish]')
        .replace("from tools import search", "from tools import search, publish")
    )
    _write(tmp_path, files)
    spec = SPEC.replace(
        "REG", "[{id: research, match: research, tools: [search]}, {id: writer, match: writer, tools: [search]}]"
    ).replace("triage", "research")
    fs = run(mode="check", loaded=load_spec_text(spec), observed=build_observed(tmp_path)).findings
    assert any(f.rule_id == "AS-206" and "writer holds publish" in f.title for f in fs), [f.title for f in fs]
    assert not [f for f in fs if f.rule_id in ("AS-308", "AS-909")]


RUNTIME = """
    import sys
    from crewai import Agent
    from tools import search
    def build(names):
        return [Agent(role=n, goal="g", backstory="b", tools=[search]) for n in names]
    crew = build(sys.argv[1:])
"""


def test_runtime_built_agents_are_reported_not_counted(tmp_path: Path) -> None:
    _write(tmp_path, {"crew.py": RUNTIME})
    obs = build_observed(tmp_path)
    assert len(obs.unresolved_agents) == 1 and "runtime" in obs.unresolved_agents[0].reason
    spec = SPEC.replace("REG", "[{id: research, tools: [search]}, {id: writer, tools: []}]")
    fs = run(mode="check", loaded=load_spec_text(spec), observed=obs).findings
    unknown = [f for f in fs if f.rule_id == "AS-909"]
    assert unknown and all(f.confidence.level.value == "unknown" for f in unknown)
    assert any("not matched to code" in f.title for f in unknown)
    assert not [f for f in fs if f.rule_id == "AS-308"], "the placeholder node is not an unregistered agent"


def test_audit_reports_runtime_agents(tmp_path: Path) -> None:
    _write(tmp_path, {"crew.py": RUNTIME})
    fs = run(mode="audit", loaded=None, observed=build_observed(tmp_path)).findings
    assert [f for f in fs if f.rule_id == "AS-909"]


def test_request_params_dict_caps_results(tmp_path: Path) -> None:
    """`limit` passed inside a params dict (not as a keyword) bounds the output: no AS-213."""
    _write(
        tmp_path,
        {
            "agent.py": """
        import requests
        from crewai import Agent
        from crewai.tools import BaseTool
        def get_json(url, params):
            return requests.get(url, params=params, timeout=5).json()
        class WikiSearch(BaseTool):
            name: str = "wikipedia_search"
            description: str = "Search Wikipedia."
            def _run(self, query: str) -> str:
                return str(get_json("https://en.wikipedia.org/w/api.php", {"search": query, "limit": 1}))
        class HNSearch(BaseTool):
            name: str = "hn_search"
            description: str = "Search HN."
            def _run(self, query: str) -> str:
                params = {"query": query, "hitsPerPage": 5}
                return str(get_json("https://hn.algolia.com/api/v1/search", params))
        class OpenSearch(BaseTool):
            name: str = "open_search"
            description: str = "Search without a cap."
            def _run(self, query: str) -> str:
                return str(get_json("https://example.org/search", {"q": query}))
        a = Agent(role="r", goal="g", backstory="b", tools=[WikiSearch(), HNSearch(), OpenSearch()])
    """
        },
    )
    fs = run(mode="audit", loaded=None, observed=build_observed(tmp_path)).findings
    flagged = {f.title.split()[0] for f in fs if f.rule_id == "AS-213"}
    assert flagged == {"open_search"}, flagged


def test_same_factory_arguments_build_one_agent(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent
        def make(model):
            return Agent(name="helper", instructions="h", model=model)
        a = make("gpt-4o")
        b = make("gpt-4o-mini")
    """
        },
    )
    assert [a.name for a in build_observed(tmp_path).declared_agents] == ["helper"]


def test_factory_calls_in_tests_are_not_agents(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
            from agents import Agent
            def make(n):
                return Agent(name=n, instructions="x")
            prod = make("research")
        """,
            "tests/test_agent.py": """
            from agent import make
            def test_it():
                assert make("throwaway")
        """,
        },
    )
    assert [a.name for a in build_observed(tmp_path).declared_agents] == ["research"]


def test_factory_in_another_file_resolves_tools_where_they_are_passed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "factory.py": """
            from agents import Agent
            def make(n, ts):
                return Agent(name=n, instructions="x", tools=ts)
        """,
            "main.py": """
            from agents import function_tool
            from factory import make
            @function_tool
            def search(q: str, max_results: int = 5) -> str:
                '''Search.'''
                return q
            research = make("research", [search])
            writer = make("writer", [])
        """,
        },
    )
    agents = {a.name: a.tools for a in build_observed(tmp_path).declared_agents}
    assert agents == {"research": ["search"], "writer": []}, agents


def test_generic_builder_name_is_not_the_agent_name(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent
        def build():
            return Agent(instructions="x")
        def researcher():
            return Agent(instructions="y")
    """
        },
    )
    assert sorted(a.name for a in build_observed(tmp_path).declared_agents) == ["Agent", "researcher"]


def test_clone_with_new_tools_is_its_own_agent(tmp_path: Path) -> None:
    """openai-agents examples/financial_research_agent: writer_agent.clone(tools=[...])."""
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent, function_tool
        @function_tool
        def search(q: str, max_results: int = 5) -> str:
            '''Search.'''
            return q
        writer = Agent(name="writer", instructions="w")
        writer_with_tools = writer.clone(tools=[search])
        cheaper = writer.clone(model="gpt-4o-mini")
    """
        },
    )
    agents = {a.name: a.tools for a in build_observed(tmp_path).declared_agents}
    assert agents == {"writer": [], "writer_with_tools": ["search"]}, agents


def test_self_attribute_agents_are_named_by_the_attribute(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from autogen_agentchat.agents import AssistantAgent
        class Team:
            def __init__(self, client):
                self.researcher = AssistantAgent(name=self.label(), model_client=client)
                self.writer = AssistantAgent(name=self.label(), model_client=client)
            def label(self):
                return input()
    """
        },
    )
    assert sorted(a.name for a in build_observed(tmp_path).declared_agents) == ["researcher", "writer"]


def test_kwargs_spec_dicts_with_tool_objects(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        from agents import Agent, function_tool
        @function_tool
        def search(q: str, max_results: int = 5) -> str:
            '''Search.'''
            return q
        SPECS = [{"name": "research", "instructions": "r", "tools": [search]},
                 {"name": "writer", "instructions": "w", "tools": []}]
        team = [Agent(**spec) for spec in SPECS]
    """
        },
    )
    agents = {a.name: a.tools for a in build_observed(tmp_path).declared_agents}
    assert agents == {"research": ["search"], "writer": []}, agents


def test_parameters_never_resolve_to_unrelated_variables(tmp_path: Path) -> None:
    """`self.config = config` is whatever the caller passed: a module-level `config = {...}` elsewhere in
    the file must not be read as its value."""
    _write(
        tmp_path,
        {
            "agent.py": """
        import sys
        from agents import Agent
        class Crew:
            def __init__(self, config):
                self.config = config
            def build(self):
                return [Agent(name=k, instructions="x") for k in self.config]
        def unrelated():
            config = {"ghost": 1}
            return config
        Crew(sys.argv).build()
    """
        },
    )
    obs = build_observed(tmp_path)
    assert "ghost" not in {a.name for a in obs.declared_agents}
    assert len(obs.unresolved_agents) == 1


def test_loop_that_repeats_one_agent_is_not_unresolved(tmp_path: Path) -> None:
    """crewAI-examples browser_tools: a summarizer agent per chunk of a runtime string is one agent."""
    _write(
        tmp_path,
        {
            "agent.py": """
        from crewai import Agent, Task
        def summarize(content: str) -> str:
            out = []
            for chunk in [content[i:i + 8000] for i in range(0, len(content), 8000)]:
                agent = Agent(role="Principal Researcher", goal="g", backstory="b")
                out.append(Task(agent=agent, description=chunk).execute())
            return "".join(out)
    """
        },
    )
    obs = build_observed(tmp_path)
    assert [a.name for a in obs.declared_agents] == ["Principal Researcher"] and not obs.unresolved_agents


def test_yaml_folded_quotes_are_not_part_of_the_name(tmp_path: Path) -> None:
    files = dict(CASES["crewai_crewbase_default_config"])
    files["config/agents.yaml"] = (
        'researcher:\n  role: >\n    "research"\n  goal: g\n  backstory: b\n'
        "reporting_analyst: {role: writer, goal: g, backstory: b}\n"
    )
    _write(tmp_path, files)
    assert sorted(a.name for a in build_observed(tmp_path).declared_agents) == ["research", "writer"]


def test_self_referential_config_terminates(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "agent.py": """
        import json
        from agents import Agent
        cfg = json.load(cfg)
        loop_a = loop_b
        loop_b = loop_a
        agents = [Agent(name=n, instructions="x") for n in cfg[loop_a]]
    """
        },
    )
    obs = build_observed(tmp_path)
    assert len(obs.unresolved_agents) == 1
