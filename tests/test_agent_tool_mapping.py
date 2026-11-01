"""Each agent gets the tools the code gives it, however the per-agent table is written: a dict, an if / elif
chain, a match statement, tools kept on self in __init__, or tool names in agents.yaml looked up in a
registry. The same shapes are checked on CrewAI, LangGraph, AutoGen and the OpenAI Agents SDK."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.ast import extract
from agenticreq.cli import app

CREW_TOOLS = """
from crewai.tools import BaseTool


class Wiki(BaseTool):
    name: str = "wiki"
    description: str = "w"

    def _run(self, q: str) -> str:
        return q


class Arxiv(BaseTool):
    name: str = "arxiv"
    description: str = "a"

    def _run(self, q: str) -> str:
        return q


class Weather(BaseTool):
    name: str = "weather"
    description: str = "w"

    def _run(self, q: str) -> str:
        return q
"""

CREW_HEAD = """
from pathlib import Path

import yaml
from crewai import Agent

from tools import Arxiv, Weather, Wiki


def _load(name):
    with open(Path(__file__).parent / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)

"""

LOOP = """
    def build(self):
        out = {}
        for key, cfg in self.agents_cfg.items():
            out[key] = Agent(role=cfg["role"], goal="g", backstory="b", tools=TOOLS, llm="gpt-4o-2024-08-06")
        return out
"""


def _cls(init_extra: str = "") -> str:
    return (
        '\nclass Crew:\n    def __init__(self):\n        self.agents_cfg = _load("agents.yaml")\n' + init_extra + "\n"
    )


CREW_VARIANTS = {
    "dict_subscript": _cls()
    + """
    def _tools_for(self, key):
        return {"researcher": [Wiki(), Arxiv()], "analyst": [Weather()], "writer": []}[key]
"""
    + LOOP.replace("TOOLS", "self._tools_for(key)"),
    "if_chain": _cls()
    + """
    def _tools_for(self, key):
        if key == "researcher":
            return [Wiki(), Arxiv()]
        elif key == "analyst":
            return [Weather()]
        return []
"""
    + LOOP.replace("TOOLS", "self._tools_for(key)"),
    "match": _cls()
    + """
    def _tools_for(self, key):
        match key:
            case "researcher":
                return [Wiki(), Arxiv()]
            case "analyst" | "quant":
                return [Weather()]
            case _:
                return []
"""
    + LOOP.replace("TOOLS", "self._tools_for(key)"),
    "self_attrs": _cls("        self.wiki = Wiki()\n        self.arxiv = Arxiv()\n        self.weather = Weather()")
    + """
    def _tools_for(self, key):
        return {"researcher": [self.wiki, self.arxiv], "analyst": [self.weather], "writer": []}[key]
"""
    + LOOP.replace("TOOLS", "self._tools_for(key)"),
    "yaml_names_registry": '\nREGISTRY = {"wiki": Wiki, "arxiv": Arxiv, "weather": Weather}\n'
    + _cls()
    + LOOP.replace("TOOLS", '[REGISTRY[n]() for n in cfg.get("tools", [])]'),
    "yaml_names_self_dict": _cls('        self.tools = {"wiki": Wiki(), "arxiv": Arxiv(), "weather": Weather()}')
    + """
    def _tools_for(self, key):
        return [self.tools[n] for n in self.agents_cfg[key].get("tools", [])]
"""
    + LOOP.replace("TOOLS", "self._tools_for(key)"),
}

ROLES = {"researcher": "Lead Researcher", "analyst": "Data Analyst", "writer": "Report Writer"}
YAML_TOOLS = {"researcher": ["wiki", "arxiv"], "analyst": ["weather"]}
CREW_EXPECT = {"Lead Researcher": {"wiki", "arxiv"}, "Data Analyst": {"weather"}, "Report Writer": set()}


def _crew(root: Path, variant: str) -> Path:
    (root / "config").mkdir(parents=True)
    cfg = {
        k: {"role": r, **({"tools": YAML_TOOLS[k]} if k in YAML_TOOLS and "yaml" in variant else {})}
        for k, r in ROLES.items()
    }
    (root / "config" / "agents.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (root / "tools.py").write_text(CREW_TOOLS, encoding="utf-8")
    (root / "crew.py").write_text(CREW_HEAD + CREW_VARIANTS[variant], encoding="utf-8")
    return root


@pytest.mark.parametrize("variant", sorted(CREW_VARIANTS))
def test_crewai_tools_land_on_the_right_agent(tmp_path: Path, variant: str) -> None:
    obs = extract(_crew(tmp_path, variant))
    assert {a.name: set(a.tools) for a in obs.declared_agents} == CREW_EXPECT


FN_TOOLS = '''

@tool
def web_search(q: str) -> str:
    """search"""
    return q


@tool
def calculator(x: str) -> str:
    """calc"""
    return x


@tool
def send_email(to: str) -> str:
    """email"""
    return to

'''

SELECTORS = {
    "if_chain": """
def tools_for(role):
    if role == "researcher":
        return [web_search]
    elif role == "analyst":
        return [calculator, web_search]
    return [send_email]
""",
    "match": """
def tools_for(role):
    match role:
        case "researcher":
            return [web_search]
        case "analyst" | "quant":
            return [calculator, web_search]
        case _:
            return [send_email]
""",
    "registry": """
REGISTRY = {"search": web_search, "calc": calculator, "email": send_email}
SPEC = {"researcher": ["search"], "analyst": ["calc", "search"], "mailer": ["email"]}


def tools_for(role):
    return [REGISTRY[n] for n in SPEC[role]]
""",
}

FRAMEWORKS = {
    "langgraph": (
        "from langchain_core.tools import tool\nfrom langgraph.prebuilt import create_react_agent\n",
        'create_react_agent("openai:gpt-4o-2024-08-06", tools=tools_for(role), name=role)',
    ),
    "autogen": (
        "from autogen_agentchat.agents import AssistantAgent\n\n\ndef tool(f):\n    return f\n",
        "AssistantAgent(name=role, model_client=None, tools=tools_for(role))",
    ),
    "openai-agents": (
        "from agents import Agent, function_tool as tool\n",
        'Agent(name=role, model="gpt-4o-2024-08-06", tools=tools_for(role))',
    ),
}
FN_EXPECT = {"researcher": {"web_search"}, "analyst": {"calculator", "web_search"}, "mailer": {"send_email"}}


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
@pytest.mark.parametrize("selector", sorted(SELECTORS))
def test_tool_selectors_work_on_every_framework(tmp_path: Path, framework: str, selector: str) -> None:
    imports, build = FRAMEWORKS[framework]
    loop = f'\nAGENTS = {{}}\nfor role in ["researcher", "analyst", "mailer"]:\n    AGENTS[role] = {build}\n'
    (tmp_path / "agent.py").write_text(imports + FN_TOOLS + SELECTORS[selector] + loop, encoding="utf-8")
    obs = extract(tmp_path)
    assert framework in obs.frameworks
    assert {a.name: set(a.tools) for a in obs.declared_agents} == FN_EXPECT


def test_undecidable_branch_keeps_both_sides_as_conditional(tmp_path: Path) -> None:
    """A branch on a runtime value never drops a tool the agent may hold: both sides count, marked conditional."""
    (tmp_path / "agent.py").write_text(
        "import os\nfrom agents import Agent, function_tool as tool\n"
        + FN_TOOLS
        + """
def tools_for():
    if os.environ["MODE"] == "full":
        return [web_search, send_email]
    return [web_search]


a = Agent(name="a", model="gpt-4o-2024-08-06", tools=tools_for())
b = Agent(name="b", model="gpt-4o-2024-08-06", tools=[calculator])
""",
        encoding="utf-8",
    )
    obs = extract(tmp_path)
    assert {a.name: set(a.tools) for a in obs.declared_agents} == {
        "a": {"web_search", "send_email"},
        "b": {"calculator"},
    }
    assert "send_email" in obs.conditional_tools


def test_console_shows_each_agents_tools(tmp_path: Path) -> None:
    _crew(tmp_path, "if_chain")
    r = CliRunner().invoke(app, ["audit", str(tmp_path), "--fail-on", "never", "--verbose"], terminal_width=200)
    out = " ".join(r.output.split())
    assert "agent Lead Researcher: wiki, arxiv" in out  # --verbose lists which tools the code gives each agent
    assert "agent Data Analyst: weather" in out
    assert "agent Report Writer: no tools" in out
