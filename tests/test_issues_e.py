"""Priority E: agents built through loops, factories and registries, in every supported framework."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from agenticreq.scaffold import render_init

FRAMEWORKS = {
    "langgraph": ("from langgraph.prebuilt import create_react_agent", "create_react_agent('m', tools=[], name={n})"),
    "crewai": ("from crewai import Agent", "Agent(role={n}, goal='g', backstory='b', llm='m')"),
    "openai_agents": ("from agents import Agent", "Agent(name={n}, instructions='i', model='m')"),
    "claude_sdk": ("from claude_agent_sdk import AgentDefinition", "AgentDefinition(description={n}, prompt='p')"),
    "google_adk": ("from google.adk.agents import LlmAgent", "LlmAgent(name={n}, model='m')"),
    "smolagents": ("from smolagents import ToolCallingAgent", "ToolCallingAgent(tools=[], model=None, name={n})"),
    "autogen": ("from autogen_agentchat.agents import AssistantAgent", "AssistantAgent({n}, model_client=None)"),
    "llamaindex": ("from llama_index.core.agent.workflow import FunctionAgent", "FunctionAgent(name={n}, llm=None)"),
    "pydantic_ai": ("from pydantic_ai import Agent", "Agent('m', name={n})"),
    "haystack": ("from haystack.components.agents import Agent", "Agent(chat_generator=None, system_prompt={n})"),
    "agno": ("from agno.agent import Agent", "Agent(name={n}, model=None)"),
    "semantic_kernel": (
        "from semantic_kernel.agents import ChatCompletionAgent",
        "ChatCompletionAgent(name={n}, service=None)",
    ),
}
FORMS = {
    "for_loop": "{imp}\nNAMES = ['alpha', 'beta']\nagents = []\nfor name in NAMES:\n    agents.append({c_name})\n",
    "comprehension": "{imp}\nNAMES = ['alpha', 'beta']\nagents = [{c_n} for n in NAMES]\n",
    "dict_registry": "{imp}\nREGISTRY = {{'alpha': {c_a}, 'beta': {c_b}}}\n",
    "factory": "{imp}\ndef build_agent(name):\n    return {c_name}\na = build_agent('alpha')\nb = build_agent('beta')\n",
    "two_hop_factory": "{imp}\ndef _new(name):\n    return {c_name}\ndef build(name):\n    return _new(name)\n"
    "a = build('alpha')\nb = build('beta')\n",
    "map": "{imp}\ndef make(name):\n    return {c_name}\nagents = list(map(make, ['alpha', 'beta']))\n",
    "yaml_list": "import yaml\n{imp}\nSPEC = yaml.safe_load(open('agents.yaml'))\nagents = [{c_s} for s in SPEC['agents']]\n",
    "json_list": "import json\n{imp}\nSPEC = json.load(open('agents.json'))\nagents = [{c_s} for s in SPEC]\n",
    "enum": "import enum\n{imp}\nclass Role(enum.Enum):\n    A = 'alpha'\n    B = 'beta'\nagents = [{c_r} for r in Role]\n",
    "dataclass": "from dataclasses import dataclass\n{imp}\n@dataclass\nclass Spec:\n    name: str\n"
    "SPECS = [Spec('alpha'), Spec('beta')]\nagents = [{c_d} for s in SPECS]\n",
}
#: Haystack's Agent has no name of its own, so its agents are counted and named by position
NO_NAME = {"haystack"}


def _write(d: Path, form: str, fw: str) -> None:
    imp, c = FRAMEWORKS[fw]
    (d / "agent.py").write_text(
        FORMS[form].format(
            imp=imp,
            c_name=c.format(n="name"),
            c_n=c.format(n="n"),
            c_a=c.format(n="'alpha'"),
            c_b=c.format(n="'beta'"),
            c_s=c.format(n="s['name']"),
            c_r=c.format(n="r.value"),
            c_d=c.format(n="s.name"),
        ),
        encoding="utf-8",
    )
    (d / "agents.yaml").write_text("agents:\n  - name: alpha\n  - name: beta\n", encoding="utf-8")
    (d / "agents.json").write_text('[{"name": "alpha"}, {"name": "beta"}]', encoding="utf-8")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_e1_agents_are_counted_and_named(tmp_path: Path, fw: str, form: str) -> None:
    _write(tmp_path, form, fw)
    obs = build_observed(tmp_path)
    names = sorted(a.name for a in obs.declared_agents)
    assert len(names) == 2 and not obs.unresolved_agents, names
    if fw not in NO_NAME:
        assert names == ["alpha", "beta"]
    assert names == sorted(a.name for a in build_observed(tmp_path).declared_agents)  # the same code, the same names


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_e1_a_runtime_only_list_is_reported_and_is_never_a_pass_or_a_gap(tmp_path: Path, fw: str) -> None:
    imp, c = FRAMEWORKS[fw]
    (tmp_path / "agent.py").write_text(
        f"import sys\n{imp}\ndef go():\n    return [{c.format(n='n')} for n in sys.argv[1:]]\n", encoding="utf-8"
    )
    obs = build_observed(tmp_path)
    assert obs.unresolved_agents
    report = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    line = next(ln for ln in results(report) if "built at" in ln.reason)
    assert line.status == "SIGN-OFF"
    assert "1+ agents built at agent.py:" in line.reason and "from runtime value `sys.argv[1:]`" in line.reason


SPEC = """
agent_id: a
risk_level: medium
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
"""


def test_e1_a_run_says_how_many_agents_it_saw(tmp_path: Path) -> None:
    from conftest import chat_call

    imp, c = FRAMEWORKS["openai_agents"]
    (tmp_path / "agent.py").write_text(
        f"import sys\n{imp}\nagents = [{c.format(n='n')} for n in sys.argv[1:]]\n", encoding="utf-8"
    )
    calls = [chat_call(0, [], system="one"), chat_call(1, [], system="two"), chat_call(2, [], system="three")]
    obs = build_observed(tmp_path, captures=calls)
    report = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    assert any("a run saw 3" in ln.reason for ln in results(report))


def test_e1_level_3_turns_on_for_two_agents_and_for_a_dynamic_pool(tmp_path: Path) -> None:
    imp, c = FRAMEWORKS["openai_agents"]
    _write(tmp_path, "comprehension", "openai_agents")
    assert "agent_registry:" in render_init(build_observed(tmp_path))
    (tmp_path / "agent.py").write_text(
        f"import sys\n{imp}\nagents = [{c.format(n='n')} for n in sys.argv[1:]]\n", encoding="utf-8"
    )
    assert "agent_registry:" in render_init(build_observed(tmp_path))
