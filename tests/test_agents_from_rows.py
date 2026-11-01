"""Agents built by a factory from a literal list of rows (a dataclass per agent) are each found by name, with their
tools, in every framework; a multi-agent init writes the Level 3 fields as real TODOs."""

from __future__ import annotations

import textwrap
from pathlib import Path

from agenticreq.observe import build_observed
from agenticreq.scaffold import render_init

CODE = """
    from dataclasses import dataclass
    from autogen_agentchat.agents import AssistantAgent

    def web_search(q: str) -> str:
        '''Search.'''
        return q[:50]

    def save_note(text: str) -> str:
        '''Save.'''
        return text[:50]

    TOOLS = {"web_search": web_search, "save_note": save_note}

    def pick(*names):
        return [TOOLS[n] for n in names]

    @dataclass(frozen=True)
    class AgentSpec:
        name: str
        system_message: str
        tools: tuple[str, ...] = ()

    SPECS = [AgentSpec("researcher", "You research.", ("web_search",)),
             AgentSpec("writer", "You write.", ("save_note",))]

    def _make(spec, client):
        kwargs = dict(name=spec.name, model_client=client, system_message=spec.system_message,
                      tools=pick(*spec.tools) if spec.tools else None)
        try:
            return AssistantAgent(max_tool_iterations=6, **kwargs)
        except TypeError:
            return AssistantAgent(**kwargs)

    def build_agents(client):
        return {s.name: _make(s, client) for s in SPECS}
"""


def test_agents_from_a_list_of_rows_are_found_by_name_with_their_tools(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(textwrap.dedent(CODE), encoding="utf-8")
    obs = build_observed(tmp_path)
    assert [(a.name, a.tools) for a in obs.declared_agents] == [
        ("researcher", ["web_search"]),
        ("writer", ["save_note"]),
    ]
    text = render_init(obs)
    for field in ("topology", "agent_registry", "declared_delegation", "conflict_resolution"):
        assert f"\n{field}: TODO" in text, field
