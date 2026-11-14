"""Fixes found by running the whole journey (audit -> init -> check -> mocks -> mock-run -> sign-off -> check) on a
multi-agent project for each framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agenticreq.capture.discover import DiscoverResult, unsupported_transports
from agenticreq.capture.responder import CannedResponder, MockConfig, _agent_choices
from agenticreq.data import load_yaml
from agenticreq.mockgen import render_mocks
from agenticreq.models.observed import CapturedCall, ToolSchema
from agenticreq.observe import build_observed


def _call(system: str, tools: list[tuple[str, str]]) -> CapturedCall:
    return CapturedCall(
        run_id="r",
        index=0,
        provider="openai_chat",
        url="https://api.openai.com/v1/chat/completions",
        model="m",
        system_prompt_sha256=system,
        tools=[ToolSchema(name=n, description=d, source="wire") for n, d in tools],
    )


def _body(system: str, tool: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "task"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool,
                    "parameters": {"type": "object", "properties": props, "required": list(props)},
                },
            }
        ],
    }


def test_google_genai_accepts_the_canned_aiohttp_reply() -> None:
    aiohttp = pytest.importorskip("aiohttp")
    from agenticreq.capture.hook import _aio_response

    r = _aio_response(200, {"content-type": "application/json"}, b'{"a": 1}', "https://x")
    assert isinstance(r, aiohttp.ClientResponse) and r.status == 200 and r.ok


def test_llamaindex_handoff_names_an_agent_from_its_description() -> None:
    desc = "Useful for handing off.\n\nCurrently available agents:\n{'billing': 'Refunds orders.'}\n"
    assert _agent_choices(desc) == ["billing"]
    r = CannedResponder(MockConfig())
    body = _body("triage", "handoff", {"to_agent": {"type": "string"}, "reason": {"type": "string"}})
    got = r.plan(_call("triage", [("handoff", desc)]), body)
    assert got[0].arguments["to_agent"] == "billing"


def test_agno_delegates_to_every_member_once() -> None:
    system = '<team_members>\n<member id="researcher" name="researcher">\n<member id="refunder" name="refunder">\n'
    r = CannedResponder(MockConfig())
    body = _body(system, "delegate_task_to_member", {"member_id": {"type": "string"}, "task": {"type": "string"}})
    call = _call("team", [("delegate_task_to_member", "Use this function to delegate a task.")])
    picked = [r.plan(call, body) for _ in range(3)]
    assert [p[0].arguments["member_id"] for p in picked if p] == ["researcher", "refunder"]
    assert picked[2] == []


def test_a_tool_that_runs_another_agent_is_never_the_entry(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from langchain_core.tools import tool\nfrom langchain.agents import create_agent\n\n"
        "billing = create_agent(model='openai:gpt-4o', tools=[])\n\n"
        "@tool\ndef ask_billing(request: str) -> str:\n    '''Ask billing.'''\n"
        "    return billing.invoke({'messages': [('user', request)]})\n\n"
        "def run(task: str) -> str:\n    return str(billing.invoke({'messages': [('user', task)]}))\n",
        encoding="utf-8",
    )
    data = yaml.safe_load(render_mocks(tmp_path, build_observed(tmp_path)))
    assert data["entry"] == "agent:run"


def test_a_tool_that_delegates_through_a_framework_is_not_patched(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from haystack.components.agents import Agent\nfrom haystack.dataclasses import ChatMessage\n"
        "from haystack.tools import tool\n\nbilling = Agent(chat_generator=None, tools=[])\n\n"
        "@tool\ndef ask_billing(request: str) -> str:\n    '''Ask billing.'''\n"
        "    return billing.run(messages=[ChatMessage.from_user(request)])['messages'][-1].text\n\n"
        "triage = Agent(chat_generator=None, tools=[ask_billing])\n",
        encoding="utf-8",
    )
    data = yaml.safe_load(render_mocks(tmp_path, build_observed(tmp_path)))
    assert "patch" not in data["tools"]["ask_billing"]


def test_dspy_native_transport_is_named_not_a_failure() -> None:
    table = load_yaml("providers.yml")
    res = DiscoverResult(entry="agent:run", llm_misses=["127.0.0.1:9"])
    res.versions = {"module:dspy._vendor.lm15.transports": "loaded"}
    names = unsupported_transports(res, table)
    assert names and "DSPy native lm15 transport" in names[0]
    res.unsupported = names
    assert res.problems() == []  # its items become sign-offs, the run is not a failure
    quiet = DiscoverResult(entry="agent:run")  # no call escaped: the library is not named
    quiet.versions = {"module:dspy._vendor.lm15.transports": "loaded"}
    assert unsupported_transports(quiet, table) == []
