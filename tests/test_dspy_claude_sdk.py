"""DSPy's own lm15 transport is captured; DSPy ReAct's submit tool is the framework's; a request is named after an
agent only when the name stands as a word; Claude Agent SDK subagents, @tool schemas, session bounds and in-process
MCP servers are read from the code."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from agenticreq.capture.hook import CaptureSession
from agenticreq.capture.responder import CannedResponder, MockConfig
from agenticreq.data import InjectedTools
from agenticreq.observe import build_observed


def _fake_lm15() -> types.ModuleType:
    """The surface of dspy._vendor.lm15.transports the hook patches."""
    mod = types.ModuleType("fake_lm15")

    class Request:
        def __init__(self, method: str, url: str, headers: list[tuple[str, str]], body: bytes) -> None:
            self.method, self.url, self.headers, self.body = method, url, headers, body

    class Resp:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

        def read(self) -> bytes:
            return b"".join(self.chunks)

    class Sync:
        def stream(self, request: Any) -> Any:
            raise AssertionError("the real transport was reached")

    class Async:
        async def _do_stream(self, request: Any) -> Any:
            raise AssertionError("the real transport was reached")

    mod.TransportRequest, mod.TransportResponse, mod.AsyncTransportResponse = Request, Resp, Resp
    mod.StdlibTransport, mod.StdlibAsyncTransport = Sync, Async
    return mod


def test_lm15_requests_are_answered_by_the_canned_model() -> None:
    mod = _fake_lm15()
    s = CaptureSession(CannedResponder(MockConfig()), run_id="r")
    s._patch_lm15(mod)
    body = b'{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}'
    req = mod.TransportRequest("POST", "https://api.openai.com/v1/chat/completions", [], body)
    resp = mod.StdlibTransport().stream(req)
    assert resp.status == 200 and b"choices" in resp.read()
    assert len(s.calls) == 1 and "lm15" in s.libs_hooked
    s.uninstall()


def test_lm15_is_patched_when_imported_after_install(monkeypatch: Any) -> None:
    from agenticreq.capture import hook

    monkeypatch.setitem(hook.OWN_TRANSPORTS, "fake_lm15_mod", "_patch_lm15")
    seen: list[str] = []
    s = CaptureSession(CannedResponder(MockConfig()), run_id="r")
    monkeypatch.setattr(s, "_patch_lm15", lambda m: seen.append(m.__name__))
    s._own_transports()
    try:
        assert sys.modules.get("fake_lm15_mod") is None and seen == []
    finally:
        s.uninstall()


def test_dspy_submit_is_the_framework_s() -> None:
    assert InjectedTools().framework_of("submit") == "dspy" and InjectedTools().framework_of("finish") == "dspy"


def test_a_name_inside_a_tool_name_does_not_name_the_request(tmp_path: Path) -> None:
    from agenticreq.graph.builder import _node_name

    (tmp_path / "a.py").write_text(
        "import dspy\n\ndef refund(x: str) -> str:\n    '''R.'''\n    return x\n\n"
        "billing = dspy.ReAct('q -> a', tools=[refund])\n",
        encoding="utf-8",
    )
    obs = build_observed(tmp_path)
    assert _node_name(obs, "Tools: ask_billing, search_orders") is None
    assert _node_name(obs, "You are billing.") == "billing"


CLAUDE = """
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, create_sdk_mcp_server, tool


@tool("issue_refund", "Refund a customer order", {"order_id": str, "amount": float})
async def issue_refund(args):
    return {"content": []}


shop = create_sdk_mcp_server(name="shop", tools=[issue_refund])
options = ClaudeAgentOptions(
    model="claude-sonnet-4-5-20250929", mcp_servers={"shop": shop}, max_turns=8,
    agents={"billing": AgentDefinition(description="Refunds orders.", prompt="Refund.", tools=["mcp__shop__issue_refund"])},
)
"""


def test_claude_sdk_subagent_tool_schema_bound_and_server(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(CLAUDE, encoding="utf-8")
    obs = build_observed(tmp_path)
    assert "billing" in {a.name for a in obs.declared_agents}
    params = {p.name: p.type for p in obs.declared_tools["issue_refund"].params}
    assert params == {"order_id": "string", "amount": "number"}
    assert [b.agent for b in obs.bounds if b.kind == "max_turns"] == [None]  # bounds the whole session
    assert not [u for u in obs.unresolved_tools if "mcp_servers" in u.reason]


def test_r1_an_external_mcp_server_is_still_a_run_time_source(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from claude_agent_sdk import ClaudeAgentOptions\n"
        "options = ClaudeAgentOptions(mcp_servers={'fs': {'type': 'stdio', 'command': 'npx'}})\n",
        encoding="utf-8",
    )
    assert [u for u in build_observed(tmp_path).unresolved_tools if "mcp_servers" in u.reason]
