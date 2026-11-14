"""The Claude Agent SDK's `claude` CLI is answered by a local canned endpoint: only that exact program may start,
the canned model never asks the CLI for its own tools (they act outside the mock run's guards), subagents are reached
by their names, and MCP tool names match the code's."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from agenticreq.capture.cli_server import CLI_ENV, start
from agenticreq.capture.guard import ProcessGuard
from agenticreq.capture.hook import CaptureSession
from agenticreq.capture.responder import CannedResponder, MockConfig, _agent_choices, _bare
from agenticreq.models.observed import CapturedCall, ToolSchema
from agenticreq.names import norm


def test_local_endpoint_answers_and_records_a_messages_request() -> None:
    s = CaptureSession(CannedResponder(MockConfig()), run_id="r")
    server, url = start(s)
    try:
        body = {"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        req = urllib.request.Request(
            f"{url}/v1/messages", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310 - the local test endpoint
            data = json.loads(r.read())
        assert data["type"] == "message" and len(s.calls) == 1 and s.calls[0].client_lib == "claude-cli"
    finally:
        server.shutdown()
    assert CLI_ENV["DISABLE_TELEMETRY"] == "1" and CLI_ENV["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_only_the_exact_cli_may_start(tmp_path: Path) -> None:
    cli = tmp_path / "claude_agent_sdk" / "_bundled" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    g = ProcessGuard([], cache_dirs=False, project=str(tmp_path), allow_exec=[str(cli)])
    g.active = True
    g._hook("subprocess.Popen", (str(cli), [str(cli), "--print"], None, None))  # allowed: no exception
    other = tmp_path / "claude"  # same name, another place: blocked
    with pytest.raises(PermissionError):
        g._hook("subprocess.Popen", (str(other), [str(other)], None, None))
    with pytest.raises(PermissionError):
        g._hook("subprocess.Popen", (None, ["bash", "-c", "x"], None, None))


def _cli_call(tools: list[str], system: str = "main") -> CapturedCall:
    return CapturedCall(
        run_id="r",
        index=0,
        provider="anthropic",
        url="https://api.anthropic.com/v1/messages",
        model="m",
        client_lib="claude-cli",
        system_prompt_sha256=system,
        tools=[ToolSchema(name=t, source="wire") for t in tools],
    )


def _body(tools: list[str], reminder: str = "") -> dict[str, Any]:
    props = {
        "Agent": {"description": {"type": "string"}, "prompt": {"type": "string"}, "subagent_type": {"type": "string"}}
    }
    return {
        "system": "You are triage.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": reminder + "\nhelp"}]}],
        "tools": [
            {"name": t, "input_schema": {"type": "object", "properties": props.get(t, {}), "required": []}}
            for t in tools
        ],
    }


def test_the_canned_model_never_asks_the_cli_for_its_own_tools() -> None:
    tools = ["Bash", "Write", "CronCreate", "Workflow", "mcp__shop__search_orders"]
    r = CannedResponder(MockConfig())
    asked = [tc.name for _ in range(5) for tc in r.plan(_cli_call(tools), _body(tools))]
    assert asked == ["mcp__shop__search_orders"]


def test_a_cli_tool_the_person_lists_in_order_is_asked_for() -> None:
    r = CannedResponder(MockConfig(order=["Bash"]))
    assert [tc.name for tc in r.plan(_cli_call(["Bash"]), _body(["Bash"]))] == ["Bash"]


def test_the_subagent_is_named_from_the_reminder_and_cli_agents_are_skipped() -> None:
    reminder = (
        "<system-reminder>\nAvailable agent types for the Agent tool:\n- general-purpose: anything\n"
        "- billing: Refunds orders. (Tools: mcp__shop__issue_refund)\n- Explore: search\n</system-reminder>"
    )
    r = CannedResponder(MockConfig())
    got = r.plan(_cli_call(["Agent"]), _body(["Agent"], reminder))
    assert got and got[0].arguments["subagent_type"] == "billing"
    assert _agent_choices("Lookups: hand over the exact command", "") == []


def test_mcp_tool_names_match_the_code() -> None:
    assert _bare("mcp__shop__search_orders") == "search_orders" and _bare("search_orders") == "search_orders"
    assert norm("mcp__shop__search_orders") == norm("search_orders")
    assert norm("Task") == norm("Agent")
