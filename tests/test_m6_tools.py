"""M6: only registered tools are tools (helpers and bases are not), a tool class is reported under its name=,
hosted prefixes are one tool, and framework-injected tools (Google ADK set_model_response, Claude Agent SDK
built-ins) are known; Bash still counts as shell execution and Write / Edit as writes."""

from __future__ import annotations

from pathlib import Path

from agenticreq.data import InjectedTools
from agenticreq.names import norm
from agenticreq.observe import build_observed

M6 = Path(__file__).parent / "fixtures" / "repro" / "m6"


def test_m6_a_helpers_and_bases_are_not_tools_b_name_attribute_wins() -> None:
    tools = build_observed(M6 / "helpers").declared_tools
    assert set(tools) == {"save_file", "search_notes"}  # not _StrictTool, _RunScoped, _clean, format_row


def test_m6_c_hosted_prefix_is_the_same_tool() -> None:
    assert norm("hosted:web_search") == norm("web_search") == norm("web_search_preview")


def test_m6_d_adk_set_model_response_is_injected() -> None:
    assert InjectedTools().framework_of("set_model_response") == "google-adk"
    assert "set_model_response" not in build_observed(M6 / "adk").declared_tools


def test_m6_d_claude_builtins_keep_their_capability(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from claude_agent_sdk import ClaudeAgentOptions\n"
        "o = ClaudeAgentOptions(allowed_tools=['Read', 'Write', 'Edit', 'Bash'])\n",
        encoding="utf-8",
    )
    tools = build_observed(tmp_path).declared_tools
    assert tools["Bash"].capability == "shell_execution"
    assert tools["Write"].effect == tools["Edit"].effect == "write" and tools["Read"].effect == "read"
