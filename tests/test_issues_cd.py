"""Priority C (counts and display) and D (framework and provider support)."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agenticreq.capture.discover import DiscoverResult, unsupported_transports
from agenticreq.capture.responder import CannedResponder, MockConfig
from agenticreq.capture.wire import detect_provider, parse_request, parse_response_tool_calls
from agenticreq.cli import app
from agenticreq.data import load_yaml
from agenticreq.mockgen import render_mocks
from agenticreq.models.verdict import distinct
from agenticreq.observe import build_observed
from test_issues_b import ISSUES, lines

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"


def _check(project: Path, *extra: str) -> str:
    r = runner.invoke(
        app,
        ["check", str(project), "--spec", str(project / "spec.yml"), "--all", "--quiet", *extra],
        terminal_width=200,
        env={"NO_COLOR": "1", "COLUMNS": "200"},
    )
    return r.output


# =========================================================================== C1, C2
def test_c1_summary_and_headings_count_the_same_rows() -> None:
    out = _check(FIX / "spike")
    m = re.search(r"(\d+) gaps? . \d+ sign-offs? left.*?(\d+) to check . (\d+) passed", out)
    assert m is not None
    assert f"SPEC VS CODE GAPS ({m.group(1)})" in out and f"PASSED ({m.group(3)})" in out


def test_c1_one_rule_counts_the_same_result_for_two_agents_once() -> None:
    from agenticreq.models.verdict import ResultLine

    a = ResultLine("a", "p", "GAP", "r", (), (), "", False)
    b = ResultLine("b", "p", "GAP", "r", (), (), "", False)
    assert len(distinct([a, b])) == 1


def test_c2_a_group_held_back_still_names_every_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(FIX / "passfail" / "clean", tmp_path / "p")
    monkeypatch.chdir(tmp_path / "p")
    r = runner.invoke(app, ["check"], terminal_width=400, env={"NO_COLOR": "1", "COLUMNS": "400"})
    held = next(ln for ln in r.output.splitlines() if ln.startswith("Check with a mock run:"))
    count = int(re.search(r"(\d+) items?", held).group(1))  # type: ignore[union-attr]
    assert len(held.split("are fixed: ")[1].split(", ")) == count


# =========================================================================== C3
def test_c3_spec_tool_effects_come_before_the_code() -> None:
    project = ISSUES / "c3_effects"
    obs = build_observed(project)
    text = render_mocks(project, obs, set(), {"lookup": "write"})
    assert "lookup: {patch:" in text and "lookup: {args" not in text


def test_c3_a_tool_the_scan_cannot_follow_is_not_called_read_only() -> None:
    project = ISSUES / "c3_effects"
    text = render_mocks(project, build_observed(project))
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("lookup:"))
    assert "read-only" not in line and "cannot tell whether it writes" in line


def test_c3_the_blocked_list_has_no_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agenticreq.capture.guard import ProcessGuard

    g = ProcessGuard([], cache_dirs=False, project=str(tmp_path))
    g.active = True
    for _ in range(3):
        with pytest.raises(PermissionError):
            g._hook("os.remove", (str(tmp_path / "x"),))
    assert g.blocked == [f"write:{tmp_path / 'x'}"]


# =========================================================================== D1
def test_d1_built_in_tools_are_read_from_allowed_tools() -> None:
    obs = build_observed(ISSUES / "d1_claude")
    assert set(obs.bound_tools) == {"Read", "Bash", "Write"}
    assert obs.declared_tools["Bash"].capability == "shell_execution"
    assert obs.declared_tools["Write"].effect == "write" and obs.declared_tools["Read"].effect == "read"


def test_d1_a_project_that_allows_bash_shows_a_gap_when_the_spec_prohibits_the_shell() -> None:
    got = lines(ISSUES / "d1_claude")
    assert got["prohibited"].status == "GAP" and "Bash" in got["prohibited"].reason
    assert got["role_boundary"].status == "GAP" and "Bash, Write" in got["role_boundary"].reason


def test_d1_a_tool_called_read_in_another_framework_is_not_the_built_in(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from agents import Agent\nagent = Agent(name='a', tools=['Read'])\n", encoding="utf-8"
    )
    assert "Read" not in build_observed(tmp_path).declared_tools


def test_d1_bash_with_a_pattern_is_bash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from claude_agent_sdk import ClaudeAgentOptions\no = ClaudeAgentOptions(allowed_tools=['Bash(git:*)'])\n",
        encoding="utf-8",
    )
    assert set(build_observed(tmp_path).declared_tools) == {"Bash"}


# =========================================================================== D2
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
GEMINI_BODY = {
    "systemInstruction": {"parts": [{"text": "be brief"}]},
    "contents": [{"role": "user", "parts": [{"text": "go"}]}],
    "tools": [{"functionDeclarations": [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]}],
}


@pytest.mark.parametrize(
    "url,want",
    [
        ("https://api.openai.com/v1/chat/completions", "openai_chat"),
        ("https://api.openai.com/v1/responses", "openai_responses"),
        ("https://api.groq.com/openai/v1/chat/completions", "openai_chat"),
        ("https://api.mistral.ai/v1/chat/completions", "openai_chat"),
        ("https://openrouter.ai/api/v1/chat/completions", "openai_chat"),
        ("http://localhost:11434/v1/chat/completions", "openai_chat"),
        ("https://x.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21", "openai_chat"),
        ("http://litellm.internal:4000/chat/completions", "openai_chat"),
        (GEMINI, "gemini"),
        (GEMINI.replace(":generateContent", ":streamGenerateContent") + "?alt=sse", "gemini"),
    ],
)
def test_d2_every_listed_host_is_recognised(url: str, want: str) -> None:
    assert detect_provider(url, {"messages": [], "model": "m"}) == want


def test_d2_anthropic_needs_its_body() -> None:
    assert detect_provider("https://api.anthropic.com/v1/messages", {"messages": []}) == "anthropic"
    assert detect_provider("https://example.com/inbox/messages", {"to": "x"}) is None


def test_d2_gemini_request_reply_and_tool_call() -> None:
    c = parse_request(0, GEMINI, GEMINI_BODY)
    assert (c.provider, c.model, c.system_prompt) == ("gemini", "gemini-2.0-flash-001", "be brief")
    assert [t.name for t in c.tools] == ["lookup"]
    r = CannedResponder(MockConfig())
    tcs = r.plan(c, GEMINI_BODY)
    raw, ctype = r.render(c, tcs, stream=False)
    body = json.loads(raw)
    assert ctype == "application/json" and [t.name for t in parse_response_tool_calls("gemini", body)] == ["lookup"]
    follow = {
        **GEMINI_BODY,
        "contents": [
            *GEMINI_BODY["contents"],
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {}}}]},
        ],
    }
    c2 = parse_request(1, GEMINI, follow)
    assert [t.name for t in c2.prior_tool_calls] == ["lookup"]
    raw2, _ = r.render(c2, r.plan(c2, follow), stream=False)
    assert "text" in json.loads(raw2)["candidates"][0]["content"]["parts"][0]


def _run_project(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    shutil.copytree(ISSUES / name, tmp_path / name)
    monkeypatch.chdir(tmp_path / name)
    return runner.invoke(app, ["mock-run", "--python", sys.executable, "--quiet"])


def test_d2_requests_is_intercepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("requests")
    r = _run_project("d2_requests", tmp_path, monkeypatch)
    assert r.exit_code == 0 and "captured 2/2" in r.output


def test_d2_aiohttp_is_intercepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("aiohttp")
    r = _run_project("d2_aiohttp", tmp_path, monkeypatch)
    assert r.exit_code == 0 and "captured 2/2" in r.output


def test_d2_a_transport_that_cannot_be_intercepted_is_named_and_does_not_fail() -> None:
    res = DiscoverResult(entry="a", llm_misses=["bedrock-runtime.us-east-1.amazonaws.com"])
    res.unsupported = unsupported_transports(res, load_yaml("providers.yml"))
    assert res.unsupported == ["AWS Bedrock (signed botocore transport)"] and res.problems() == []


def test_d2_after_an_unsupported_run_checks_are_sign_offs(tmp_path: Path) -> None:
    from agenticreq.engine import run
    from agenticreq.loader import load_spec
    from agenticreq.models.verdict import overall, results

    project = ISSUES / "d4_code"
    report = run(mode="check", loaded=load_spec(project / "spec.yml"), observed=build_observed(project))
    report.metadata["mock_run"] = {"captured": 0, "total": 0, "stale": False, "unsupported": ["X (gRPC)"]}
    got = results(report)
    assert not [ln for ln in got if ln.status == "CHECK"]
    assert any("mock run is not supported for X (gRPC)" in ln.reason for ln in got)
    assert overall(report) == "FAIL"  # the gaps of this project stay gaps


def test_d2_a_dummy_key_stays_in_the_run() -> None:
    from agenticreq.capture.guard import FAKE_PREFIX, build_child_env

    env, _ = build_child_env({}, Path("."))
    assert env["XAI_API_KEY"].startswith(FAKE_PREFIX) and env["AZURE_OPENAI_ENDPOINT"]


# =========================================================================== D3
def test_d3_adk_sub_agents_agent_tool_and_whole_run_limit() -> None:
    obs = build_observed(ISSUES / "d3_adk")
    root = next(a for a in obs.declared_agents if a.name == "root")
    assert root.handoffs == ["writer"] and "researcher" in root.tools
    assert [(b.name, b.value) for b in obs.bounds] == [("max_llm_calls", 30.0)]


# =========================================================================== D4
def test_d4_a_code_agent_is_a_prompt_only_sign_off_and_never_asks_for_more_runs() -> None:
    project = ISSUES / "d4_code"
    got = lines(project, project / "captures.json")
    for name in ("lookup", "save_note"):
        assert got[name].status == "SIGN-OFF" and "Prompt-only" in got[name].reason
        assert "runs:" not in got[name].reason and "Add a run" not in got[name].reason
    assert got["execution_model"].status == "SIGN-OFF" and "lookup, save_note" in got["execution_model"].reason
    assert got["execution_model"].reason.startswith("Prompt-only")
