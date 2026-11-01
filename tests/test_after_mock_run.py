"""After a mock run that is newer than the code, "Check with a mock run" keeps only what editing mocks.yml could
settle, and says what to change. Everything else moves: a number with no limit in code to a gap, what agenticreq
cannot read to a sign-off, a gate inside a mocked tool to a sign-off. No line asks for another identical run."""

from __future__ import annotations

import json
import shutil
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import chat_call
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import Verdict, results, settle_after_run
from agenticreq.observe import build_observed

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
PROJECTS = sorted(
    [
        *(FIX / "agents").iterdir(),
        *(FIX / "agents_crewai").iterdir(),
        FIX / "e2e" / "langgraph_team",
        FIX / "agents_misc" / "finance_like",
        FIX / "agents_misc" / "mock_run_app",
    ]
)
RUN = "agenticreq mock-run"

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
bounds: {max_cost_usd: 2, max_duration_s: 600}
progress_guard: {max_no_progress_steps: 5}
output_size_limits: {max_chars: 20000}
"""


@pytest.mark.parametrize(
    "reason,rules,status,starts",
    [
        (
            "spec.yml says 2; not readable in code (or sign it off under confirmed:). Run: agenticreq mock-run",
            ("AS-501",),
            "GAP",
            "spec.yml says 2; the code sets no limit. Add it in code, or confirm where it is enforced under confirmed:.",
        ),
        (
            "spec.yml says 5; the code sets this limit from a value agenticreq cannot read (or sign it off under "
            "confirmed:). Run: agenticreq mock-run",
            ("AS-501",),
            "SIGN-OFF",
            "spec.yml says 5; the code sets this limit",
        ),
        (
            "Cannot tell whether Web Search changes anything: it calls DDGS, which agenticreq cannot follow (line 111). "
            "Declare it under tool_effects: in spec.yml, or run: agenticreq mock-run (+3 more)",
            ("AS-212",),
            "SIGN-OFF",
            "Cannot tell whether Web Search changes anything",
        ),
        (
            "Code pattern only: Save File (approval_call at tools.py:317). Add a mock that exercises it, then run: "
            "agenticreq mock-run",
            ("AS-205",),
            "SIGN-OFF",
            "The mock replaced Save File, so the gate did not run",
        ),
        (
            "Checkpointing cannot be verified for this framework. Run: agenticreq mock-run",
            ("AS-226",),
            "SIGN-OFF",
            "Checkpointing cannot be verified for this framework; a mock run cannot show more. Confirm it under confirmed:",
        ),
    ],
)
def test_each_kind_moves_where_it_can_be_settled(reason: str, rules: tuple[str, ...], status: str, starts: str) -> None:
    v = settle_after_run(Verdict("CHECK", reason), rules, "p")
    assert v.status == status and v.reason.startswith(starts) and RUN not in v.reason, v
    if "(+3 more)" in reason:
        assert v.reason.endswith("(+3 more)")


def test_a_line_that_mocks_yml_settles_stays_and_says_what_to_change() -> None:
    v = Verdict(
        "CHECK",
        "Never offered to the model: the run never reached billing. Under tools: in mocks.yml, give "
        "billing a task, then run: agenticreq mock-run",
    )
    assert settle_after_run(v, ("AS-201",), "lookup_order") == v


def _all_offered(project: Path) -> Any:
    obs = build_observed(project)
    tools = [
        {"name": t, "description": "d", "parameters": {"type": "object", "properties": {}}} for t in obs.declared_tools
    ]
    return build_observed(project, captures=[chat_call(0, tools)])


@pytest.mark.parametrize("project", PROJECTS, ids=[p.name for p in PROJECTS])
def test_after_a_mock_run_no_line_asks_for_another_identical_run(project: Path) -> None:
    obs = _all_offered(project)
    report = run(mode="check", loaded=load_spec_text(SPEC), observed=obs, today=date(2026, 10, 30))
    report.metadata["mock_run"] = {"captured": 1, "total": 1, "stale": False}
    lines = results(report)
    for ln in lines:
        if RUN in ln.reason:
            assert "mocks.yml" in ln.reason, (ln.param, ln.reason)
        if ln.status == "CHECK":
            assert "mocks.yml" in ln.reason or ln.reason.startswith("Held in only"), (ln.param, ln.reason)
    keys = [(ln.agent, ln.param) for ln in lines]
    assert len(keys) == len(set(keys))  # one line per parameter, in one group


def _json(args: list[str]) -> dict[str, Any]:
    r = runner.invoke(app, args)
    return json.loads(r.stdout)  # type: ignore[no-any-return]


def test_mock_run_twice_gives_the_same_report_and_step_4_can_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(FIX / "agents_misc" / "mock_run_app", tmp_path / "app")
    monkeypatch.chdir(tmp_path / "app")
    assert runner.invoke(app, ["mocks"]).exit_code == 0
    assert runner.invoke(app, ["mock-run", "--quiet"]).exit_code == 0
    first = _json(["check", "-f", "json"])
    assert runner.invoke(app, ["mock-run", "--quiet"]).exit_code == 0
    second = _json(["check", "-f", "json"])
    assert first["results"] == second["results"] and first["counts"] == second["counts"]
    for r in second["results"]:
        if r["status"] == "CHECK":  # only what a change to mocks.yml settles
            assert "mocks.yml" in r["reason"], r
        assert RUN not in r["reason"] or "mocks.yml" in r["reason"], r
    out = runner.invoke(app, ["check"]).output
    assert "Next: run your agent with mock tools" not in out


def test_a_mocked_approval_gate_is_a_sign_off_after_the_run(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        textwrap.dedent("""
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def save_file(path: str, text: str) -> str:
            '''Save a file after a person approves.'''
            if input(f"save {path}? ") != "y":
                return "denied"
            with open(path, "w") as fh:
                fh.write(text)
            return "ok"

        agent = create_react_agent("openai:gpt-4o-2024-08-06", [save_file])
    """),
        encoding="utf-8",
    )
    spec = SPEC + "approval_required: [{tool: save_file, approver: lead, timeout: 4h, on_timeout: safe_stop}]\n"
    obs = _all_offered(tmp_path)
    report = run(mode="check", loaded=load_spec_text(spec.replace("approval_required: []\n", "")), observed=obs)
    report.metadata["mock_run"] = {"captured": 1, "total": 1, "stale": False}
    [ln] = [x for x in results(report) if x.param == "approval_required"]
    assert ln.status != "CHECK" and RUN not in ln.reason, ln
