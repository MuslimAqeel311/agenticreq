"""A run is evidence for `prohibited` and `approval_required`, so both lines can PASS on proof, never on a guess.

* prohibited: the model's tool list is captured in every run and the tool is never in it -> "held in N of N runs".
* approval_required: each time the model asked for the tool, an approval gate (input(), LangGraph interrupt(),
  OpenAI Agents needs_approval) fired before the tool's result reached the model -> "held in N of N runs".
  The result came back with no approval recorded -> FAIL "ran without a recorded approval".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture.discover import load_captures
from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.observed import CapturedCall, ToolCall, ToolResult, ToolSchema
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

FIX = Path(__file__).parent / "fixtures" / "runtime_proof"
URL = "https://api.openai.com/v1/chat/completions"


def _lines(report: object) -> dict[str, tuple[str, str]]:
    return {ln.param: (ln.status, ln.reason) for ln in results(report)}  # type: ignore[arg-type]


@pytest.mark.parametrize("variant,approval", [("gated", "PASS"), ("ungated", "GAP")])
def test_discover_run_decides_prohibited_and_approval(tmp_path: Path, variant: str, approval: str) -> None:
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    caps = tmp_path / "c.json"
    r = CliRunner().invoke(
        app,
        [
            "mock-run",
            "--entry",
            f"{FIX / variant / 'agent.py'}:main",
            "--mocks",
            str(tmp_path / "mocks.yml"),
            "--out",
            str(caps),
        ],
    )
    assert r.exit_code == 0, r.output
    obs = build_observed(FIX / variant, captures=load_captures(caps))
    rep = run(mode="check", loaded=load_spec_text((FIX / "spec.yml").read_text(), "spec.yml"), observed=obs)
    lines = _lines(rep)
    assert lines["prohibited"][0] == "PASS" and "Held in 1 of 1 runs" in lines["prohibited"][1]
    assert lines["approval_required"][0] == approval, lines["approval_required"]
    if approval == "CHECK":  # the run shows no gate fired: a gate agenticreq does not record waits for a sign-off
        assert "delete_record ran without a recorded approval" in lines["approval_required"][1]


def _call(
    i: int,
    offered: list[str],
    *,
    asks: str | None = None,
    results_for: list[str] = (),  # type: ignore[assignment]
    gates: list[str] = (),
    run_id: str = "run-0",
) -> CapturedCall:  # type: ignore[assignment]
    return CapturedCall(
        index=i,
        run_id=run_id,
        provider="openai_chat",
        url=URL,
        model="gpt-4o-2024-08-06",
        tools=[ToolSchema(name=n) for n in offered],
        synthetic=True,
        tool_calls=[ToolCall(name=asks, id=f"c_{asks}")] if asks else [],
        tool_results=[ToolResult(call_id=f"c_{n}", name=n) for n in results_for],
        gates_after=list(gates),
    )


def _check(captures: list[CapturedCall], variant: str = "gated") -> dict[str, tuple[str, str]]:
    obs = build_observed(FIX / variant, captures=captures)
    return _lines(run(mode="check", loaded=load_spec_text((FIX / "spec.yml").read_text(), "spec.yml"), observed=obs))


TOOLS = ["search_records", "delete_record"]


def test_an_interrupt_that_stops_the_run_counts_as_held() -> None:
    lines = _check([_call(0, TOOLS, asks="delete_record", gates=["interrupt"])])
    assert lines["approval_required"][0] == "PASS"


def test_a_run_that_never_asks_for_the_tool_is_no_evidence() -> None:
    lines = _check([_call(0, TOOLS, asks="search_records"), _call(1, TOOLS, results_for=["search_records"])])
    assert lines["approval_required"][0] != "PASS"
    assert "Held in" not in lines["approval_required"][1]


def test_no_gate_and_no_result_is_no_evidence() -> None:
    lines = _check([_call(0, TOOLS, asks="delete_record")])
    assert "Held in" not in lines["approval_required"][1]
    assert "ran without a recorded approval" not in lines["approval_required"][1]


def test_one_ungated_run_out_of_two_fails_the_line() -> None:
    lines = _check(
        [
            _call(0, TOOLS, asks="delete_record", gates=["human_input"]),
            _call(1, TOOLS, results_for=["delete_record"]),
            _call(0, TOOLS, asks="delete_record", run_id="run-1"),
            _call(1, TOOLS, results_for=["delete_record"], run_id="run-1"),
        ]
    )
    assert lines["approval_required"][0] in ("GAP", "CHECK")
    assert "1 of 2 run(s)" in lines["approval_required"][1]


def test_a_prohibited_tool_offered_on_the_wire_still_fails() -> None:
    lines = _check([_call(0, [*TOOLS, "wire_transfer"])])
    assert lines["prohibited"][0] == "GAP"


def test_without_runs_prohibited_stays_a_guess() -> None:
    lines = _check([])
    assert lines["prohibited"][0] == "CHECK" and "Code pattern only" in lines["prohibited"][1]
    assert lines["prohibited"][1].endswith("agenticreq mock-run")


def test_gate_probe_reaches_names_imported_by_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework that did `from x import approve` is still probed; a "true" probe records only a True answer."""
    import asyncio
    import sys

    from agenticreq.capture import gates
    from agenticreq.capture.hook import CaptureSession

    (tmp_path / "fakefw").mkdir()
    (tmp_path / "fakefw" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "fakefw" / "hitl.py").write_text(
        "def ask(x):\n    return x\n\nasync def needs(x):\n    return x\n", encoding="utf-8"
    )
    (tmp_path / "fakefw" / "runner.py").write_text("from fakefw.hitl import ask, needs\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import fakefw.runner as runner  # imported BEFORE the probe, like a framework's internals

    monkeypatch.setattr(
        gates, "PROBES", (("fakefw.hitl", "ask", "ask", "call"), ("fakefw.hitl", "needs", "needs", "true"))
    )
    session = CaptureSession(None)
    session.calls.append(_call(0, TOOLS))
    probe = gates.GateProbe(session)
    probe.install()
    try:
        runner.ask(1)
        asyncio.run(runner.needs(False))  # no approval needed: not a gate
        asyncio.run(runner.needs(True))
        assert input("approve? ") == "yes"
    finally:
        probe.uninstall()
    assert session.calls[0].gates_after == ["ask", "needs", "human_input"]
    assert runner.ask is sys.modules["fakefw.hitl"].ask and not hasattr(runner.ask, "__wrapped__")  # undone
