"""Every number comes from the user: it fails only when it is missing or when code and spec.yml disagree.

Three layers, every one named so a failure shows parameter, risk level and row of the decision table
(docs/numbers.md):

* ``test_decision_table``  - the pure decision (thresholds.judge) for EVERY parameter x EVERY risk x EVERY row;
* ``test_end_to_end``      - the same rows through real code + spec.yml + the engine, wherever the rule can see them;
* the key cases, the yes/no floors, the shape errors, ``init``, ``params`` and determinism.

The expected texts below are written out here, not read from the code under test. The library proposes no number.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Callable

import pytest
from typer.testing import CliRunner

from agenticreq import thresholds as th
from agenticreq.cli import app
from agenticreq.defaults import _FLOORS
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.observed import CapturedCall, ToolCall
from agenticreq.models.verdict import ResultLine, results
from agenticreq.observe import build_observed

RISKS = ("low", "medium", "high", "critical")
DOCS = Path(__file__).parent.parent / "docs" / "numbers.md"
NOWHERE = ": write a value, or not_needed with a reason."

#: the name a reason uses
NAME = {
    "bounds.max_steps": "max_steps",
    "bounds.max_cost_usd": "max_cost_usd",
    "bounds.max_duration_s": "max_duration_s",
    "bounds.max_retries": "max_retries",
    "bounds.max_tool_calls": "max_tool_calls",
    "max_delegation_depth": "max_delegation_depth",
    "progress_guard.max_no_progress_steps": "max_no_progress_steps",
    "output_size_limits.max_chars": "max_chars",
    "confidence_threshold": "confidence_threshold",
    "rate_limits": "rate_limits",
    "aggregate_budget.max_cost_usd": "aggregate_budget.max_cost_usd",
    "aggregate_budget.max_steps": "aggregate_budget.max_steps",
    "aggregate_budget.max_tool_calls": "aggregate_budget.max_tool_calls",
    "approval_required[].timeout": "approval_required[].timeout",
}
#: does the library look for this limit in code (so "code sets no limit" is something it established)?
LOOKED = {
    "bounds.max_steps": True,
    "bounds.max_retries": True,
    "bounds.max_tool_calls": True,
    "max_delegation_depth": True,
    "aggregate_budget.max_steps": True,
}
#: spec value, a code value at least as strict, a code value that is looser
VALUES: dict[str, tuple[object, object, object]] = {
    "bounds.max_steps": (40, 30, 80),
    "bounds.max_cost_usd": (5.0, 4.0, 8.0),
    "bounds.max_duration_s": (600, 400, 900),
    "bounds.max_retries": (3, 2, 6),
    "bounds.max_tool_calls": (40, 30, 80),
    "max_delegation_depth": (3, 2, 5),
    "progress_guard.max_no_progress_steps": (5, 3, 9),
    "output_size_limits.max_chars": (5000, 4000, 9000),
    "confidence_threshold": (0.5, 0.7, 0.2),
    "rate_limits": (30, 20, 50),
    "aggregate_budget.max_cost_usd": (5.0, 4.0, 8.0),
    "aggregate_budget.max_steps": (60, 40, 90),
    "aggregate_budget.max_tool_calls": (60, 40, 90),
    "approval_required[].timeout": ("4h", None, None),
}


def fmt(v: object) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def rec_suffix(key: str, risk: str) -> str:
    return ""  # a result never shows a number the library picked


def unreadable(spec: object, code_present: bool = False) -> str:
    if code_present:  # a limit of this kind is in code, but its value is not readable
        return (
            f"spec.yml says {fmt(spec)}; the code sets this limit from a value agenticreq cannot read (or sign it "
            "off under confirmed:). Run: agenticreq mock-run"
        )
    return (
        f"spec.yml says {fmt(spec)}; not readable in code (or sign it off under confirmed:). Run: agenticreq mock-run"
    )


#: what a number counts, in "Code allows 80 steps; spec.yml says 40."
UNIT = {
    "bounds.max_steps": " steps",
    "bounds.max_retries": " retries",
    "bounds.max_tool_calls": " tool calls",
    "aggregate_budget.max_steps": " steps",
}
#: the line a sign-off under confirmed: gives (ATT below)
SIGNED_BY = "Signed by lead@example.com on 2026-10-01."


def looser(key: str, code: object, spec: object) -> str:
    return f"Code allows {fmt(code)}{UNIT.get(key, '')}; spec.yml says {fmt(spec)}."


# --------------------------------------------------------------------------- layer 1: the decision table
ROWS = (
    "spec_and_code_match",
    "code_only",
    "spec_and_code_looser",
    "spec_only",
    "spec_only_code_unreadable",
    "nowhere",
    "spec_is_todo",
    "unreadable_confirmed",
    "unreadable_seen_in_runs",
)


def _decision(key: str, risk: str, row: str) -> tuple[dict[str, object], str, str] | None:
    """(judge kwargs, expected status, expected reason), or None when the row cannot exist for this parameter."""
    spec, ok, bad = VALUES[key]
    name = NAME[key]
    if row in ("spec_and_code_match", "code_only", "spec_and_code_looser") and ok is None:
        return None  # a timeout is text: code cannot hold a comparable value
    if row == "spec_and_code_match":
        return dict(spec=spec, code=ok), "PASS", f"Found in code: {fmt(ok)}{rec_suffix(key, risk)}."
    if row == "code_only":
        return dict(code=ok), "PASS", f"Found in code: {fmt(ok)}{rec_suffix(key, risk)}."
    if row == "spec_and_code_looser":
        return dict(spec=spec, code=bad), "FAIL", looser(key, bad, spec)
    if row == "spec_only":
        if LOOKED.get(key):
            return dict(spec=spec), "FAIL", f"spec.yml says {fmt(spec)}; the code sets no limit."
        return dict(spec=spec), "FAIL", unreadable(spec)
    if row == "spec_only_code_unreadable":
        return dict(spec=spec, code_present=True), "FAIL", unreadable(spec, True)
    if row == "nowhere":
        return dict(), "FAIL", f"No {name} set{NOWHERE}"
    if row == "spec_is_todo":
        return dict(spec_todo=True), "FAIL", f"{name} is still TODO in spec.yml."
    if row == "unreadable_confirmed":
        return dict(spec=spec, code_present=True, confirmed=True), "PASS", "Signed by you."
    return dict(spec=spec, code_present=True, observed_ok=True), "PASS", "Seen when the agent ran."


DECISIONS = [
    pytest.param(key, risk, row, id=f"{key}|{risk}|{row}")
    for key in NAME
    for risk in RISKS
    for row in ROWS
    if _decision(key, risk, row) is not None
]


@pytest.mark.parametrize("key,risk,row", DECISIONS)
def test_decision_table(key: str, risk: str, row: str) -> None:
    kwargs, status, reason = _decision(key, risk, row)  # type: ignore[misc]
    d = th.judge(th.BY_KEY[key], risk, **kwargs)  # type: ignore[arg-type]
    assert (d.status, d.reason) == (status, reason), f"{key} @ {risk}: {row}"
    readable = reason not in (unreadable(VALUES[key][0]), unreadable(VALUES[key][0], True))
    assert d.kind == (None if status == "PASS" else "missing" if readable else "not_proven")
    # the report group: a number that is wrong or missing is a gap, one code cannot show waits for a mock run
    group = {"PASS": "PASS", "FAIL": "GAP" if readable else "CHECK"}[status]
    assert d.line_status == ("SIGNED" if row == "unreadable_confirmed" else group), f"{key} @ {risk}: {row}"


def test_every_parameter_is_in_the_task_list() -> None:
    assert set(th.BY_KEY) == set(NAME)  # 9 numbers + rate_limits + aggregate_budget (3) + approval timeout


@pytest.mark.parametrize("key", sorted(NAME))
@pytest.mark.parametrize("risk", RISKS)
def test_no_parameter_has_a_recommended_value(key: str, risk: str) -> None:
    d = th.judge(th.BY_KEY[key], risk)
    assert d.reason == f"No {NAME[key]} set{NOWHERE}" and "recommended" not in d.reason


def test_stricter_means_lower_for_max_and_higher_for_confidence() -> None:
    steps, conf = th.BY_KEY["bounds.max_steps"], th.BY_KEY["confidence_threshold"]
    assert th.judge(steps, "high", spec=40, code=40).status == "PASS"  # equal is at least as strict
    assert th.judge(steps, "high", spec=40, code=41).status == "FAIL"
    assert th.judge(conf, "high", spec=0.5, code=0.5).status == "PASS"
    assert th.judge(conf, "high", spec=0.5, code=0.49).status == "FAIL"


def test_the_numbers_page_says_the_library_suggests_none_and_lists_no_value_table() -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert "**agenticreq never suggests numbers.**" in text and "Found in code: <value>." in text
    assert "No <name> set: write a value, or not_needed with a reason." in text
    assert "| low | medium | high | critical |" not in text  # no per-risk table of values


def test_numeric_floors_are_gone_and_safeguard_floors_stay() -> None:
    keys = {k for floors in _FLOORS.values() for k, _ in floors}
    assert not {k for k in keys if k.startswith("bounds.") or k == "max_delegation_depth"}
    assert {
        "self_modification",
        "fail_closed",
        "on_approval_timeout",
        "kill_switch",
        "log_immutable",
        "version_pinning",
        "audit_log",
        "dry_run_required",
        "transitive_prohibition",
    } <= keys


# --------------------------------------------------------------------------- layer 2: through code + spec + engine
SPEC = """agent_id: a
risk_level: {risk}
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: []
approval_required: {approval}
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
{extra}
"""
ATT = 'confirmed:\n  - {{clause: "{c}", by: lead@example.com, date: 2026-10-01, evidence: T-1, expires: 2099-01-01}}\n'
TIMEOUT = "[{tool: lookup_order, approver: lead, timeout: 4h, on_timeout: safe_stop}]"

OAI = '''
from agents import Agent, Runner, function_tool
{module}
@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status."""
    return "x"
support = Agent(name="support", instructions="i", tools=[lookup_order], model="gpt-4o-2024-08-06")
def main(q):
    return Runner.run_sync(support, q{kwargs}).final_output
'''
LANGGRAPH = '''
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
@tool
def lookup_order(order_id: str) -> str:
    """Return the status."""
    return "x"
graph = create_react_agent(ChatOpenAI(model="gpt-4o-2024-08-06"), [lookup_order])
def main(q):
    return graph.invoke({"messages": [("user", q)]})
'''
CREW = '''
from crewai import Agent
from crewai.tools import tool
@tool("lookup_order")
def lookup_order(order_id: str) -> str:
    """Return the status."""
    return "x"
a = Agent(role="r", goal="g", backstory="b", tools=[lookup_order], max_iter=5, max_execution_time={t}, llm="gpt-4o-2024-08-06")
'''
HANDOFFS = '''
from agents import Agent, Runner, function_tool
@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status."""
    return "x"
expert = Agent(name="expert", instructions="i", tools=[lookup_order], model="gpt-4o-2024-08-06")
specialist = Agent(name="specialist", instructions="i", handoffs=[expert], model="gpt-4o-2024-08-06")
triage = Agent(name="triage", instructions="i", handoffs=[specialist], model="gpt-4o-2024-08-06")
def main(q):
    return Runner.run_sync(triage, q, max_turns=10).final_output
'''


def oai(turns: int | None = 30, module: str = "", **kw: int) -> str:
    kwargs = "".join(f", {k}={v}" for k, v in (({"max_turns": turns} if turns else {}) | kw).items())
    return OAI.format(module=module, kwargs=kwargs)


def lines_of(tmp: Path, src: str, extra: str | None, risk: str, approval: str = "[]") -> list[ResultLine]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    obs = build_observed(tmp)
    if extra is None:  # audit: no spec, so risk_level medium
        return results(run(mode="audit", observed=obs, today=date(2026, 10, 15)))
    loaded = load_spec_text(SPEC.format(risk=risk, approval=approval, extra=extra))
    return results(run(mode="check", loaded=loaded, observed=obs, today=date(2026, 10, 15)))


Expect = tuple[str, str] | None  # (status, reason), or None = no line for this parameter
Scenario = tuple[str, str | None, Expect, str]  # (source, spec extra, expected, spec approval_required)


def _scn(key: str) -> dict[str, Callable[[str], Scenario]]:
    """row -> builder(risk) -> scenario, for the rows this parameter's rule can produce end to end."""
    spec, ok, bad = VALUES[key]
    name = NAME[key]
    out: dict[str, Callable[[str], Scenario]] = {}

    def row(r: str, src: str, extra: str | None, status: str, reason: str, approval: str = "[]") -> None:
        out[r] = lambda risk, s=src, e=extra, st=status, rs=reason, ap=approval: (s, e, (st, rs) if st else None, ap)

    def gated(r: str, src: str, extra: str | None, approval: str = "[]") -> None:
        out[r] = lambda risk, s=src, e=extra, ap=approval: (s, e, None, ap)

    def todo(top: str, leaf: str | None) -> tuple[str, str]:
        body = f"{top}: {{{leaf}: TODO}}" if leaf else f"{top}: TODO"
        return body, f"{name} is still TODO in spec.yml."

    def attest(c: str) -> str:
        return ATT.format(c=c)

    if key == "bounds.max_steps":
        row("spec_and_code_match", oai(30), f"bounds: {{max_steps: {spec}}}", "PASS", "")
        row("code_only", oai(30), "", "PASS", "")
        row("spec_and_code_looser", oai(80), f"bounds: {{max_steps: {spec}}}", "FAIL", looser(key, "80", spec))
        row(
            "spec_only",
            LANGGRAPH,
            f"bounds: {{max_steps: {spec}}}",
            "FAIL",
            f"spec.yml says {spec}; the code sets no limit.",
        )
        row("nowhere", LANGGRAPH, "", "FAIL", "")
        b, r = todo("bounds", "max_steps")
        row("spec_is_todo", oai(30), b, "FAIL", r)
    elif key in ("bounds.max_retries", "bounds.max_tool_calls"):
        kw = "max_retries" if key.endswith("retries") else "max_tool_calls"
        mod = "from openai import AsyncOpenAI\nclient = AsyncOpenAI(max_retries={n})\n"
        src = (
            (lambda n: oai(30, module=mod.format(n=n)))
            if kw == "max_retries"
            else (lambda n: oai(30, max_tool_calls=n))
        )
        row("spec_and_code_match", src(ok), f"bounds: {{{kw}: {spec}}}", "PASS", "")
        row("code_only", src(ok), "", "PASS", "")
        row("spec_and_code_looser", src(bad), f"bounds: {{{kw}: {spec}}}", "FAIL", looser(key, bad, spec))
        row("spec_only", oai(30), f"bounds: {{{kw}: {spec}}}", "FAIL", f"spec.yml says {spec}; the code sets no limit.")
        gated("nowhere", oai(30), "")  # these two never had a risk_level requirement: no number, no check
        b, r = todo("bounds", kw)
        row("spec_is_todo", oai(30), b, "FAIL", r)
    elif key == "bounds.max_cost_usd":
        pyd = (
            "from pydantic_ai import Agent\nfrom pydantic_ai.usage import UsageLimits\n"
            'agent = Agent("openai:gpt-4o-2024-08-06")\n@agent.tool_plain\ndef lookup_order(order_id: str) -> str:\n'
            '    """Return the status."""\n    return "x"\n'
            "def main(q):\n    return agent.run_sync(q, usage_limits=UsageLimits(total_tokens_limit=5000))\n"
        )
        row("spec_only", oai(30), "bounds: {max_cost_usd: 5}", "FAIL", unreadable(5))
        row("spec_only_code_unreadable", pyd, "bounds: {max_cost_usd: 5}", "FAIL", unreadable(5, True))
        row(
            "unreadable_confirmed",
            oai(30),
            "bounds: {max_cost_usd: 5}\n" + attest("bounds.max_cost_usd"),
            "SIGNED",
            SIGNED_BY,
        )
        gated("nowhere", oai(30), "")
        b, r = todo("bounds", "max_cost_usd")
        row("spec_is_todo", oai(30), b, "FAIL", r)
    elif key == "bounds.max_duration_s":
        row("spec_and_code_match", CREW.format(t=400), f"bounds: {{max_duration_s: {spec}}}", "PASS", "")
        row("code_only", CREW.format(t=400), "", "PASS", "")
        row(
            "spec_and_code_looser",
            CREW.format(t=900),
            f"bounds: {{max_duration_s: {spec}}}",
            "FAIL",
            looser(key, "900", spec),
        )
        row("spec_only", oai(30), f"bounds: {{max_duration_s: {spec}}}", "FAIL", unreadable(spec))
        row(
            "unreadable_confirmed",
            oai(30),
            f"bounds: {{max_duration_s: {spec}}}\n" + attest("max_duration_s"),
            "SIGNED",
            SIGNED_BY,
        )
        gated("nowhere", oai(30), "")
        b, r = todo("bounds", "max_duration_s")
        row("spec_is_todo", oai(30), b, "FAIL", r)
    elif key == "max_delegation_depth":
        row("spec_and_code_match", HANDOFFS, f"max_delegation_depth: {spec}", "PASS", "")  # depth 2
        row("code_only", HANDOFFS, "", "PASS", "")
        row("spec_and_code_looser", HANDOFFS, "max_delegation_depth: 1", "FAIL", looser(key, 2, 1))
        b, r = todo("max_delegation_depth", None)
        row("spec_is_todo", HANDOFFS, b, "FAIL", r)
    elif key in (
        "progress_guard.max_no_progress_steps",
        "output_size_limits.max_chars",
        "confidence_threshold",
        "rate_limits",
    ):
        top, _, leaf = key.partition(".")
        body = {
            "rate_limits": "rate_limits: {requests_per_minute: 30}",
            "confidence_threshold": "confidence_threshold: 0.5",
        }.get(key, f"{top}: {{{leaf}: {spec}}}")
        shown = {"rate_limits": 30, "confidence_threshold": 0.5}.get(key, spec)
        clause = {"rate_limits": "rate_limits", "confidence_threshold": "confidence_threshold"}.get(key, top)
        row("spec_only", oai(30), body, "FAIL", unreadable(shown))
        row("unreadable_confirmed", oai(30), body + "\n" + attest(clause), "SIGNED", SIGNED_BY)
        b, r = (
            todo(top, leaf or None)
            if key != "rate_limits"
            else ("rate_limits: {requests_per_minute: TODO}", "rate_limits is still TODO in spec.yml.")
        )
        row("spec_is_todo", oai(30), b, "FAIL", r)
        if key == "progress_guard.max_no_progress_steps":  # replaced the "built-in safeguard" confirmation
            row("nowhere", oai(30), "", "FAIL", "")
        elif key == "confidence_threshold":  # only where a risk level recommends a gate
            out["nowhere"] = lambda risk: (oai(30), "", ("FAIL", "") if risk == "critical" else None, "[]")
        else:
            gated("nowhere", oai(30), "")
    elif key.startswith("aggregate_budget."):
        leaf = key.split(".")[1]
        agg_spec = {"max_cost_usd": 5, "max_steps": 60, "max_tool_calls": 60}[leaf]
        body = f"aggregate_budget: {{{leaf}: {agg_spec}}}"
        if leaf == "max_steps":
            row("spec_and_code_match", HANDOFFS, body, "PASS", "Found in code: 10.")  # the run-level max_turns=10
            row("spec_and_code_looser", HANDOFFS, "aggregate_budget: {max_steps: 5}", "FAIL", looser(key, 10, 5))
        else:
            row("spec_only", HANDOFFS, body, "FAIL", unreadable(agg_spec))
            row("unreadable_confirmed", HANDOFFS, body + "\n" + attest(key), "SIGNED", SIGNED_BY)
        row(
            "spec_is_todo",
            HANDOFFS,
            f"aggregate_budget: {{{leaf}: TODO}}",
            "FAIL",
            f"{name} is still TODO in spec.yml.",
        )
        gated("nowhere", HANDOFFS, "")  # no aggregate_budget at all: AS-306 reports that, not a missing number
    elif key == "approval_required[].timeout":
        row("spec_only", HANDOFFS, "", "FAIL", unreadable("4h"), TIMEOUT)
        row("unreadable_confirmed", HANDOFFS, attest("approval_required[].timeout"), "SIGNED", SIGNED_BY, TIMEOUT)
        row(
            "spec_is_todo",
            HANDOFFS,
            "",
            "FAIL",
            "approval_required[].timeout is still TODO in spec.yml.",
            "[{tool: lookup_order, approver: lead, timeout: TODO, on_timeout: safe_stop}]",
        )
        gated("nowhere", HANDOFFS, "", "[{tool: lookup_order, approver: lead}]")
    return out


def _line(ls: list[ResultLine], key: str) -> ResultLine | None:
    got = [ln for ln in ls if ln.param == key]
    return got[0] if got else None


E2E = [pytest.param(key, risk, row, id=f"{key}|{risk}|{row}") for key in NAME for risk in RISKS for row in _scn(key)]


@pytest.mark.parametrize("key,risk,row", E2E)
def test_end_to_end(tmp_path: Path, key: str, risk: str, row: str) -> None:
    src, extra, expect, approval = _scn(key)[row](risk)
    got = _line(lines_of(tmp_path, src, extra, risk, approval), key)
    if expect is None:
        assert got is None, f"{key} @ {risk}: {row} must produce no line, got {got}"
        return
    status, reason = expect
    spec, ok, bad = VALUES[key]
    if not reason:  # the reasons that carry the recommendation, or a code value, are built from the table
        if row in ("spec_and_code_match", "code_only"):
            code_val = {
                "bounds.max_steps": 30,
                "bounds.max_retries": ok,
                "bounds.max_tool_calls": ok,
                "bounds.max_duration_s": 400,
                "max_delegation_depth": 2,
            }[key]
            reason = f"Found in code: {fmt(code_val)}{rec_suffix(key, risk)}."
        else:  # nowhere
            reason = f"No {NAME[key]} set{NOWHERE}"
    if status == "FAIL":  # the group: a number code cannot show waits for a mock run; anything else is a gap
        unreadable_value = "not readable" in reason or "cannot read" in reason
        status = "CHECK" if reason.startswith("spec.yml says") and unreadable_value else "GAP"
    assert got is not None, f"{key} @ {risk}: {row}: no line"
    assert (got.status, got.reason) == (status, reason), f"{key} @ {risk}: {row}"
    assert got.group == {"GAP": "gap", "CHECK": "mock_run", "SIGNED": "sign_off", "PASS": "passed"}[status]


def test_audit_without_a_spec_asks_for_a_cost_and_duration_limit(tmp_path: Path) -> None:
    ls = lines_of(tmp_path, oai(30), None, "medium")
    for key, rec in (("bounds.max_cost_usd", "5"), ("bounds.max_duration_s", "600")):
        ln = _line(ls, key)
        assert ln is not None and ln.status == "GAP"
        assert ln.reason == f"No {NAME[key]} set{NOWHERE}"


# --------------------------------------------------------------------------- the key cases
def test_high_risk_code_120_without_a_spec_value_passes(tmp_path: Path) -> None:
    ls = lines_of(tmp_path, oai(120), "", "high")
    assert (_line(ls, "bounds.max_steps").status, _line(ls, "bounds.max_steps").reason) == (  # type: ignore[union-attr]
        "PASS",
        "Found in code: 120.",
    )


def test_critical_spec_200_and_code_200_passes(tmp_path: Path) -> None:
    ls = lines_of(tmp_path, oai(200), "bounds: {max_steps: 200}", "critical")
    ln = _line(ls, "bounds.max_steps")
    assert ln is not None and ln.status == "PASS" and ln.reason == "Found in code: 200."


def test_spec_40_code_80_fails_naming_both(tmp_path: Path) -> None:
    ln = _line(lines_of(tmp_path, oai(80), "bounds: {max_steps: 40}", "medium"), "bounds.max_steps")
    assert ln is not None and (ln.status, ln.reason) == ("GAP", "Code allows 80 steps; spec.yml says 40.")


def test_langgraph_without_recursion_limit_is_missing(tmp_path: Path) -> None:
    ln = _line(lines_of(tmp_path, LANGGRAPH, "", "high"), "bounds.max_steps")
    assert ln is not None and ln.status == "GAP" and ln.group == "gap"
    assert ln.reason == "No max_steps set: write a value, or not_needed with a reason."


def test_cost_in_spec_is_not_proven_until_it_is_attested(tmp_path: Path) -> None:
    before = _line(lines_of(tmp_path, oai(30), "bounds: {max_cost_usd: 5}", "high"), "bounds.max_cost_usd")
    assert before is not None and before.status == "CHECK" and before.group == "mock_run"
    assert before.reason == unreadable(5)
    after = _line(
        lines_of(tmp_path, oai(30), "bounds: {max_cost_usd: 5}\n" + ATT.format(c="bounds.max_cost_usd"), "high"),
        "bounds.max_cost_usd",
    )
    assert after is not None and (after.status, after.reason) == ("SIGNED", SIGNED_BY)
    assert after.group == "sign_off"  # signed, never counted as passed


def test_an_expired_attestation_does_not_confirm(tmp_path: Path) -> None:
    att = ATT.format(c="bounds.max_cost_usd").replace("2099-01-01", "2026-01-01")
    ln = _line(lines_of(tmp_path, oai(30), "bounds: {max_cost_usd: 5}\n" + att, "high"), "bounds.max_cost_usd")
    assert ln is not None and ln.status == "SIGN-OFF"  # the sign-off expired: open again, never passed
    assert ln.reason.startswith("Signed before, but the sign-off for bounds.max_cost_usd by lead@example.com expired")


@pytest.mark.parametrize("risk", RISKS)
def test_any_value_is_valid_at_any_risk_level(tmp_path: Path, risk: str) -> None:
    extra = (
        "bounds: {max_steps: 500, max_cost_usd: 900.0, max_duration_s: 99999, max_retries: 50, "
        "max_tool_calls: 700}\nmax_delegation_depth: 9\nprogress_guard: {max_no_progress_steps: 40}\n"
        "output_size_limits: {max_chars: 900000}\n"
    )
    loaded = load_spec_text(SPEC.format(risk=risk, approval="[]", extra=extra))
    assert loaded.status != "invalid"
    (tmp_path / "agent.py").write_text(oai(500), encoding="utf-8")
    rep = run(mode="check", loaded=loaded, observed=build_observed(tmp_path), today=date(2026, 10, 15))
    assert not [f for f in rep.findings if f.rule_id == "AS-111"], f"a number is not a floor at {risk}"
    ln = _line(results(rep), "bounds.max_steps")
    assert ln is not None and ln.status == "PASS"  # far from the recommendation is never a FAIL by itself


# --------------------------------------------------------------------------- yes/no floors still work
@pytest.mark.parametrize("risk", ("high", "critical"))
def test_the_yes_no_floors_still_fail(risk: str) -> None:
    loaded = load_spec_text(SPEC.format(risk=risk, approval="[]", extra="kill_switch: false\n"))
    rep = run(mode="validate", loaded=loaded, groups=["validate"])
    floors = [f for f in rep.findings if f.rule_id == "AS-111"]
    assert floors and "kill_switch" in floors[0].detail and floors[0].severity == "blocker"
    assert rep.exit_code == 1


def test_the_yes_no_floors_in_a_check_report_fail(tmp_path: Path) -> None:
    loaded = load_spec_text(SPEC.format(risk="high", approval="[]", extra="kill_switch: false\n"))
    (tmp_path / "agent.py").write_text(oai(10), encoding="utf-8")
    rep = run(mode="check", loaded=loaded, observed=build_observed(tmp_path), today=date(2026, 10, 15))
    ln = _line(results(rep), "risk_level")  # AS-111 is filed under risk_level
    assert ln is not None and ln.status == "GAP" and "kill_switch=False" in ln.reason
    assert rep.exit_code == 1


# --------------------------------------------------------------------------- shape errors
@pytest.mark.parametrize(
    "bad",
    [
        "bounds: {max_steps: 0}",
        "bounds: {max_steps: -5}",
        "bounds: {max_cost_usd: 0}",
        "bounds: {max_duration_s: -1}",
        "bounds: {max_retries: 0}",
        "bounds: {max_tool_calls: 0}",
        "bounds: {max_steps: abc}",
        "bounds: {max_steps: 2.5}",
        "max_delegation_depth: 0",
        "progress_guard: {max_no_progress_steps: 0}",
        "output_size_limits: {max_chars: -3}",
        "aggregate_budget: {max_steps: 0}",
        "aggregate_budget: {max_cost_usd: -1}",
        "confidence_threshold: 1.5",
        "confidence_threshold: -0.1",
        "confidence_threshold: high",
    ],
)
def test_a_number_that_cannot_be_a_limit_is_a_spec_shape_error(tmp_path: Path, bad: str) -> None:
    p = tmp_path / "spec.yml"
    p.write_text(SPEC.format(risk="high", approval="[]", extra=bad), encoding="utf-8")
    (tmp_path / "agent.py").write_text(oai(10), encoding="utf-8")
    res = CliRunner().invoke(app, ["check", str(tmp_path), "--spec", str(p)])
    assert res.exit_code == 2, (bad, res.output)
    assert "SPEC INVALID" in res.output


@pytest.mark.parametrize(
    "good",
    [
        "bounds: {max_steps: 1}",
        "confidence_threshold: 0",
        "confidence_threshold: 1",
        "bounds: {max_cost_usd: 0.01}",
        "max_delegation_depth: 1",
    ],
)
def test_the_smallest_valid_numbers_load(good: str) -> None:
    assert load_spec_text(SPEC.format(risk="critical", approval="[]", extra=good)).status != "invalid"


# --------------------------------------------------------------------------- runs (observed within the limit)
def _call(i: int, tools: list[tuple[str, dict]], t: float) -> CapturedCall:  # type: ignore[type-arg]
    return CapturedCall(
        index=i,
        run_id="r1",
        provider="openai_chat",
        url="https://api.openai.com/v1/chat",
        model="gpt-4o-2024-08-06",
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in tools],
        timestamp=t,
        synthetic=False,
    )


LANG_TOOL = (
    "from langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
    "@tool\ndef search_docs(q: str) -> str:\n    '''Search.'''\n    return q\n"
    "llm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([search_docs])\n"
)
CLEAN = [
    _call(0, [("search_docs", {"q": "a"})], 0.0),
    _call(1, [("search_docs", {"q": "b"})], 30.0),
    _call(2, [], 61.0),
]
LOOP = [_call(i, [("search_docs", {"q": "x"})], 10.0 * i) for i in range(4)] + [_call(4, [], 40.0)]


@pytest.mark.parametrize("risk", RISKS)
@pytest.mark.parametrize(
    "extra,calls,status,startswith,param",
    [
        (
            "progress_guard: {max_no_progress_steps: 2}",
            CLEAN,
            "PASS",
            "Held in 1 of 1 runs",
            "progress_guard.max_no_progress_steps",
        ),
        ("rate_limits: {requests_per_minute: 3}", CLEAN, "PASS", "Held in 1 of 1 runs", "rate_limits"),
        ("bounds: {max_steps: 6, max_tool_calls: 5}", CLEAN, "PASS", "Found in code", "bounds.max_tool_calls"),
        ("progress_guard: {max_no_progress_steps: 2}", LOOP, "GAP", "", "progress_guard.max_no_progress_steps"),
    ],
)
def test_runs_prove_a_number_that_code_cannot_show(
    tmp_path: Path, risk: str, extra: str, calls: list[CapturedCall], status: str, startswith: str, param: str
) -> None:
    (tmp_path / "agent.py").write_text(LANG_TOOL, encoding="utf-8")
    rep = run(
        mode="check",
        loaded=load_spec_text(SPEC.format(risk=risk, approval="[]", extra=extra)),
        observed=build_observed(tmp_path, captures=calls),
        today=date(2026, 10, 15),
    )
    ln = _line(results(rep), param)
    if param == "bounds.max_tool_calls":  # measured against spec.yml's number, whatever the risk level recommends
        seen = [p for p in rep.passed if p.rule_id == "AS-229" and p.param == param]
        assert seen and not [f for f in rep.findings if f.rule_id == "AS-229"]
        return
    assert ln is not None and ln.status == status and ln.reason.startswith(startswith), (param, ln)


# --------------------------------------------------------------------------- several agents: per agent
def test_each_agent_is_judged_on_its_own_bound(tmp_path: Path) -> None:
    src = CREW.format(t=60).replace(
        "a = Agent(",
        "reporter = Agent(role='reporter', goal='g', backstory='b', tools=[], "
        "max_iter=40, llm='gpt-4o-2024-08-06')\na = Agent(",
    )
    ls = lines_of(tmp_path, src, "bounds: {max_steps: 20}", "high")
    steps = [ln for ln in ls if ln.param == "bounds.max_steps"]
    by_agent = {ln.agent: (ln.status, ln.reason) for ln in steps}
    assert len(by_agent) == 2, by_agent
    assert ("PASS", "Found in code: 5.") in by_agent.values()
    assert ("GAP", "Code allows 40 steps; spec.yml says 20.") in by_agent.values()


def test_an_agent_without_its_own_bound_has_none(tmp_path: Path) -> None:
    src = (
        "from crewai import Agent\nfrom crewai.tools import tool\n@tool('t1')\ndef t1(x: str) -> str:\n"
        "    '''One.'''\n    return x\n"
        "a = Agent(role='a', goal='g', backstory='b', tools=[t1], max_iter=5, llm='gpt-4o-2024-08-06')\n"
        "b = Agent(role='b', goal='g', backstory='b', tools=[t1], llm='gpt-4o-2024-08-06')\n"
    )
    ls = [ln for ln in lines_of(tmp_path, src, "", "high") if ln.param == "bounds.max_steps"]
    by_agent = {ln.agent: ln.status for ln in ls}
    assert by_agent.get("a") == "PASS", by_agent


# --------------------------------------------------------------------------- init, params, determinism
def test_init_writes_no_number_only_a_comment_naming_them(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(oai(30), encoding="utf-8")
    out = tmp_path / "spec.yml"
    res = CliRunner().invoke(app, ["init", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8")
    assert "recommended" not in text and "# Numbers: write your own value for each one you need" in text
    for key in ("max_steps", "max_cost_usd", "max_duration_s", "max_retries", "max_tool_calls"):
        assert not re.search(rf"^\s*{key}:", text, re.M), key  # named in a comment only, never as a key
    assert load_spec_text(text).status != "invalid"  # still loads: only the user's own decisions are TODO


def test_params_show_what_is_valid_never_a_value() -> None:
    res = CliRunner().invoke(app, ["params", "--field", "bounds", "--brief"])
    assert res.exit_code == 0
    assert "bounds.max_steps: you write it (a positive number), or not_needed with a reason" in " ".join(
        res.output.split()
    )
    assert "recommended" not in res.output
    data = json.loads(CliRunner().invoke(app, ["params", "--field", "bounds", "-f", "json"]).output)
    assert "recommended" not in data[0]


def test_json_output_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(HANDOFFS, encoding="utf-8")
    spec = tmp_path / "spec.yml"
    spec.write_text(
        SPEC.format(
            risk="high",
            approval=TIMEOUT,
            extra="bounds: {max_steps: 20, max_cost_usd: 5}\n"
            "aggregate_budget: {max_steps: 60}\nmax_delegation_depth: 3\n",
        ),
        encoding="utf-8",
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run(
            [sys.executable, "-m", "agenticreq.cli", "check", str(tmp_path), "--spec", str(spec), "-f", "json"],
            capture_output=True,
            text=True,
            env=env,
        )
        outs.append(p.stdout)
    assert outs[0] and json.loads(outs[0])["results"]
    assert outs[0] == outs[1] == outs[2]
