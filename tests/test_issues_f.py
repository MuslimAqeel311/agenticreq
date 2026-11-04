"""Priority F: end_to_end_trace_id, from the code and from runs, for every framework."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from conftest import chat_call

from agenticreq.data import load_yaml
from agenticreq.derived import enforceability, trace_coverage
from agenticreq.engine import run
from agenticreq.loader import load_spec, load_spec_text
from agenticreq.models.observed import TraceSpan
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from agenticreq.rules.graph import one_trace_across_agents
from test_issues_b import ISSUES

SPEC = ISSUES / "f1_trace" / "spec.yml"
AGENTS = "from crewai import Agent\na = Agent(role='a', goal='g', backstory='b', allow_delegation=True)\nb = Agent(role='b', goal='g', backstory='b')\n"


def _line(project: Path, **kw: object) -> object:
    obs = build_observed(project, **kw)  # type: ignore[arg-type]
    report = run(mode="check", loaded=load_spec(SPEC), observed=obs, today=date(2026, 11, 4))
    return {ln.param: ln for ln in results(report)}.get("end_to_end_trace_id")


def test_f1_a_project_id_is_read_in_order_with_its_places() -> None:
    ev = next(e for e in build_observed(ISSUES / "f1_trace").trace_evidence if e.mechanism == "correlation id")
    assert ev.name == "run_id"
    assert [str(r) for r in ev.created] == ["agent.py:14"]
    assert [str(r) for r in ev.logged] == ["agent.py:15"]
    assert [str(r) for r in ev.used] == ["agent.py:16"]


def test_f4_a_mechanism_found_is_a_sign_off_with_its_evidence() -> None:
    line = _line(ISSUES / "f1_trace")
    assert line.status == "SIGN-OFF"  # type: ignore[attr-defined]
    assert "created at agent.py:14" in line.reason and "in log records at agent.py:15" in line.reason  # type: ignore[attr-defined]


def test_f4_no_mechanism_anywhere_is_a_gap(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(AGENTS, encoding="utf-8")
    line = _line(tmp_path)
    assert line is not None and line.status == "GAP"  # type: ignore[attr-defined]


def test_f2_opentelemetry_counts_for_any_framework(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from opentelemetry import trace, propagate\n"
        + AGENTS
        + "tracer = trace.get_tracer('x')\ndef go():\n    with tracer.start_as_current_span('run'):\n        propagate.inject({})\n",
        encoding="utf-8",
    )
    ev = build_observed(tmp_path).trace_evidence
    assert [e.mechanism for e in ev] == ["OpenTelemetry"] and ev[0].created and ev[0].used


@pytest.mark.parametrize(
    "imp,use,mechanism",
    [
        ("from google.adk.agents import Agent", "def f(ctx):\n    print(ctx.invocation_id)\n", "ADK invocation"),
        ("from pydantic_ai import Agent", "def f(ctx):\n    print(ctx.run_id)\n", "PydanticAI instrumentation"),
        ("from claude_agent_sdk import ClaudeAgentOptions", "def f(m):\n    print(m.session_id)\n", "Claude session"),
        ("from agno.agent import Agent", "def f(r):\n    print(r.run_id)\n", "Agno run"),
        ("from agents import Agent", "def f(r):\n    print(r.trace_id)\n", "Agents SDK tracing"),
        (
            "from langgraph.prebuilt import create_react_agent as Agent",
            "def f(c):\n    print(c.run_id)\n",
            "LangChain run tree",
        ),
        (
            "from autogen_agentchat.agents import AssistantAgent as Agent",
            "x2 = Agent('b', tracer_provider=None)\n",
            "AutoGen runtime tracing",
        ),
        (
            "from llama_index.core.agent.workflow import FunctionAgent as Agent",
            "def f(h):\n    print(h.run_id)\n",
            "workflow run",
        ),
        ("from haystack.components.agents import Agent", "def f(t):\n    t.tracer.trace('x')\n", "Haystack tracer"),
        (
            "from semantic_kernel.agents import ChatCompletionAgent as Agent",
            "x2 = Agent(name='b', tracer_provider=None)\n",
            "OpenTelemetry in Semantic Kernel",
        ),
    ],
)
def test_f2_each_framework_reads_from_the_table(tmp_path: Path, imp: str, use: str, mechanism: str) -> None:
    ctor = "ClaudeAgentOptions(model='m')" if "Claude" in imp else "Agent(name='a', model='m')"
    (tmp_path / "a.py").write_text(f"{imp}\nx = {ctor}\n{use}", encoding="utf-8")
    assert mechanism in [e.mechanism for e in build_observed(tmp_path).trace_evidence]


def test_f2_the_table_only_lists_what_it_describes() -> None:
    assert {"mechanism", "id", "tokens"} <= set(load_yaml("tracing.yml")["frameworks"]["google-adk"])


def test_f3_one_trace_id_across_two_agents_passes_from_spans(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(AGENTS, encoding="utf-8")
    spans = [TraceSpan(trace_id="t1", agent_name="a"), TraceSpan(trace_id="t1", agent_name="b")]
    line = _line(tmp_path, spans=spans)
    assert line is not None and line.status == "PASS"  # type: ignore[attr-defined]


def test_f3_one_trace_id_across_two_agents_passes_from_captures(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(AGENTS, encoding="utf-8")
    calls = [chat_call(0, [], system="one"), chat_call(1, [], system="two")]
    for c in calls:
        c.trace_id = "abc"
    obs = build_observed(tmp_path, captures=calls)
    assert one_trace_across_agents(obs) == ("abc", 2)


def test_f3_two_trace_ids_are_not_one(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(AGENTS, encoding="utf-8")
    spans = [TraceSpan(trace_id="t1", agent_name="a"), TraceSpan(trace_id="t2", agent_name="b")]
    assert one_trace_across_agents(build_observed(tmp_path, spans=spans)) is None


def test_f3_a_traceparent_header_becomes_the_trace_id() -> None:
    from agenticreq.capture.hook import trace_header

    assert trace_header({"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}) == (
        "4bf92f3577b34da6a3ce929d0e0e4736"
    )
    assert trace_header({"x-correlation-id": "c1"}) == "c1" and trace_header({}) is None


def test_f5_trace_coverage_and_enforceability_use_the_same_evidence(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(AGENTS, encoding="utf-8")
    spans = [TraceSpan(trace_id="t1", agent_name="a"), TraceSpan(trace_id="t1", agent_name="b")]
    obs = build_observed(tmp_path, spans=spans)
    loaded = load_spec_text(SPEC.read_text(encoding="utf-8"))
    report = run(mode="check", loaded=loaded, observed=obs, today=date(2026, 11, 4))
    assert loaded.spec is not None
    assert trace_coverage(loaded.spec, obs)["end_to_end"] is True
    rows = enforceability(loaded.spec, obs, report.passed)["rows"]
    assert {"clause": "end_to_end_trace_id", "code": "one trace id across agents", "export": ""} in rows
