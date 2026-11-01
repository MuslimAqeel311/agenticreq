"""AS-228..231: stop_conditions / progress_guard / bounds / rate_limits from REAL runs only."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.observed import CapturedCall, ToolCall
from agenticreq.observe import build_observed

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
bounds: {max_steps: 6, max_tool_calls: 5}
progress_guard: {max_no_progress_steps: 2}
rate_limits: {llm_requests_per_minute: 3}
"""
AGENT = (
    "from langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
    "@tool\ndef search_docs(q: str, limit: int = 5) -> str:\n    '''Search.'''\n    return q\n"
    "llm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([search_docs])\n"
)


def call(
    i: int,
    tools: list[tuple[str, dict]],
    t: float,
    synthetic: bool = False,  # type: ignore[type-arg]
    run_id: str = "r1",
) -> CapturedCall:
    return CapturedCall(
        index=i,
        run_id=run_id,
        provider="openai_chat",
        url="https://api.openai.com/v1/chat",
        model="gpt-4o-2024-08-06",
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in tools],
        timestamp=t,
        synthetic=synthetic,
    )


def findings(tmp: Path, calls: list[CapturedCall]) -> tuple[set[str], set[str], list]:  # type: ignore[type-arg]
    (tmp / "agent.py").write_text(AGENT, encoding="utf-8")
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=build_observed(tmp, captures=calls))
    return {f.rule_id for f in r.findings}, {p.rule_id for p in r.passed}, r.findings


LOOP = [call(i, [("search_docs", {"q": "x"})], 10.0 * i) for i in range(4)] + [call(4, [], 40.0)]
CLEAN = [call(0, [("search_docs", {"q": "a"})], 0.0), call(1, [("search_docs", {"q": "b"})], 30.0), call(2, [], 61.0)]


def test_progress_guard(tmp_path: Path) -> None:
    ids, _, fs = findings(tmp_path, LOOP)
    assert "AS-228" in ids
    assert "4 times" in next(f for f in fs if f.rule_id == "AS-228").title
    ids, passed, _ = findings(tmp_path, CLEAN)
    assert "AS-228" not in ids and "AS-228" in passed


def test_observed_bounds(tmp_path: Path) -> None:
    many = [call(i, [("search_docs", {"q": str(i)})], float(i * 30)) for i in range(8)] + [call(8, [], 300.0)]
    ids, _, _ = findings(tmp_path, many)
    assert "AS-229" in ids  # 9 LLM calls > max_steps 6, 8 tool calls > 5
    ids, passed, _ = findings(tmp_path, CLEAN)
    assert "AS-229" not in ids and "AS-229" in passed


def test_stop_conditions(tmp_path: Path) -> None:
    cut = [call(0, [("search_docs", {"q": "a"})], 0.0), call(1, [("search_docs", {"q": "b"})], 30.0)]
    ids, _, _ = findings(tmp_path, cut)  # the last response still asked for a tool, far below the bound
    assert "AS-230" in ids
    ids, passed, _ = findings(tmp_path, CLEAN)
    assert "AS-230" not in ids and "AS-230" in passed


def test_rate_limit(tmp_path: Path) -> None:
    burst = [call(i, [("search_docs", {"q": str(i)})], float(i)) for i in range(4)] + [call(4, [], 5.0)]
    ids, _, _ = findings(tmp_path, burst)  # 5 requests within 60 s > 3/min
    assert "AS-231" in ids
    ids, passed, _ = findings(tmp_path, CLEAN)
    assert "AS-231" not in ids and "AS-231" in passed


def test_synthetic_runs_prove_nothing(tmp_path: Path) -> None:
    fake = [call(i, [("search_docs", {"q": "x"})], float(i), synthetic=True) for i in range(9)]
    ids, passed, fs = findings(tmp_path, fake)
    assert not ids & {"AS-228", "AS-229", "AS-230", "AS-231"}
    assert not passed & {"AS-228", "AS-229", "AS-230", "AS-231"}
    # without real runs these clauses stay attestation-only
    assert any(f.rule_id == "AS-501" and "progress_guard" in f.detail for f in fs)


@pytest.mark.parametrize("clause", ["stop_conditions", "progress_guard", "rate_limits"])
def test_real_runs_replace_the_attestation(tmp_path: Path, clause: str) -> None:
    _, _, fs = findings(tmp_path, CLEAN)
    assert not any(f.rule_id == "AS-501" and clause in f.detail for f in fs)


def test_traces_feed_the_same_checks(tmp_path: Path) -> None:
    from agenticreq.models.observed import TraceSpan

    spans = [
        TraceSpan(
            trace_id="t1",
            span_id=str(i),
            name="tool",
            tool_name="search_docs",
            operation="execute_tool",
            attributes={"gen_ai.tool.call.arguments": '{"q": "x"}'},
            start_time=float(i),
        )
        for i in range(4)
    ]
    (tmp_path / "agent.py").write_text(AGENT, encoding="utf-8")
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=build_observed(tmp_path, spans=spans))
    assert "AS-228" in {f.rule_id for f in r.findings}
