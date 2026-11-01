"""Library fixes found on a real LangGraph multi-agent project, each checked on three different scenarios:

1. structured-output schemas (with_structured_output(Plan), response_format=, pydantic_object=, X.model_validate_json)
   are sent to the model as tools, and are not tools: no role_boundary / registry / inventory gap for them
2. a numeric limit makes a tool an action only when its name does not already say it reads
3. a tool never offered to the model: which agent never ran (or ran without it), from the code and the run
4. once a fresh mock run has happened, a line it did not settle names the step left, not "run the mock run"
5. the console label column follows the longest label and keeps every reason in one column
6. the framework/version tag stays in the JSON evidence, out of the report lines

Scenario A mirrors the project (LangGraph supervisor + workers, Plan / RouteDecision schemas, read_file with a
limit, an analyst the run never reaches). Scenario B is an OpenAI-SDK style agent with keyword schemas and a refund
tool. Scenario C has a tool the code leaves out at run time while its agent runs."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from agenticreq.ast.extractor import extract
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.observed import CapturedCall, ToolSchema
from agenticreq.models.verdict import AFTER_RUN, after_mock_run, mock_run_done, results
from agenticreq.observe import build_observed
from agenticreq.report.console import _wrap, render_console

MOCK_RUN_INFO = {"time": "2026-10-28 10:00", "captured": 4, "total": 4, "stale": False, "file": "captures.json"}

# ------------------------------------------------------------------ scenario A: the project's shape (LangGraph)
AGENT_A = '''
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class Plan(BaseModel):
    steps: list[str]


class RouteDecision(BaseModel):
    next: str


@tool
def read_file(filename: str, max_chars: int = 8000) -> str:
    """Read a text file from the workspace."""
    return open(filename, encoding="utf-8").read()[:max_chars]


@tool
def write_file(filename: str, content: str) -> str:
    """Write a text file in the workspace."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return "saved"


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic."""
    return expression


@tool
def text_stats(text: str) -> str:
    """Count words."""
    return str(len(text.split()))


llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
planner = llm.with_structured_output(Plan)
router = llm.with_structured_output(schema=RouteDecision)
researcher = create_react_agent(llm, [read_file], name="researcher")
writer = create_react_agent(llm, [read_file, write_file, text_stats], name="writer")
analyst = create_react_agent(llm, [read_file, calculator], name="analyst")
'''
SPEC_A = """
agent_id: team
risk_level: high
goal: {statement: write a report, done_criteria: [report written]}
out_of_scope: []
prohibited: []
approval_required:
  - {tool: write_file, approver: technical_owner, timeout: 4h, on_timeout: safe_stop}
limits:
  - {tool: read_file, arg: max_chars, limit: {max: 50000}}
  - {tool: write_file, arg: content, limit: {max: 1000000}}
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
role_boundary: {allowed_tools: [read_file, write_file, calculator, text_stats]}
agent_registry:
  - {id: researcher, tools: [read_file]}
  - {id: writer, tools: [read_file, write_file, text_stats]}
  - {id: analyst, tools: [read_file, calculator]}
"""
SCHEMAS = ("Plan", "RouteDecision")


def _tool(name: str) -> ToolSchema:
    return ToolSchema(name=name, source="wire", kind="wire:anthropic")


def _call(i: int, prompt: str, tools: list[str]) -> CapturedCall:
    return CapturedCall(
        index=i,
        provider="anthropic",
        url="https://api.anthropic.com/v1/messages",
        model="claude-sonnet-4-5-20250929",
        system_prompt=prompt,
        system_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        tools=[_tool(t) for t in tools],
        synthetic=True,
    )


#: the run: planner and router (schemas only), then researcher and writer; analyst is never reached
CALLS_A = [
    _call(0, "You plan the work.", ["Plan"]),
    _call(1, "You route to the next worker.", ["RouteDecision"]),
    _call(2, "You are the researcher.", ["read_file"]),
    _call(3, "You are the writer.", ["read_file", "write_file", "text_stats"]),
]


def _report(tmp: Path, code: str, spec: str, calls: list[CapturedCall], *, ran: bool = True):  # type: ignore[no-untyped-def]
    (tmp / "agent.py").write_text(code, encoding="utf-8")
    r = run(mode="check", loaded=load_spec_text(spec), observed=build_observed(tmp, captures=calls))
    if ran:
        r.metadata["mock_run"] = dict(MOCK_RUN_INFO)
    return r


def _text(f) -> str:  # type: ignore[no-untyped-def]
    return f"{f.title} {f.detail} {f.reason or ''}"


def test_a_schemas_are_recorded_from_positional_and_keyword_calls(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(AGENT_A, encoding="utf-8")
    obs = extract(tmp_path)
    assert set(SCHEMAS) <= set(obs.output_types)
    assert all(obs.is_output_schema(s) for s in SCHEMAS)
    assert not any(obs.is_output_schema(t) for t in ("read_file", "write_file", "calculator"))


def test_a_no_gap_names_a_schema(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    for f in r.findings:  # role_boundary (AS-206), registry (AS-206), inventory (AS-202): none about a schema
        assert not any(s in _text(f) for s in SCHEMAS), (f.rule_id, f.title)
    inventory = {p.subject: p for p in r.passed if p.rule_id == "AS-202"}
    assert set(SCHEMAS) <= set(inventory)
    assert all(inventory[s].confidence.level.value == "proven" for s in SCHEMAS)  # read from the code, not a pattern
    lines = {(ln.agent, ln.param): ln for ln in results(r)}
    assert not any(s in ln.reason for ln in lines.values() for s in SCHEMAS if ln.status == "GAP")


def test_a_schemas_are_not_held_by_any_agent_node(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(AGENT_A, encoding="utf-8")
    obs = build_observed(tmp_path, captures=CALLS_A)
    held = {t for n in obs.nodes for t in n.tools}
    assert held and not held & set(SCHEMAS)


def test_a_limit_on_a_read_tool_does_not_make_it_a_write(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    assert not [f for f in r.findings if f.rule_id == "AS-212" and "read_file" in f.title]
    # the approval tool is still checked: write_file has an approval_required entry and no gate in code
    assert any("write_file" in _text(f) for f in r.findings if f.rule_id.startswith("AS-2"))


def test_a_never_offered_names_the_agent_that_never_ran(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    calc = [f for f in r.findings if f.rule_id == "AS-201" and f.title.startswith("calculator ")]
    assert len(calc) == 1 and calc[0].status_hint == "CHECK"
    assert "the run never reached analyst" in (calc[0].reason or "")
    assert "order in mocks.yml" not in (calc[0].reason or "")  # never offered: the order cannot help
    assert (calc[0].reason or "").endswith("then run: agenticreq mock-run")


def test_a_a_routing_schema_is_named_as_the_way_to_reach_the_agent(tmp_path: Path) -> None:
    calls = [c.model_copy(deep=True) for c in CALLS_A]
    route = calls[1].tools[0]  # RouteDecision, as the wire carried it: its choices name the workers
    route.raw = {
        "name": "RouteDecision",
        "input_schema": {"properties": {"next": {"enum": ["researcher", "writer", "analyst"], "type": "string"}}},
    }
    r = _report(tmp_path, AGENT_A, SPEC_A, calls)
    calc = next(f for f in r.findings if f.rule_id == "AS-201" and f.title.startswith("calculator "))
    assert "whose RouteDecision args send the run to it" in (calc.reason or "")  # the canned model routes
    assert "Plan" not in (calc.reason or "")  # Plan's wire schema names no agent here


def test_a_after_a_mock_run_no_line_just_says_run_it_again(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    checks = [ln for ln in results(r) if ln.status == "CHECK"]
    assert checks
    assert not [ln.reason for ln in checks if ln.reason.endswith(". Run: agenticreq mock-run")]
    before = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A, ran=False)  # no mock run yet: the advice stays
    assert any(ln.reason.endswith(". Run: agenticreq mock-run") for ln in results(before) if ln.status == "CHECK")


# ------------------------------------------------------------------ scenario B: OpenAI SDK, keyword schemas
AGENT_B = '''
from openai import OpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel


class Answer(BaseModel):
    text: str


class Summary(BaseModel):
    points: list[str]


class Report(BaseModel):
    body: str


@tool
def search_orders(query: str, limit: int = 10) -> str:
    """Search orders."""
    return query


@tool
def process_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return order_id


client = OpenAI()
parser = PydanticOutputParser(pydantic_object=Answer)


def summarise(raw: str) -> Summary:
    out = client.beta.chat.completions.parse(model="gpt-4o-2024-08-06", messages=[], response_format=Summary)
    report = Report.model_validate_json(raw)
    return out.choices[0].message.parsed
'''
SPEC_B = (
    SPEC_A.split("approval_required:")[0]
    + """approval_required: []
limits:
  - {tool: search_orders, arg: limit, limit: {max: 50}}
  - {tool: process_refund, arg: amount, limit: {max: 500}}
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
"""
)


def test_b_keyword_and_receiver_schemas_are_recorded_and_text_args_are_not(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(AGENT_B, encoding="utf-8")
    obs = extract(tmp_path)
    assert {"Answer", "Summary", "Report"} <= set(obs.output_types)
    assert "raw" not in obs.output_types  # model_validate_json(raw): raw is the model's text, Report the schema


def test_b_limit_still_marks_an_action_whose_name_is_unclear(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_B, SPEC_B, [], ran=False)
    write_gaps = {f.title for f in r.findings if f.rule_id == "AS-212"}
    assert any("process_refund" in t for t in write_gaps)  # "process" says nothing: the spec's limit decides
    assert not any("search_orders" in t for t in write_gaps)  # "search" reads: its limit caps a read


def test_b_a_real_tool_is_never_taken_for_a_schema(tmp_path: Path) -> None:
    code = (
        AGENT_B
        + '''

@tool
def summary(order_id: str) -> str:
    """Summarise an order."""
    return order_id
'''
    )
    (tmp_path / "agent.py").write_text(code, encoding="utf-8")
    obs = extract(tmp_path)
    assert "Summary" in obs.output_types and "summary" in obs.declared_tools
    assert not obs.is_output_schema("summary")  # a tool in code by that name is a tool


# ------------------------------------------------------------------ scenario C: a tool left out at run time
AGENT_C = '''
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic."""
    return expression


@tool
def python_repl(code: str) -> str:
    """Run Python."""
    return code


@tool
def lookup(term: str) -> str:
    """Look up a term."""
    return term


llm = ChatOpenAI(model="gpt-4o-2024-08-06")
tools = [calculator, python_repl] if os.getenv("ALLOW_CODE") else [calculator]
analyst = create_react_agent(llm, [calculator, python_repl], name="analyst")
librarian = create_react_agent(llm, [lookup], name="librarian")
'''
SPEC_C = (
    SPEC_B.split("approval_required:")[0]
    + """approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
"""
)


def _openai_call(i: int, tools: list[str]) -> CapturedCall:
    return CapturedCall(
        index=i,
        provider="openai_chat",
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-2024-08-06",
        system_prompt="You are the analyst.",
        system_prompt_sha256=hashlib.sha256(b"analyst").hexdigest(),
        tools=[_tool(t) for t in tools],
        synthetic=True,
    )


def test_c_agent_ran_without_the_tool_points_at_the_settings(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_C, SPEC_C, [_openai_call(0, ["calculator"])])
    by_tool = {f.title.split(" ")[0]: f.reason or "" for f in r.findings if f.rule_id == "AS-201"}
    assert "analyst ran without it" in by_tool["python_repl"] and "env:" in by_tool["python_repl"]
    assert "the run never reached librarian" in by_tool["lookup"]


# ------------------------------------------------------------------ the console and the wording helpers
def test_after_mock_run_rewrites_only_a_bare_run_step() -> None:
    bare = "spec.yml says 2; not readable in code (or sign it off under confirmed:). Run: agenticreq mock-run"
    assert after_mock_run(bare) == f"spec.yml says 2; not readable in code; the mock run did not settle it. {AFTER_RUN}"
    own_step = "Never offered to the model: x. Give args: in mocks.yml a task, then run: agenticreq mock-run"
    assert after_mock_run(own_step) == own_step
    noted = "recursion_limit is not a literal (+3 more). Run: agenticreq mock-run"
    assert after_mock_run(noted).startswith("recursion_limit is not a literal (+3 more); the mock run")


def test_mock_run_done_needs_calls_and_fresh_code(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    assert mock_run_done(r)
    r.metadata["mock_run"]["stale"] = True  # the code changed since: running it again is the right step
    assert not mock_run_done(r)
    r.metadata["mock_run"] = {**MOCK_RUN_INFO, "captured": 0}
    assert not mock_run_done(r)


def test_long_text_wraps_to_the_text_column_and_is_never_cut() -> None:
    lines = _wrap("progress_guard.max_no_progress_steps is never cut with an ellipsis at any width", 40, 10)
    assert all(len(ln) <= 40 for ln in lines) and all(ln.startswith(" " * 10) for ln in lines[1:])
    joined = "".join(ln.strip() for ln in lines).replace(" ", "")  # every character is still there, in order
    assert joined == "progress_guard.max_no_progress_steps is never cut with an ellipsis at any width".replace(" ", "")
    assert _wrap("short", 30) == ["short"]


#: a result row: two spaces, the mark and label, two or more spaces, the reason
_ROW = re.compile(r"^  \S.*? {2,}\S")


def test_every_reason_starts_in_one_column_at_any_width(tmp_path: Path) -> None:
    from rich.console import Console

    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    for width in (60, 80, 110, 160):
        con = Console(width=width, force_terminal=False, color_system=None, record=True, legacy_windows=False)
        render_console(r, console=con)
        text = con.export_text()
        assert all(len(line) <= width for line in text.splitlines()), width
        for block in text.split("\n\n"):  # inside one group, the reason column is the same for every row
            starts = {m.end() - 1 for m in (_ROW.match(ln) for ln in block.splitlines()) if m}
            assert len(starts) <= 1, (width, block)


def test_framework_tag_stays_out_of_report_lines(tmp_path: Path) -> None:
    r = _report(tmp_path, AGENT_A, SPEC_A, CALLS_A)
    lines = results(r)
    assert not [ln.reason for ln in lines if "(runtime)]" in ln.reason or "(import)]" in ln.reason]


def test_new_code_is_valid_python() -> None:  # the scenarios themselves must parse, or they test nothing
    for code in (AGENT_A, AGENT_B, AGENT_C):
        ast.parse(code)
