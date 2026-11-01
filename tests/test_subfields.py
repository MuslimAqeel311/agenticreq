"""Every sub-field a user writes is checked by a rule of its own, or becomes a plain question for a person; a declared
approval timeout is checked against an approval call that can wait forever."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from agenticreq.subfields import QUESTIONS, SUBFIELDS, all_subfields, question, sign_off_only

SPEC = """
agent_id: a
risk_level: medium
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: []
approval_required: {approval}
limits: {limits}
escalation: []
untrusted_inputs: {untrusted}
owner: o
technical_owner: t
{extra}
"""
CODE = """
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from pydantic import Field
    from typing import Annotated

    def request_human_approval(what: str) -> bool:
        return input(f"approve {what}? ") == "y"

    @tool
    def save_file(path: str, text: str) -> str:
        '''Save a file after a person approves.'''
        if not request_human_approval(path):
            return "denied"
        with open(path, "w") as fh:
            fh.write(text)
        return "ok"

    @tool
    def search(query: str, max_results: Annotated[int, Field(le=50)] = 10) -> str:
        '''Search.'''
        return query[:100]

    agent = create_react_agent("openai:gpt-4o-2024-08-06", [save_file, search])
"""


def _lines(
    tmp: Path, approval: str = "[]", limits: str = "[]", untrusted: str = "[]", extra: str = "", code: str = CODE
) -> list[tuple[str, str, str]]:
    (tmp / "agent.py").write_text(textwrap.dedent(code), encoding="utf-8")
    loaded = load_spec_text(SPEC.format(approval=approval, limits=limits, untrusted=untrusted, extra=extra))
    report = run(mode="check", loaded=loaded, observed=build_observed(tmp), today=date(2026, 10, 30))
    return [(ln.param, ln.status, ln.reason) for ln in results(report)]


def test_an_approval_timeout_with_an_approval_call_that_waits_forever_is_a_gap(tmp_path: Path) -> None:
    lines = _lines(tmp_path, approval="[{tool: save_file, approver: lead, timeout: 4h, on_timeout: safe_stop}]")
    [t] = [x for x in lines if x[0] == "approval_required[].timeout"]
    assert t[1] == "GAP" and t[2].startswith("spec.yml says timeout 4h; the approval call can wait forever"), t


def test_an_approval_call_with_a_time_limit_is_not_reported_as_waiting_forever(tmp_path: Path) -> None:
    code = CODE.replace(
        'return input(f"approve {what}? ") == "y"', 'return wait_for_answer(f"approve {what}?", timeout=60) == "y"'
    )
    assert code != CODE
    lines = _lines(
        tmp_path, approval="[{tool: save_file, approver: lead, timeout: 4h, on_timeout: safe_stop}]", code=code
    )
    assert not any("wait forever" in r for _, _, r in lines), lines


def test_a_sub_field_the_code_cannot_show_becomes_a_plain_question(tmp_path: Path) -> None:
    lines = _lines(tmp_path, untrusted="[{source: search, kind: web_page}]")
    [nc] = [x for x in lines if x[0] == "untrusted_inputs[].kind"]
    q = question("untrusted_inputs[].kind")
    assert nc[1] == "SIGN-OFF" and nc[2].startswith(q) and "Confirm under confirmed:" in nc[2], nc
    assert not any(x[0] == "untrusted_inputs[].source" and "Confirm under confirmed: (untrusted" in x[2] for x in lines)


def test_a_sign_off_sub_field_is_confirmed_by_its_own_name(tmp_path: Path) -> None:
    extra = (
        'confirmed:\n  - {clause: "untrusted_inputs[].kind", by: lead@x.com, date: 2026-10-01, '
        "expires: 2027-10-01, evidence: T-1}"
    )
    lines = _lines(tmp_path, untrusted="[{source: search, kind: web_page}]", extra=extra)
    [nc] = [x for x in lines if x[0] == "untrusted_inputs[].kind"]
    assert nc[1] == "SIGNED", nc


def test_a_fully_checked_clause_still_passes(tmp_path: Path) -> None:
    lines = _lines(tmp_path, limits="[{tool: search, arg: max_results, limit: {max: 50}}]")
    assert not any(r.startswith(tuple(QUESTIONS.values())) for _, _, r in lines)
    [lim] = [x for x in lines if x[0] == "limits"]
    assert lim[1] == "PASS", lim


def test_every_sub_field_of_the_model_is_in_the_catalog() -> None:
    assert set(all_subfields()) == set(SUBFIELDS)  # a new sub-field must say what checks it (or "")


def test_the_sub_fields_a_person_answers_today() -> None:
    assert sign_off_only() == [
        "role_boundary.capabilities",
        "role_boundary.data_access",
        "untrusted_inputs[].kind",
        "output_size_limits.max_items",
        "write_safety.compensating_action",
        "agent_registry[].role",
    ]


def test_each_sign_off_sub_field_has_its_own_plain_question_and_no_old_words() -> None:
    assert set(sign_off_only()) == set(QUESTIONS)
    assert question("role_boundary.data_access") == (
        "Does the agent only read the data listed in role_boundary.data_access?"
    )
    for path in sign_off_only():
        assert question(path).endswith("?") and "NOT CHECKED" not in question(path) and "gap" not in question(path)


def test_the_three_fields_become_sign_off_questions_in_the_check_output(tmp_path: Path) -> None:
    extra = "role_boundary: {allowed_tools: [save_file], capabilities: ['write:files'], data_access: [reports]}"
    lines = _lines(tmp_path, untrusted="[{source: search, kind: web_page}]", extra=extra)
    by_param = {p: (status, reason) for p, status, reason in lines}
    want = {
        "role_boundary.capabilities": "Does the agent only do what is listed in role_boundary.capabilities?",
        "role_boundary.data_access": "Does the agent only read the data listed in role_boundary.data_access?",
        "untrusted_inputs[].kind": "Is every source under untrusted_inputs given the right kind?",
    }
    for param, question_text in want.items():
        status, reason = by_param[param]
        assert status == "SIGN-OFF" and reason == f"{question_text} Confirm under confirmed: ({param}).", (
            param,
            reason,
        )
    for _, _, reason in lines:
        assert "NOT CHECKED" not in reason and "library gap" not in reason and "not checked" not in reason.lower()
