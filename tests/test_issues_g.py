"""Priority G: messages and sign-off entries."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.spec import Attestation, Waiver
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from agenticreq.signoff import assess, why_text
from test_issues_b import ISSUES, lines

runner = CliRunner()
TODAY = dt.date(2026, 11, 4)
D = dt.date(2026, 11, 1)


def _entry(**kw: object) -> Attestation:
    return Attestation(clause="owner", by="me", date=D, expires=dt.date(2027, 1, 1), **kw)  # type: ignore[arg-type]


# =========================================================================== G1
def test_g1_a_limit_mismatch_prints_both_numbers() -> None:
    line = lines(ISSUES / "g1_limit")["limits"]
    assert line.status == "GAP" and "code allows 600; spec.yml says 500" in line.reason


# =========================================================================== G2
@pytest.mark.parametrize("answer", ["yes", "no", "not_needed"])
@pytest.mark.parametrize("key", ["reason", "evidence"])
def test_g2_either_key_answers_for_every_answer(answer: str, key: str) -> None:
    a = _entry(answer=answer, **{key: "the owner told me"})
    assert assess(a, TODAY).state == {"yes": "signed", "no": "no", "not_needed": "not_needed"}[answer]
    assert why_text(a) == "the owner told me"


def test_g2_both_are_shown() -> None:
    assert why_text(_entry(answer="no", reason="not yet", evidence="ticket 9")) == "not yet (evidence: ticket 9)"


@pytest.mark.parametrize("answer", ["yes", "no", "not_needed"])
def test_g2_neither_names_the_accepted_keys(answer: str) -> None:
    a = _entry(answer=answer)
    assert assess(a, TODAY).state == "open" and "reason or evidence is missing" in assess(a, TODAY).message


def test_g2_a_waiver_takes_either_key_and_needs_one() -> None:
    assert Waiver(rule_id="AS-1", evidence="e").why == "e"
    assert Waiver(rule_id="AS-1", reason="r", evidence="e").why == "r (evidence: e)"
    with pytest.raises(ValueError, match="reason: or evidence:"):
        Waiver(rule_id="AS-1")


def test_g2_not_needed_with_evidence_shows_it_and_not_the_placeholder() -> None:
    from agenticreq.not_needed import entries
    from agenticreq.models.spec import AgentSpec

    spec = AgentSpec.model_construct(confirmed=[_entry(answer="not_needed", evidence="no such process here")])
    got = entries({}, spec, TODAY)
    assert "no reason given" not in got[0].reason and "no such process here".capitalize() in got[0].reason


# =========================================================================== G3
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
confirmed:
  - {clause: save_note, by: me, date: 2026-11-01, expires: 2027-01-01, evidence: x}
"""


def test_g3_a_confirmed_entry_does_not_close_a_mock_run_item_and_the_line_says_so(tmp_path: Path) -> None:
    from conftest import chat_call

    (tmp_path / "a.py").write_text(
        "from agents import Agent, function_tool\n@function_tool\ndef save_note(t: str) -> str:\n    '''Save.'''\n"
        "    return t\n@function_tool\ndef other(t: str) -> str:\n    '''Other.'''\n    return t\n"
        "agent = Agent(name='a', tools=[save_note, other])\n",
        encoding="utf-8",
    )
    obs = build_observed(
        tmp_path, captures=[chat_call(0, [{"name": "other", "description": "Other.", "parameters": {}}])]
    )
    report = run(mode="check", loaded=load_spec_text(SPEC), observed=obs, today=TODAY)
    line = {ln.param: ln for ln in results(report)}["save_note"]
    assert line.status == "CHECK" and "This is settled by a mock run, not by a sign-off." in line.reason


# =========================================================================== G4
def test_g4_a_commented_out_entry_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mocks.yml").write_text("# entry: agent.py:main\ntools: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["mock-run"])
    assert "commented out" in " ".join(r.output.split())


def test_g4_validate_says_which_spec_and_nothing_about_a_run() -> None:
    r = runner.invoke(app, ["validate", str(ISSUES / "g1_limit" / "spec.yml")], env={"NO_COLOR": "1"})
    head = r.output.splitlines()[0]
    assert "no spec" not in head and "mock run" not in head and "g1_limit/spec.yml" in head


def test_g4_a_missing_package_says_which_interpreter_and_executable() -> None:
    from agenticreq.capture.discover import DiscoverResult, explain

    f = {"type": "ModuleNotFoundError", "message": "x", "missing_module": "agenticreq.capture"}
    text = explain("r", f, "", 0, DiscoverResult(entry="a"), "/env/bin/python")
    assert "/env/bin/python" in text and "agenticreq itself runs from" in text


# =========================================================================== G5
def test_g5_a_reference_to_a_missing_file_stays_unsigned(tmp_path: Path) -> None:
    a = _entry(evidence="see gone.py:3")
    st = assess(a, TODAY, tmp_path)
    assert st.state == "open" and "gone.py:3: no such file" in st.message


def test_g5_a_line_past_the_end_stays_unsigned(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("a = 1\n", encoding="utf-8")
    st = assess(_entry(evidence="x.py:40"), TODAY, tmp_path)
    assert st.state == "open" and "x.py:40: the file has 1 lines" in st.message


def test_g5_a_good_reference_signs(tmp_path: Path) -> None:
    shutil.copy(ISSUES / "g1_limit" / "agent.py", tmp_path / "agent.py")
    import os

    os.utime(tmp_path / "agent.py", (0, 0))
    assert assess(_entry(evidence="agent.py:2"), TODAY, tmp_path).state == "signed"
