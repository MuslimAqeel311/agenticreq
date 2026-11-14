"""M14: reason: and evidence: are interchangeable for every answer, both are shown when both exist, the accepted
keys are named when neither exists, and a file:line reference is checked for the file and the line only."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from agenticreq.models.spec import Attestation
from agenticreq.not_needed import NO_REASON
from agenticreq.signoff import assess, why_text

TODAY = dt.date(2026, 11, 14)


def _a(**kw: object) -> Attestation:
    base: dict[str, object] = {"clause": "kill_switch", "by": "a@b.com", "date": "2026-11-01", "expires": "2027-11-01"}
    return Attestation.model_validate({**base, **kw})


@pytest.mark.parametrize("answer", ["yes", "no", "not_needed"])
@pytest.mark.parametrize("key", ["reason", "evidence"])
def test_m14_either_key_answers_every_answer(answer: str, key: str) -> None:
    st = assess(_a(answer=answer, **{key: "checked by hand"}), TODAY)
    assert "missing" not in st.message


def test_m14_both_are_shown() -> None:
    assert (
        why_text(_a(answer="no", reason="pipeline only", evidence="see README"))
        == "pipeline only (evidence: see README)"
    )


def test_m14_neither_names_the_accepted_keys() -> None:
    st = assess(_a(answer="no"), TODAY)
    assert st.state == "open" and st.message == "reason or evidence is missing"
    assert "reason:" in NO_REASON and "evidence:" in NO_REASON


def test_m14_a_bad_file_line_stays_unsigned(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    later = (dt.date.today() + dt.timedelta(days=1)).isoformat()  # signed after the file was written
    ok = assess(_a(answer="yes", evidence="agent.py:2 sets it", date=later, expires="2099-01-01"), TODAY, tmp_path)
    past = assess(_a(answer="yes", evidence="agent.py:9 sets it"), TODAY, tmp_path)
    gone = assess(_a(answer="yes", evidence="nope.py:1"), TODAY, tmp_path)
    assert ok.state == "signed"
    assert past.state == "open" and "the file has 2 lines" in past.message
    assert gone.state == "open" and "no such file" in gone.message
