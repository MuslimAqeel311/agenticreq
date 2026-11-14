"""M9: agenticreq never expands environment variables. A value that is only a variable reference ($HOME, ${OWNER})
is reported as a placeholder (the field stays undeclared); text that contains a $ is kept exactly as written."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenticreq.loader import load_spec
from agenticreq.loader.issues import is_placeholder

SPEC = Path(__file__).parent / "fixtures" / "repro" / "h1_input" / "spec.yml"


@pytest.mark.parametrize("value", ["$HOME", "${OWNER}", " ${TEAM_LEAD} ", "${OWNER:-a@b.c}", "$owner"])
def test_m9_a_whole_value_reference_is_a_placeholder(value: str) -> None:
    assert is_placeholder(value)


@pytest.mark.parametrize(
    "value", ["SYS.V$VERSION", "price is $5", "$$HOME", "costs $ 10", "a$b", "use ${OWNER} or the lead", "$5"]
)
def test_m9_text_with_a_dollar_is_a_value(value: str) -> None:
    assert not is_placeholder(value)


def test_m9_spec_keeps_free_text_names_and_reasons_verbatim(tmp_path: Path) -> None:
    text = SPEC.read_text(encoding="utf-8").replace(
        "out_of_scope: [legal advice]",
        "out_of_scope: [legal advice, 'querying SYS.V$VERSION', 'price is $5', '$$HOME is literal']",
    )
    (tmp_path / "spec.yml").write_text(text, encoding="utf-8")
    loaded = load_spec(tmp_path / "spec.yml")
    assert loaded.spec is not None and loaded.status != "invalid"
    assert loaded.spec.out_of_scope == ["legal advice", "querying SYS.V$VERSION", "price is $5", "$$HOME is literal"]
    assert not [i for i in loaded.issues if i.kind == "placeholder"]


def test_m9_a_variable_owner_is_still_reported(tmp_path: Path) -> None:
    text = SPEC.read_text(encoding="utf-8").replace("owner: owner@example.com", "owner: $HOME")
    (tmp_path / "spec.yml").write_text(text, encoding="utf-8")
    loaded = load_spec(tmp_path / "spec.yml")
    ph = [i for i in loaded.issues if i.kind == "placeholder"]
    assert ph and ph[0].items == ["owner: $HOME"]
