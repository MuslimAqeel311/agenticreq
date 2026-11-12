"""H1: framework approval gates are recognised (data/approval_gates.yml), an unverified gate goes to "Check with a
mock run", and every genuine missing gate is still a gap (R1)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from agenticreq.ast.extract_support import GATE_HOOK_KWARG
from agenticreq.data import load_yaml
from agenticreq.engine import run
from agenticreq.loader import load_spec
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

REPRO = Path(__file__).parent / "fixtures" / "repro"
TODAY = date(2026, 11, 12)


def _check(project: Path) -> dict[str, str]:
    report = run(mode="check", loaded=load_spec(project / "spec.yml"), observed=build_observed(project), today=TODAY)
    return {ln.param: ln.status for ln in results(report)}


def _audit(project: Path) -> dict[str, tuple[str, str]]:
    report = run(mode="audit", observed=build_observed(project), today=TODAY)
    out: dict[str, tuple[str, str]] = {}
    for ln in results(report):
        if ln.param == "write_safety" and "save_record" in ln.reason:
            out[ln.status] = (ln.param, ln.reason)
    return out


def _copy(name: str, tmp_path: Path, old: str = "", new: str = "") -> Path:
    d = tmp_path / name
    shutil.copytree(REPRO / name, d)
    if old:
        p = d / "agent.py"
        text = p.read_text(encoding="utf-8")
        assert old in text
        p.write_text(text.replace(old, new), encoding="utf-8")
    return d


# a hook decided at run time: Check with a mock run, never a gap, never a pass
HOOKS = ["h1_llamaindex", "h1_strands", "h1_haystack", "h1_claude", "h1_adk", "h1_helper", "h1_input"]
# a gate the framework enforces from configuration: no gap
CONFIG = ["h1_lc_mw", "h1_agno", "h1_oai", "h1_langgraph"]


@pytest.mark.parametrize("name", HOOKS)
def test_h1_unverified_gate_is_a_mock_run_check(name: str) -> None:
    assert _check(REPRO / name)["approval_required"] == "CHECK"


@pytest.mark.parametrize("name", CONFIG)
def test_h1_configured_gate_is_not_a_gap(name: str) -> None:
    assert _check(REPRO / name)["approval_required"] not in ("GAP", "PASS")  # code pattern only: a mock run shows it


@pytest.mark.parametrize("name", HOOKS + CONFIG)
def test_h1_audit_never_reports_a_recognised_gate_as_a_gap(name: str) -> None:
    assert "GAP" not in _audit(REPRO / name)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("h1_llamaindex", "framework_hook"),
        ("h1_strands", "framework_hook"),
        ("h1_lc_mw", "hitl_middleware"),
        ("h1_helper", "approval_call"),
        ("h1_input", "human_input"),
    ],
)
def test_h1_gate_is_scoped_to_its_tool(name: str, kind: str) -> None:
    gates = [g for g in build_observed(REPRO / name).gates if g.kind == kind]
    assert gates and gates[0].tools == ["save_record"]


def test_h1_d_bare_input_without_a_time_limit_is_still_a_timeout_gap() -> None:
    lines = _check(REPRO / "h1_input")
    assert lines["approval_required[].timeout"] == "GAP"


def test_h1_helper_asking_with_input_is_still_a_timeout_gap() -> None:
    assert _check(REPRO / "h1_helper")["approval_required[].timeout"] == "GAP"


# ------------------------------------------------------------------ R1: the genuine defect is still reported
def test_h1_r1_a_write_with_no_gate_is_still_a_gap(tmp_path: Path) -> None:
    d = _copy("h1_input", tmp_path, '    if input("approve? ") != "y":\n        return "not approved"\n', "")
    assert _check(d)["approval_required"] == "GAP"
    assert "GAP" in _audit(d)


def test_h1_r1_a_middleware_set_to_false_gates_nothing(tmp_path: Path) -> None:
    d = _copy("h1_lc_mw", tmp_path, '{"save_record": True}', '{"save_record": False}')
    assert _check(d)["approval_required"] == "GAP"
    assert "GAP" in _audit(d)


def test_h1_r1_a_hook_naming_another_tool_does_not_gate_this_one(tmp_path: Path) -> None:
    d = _copy("h1_strands", tmp_path, 'event.tool_use["name"] == "save_record"', 'event.tool_use["name"] == "other"')
    gates = build_observed(d).gates
    assert all(g.tools != ["save_record"] for g in gates)
    # scope unknown: a check, never a pass
    assert _check(d)["approval_required"] == "CHECK"


def test_h1_r1_a_gate_after_the_write_is_a_gap(tmp_path: Path) -> None:
    d = _copy(
        "h1_helper",
        tmp_path,
        '    if not _approved("save_record"):\n        return "not approved"\n    Path("record.txt").write_text(text)\n',
        '    Path("record.txt").write_text(text)\n    if not _approved("save_record"):\n        return "not approved"\n',
    )
    assert _check(d)["approval_required"] == "GAP"


def test_h1_r1_a_hook_skipped_on_one_branch_stays_a_gap_in_audit(tmp_path: Path) -> None:
    d = _copy(
        "h1_llamaindex",
        tmp_path,
        '    answer = await ctx.wait_for_event(HumanResponseEvent)\n    if answer.response != "yes":\n'
        '        return "not approved"\n',
        "    if len(text) > 5:\n        answer = await ctx.wait_for_event(HumanResponseEvent)\n"
        '        if answer.response != "yes":\n            return "not approved"\n',
    )
    assert [g.coverage for g in build_observed(d).gates] == ["partial"]
    assert "GAP" in _audit(d)


def test_h1_r1_input_used_for_data_is_not_a_gate(tmp_path: Path) -> None:
    d = _copy(
        "h1_input",
        tmp_path,
        '    if input("approve? ") != "y":\n        return "not approved"\n',
        "    text = input('more? ') + text\n",
    )
    assert not build_observed(d).gates
    assert _check(d)["approval_required"] == "GAP"


# ------------------------------------------------------------------ the data file
def test_h1_every_listed_framework_has_a_mechanism_in_the_tables() -> None:
    data = load_yaml("approval_gates.yml")
    known: set[str] = set(data["gate_calls"]) | set(data["gate_kwargs"]) | set(data["gate_decorators"])
    known |= (
        set(data["hook_kwargs"]) | set(data["hook_dict_keys"]) | set(data["person_in_loop"]) | set(data["middleware"])
    )
    known |= {f"{call}({arg})" for call, args in data["hook_calls"].items() for arg in args}
    for fw, mechanisms in data["frameworks"].items():
        missing = [m for m in mechanisms if m not in known and not GATE_HOOK_KWARG.match(m)]
        assert not missing, f"{fw}: {missing} are not in any gate table"
    for fw in ("langgraph", "langchain", "llamaindex", "strands", "haystack", "agno", "openai-agents", "google-adk"):
        assert fw in data["frameworks"]
    assert {"crewai", "claude-agent-sdk"} <= set(data["frameworks"])
