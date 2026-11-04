"""Priority B: false gaps and wrong evidence. Each removed gap has a test that the real gap is still reported."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agenticreq.ast.constants import Several, lookup
from agenticreq.capture.discover import load_capture_bundle
from agenticreq.engine import run
from agenticreq.loader import load_spec
from agenticreq.models.verdict import results
from agenticreq.names import norm
from agenticreq.observe import build_observed

ISSUES = Path(__file__).parent / "fixtures" / "issues"


def lines(project: Path, captures: Path | None = None) -> dict[str, Any]:
    calls = load_capture_bundle(captures)[0] if captures else None
    obs = build_observed(project, captures=calls)
    report = run(mode="check", loaded=load_spec(project / "spec.yml"), observed=obs, today=date(2026, 11, 4))
    return {ln.param: ln for ln in results(report)}


def _tweak(name: str, tmp_path: Path, file: str, old: str, new: str) -> Path:
    d = tmp_path / name
    shutil.copytree(ISSUES / name, d)
    p = d / file
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new), encoding="utf-8")
    return d


# =========================================================================== B1
def test_b1_retries_rate_and_dataclass_defaults_are_read_with_their_lines() -> None:
    got = lines(ISSUES / "b1_limits")
    assert got["bounds.max_retries"].status == "PASS" and "(agent.py:26)" in got["bounds.max_retries"].reason
    assert got["rate_limits"].status == "PASS" and "10 (agent.py:29)" in got["rate_limits"].reason
    no_progress = got["progress_guard.max_no_progress_steps"]
    assert no_progress.status == "PASS" and "5 (agent.py:13)" in no_progress.reason
    assert got["bounds.max_duration_s"].status == "PASS" and "120 (agent.py:27)" in got["bounds.max_duration_s"].reason


def test_b1_max_rpm_is_never_a_tool_call_cap() -> None:
    line = lines(ISSUES / "b1_limits")["bounds.max_tool_calls"]
    assert line.status == "GAP" and "10" not in line.reason


def test_b1_real_gaps_stay(tmp_path: Path) -> None:
    d = _tweak("b1_limits", tmp_path, "spec.yml", "max_retries: 5", "max_retries: 1")
    assert lines(d)["bounds.max_retries"].status == "GAP"
    d = _tweak("b1_limits", tmp_path / "2", "spec.yml", "requests_per_minute: 20", "requests_per_minute: 5")
    assert "Code allows 10" in lines(d)["rate_limits"].reason
    d = _tweak("b1_limits", tmp_path / "3", "spec.yml", "max_no_progress_steps: 6", "max_no_progress_steps: 2")
    assert lines(d)["progress_guard.max_no_progress_steps"].status == "GAP"


def test_b1_a_dry_run_flag_in_the_body_is_no_gap_and_no_pass() -> None:
    assert lines(ISSUES / "b1_limits")["dry_run_required"].status == "CHECK"


def test_b1_a_tool_with_no_dry_run_flag_stays_a_gap(tmp_path: Path) -> None:
    d = _tweak("b1_limits", tmp_path, "agent.py", '    if DRY_RUN:\n        return f"would send {ref}"\n', "")
    assert lines(d)["dry_run_required"].status == "GAP"


def test_b1_retries_on_a_non_agent_call_is_not_read(tmp_path: Path) -> None:
    d = _tweak("b1_limits", tmp_path, "agent.py", "retries=3, output_retries=2, ", "")
    (d / "agent.py").write_text((d / "agent.py").read_text() + "\nsession = make_session(retries=9)\n")
    assert "Found in code: 9" not in lines(d)["bounds.max_retries"].reason


# =========================================================================== B2
def test_b2_a_guard_before_the_write_is_a_gate_to_verify_not_a_gap() -> None:
    obs = build_observed(ISSUES / "b2_gate")
    guards = {t for g in obs.gates if g.kind == "guard_call" for t in g.tools or []}
    assert guards == {"pay_invoice"} and not any(g.verified for g in obs.gates)


def test_b2_a_check_after_the_write_or_a_plain_empty_test_stays_a_gap() -> None:
    reason = lines(ISSUES / "b2_gate")["approval_required"]
    assert reason.status == "GAP" and "refund" in reason.reason and "archive" in reason.reason


@pytest.mark.parametrize("call", ["can_use_tool=check", "confirmation_strategies=strategies"])
def test_b2_framework_hooks_are_gates(tmp_path: Path, call: str) -> None:
    (tmp_path / "a.py").write_text(
        f"from agents import Agent\ndef t(q): return q\nagent = Agent(name='a', tools=[t], {call})\n", encoding="utf-8"
    )
    assert build_observed(tmp_path).gates


# =========================================================================== B3
def test_b3_hosted_prefix_and_dated_names_are_the_same_tool() -> None:
    assert norm("hosted:web_search") == norm("web_search") == norm("web_search_20250305")


def test_b3_a_hosted_tool_is_offered_under_its_code_name() -> None:
    project = ISSUES / "b3_hosted"
    got = lines(project, project / "captures.json")
    assert "web_search" not in got or "Never offered" not in got["web_search"].reason
    assert "schema" not in got.get("tool_schema_strictness", got["bounds.max_steps"]).reason


# =========================================================================== B4
def test_b4_a_helper_base_class_is_not_a_tool() -> None:
    assert set(build_observed(ISSUES / "b4_base").declared_tools) == {"save_file"}


def test_b4_a_class_in_a_tool_list_is_still_a_tool(tmp_path: Path) -> None:
    d = _tweak("b4_base", tmp_path, "agent.py", "tools=[SaveFile()]", "tools=[SaveFile(), Unused()]")
    assert set(build_observed(d).declared_tools) == {"save_file", "unused_tool"}


def test_b4_a_project_that_binds_nothing_keeps_every_candidate(tmp_path: Path) -> None:
    d = _tweak("b4_base", tmp_path, "agent.py", "tools=[SaveFile()]", "tools=load_tools()")
    assert "save_file" in build_observed(d).declared_tools


# =========================================================================== B5
def test_b5_a_tool_call_cap_is_not_a_step_limit_and_a_retry_loop_is_retries() -> None:
    got = lines(ISSUES / "b5_found")
    assert "7" not in got["bounds.max_steps"].reason
    assert got["bounds.max_tool_calls"].status == "PASS" and "(agent.py:14)" in got["bounds.max_tool_calls"].reason
    assert got["bounds.max_retries"].reason.startswith("Code allows 3 retries")


def test_b5_a_real_step_loop_is_still_a_step_limit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "import anthropic\nclient = anthropic.Anthropic()\ndef go():\n    for step in range(9):\n"
        "        client.messages.create(model='m', max_tokens=1, messages=[])\n",
        encoding="utf-8",
    )
    b = build_observed(tmp_path).bounds
    assert [(x.kind, x.value) for x in b] == [("max_steps", 9.0)]


def test_b5_a_framework_default_is_labelled_not_called_code() -> None:
    reason = lines(ISSUES / "b5_found")["bounds.max_steps"].reason
    assert reason.startswith("No bound in code: framework default") and "Code allows" not in reason


def test_b5_delegation_depth_prints_where_it_came_from(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "from agents import Agent\nb = Agent(name='b', instructions='i')\na = Agent(name='a', instructions='i', handoffs=[b])\n",
        encoding="utf-8",
    )
    (tmp_path / "spec.yml").write_text(
        (ISSUES / "b9_adk" / "spec.yml").read_text(encoding="utf-8") + "max_delegation_depth: 3\n", encoding="utf-8"
    )
    reason = lines(tmp_path)["max_delegation_depth"].reason
    assert "a → b" in reason and "a.py:3" in reason


# =========================================================================== B6
def test_b6_several_definitions_say_so() -> None:
    hit = lookup(ISSUES / "b6_const", "no", "progress")
    assert isinstance(hit, Several) and hit.files == ("config.py", "other.py")
    line = lines(ISSUES / "b6_const")["progress_guard.max_no_progress_steps"]
    assert "several values for MAX_NO_PROGRESS found in config.py, other.py; cannot pick" in line.reason
    assert "the code sets no limit" not in line.reason


def test_b6_the_definition_next_to_the_agent_wins(tmp_path: Path) -> None:
    d = _tweak("b6_const", tmp_path, "agent.py", '"""Limits', 'MAX_NO_PROGRESS = 3\n"""Limits')
    hit = lookup(d, "no", "progress", near=("agent.py",))
    assert hit is not None and not isinstance(hit, Several) and hit[0] == 3


def test_b6_imported_constants_and_settings_objects_are_read() -> None:
    bounds = {(b.name, b.value, b.assumed_from) for b in build_observed(ISSUES / "b6_const").bounds}
    assert ("max_iter", 8.0, None) in bounds and ("max_turns", 12.0, None) in bounds


def test_b6_a_pydantic_settings_default_stays_assumed(tmp_path: Path) -> None:
    d = _tweak("b6_const", tmp_path, "config.py", "class Settings:", "class Settings(BaseSettings):")
    assert any(b.name == "max_turns" and b.assumed_from for b in build_observed(d).bounds)


# =========================================================================== B7
def test_b7_tool_names_come_from_the_name_argument_or_attribute() -> None:
    assert set(build_observed(ISSUES / "b7_name").declared_tools) == {"save_file", "load_file", "lookup_item"}


# =========================================================================== B8
def test_b8_a_logger_named_for_audit_counts() -> None:
    line = lines(ISSUES / "b8_audit")["audit_log"]
    assert line.status != "GAP"


def test_b8_a_tool_that_logs_to_nothing_stays_a_gap(tmp_path: Path) -> None:
    d = _tweak("b8_audit", tmp_path, "agent.py", '    audit_trail.info("pay %s", ref)\n', "")
    assert lines(d)["audit_log"].status == "GAP"


def test_b8_a_logger_created_for_audit_counts_under_any_name(tmp_path: Path) -> None:
    d = _tweak("b8_audit", tmp_path, "agent.py", "audit_trail", "trail")
    assert lines(d)["audit_log"].status != "GAP"


# =========================================================================== B9
def test_b9_a_whole_run_limit_of_a_framework_is_read() -> None:
    line = lines(ISSUES / "b9_adk")["aggregate_budget"]
    assert line.status != "GAP" and "max_llm_calls" in line.reason


def test_b9_a_framework_with_no_whole_run_setting_is_a_sign_off() -> None:
    line = lines(ISSUES / "b9_crew")["aggregate_budget"]
    assert line.status == "SIGN-OFF" and "What limits a whole run" in line.reason


def test_b9_a_framework_that_has_one_and_does_not_use_it_stays_a_gap(tmp_path: Path) -> None:
    d = _tweak("b9_adk", tmp_path, "agent.py", "run_config=CONFIG", "run_config=None")
    d2 = d / "agent.py"
    d2.write_text(d2.read_text().replace("CONFIG = RunConfig(max_llm_calls=40)\n", ""))
    assert lines(d)["aggregate_budget"].status == "GAP"


def test_every_found_number_prints_its_place() -> None:
    import re

    for name in ("b1_limits", "b5_found"):
        for ln in lines(ISSUES / name).values():
            for m in re.finditer(r"Found in code: ([\d.]+)(.{0,3})", ln.reason):
                assert m.group(2).startswith(" ("), (name, ln.param, ln.reason)
