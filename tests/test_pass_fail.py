"""The four result groups: every line is a spec vs code gap (GAP), a human sign-off (SIGN-OFF / SIGNED), a check a
mock run can settle (CHECK), or passed (PASS). Exit codes follow --fail-on gaps | all | never."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import FailOn, _exit_code, app
from agenticreq.models.confidence import Confidence, ConfidenceLevel
from agenticreq.models.finding import Finding, PassedCheck, Report
from agenticreq.models.verdict import GROUP_OF, finding_verdict, passed_verdict, results, totals

FIX = Path(__file__).parent / "fixtures" / "passfail"
runner = CliRunner()


def _json(*args: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    res = runner.invoke(app, [*args, "-f", "json"])
    return res.exit_code, json.loads(res.stdout)


def _lines(data: dict, param: str) -> list[dict]:  # type: ignore[type-arg]
    return [r for r in data["results"] if r["param"] == param]


def _check(name: str, *extra: str, spec: Path | None = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    return _json("check", str(FIX / name), "--spec", str(spec or FIX / name / "spec.yml"), *extra)


def _every_line_has_one_group(data: dict) -> None:  # type: ignore[type-arg]
    for r in data["results"]:
        assert r["group"] == GROUP_OF[r["status"]], r
        assert r["reason"] and not r["reason"].lower().startswith(("missing:", "not proven:")), r
        if r["group"] == "mock_run":  # every line that waits for a mock run ends with the next command
            assert r["reason"].endswith(("agenticreq mock-run", "agenticreq init")), r


# --------------------------------------------------------------------------- a. everything proven
def test_all_proven_is_all_pass_exit_0() -> None:
    report = Report(
        agent_id="support-agent",
        passed=[
            PassedCheck(
                rule_id="AS-209", subject="max_turns", confidence=Confidence.proven(), note="max_turns=10 <= 100"
            ),
            PassedCheck(rule_id="AS-210", subject="gpt-4o-2024-08-06", confidence=Confidence.proven()),
            PassedCheck(rule_id="AS-206", subject="lookup_order", confidence=Confidence.proven()),
        ],
        metadata={"analysis": {"files": 1, "framework": "openai-agents", "tools": 1, "agents": 1}},
    )
    lines = results(report)
    assert [ln.status for ln in lines] == ["PASS"] * 3
    assert all(ln.reason.startswith("Found in code") for ln in lines)
    assert totals(lines) == {"gaps": 0, "sign_off_left": 0, "signed": 0, "not_needed": 0, "mock_run": 0, "passed": 3}
    assert _exit_code(report, FailOn.gaps) == 0 and _exit_code(report, FailOn.all) == 0
    assert report.exit_code == 0


def test_real_run_without_capture_is_never_all_pass() -> None:
    # the clean example still lacks a run of the agent (AS-903) and sign-offs: open items, not PASSes
    code, data = _check("clean", "--fail-on", "all")
    assert code == 1 and data["result"] == "FAIL"
    assert data["counts"]["mock_run"] + data["counts"]["sign_off_left"] > 0
    assert {r["param"] for r in data["results"] if r["status"] == "PASS"} >= {"version_pinning", "bounds.max_steps"}
    _every_line_has_one_group(data)


# --------------------------------------------------------------------------- b. prohibited tool bound in code
def test_prohibited_tool_bound_is_a_gap_exit_1() -> None:
    code, data = _check("prohibited")
    assert code == 1
    (line,) = _lines(data, "prohibited")
    assert line["status"] == "GAP" and line["group"] == "gap"
    assert "issue_refund" in line["reason"]
    _every_line_has_one_group(data)


# --------------------------------------------------------------------------- c. MCP tools, no list supplied
def test_mcp_tools_without_list_wait_for_a_mock_run_never_pass() -> None:
    code, data = _check("mcp", "--fail-on", "all")
    assert code == 1
    (line,) = _lines(data, "prohibited")
    assert line["status"] == "CHECK" and line["group"] == "mock_run"
    assert "--mcp-tools" in line["reason"] and line["reason"].endswith("agenticreq mock-run")
    assert all(r["status"] != "PASS" for r in data["results"] if r["param"] in ("prohibited", "extraction"))
    _every_line_has_one_group(data)


# --------------------------------------------------------------------------- d. model from an env var
def test_env_model_waits_for_a_mock_run_and_is_counted() -> None:
    code, data = _check("envmodel", "--fail-on", "all")
    assert code == 1
    (line,) = _lines(data, "version_pinning")
    assert line["status"] == "CHECK" and "env var" in line["reason"]
    assert data["counts"]["mock_run"] >= 1
    assert data["coverage"]["total_clauses"] > data["coverage"]["extracted_clauses"]  # still in the total


# --------------------------------------------------------------------------- e. sign-off open -> signed
def test_sign_off_open_then_signed_never_passed(tmp_path: Path) -> None:
    _, before = _check("clean")
    (owner,) = _lines(before, "owner")
    assert owner["status"] == "SIGN-OFF" and owner["group"] == "sign_off"
    assert owner["reason"] == "Who owns this agent? Confirm under confirmed: in spec.yml."

    spec = tmp_path / "spec.yml"
    spec.write_text(
        (FIX / "clean" / "spec.yml").read_text()
        + "confirmed:\n  - {clause: owner, by: lead@example.com, date: 2026-10-01, "
        "evidence: TICKET-1, expires: 2099-01-01}\n"
    )
    _, after = _check("clean", spec=spec)
    (owner,) = _lines(after, "owner")
    assert owner["status"] == "SIGNED" and owner["group"] == "sign_off"  # signed is never counted as passed
    assert owner["reason"] == "Signed by lead@example.com on 2026-10-01."
    assert after["counts"]["signed"] >= 1


# --------------------------------------------------------------------------- f. --fail-on
def test_fail_on_gaps_ignores_open_sign_offs_and_checks() -> None:
    code, data = _check("envmodel")  # default: gaps
    assert data["counts"]["gaps"] == 0 and data["counts"]["mock_run"] > 0
    assert code == 0 and data["result"] == "PASS"


def test_fail_on_all_fails_on_open_items() -> None:
    code, data = _check("envmodel", "--fail-on", "all")
    assert code == 1 and data["result"] == "FAIL" and data["fail_on"] == "all"


def test_fail_on_gaps_fails_on_a_gap() -> None:
    assert _check("prohibited", "--fail-on", "gaps")[0] == 1
    assert _check("prohibited")[0] == 1


def test_fail_on_never() -> None:
    code, data = _check("prohibited", "--fail-on", "never")
    assert code == 0 and data["counts"]["gaps"] > 0


def test_old_fail_on_values_are_rejected() -> None:
    for old in ("any", "missing"):
        res = runner.invoke(
            app, ["check", str(FIX / "clean"), "--spec", str(FIX / "clean" / "spec.yml"), "--fail-on", old]
        )
        assert res.exit_code == 2, old  # a usage error: the values are gaps | all | never


# --------------------------------------------------------------------------- g. two agents, no delegation
def test_two_agents_are_both_shown() -> None:
    res = runner.invoke(app, ["audit", str(FIX / "two_agents")], terminal_width=200)
    assert "[reporter]" in res.stdout
    _, data = _json("audit", str(FIX / "two_agents"))
    by_agent = {r["agent"] for r in data["results"]}
    assert "reporter" in by_agent and {a["name"] for a in data["metadata"]["agents"]} >= {"support", "reporter"}
    (write,) = [r for r in data["results"] if r["param"] == "write_safety"]
    assert write["agent"] == "reporter" and write["status"] == "GAP"


# --------------------------------------------------------------------------- mapping invariants
@pytest.mark.parametrize("level", [ConfidenceLevel.UNKNOWN, ConfidenceLevel.INSUFFICIENT_EVIDENCE])
@pytest.mark.parametrize("severity", ["blocker", "warning", "info"])
def test_undecided_is_never_pass(level: ConfidenceLevel, severity: str) -> None:
    f = Finding(
        rule_id="AS-203",
        severity=severity,
        confidence=Confidence(level=level),  # type: ignore[arg-type]
        title="Cannot rule out prohibited tool x",
        detail="d",
        evidence="e",
    )
    v = finding_verdict(f)
    assert v.status == "CHECK" and v.group == "mock_run" and v.reason.endswith("agenticreq mock-run")


def test_code_pattern_pass_waits_for_a_mock_run() -> None:
    v = passed_verdict(
        PassedCheck(
            rule_id="AS-219",
            subject="tool error handling",
            confidence=Confidence.heuristic(),
            note="no swallowed exception in 3 tool bodies",
        )
    )
    assert v.status == "CHECK"
    assert v.reason == (
        "Code pattern only (no swallowed exception in 3 tool bodies). Add a mock that returns an "
        "error, then run: agenticreq mock-run"
    )


def test_observed_pass_is_pass() -> None:
    v = passed_verdict(PassedCheck(rule_id="AS-229", subject="max_steps", confidence=Confidence.observed(0, 3)))
    assert v.status == "PASS" and v.reason.startswith("Held in 3 of 3 runs")


def test_verbose_shows_rule_ids_default_does_not() -> None:
    args = ["check", str(FIX / "prohibited"), "--spec", str(FIX / "prohibited" / "spec.yml")]
    plain = runner.invoke(app, args, terminal_width=200)
    assert "AS-203" not in plain.stdout and "SPEC VS CODE GAPS" in plain.stdout
    assert "Result: FAIL" in plain.stdout
    verbose = runner.invoke(app, [*args, "--verbose"], terminal_width=200)
    assert "AS-203" in verbose.stdout and "agent.py:17" in verbose.stdout


def test_sarif_levels_follow_the_group() -> None:
    res = runner.invoke(app, ["check", str(FIX / "mcp"), "--spec", str(FIX / "mcp" / "spec.yml"), "-f", "sarif"])
    run = json.loads(res.stdout)["runs"][0]
    want = {"GAP": "error", "CHECK": "note", "SIGN-OFF": "note", "SIGNED": "none", "PASS": "none"}
    assert run["results"]
    for r in run["results"]:
        assert r["level"] == want[r["properties"]["status"]]
    assert {r["level"] for r in run["results"]} >= {"note"}


def test_spec_invalid_keeps_exit_2(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yml"
    spec.write_text("agent_id: x\nrisk_level: [not, a, level]\n")
    for fail_on in ("gaps", "all"):
        assert (
            runner.invoke(app, ["check", str(FIX / "clean"), "--spec", str(spec), "--fail-on", fail_on]).exit_code == 2
        )


def test_nothing_checked_keeps_exit_3(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("no code here")
    shutil.copy(FIX / "clean" / "spec.yml", tmp_path / "spec.yml")
    res = runner.invoke(app, ["audit", str(tmp_path)])
    assert res.exit_code == 3
    assert "Nothing was checked" in res.stdout and "Result: FAIL" in res.stdout
    assert "PASSED" not in res.stdout and "SPEC VS CODE GAPS" not in res.stdout  # no groups: they would read as a pass


# --------------------------------------------------------------------------- scenario table, continued
def test_observed_all_runs_pass_some_runs_wait() -> None:
    held = passed_verdict(
        PassedCheck(rule_id="AS-230", subject="stop_conditions", confidence=Confidence.observed(0, 4))
    )
    assert held.status == "PASS" and held.reason.startswith("Held in 4 of 4 runs")
    some = passed_verdict(
        PassedCheck(rule_id="AS-230", subject="stop_conditions", confidence=Confidence.observed(1, 4))
    )
    assert some.status == "CHECK" and some.reason.startswith("Held in only 3 of 4 runs")


def test_spec_field_still_todo_is_a_gap(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yml"
    spec.write_text((FIX / "clean" / "spec.yml").read_text().replace("owner: owner@example.com", "owner: TODO"))
    code, data = _check("clean", spec=spec)
    assert code == 1
    reasons = [r["reason"] for r in _lines(data, "owner")]
    assert "owner is still TODO in spec.yml." in reasons, reasons
    assert all(r["status"] == "GAP" for r in _lines(data, "owner"))


_WAIVER = (
    "waivers:\n  - {rule_id: AS-223, reason: single environment by design, approved_by: lead@example.com, "
    "expires: %s}\n"
)


def test_valid_waiver_passes_expired_waiver_fails(tmp_path: Path) -> None:
    base = (FIX / "clean" / "spec.yml").read_text() + "environment_separation: true\n"
    valid, expired = tmp_path / "valid.yml", tmp_path / "expired.yml"
    valid.write_text(base + _WAIVER % "2099-01-01")
    expired.write_text(base + _WAIVER % "2020-01-01")
    unwaived = tmp_path / "unwaived.yml"
    unwaived.write_text(base)
    _, before = _check("clean", spec=unwaived)
    (line,) = _lines(before, "environment_separation")
    assert line["status"] == "GAP" and "AS-223" in line["rule_ids"]  # AS-223 itself
    _, ok = _check("clean", spec=valid)
    (line,) = _lines(ok, "environment_separation")
    assert line["status"] != "GAP" and "AS-223" in line["rule_ids"]  # the waiver is what turned it
    waived = Report(agent_id="a", metadata={"waived": ok["metadata"]["waived"]})
    (only,) = results(waived)  # the waived result on its own line reads "waived by you"
    assert only.status == "PASS" and only.reason == "waived by you: single environment by design"
    _, bad = _check("clean", spec=expired)
    lines = [r for r in bad["results"] if "AS-223" in r["rule_ids"]]
    assert lines and all(r["status"] == "GAP" for r in lines)


def test_audit_has_no_spec_field_lines_and_the_result_line_second() -> None:
    from agenticreq.params import level

    spec_fields = {p.name for p in level(1)}
    for name in ("clean", "two_agents", "prohibited"):
        res = runner.invoke(app, ["audit", str(FIX / name)])
        out = [ln for ln in res.stdout.splitlines() if ln.strip()]
        assert out[0].startswith("agenticreq audit") and out[1].startswith("Result: "), out[:2]
        _, data = _json("audit", str(FIX / name))
        assert not {r["param"] for r in data["results"]} & spec_fields, name


def test_invalid_spec_exit_2_without_groups(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yml"
    spec.write_text("agent_id: x\nrisk_level: [not, a, level]\n")
    res = runner.invoke(app, ["check", str(FIX / "clean"), "--spec", str(spec)])
    assert res.exit_code == 2
    assert "SPEC INVALID" in res.stdout and "PASSED (" not in res.stdout and "SPEC VS CODE GAPS" not in res.stdout


def test_real_run_with_captures_passes_with_everything_signed() -> None:
    d = FIX / "captured"
    code, data = _json(
        "check", str(d), "--spec", str(d / "spec.yml"), "--captures", str(d / "captures.json"), "--fail-on", "all"
    )
    assert code == 0 and data["result"] == "PASS"
    assert data["results"] and all(r["status"] in ("PASS", "SIGNED") for r in data["results"])
    assert data["counts"]["gaps"] == data["counts"]["mock_run"] == data["counts"]["sign_off_left"] == 0
    reasons = {r["param"]: r["reason"] for r in data["results"]}
    assert reasons["stop_conditions"].startswith("Held in 1 of 1 runs")  # from the capture
    assert reasons["version_pinning"].startswith("Found in code")
    assert reasons["owner"].startswith("Signed by ")
    assert data["counts"]["signed"] >= 1 and data["counts"]["passed"] >= 1
