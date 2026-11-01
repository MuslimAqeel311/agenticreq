"""Points 2 + 3 from the fin-qa-agent run: an invalid spec is ONE hard error (SPEC INVALID), never a pile of
blockers; repeated defects collapse; every error shows what to write; prose is accepted as a note and
reported UNKNOWN, never satisfied and never a crash; `validate --fix` rewrites only the unambiguous."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run, validate_spec
from agenticreq.loader import load_spec, load_spec_text
from agenticreq.models.finding import Bucket
from agenticreq.report import render_console, render_sarif

FIN = Path(__file__).parent / "fixtures" / "specs" / "real_world" / "fin_qa_agent.yml"
FIN_SHA = "26823d4dec513b8a6a9617531cd2564e338e2888843311c2a277a11c40ae700e"  # the file exactly as written
runner = CliRunner()
RAW_PYDANTIC = ("Input should be", "instance of", "valid dictionary", "Field required", "extra_forbidden")

GOOD = """
agent_id: a
risk_level: medium
goal: {statement: s, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@x.com
technical_owner: t@x.com
"""


def _v(tmp: Path, text: str):  # type: ignore[no-untyped-def]
    p = tmp / "spec.yml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return validate_spec(p), p


# ------------------------------------------------------------------ the fin-qa-agent spec, unchanged
def test_fixture_is_the_original_file() -> None:
    assert hashlib.sha256(FIN.read_bytes()).hexdigest() == FIN_SHA


def test_fin_spec_is_one_spec_invalid() -> None:
    r = validate_spec(FIN)
    assert r.spec_status == "invalid" and r.exit_code == 2
    assert r.findings == [] and r.by_bucket()[Bucket.BLOCKERS] == []  # no rule ran: nothing is "unsafe" yet
    invalid = [i for i in r.spec_issues if i.kind == "invalid"]
    assert [i.field for i in invalid] == ["limits"] and invalid[0].count == 5  # limits.0-4 are ONE issue
    prose = {i.field for i in r.spec_issues if i.kind == "prose"}
    assert prose == {"role_boundary", "prohibited", "approval_required", "escalation", "untrusted_inputs", "data_rules"}
    prohibited = next(i for i in r.spec_issues if i.field == "prohibited")
    assert prohibited.count == 6  # the six sentences are one issue, not six
    [ph] = [i for i in r.spec_issues if i.kind == "placeholder"]
    assert ph.count == 2 and "${OWNER}" in " ".join(ph.items)
    for i in r.spec_issues:
        assert i.example and i.expected, i.field  # every problem shows exactly what to write
        assert not any(x in i.message for x in RAW_PYDANTIC), i.message


def test_fin_spec_console_shows_one_result_with_examples() -> None:
    text = render_console(validate_spec(FIN))
    assert text.count("SPEC INVALID (1)") == 2  # the header and its own section: one hard error
    assert "BLOCKERS" not in text and "WARNINGS" not in text
    assert "write it like:" in text and "tool: issue_refund" in text
    assert not any(x in text for x in RAW_PYDANTIC)


def test_check_stops_before_rules_on_an_invalid_spec(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    r = runner.invoke(app, ["check", str(tmp_path), "--spec", str(FIN), "-f", "json"])
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["spec"]["status"] == "invalid" and not d["findings"]


def test_sarif_has_one_spec_invalid_result() -> None:
    sarif = json.loads(render_sarif(validate_spec(FIN)))
    [res] = sarif["runs"][0]["results"]
    assert res["ruleId"] == "AS-100" and "limits" in res["message"]["text"]


# ------------------------------------------------------------------ collapse
def test_six_missing_reasons_are_one_finding(tmp_path: Path) -> None:
    r, _ = _v(tmp_path, GOOD.replace("prohibited: []", "prohibited: [a, b, c, d, e, f]"))
    [f] = [f for f in r.findings if f.rule_id == "AS-109"]
    assert "(6 entries)" in f.title and f.detail.count("\n  - ") == 6


def test_repeated_shape_errors_are_one_issue(tmp_path: Path) -> None:
    lims = ", ".join(f"{{tool: t{i}, arg: x, limit: {i}}}" for i in range(5))
    r, _ = _v(tmp_path, GOOD.replace("limits: []", f"limits: [{lims}]"))
    [i] = [i for i in r.spec_issues if i.kind == "invalid"]
    assert i.count == 5 and "limits[].limit" in i.message and "must be a mapping" in i.message
    assert "limit: {min: <number>, max: <number>}" in i.example  # examples show <number>, never a value


@pytest.mark.parametrize(
    "patch,expect",
    [
        ("risk_level: medium", "risk_level: extreme"),
        ("owner: o@x.com", "ownr: o@x.com\nowner: o@x.com"),
        ("out_of_scope: []", "out_of_scope: {a: 1}"),
    ],
)
def test_shape_errors_are_human(tmp_path: Path, patch: str, expect: str) -> None:
    r, _ = _v(tmp_path, GOOD.replace(patch, expect))
    [i] = [i for i in r.spec_issues if i.kind == "invalid"]
    assert not any(x in i.message for x in RAW_PYDANTIC), i.message
    if "extreme" in expect:
        assert "must be one of low | medium | high | critical" in i.message and i.example == "risk_level: high"
    if "ownr" in expect:
        assert "Did you mean `owner`?" in i.message
    if "{a: 1}" in expect:
        assert "must be a list" in i.message


def test_bad_yaml_is_one_invalid_issue(tmp_path: Path) -> None:
    r, _ = _v(tmp_path, "agent_id: a\n\tbad: tab\n")
    assert r.spec_status == "invalid" and r.findings == []
    assert "not valid YAML" in r.spec_issues[0].message


# ------------------------------------------------------------------ prose is accepted, never satisfied
def test_prose_is_unknown_not_satisfied_not_a_crash(tmp_path: Path) -> None:
    text = GOOD.replace("escalation: []", "escalation: ask a human when unsure").replace(
        "prohibited: []", "prohibited: [shell execution, delete_db]"
    )
    r, p = _v(tmp_path, text)
    assert r.spec_status == "incomplete" and r.exit_code == 0  # loads: validate passes, nothing invalid
    esc = next(f for f in r.findings if f.rule_id == "AS-108")
    assert esc.bucket is Bucket.UNKNOWNS and "cannot be verified" in esc.detail and "metric" in (esc.remediation or "")
    pro = next(f for f in r.findings if f.rule_id == "AS-118")
    assert pro.bucket is Bucket.UNKNOWNS and "shell execution" in pro.evidence
    loaded = load_spec(p)
    assert loaded.spec is not None and loaded.spec.escalation is None  # undeclared, not "satisfied"
    assert loaded.spec.prohibited_tools() == {"delete_db"}  # the name is kept; the sentence is not a tool
    assert loaded.notes["escalation"] == "ask a human when unsure"


def test_check_refuses_a_spec_with_prose(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    _, p = _v(tmp_path, GOOD.replace("escalation: []", "escalation: ask a human"))
    assert runner.invoke(app, ["check", str(tmp_path), "--spec", str(p)]).exit_code == 1


def test_note_form_loads_and_keeps_the_sentence() -> None:
    loaded = load_spec_text(
        textwrap.dedent(
            GOOD.replace(
                "escalation: []",
                """escalation:
  note: "ask a human when unsure"
  rules: [{metric: confidence, op: "<", threshold: 0.5, to: analyst}]""",
            )
        )
    )
    assert loaded.status == "valid" and loaded.spec and loaded.spec.escalation
    assert loaded.notes["escalation"] == "ask a human when unsure"


def test_placeholder_is_unknown(tmp_path: Path) -> None:
    r, p = _v(tmp_path, GOOD.replace("owner: o@x.com", "owner: ${OWNER}"))
    assert r.spec_status == "incomplete"
    f = next(f for f in r.findings if "Placeholder" in f.title)
    assert f.bucket is Bucket.UNKNOWNS
    assert load_spec(p).spec.owner is None  # type: ignore[union-attr]


def test_todos_are_one_unknown_not_blockers(tmp_path: Path) -> None:
    r, _ = _v(tmp_path, GOOD.replace("agent_id: a", "agent_id: TODO").replace("owner: o@x.com", "owner: TODO"))
    [f] = [f for f in r.findings if f.rule_id == "AS-101"]
    assert f.bucket is Bucket.UNKNOWNS and f.title == "2 fields still TODO"
    assert not r.by_bucket()[Bucket.BLOCKERS]


# ------------------------------------------------------------------ validate --fix
def test_fix_is_dry_run_by_default(tmp_path: Path) -> None:
    p = tmp_path / "spec.yml"
    p.write_bytes(FIN.read_bytes())
    r = runner.invoke(app, ["validate", str(p), "--fix"])
    assert r.exit_code == 0 and "+  note:" in r.stdout and "nothing written" in r.output
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FIN_SHA  # nothing written


def test_fix_apply_gives_a_loadable_spec_and_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "spec.yml"
    p.write_bytes(FIN.read_bytes())
    assert runner.invoke(app, ["validate", str(p), "--fix", "--apply"]).exit_code == 0
    after = load_spec(p)
    assert after.status == "incomplete" and not after.invalid  # no shape error left; only TODOs and placeholders
    text = p.read_text(encoding="utf-8")
    assert "# read from env OWNER" in text  # comments outside the rewritten fields are kept
    assert '"giving investment advice"' in text and "rules: TODO" in text  # verbatim note, structure TODO
    assert "max_iterations: 8" in text  # ambiguous limits: kept as a note, never guessed into a rule
    assert {"limits", "escalation"} <= set(after.notes)
    assert "nothing unambiguous" in runner.invoke(app, ["validate", str(p), "--fix"]).output


def test_fix_keeps_crlf_and_bom(tmp_path: Path) -> None:
    p = tmp_path / "spec.yml"
    p.write_bytes(
        b"\xef\xbb\xbf"
        + textwrap.dedent(GOOD)
        .replace("escalation: []", "escalation: ask a human")
        .replace("\n", "\r\n")
        .encode("utf-8")
    )
    assert runner.invoke(app, ["validate", str(p), "--fix", "--apply"]).exit_code == 0
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert load_spec(p).notes["escalation"] == "ask a human"


def test_fix_only_rewrites_by_shape(tmp_path: Path) -> None:
    from agenticreq.loader.fix_spec import plan

    rw = {
        r.field: r
        for r in plan(
            textwrap.dedent("""
        goal: answer questions
        role_boundary: [search_kb, lookup]
        out_of_scope: price predictions
        escalation: {metric: confidence, op: "<", threshold: 0.5, to: analyst}
        limits: [{tool: pay, arg: amount, limit: {max: 5}}]
        """)
        )
    }
    assert set(rw) == {"goal", "role_boundary", "out_of_scope", "escalation"}  # a valid limits list is left alone
    assert 'statement: "answer questions"' in rw["goal"].text and "done_criteria: TODO" in rw["goal"].text
    assert 'allowed_tools: ["search_kb", "lookup"]' in rw["role_boundary"].text


def test_engine_run_on_invalid_spec_runs_no_rule() -> None:
    r = run(mode="check", loaded=load_spec(FIN))
    assert r.spec_status == "invalid" and not r.findings and not r.passed


# ------------------------------------------------------------------ per-entry routing (no blanket hint)
from agenticreq.loader.routes import ROUTES, UNKNOWN, describe, route  # noqa: E402


@pytest.mark.parametrize(
    "key,field",
    [
        ("max_iter", "bounds.max_steps"),
        ("max_iterations", "bounds.max_steps"),
        ("max_steps", "bounds.max_steps"),
        ("max_turns", "bounds.max_steps"),
        ("recursion_limit", "bounds.max_steps"),
        ("max_tool_calls", "bounds.max_tool_calls"),
        ("max_tool_calls_per_question", "bounds.max_tool_calls"),
        ("max_cost", "bounds.max_cost_usd"),
        ("max_cost_usd", "bounds.max_cost_usd"),
        ("max_duration", "bounds.max_duration_s"),
        ("timeout", "bounds.max_duration_s"),
        ("rate_limit", "rate_limits"),
        ("sec_rate_limit_per_second", "rate_limits"),
        ("calls_per_minute", "rate_limits"),
        ("llm_rpm", "rate_limits"),
        ("temperature", "(no spec field)"),
        ("top_p", "(no spec field)"),
        ("seed", "(no spec field)"),
        ("max_tokens", "context_policy.max_tokens"),
        ("max_question_chars", "limits (a tool argument)"),
        ("prompt_length", "limits (a tool argument)"),
    ],
)
def test_route_table(key: str, field: str) -> None:
    r = route(key)
    assert r is not None and r.field == field


@pytest.mark.parametrize("key", ["frobnicate", "max_widgets", "quota", "budget_owner", "max_question"])
def test_unmatched_key_falls_back_instead_of_guessing(key: str) -> None:
    assert route(key) is None
    text = describe(key, 3)
    assert UNKNOWN in text and not any(r.field in text for r in ROUTES)


def test_rate_per_second_is_converted_to_per_minute() -> None:
    assert "requests_per_minute: 600" in describe("sec_rate_limit_per_second", 10)
    assert "requests_per_minute: 30" in describe("llm_rpm", 30)
    assert "<requests per minute>" in describe("rate_limit", 5)  # no unit in the name: not guessed


def test_fin_limits_entries_each_name_their_destination() -> None:
    [i] = [i for i in validate_spec(FIN).spec_issues if i.kind == "invalid"]
    got = {x.split(":", 1)[0]: x for x in i.items}
    assert "bounds: {max_steps: 8}" in got["limits.max_iterations"]
    assert "bounds: {max_tool_calls: 12}" in got["limits.max_tool_calls_per_question"]
    assert "max_length: 1000" in got["limits.max_question_chars"] and "<which tool>" in got["limits.max_question_chars"]
    assert "rate_limits: {requests_per_minute: 600}" in got["limits.sec_rate_limit_per_second"]
    assert "bounds" not in got["limits.sec_rate_limit_per_second"]
    assert "(no spec field)" in got["limits.temperature"] and "rate_limits:" not in got["limits.temperature"]
    assert "Run-wide caps" not in i.message  # no blanket hint for the whole group


def test_unknown_key_in_limits_falls_back(tmp_path: Path) -> None:
    r, _ = _v(tmp_path, GOOD.replace("limits: []", "limits: {frobnicate: 7, max_iterations: 4}"))
    [i] = [i for i in r.spec_issues if i.kind == "invalid"]
    items = {x.split(":", 1)[0]: x for x in i.items}
    assert UNKNOWN in items["limits.frobnicate"] and "bounds: {max_steps: 4}" in items["limits.max_iterations"]


def test_console_sections_are_separate_and_numbered_from_one() -> None:
    text = render_console(validate_spec(FIN))
    hard, _, soft = text.partition("ALSO TO WRITE (7)  ")
    assert "SPEC INVALID (1)  blocks everything" in hard and "1. limits - wrong shape" in hard
    assert "2. " not in hard.split("SPEC INVALID (1)  blocks everything", 1)[1]  # only one hard error
    assert "1. owner, technical_owner" in soft and "7. data_rules" in soft and "8. " not in soft
    assert "ALSO TO WRITE (7) - fields that are TODO" in text  # header count == section count
