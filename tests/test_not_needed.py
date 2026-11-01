"""`not_needed` is the one way to say "this does not apply": any field, any level, any type, a nested key too. It counts
as answered (no AS-100 / AS-101 / AS-102), shows as NOT NEEDED under Human sign-off, and needs a reason under
confirmed: to count as done."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.loader.yaml_loader import load_spec_text
from agenticreq.not_needed import SHAPE_HINT

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"
BASE = (FIXTURES / "specs" / "good.yml").read_text(encoding="utf-8").replace("risk_level: high", "risk_level: medium")


def _spec(extra: str, base: str = BASE) -> str:
    return base + "\n" + extra + "\n"


@pytest.mark.parametrize(
    "field,kind",
    [
        ("conflict_resolution", "string"),
        ("stop_conditions", "list"),
        ("memory_policy", "mapping"),
        ("confidence_threshold", "number"),
        ("fail_closed", "boolean"),
        ("tool_schema_strictness", "enum"),
        ("aggregate_budget", "mapping, Level 3"),
        ("topology", "enum, Level 3"),
    ],
)
def test_not_needed_on_every_field_type(field: str, kind: str) -> None:
    ld = load_spec_text(_spec(f"{field}: not_needed"))
    assert ld.status == "valid", (kind, [i.message for i in ld.issues])
    assert field in ld.not_needed and field not in ld.raw


def test_not_needed_on_a_level_1_list_field_is_not_a_missing_required_field() -> None:
    text = BASE.replace("out_of_scope: [vendor onboarding, price negotiation]", "out_of_scope: not_needed")
    out = _check(text)
    assert not any(r["param"] == "spec" and "missing" in r["reason"] for r in out["results"])
    assert _line(out, "out_of_scope")["status"] == "NOT NEEDED"


def test_conflict_resolution_mapping_still_invalid_but_not_needed_is_valid() -> None:
    assert load_spec_text(_spec("conflict_resolution: {}")).status == "invalid"
    assert load_spec_text(_spec("conflict_resolution: not_needed")).status == "valid"


def test_a_nested_key() -> None:
    ld = load_spec_text(_spec("bounds: {max_cost_usd: not_needed, max_steps: 20}"))
    assert ld.status == "valid" and "bounds.max_cost_usd" in ld.not_needed
    assert ld.spec is not None and ld.spec.bounds is not None and ld.spec.bounds.max_steps == 20
    assert ld.spec.bounds.max_cost_usd is None


@pytest.mark.parametrize(
    "field",
    [
        "kill_switch",
        "fail_closed",
        "audit_log",
        "version_pinning",
        "self_modification",
        "log_immutable",
        "transitive_prohibition",
    ],
)
def test_a_safety_item_at_high_risk_is_a_spec_error(field: str) -> None:
    ld = load_spec_text(_spec(f"{field}: not_needed", BASE.replace("risk_level: medium", "risk_level: high")))
    assert ld.status == "invalid"
    msg = next(i.message for i in ld.issues if i.field == field)
    assert msg.startswith(f"{field} cannot be not_needed at high risk: a high-risk agent must have it in place")


@pytest.mark.parametrize("field", ["agent_id", "risk_level", "goal", "owner"])
def test_never_for_the_fields_every_project_has(field: str) -> None:
    import re

    text = re.sub(rf"^{field}:.*(\n  .*)*", f"{field}: not_needed", BASE, count=1, flags=re.M)
    ld = load_spec_text(text)
    assert ld.status == "invalid" and any(
        f"{field} cannot be not_needed: every project has" in i.message for i in ld.issues
    )


def test_a_confirmed_not_needed_on_a_blocked_field_is_a_spec_error() -> None:
    ld = load_spec_text(
        _spec("confirmed:\n  - {clause: owner, answer: not_needed, by: a, date: 2026-10-01, reason: x}")
    )
    assert ld.status == "invalid" and any("owner cannot be not_needed" in i.message for i in ld.issues)


def test_the_shape_error_says_how_to_fix_it() -> None:
    ld = load_spec_text(_spec("conflict_resolution: {}"))
    assert all(i.message.endswith(SHAPE_HINT) for i in ld.issues if i.rule_id == "AS-107"), ld.issues
    assert SHAPE_HINT == "write a value as shown, remove the line to use the default, or write `not_needed`"


def test_empty_list_still_means_answered_nothing() -> None:
    ld = load_spec_text(_spec("stop_conditions: []"))
    assert ld.status == "valid" and ld.spec is not None and ld.spec.stop_conditions == [] and not ld.not_needed


# --------------------------------------------------------------------------- the report
def _check(text: str) -> dict:  # type: ignore[type-arg]
    import os
    import tempfile

    d = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES / "agents_crewai" / "crew_po", d, dirs_exist_ok=True)
    (d / "spec.yml").write_text(text, encoding="utf-8")
    cwd = os.getcwd()
    try:
        os.chdir(d)
        r = runner.invoke(app, ["check", ".", "--spec", "spec.yml", "-f", "json"])
    finally:
        os.chdir(cwd)
    return json.loads(r.stdout)  # type: ignore[no-any-return]


def _line(out: dict, param: str) -> dict:  # type: ignore[type-arg]
    lines = [r for r in out["results"] if r["param"] == param]
    assert len(lines) == 1, lines
    return lines[0]  # type: ignore[no-any-return]


def test_without_a_reason_it_stays_in_the_left_count() -> None:
    out = _check(_spec("conflict_resolution: not_needed"))
    ln = _line(out, "conflict_resolution")
    assert ln["status"] == "NOT NEEDED" and ln["group"] == "sign_off" and ln["reason"] == "(no reason given)"
    assert ln.get("open") is True and out["counts"]["not_needed"] == 0


def test_with_a_reason_it_is_done() -> None:
    out = _check(
        _spec(
            "conflict_resolution: not_needed\nconfirmed:\n  - {clause: conflict_resolution, answer: "
            'not_needed, by: me@x.com, date: 2026-10-30, reason: "pipeline, no agent-vs-agent conflicts"}'
        )
    )
    ln = _line(out, "conflict_resolution")
    assert ln["status"] == "NOT NEEDED" and ln["reason"] == "pipeline, no agent-vs-agent conflicts."
    assert "open" not in ln and out["counts"]["not_needed"] == 1


def test_the_confirmed_entry_alone_is_enough() -> None:
    out = _check(
        _spec(
            "confirmed:\n  - {clause: memory_policy, answer: not_needed, by: me@x.com, "
            "date: 2026-10-30, reason: stateless}"
        )
    )
    assert _line(out, "memory_policy")["reason"] == "Stateless."


def test_a_not_needed_nested_key_replaces_only_its_own_line() -> None:
    out = _check(_spec("bounds: {max_cost_usd: not_needed, max_steps: 20}"))
    assert _line(out, "bounds.max_cost_usd")["status"] == "NOT NEEDED"
    assert all(r["status"] != "NOT NEEDED" for r in out["results"] if r["param"] == "bounds.max_steps")


def test_a_not_needed_parameter_is_never_a_gap_nor_a_pass_and_appears_once() -> None:
    out = _check(_spec("fail_closed: not_needed\ntool_schema_strictness: not_needed"))
    for p in ("fail_closed", "tool_schema_strictness"):
        assert _line(out, p)["status"] == "NOT NEEDED"


def test_params_says_it_once_at_the_top() -> None:
    r = runner.invoke(app, ["params", "--brief"], terminal_width=200)
    assert r.output.count("not_needed") >= 1 and r.output.lstrip().startswith("Any field")
