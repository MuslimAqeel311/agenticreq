"""Phase 5 acceptance: `init` and `audit` work on LangGraph / CrewAI agents with no spec file."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from agenticreq.cli import app

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
LG = FIX / "agents" / "langgraph_po"
CREW = FIX / "agents_crewai" / "crew_po"
CREW_CAPTURES = FIX / "captures" / "crewai_po.json"


def test_audit_langgraph_ast_only_no_spec() -> None:
    r = runner.invoke(app, ["audit", str(LG), "--format", "json"])
    d = json.loads(r.stdout)
    ids = {f["rule_id"] for f in d["findings"]}
    assert {"AS-210", "AS-903"} <= ids  # unpinned model, no runtime evidence
    # the fixture's submit_order / pay_invoice / issue_refund are stubs whose bodies write nothing: by the body
    # (not the name) they are read-only, so no "write without approval" gap
    assert "AS-212" not in ids
    assert "AS-211" not in ids  # submit_order's body shows no repeatable side effect: name alone is not evidence
    assert "AS-208" not in ids  # recursion_limit found


def test_audit_crewai_with_captures_no_spec() -> None:
    r = runner.invoke(app, ["audit", str(CREW), "--captures", str(CREW_CAPTURES), "--format", "json"])
    d = json.loads(r.stdout)
    rules = [f["rule_id"] for f in d["findings"]]
    assert rules.count("AS-602") == 9  # every parameter description dropped by CrewAI
    assert rules.count("AS-604") == 4  # optional params sent as required while keeping their default
    assert rules.count("AS-603") == 0  # superseded by AS-604 on the same field
    assert "AS-202" not in rules  # delegate_work_to_coworker / ask_question_to_coworker allowlisted
    assert d["metadata"]["framework"] == "crewai"


def test_audit_sarif_output(tmp_path: Path) -> None:
    out = tmp_path / "r.sarif"
    runner.invoke(app, ["audit", str(LG), "--format", "sarif", "--output", str(out)])
    s = json.loads(out.read_text(encoding="utf-8"))
    assert s["version"] == "2.1.0" and s["runs"][0]["results"]


def test_init_scaffold_shape(tmp_path: Path) -> None:
    out = tmp_path / "spec.yml"
    r = runner.invoke(
        app, ["init", str(CREW), "--captures", str(CREW_CAPTURES), "--out", str(out), "--agent-id", "crew-po"]
    )
    assert r.exit_code == 0, r.output
    text = out.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    d = data["_derived"]
    assert d["framework"].startswith("crewai 1.")
    assert d["execution_model"] == "schema-based"
    assert d["declared_tools"] == ["get_inventory", "lookup_clause", "submit_order"]
    assert d["injected_by_framework"] == ["ask_question_to_coworker", "delegate_work_to_coworker"]
    assert d["write_tools"] == []  # crew_po's submit_order is a stub that writes nothing: the body decides
    assert len(d["system_prompt_hashes"]) == 2
    assert "NOT pinned" in text
    for key in (
        "risk_level",
        "goal",
        "out_of_scope",
        "prohibited",
        "approval_required",
        "limits",
        "escalation",
        "untrusted_inputs",
        "owner",
    ):
        assert data[key] == "TODO", key


def test_init_without_runtime_marks_observed_unknown(tmp_path: Path) -> None:
    out = tmp_path / "spec.yml"
    runner.invoke(app, ["init", str(LG), "--out", str(out)])
    d = yaml.safe_load(out.read_text(encoding="utf-8"))["_derived"]
    assert d["observed_tools"] == "not seen yet" and d["never_reached"] == "not seen yet"


def test_init_refuses_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "spec.yml"
    out.write_text("x: 1\n")
    r = runner.invoke(app, ["init", str(LG), "--out", str(out)])
    assert r.exit_code == 2 and out.read_text(encoding="utf-8") == "x: 1\n"


def test_check_fails_on_scaffold_todos(tmp_path: Path) -> None:
    out = tmp_path / "spec.yml"
    runner.invoke(app, ["init", str(LG), "--out", str(out)])
    r = runner.invoke(app, ["check", str(LG), "--spec", str(out), "--format", "json"])
    assert r.exit_code == 1  # check refuses an unfinished spec
    d = json.loads(r.stdout)
    [todos] = [f for f in d["findings"] if f["rule_id"] == "AS-101"]  # one finding, not 13
    assert todos["group"] == "gap" and todos["severity"] == "blocker"  # a TODO is a spec vs code gap
    # the 12 Level-1 fields (owner + technical_owner: 13 keys) and agent_registry: names are the user's requirement
    assert todos["title"] == "14 fields still TODO"
    assert d["spec"]["status"] == "incomplete"
