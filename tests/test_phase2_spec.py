"""Phase 2 acceptance: deliberately broken specs — the library must catch every one."""

from datetime import date
from pathlib import Path

import pytest

from agenticreq.defaults import floor_violations, tier2_defaults
from agenticreq.engine import validate_spec
from agenticreq.loader import SpecBuilder, load_spec, load_spec_text

SPECS = Path(__file__).parent / "fixtures" / "specs"
TODAY = date(2026, 10, 15)

EXPECTED = {
    "01_prohibited_and_allowed.yml": "AS-103",
    "02_prohibited_and_approval.yml": "AS-103",
    "03_approver_missing.yml": "AS-104",
    "04_todo_left.yml": "AS-101",
    "05_limit_overlap.yml": "AS-105",
    "06_limit_min_gt_max.yml": "AS-105",
    "07_expired_waiver.yml": "AS-106",
    "08_bad_risk_level.yml": "AS-107",
    "09_prose_escalation.yml": "AS-108",
    "10_timeout_proceeds.yml": "AS-110",
    "11_below_floor.yml": "AS-111",
    "12_missing_untrusted_inputs.yml": "AS-102",
    "13_unregistered_delegate.yml": "AS-114",
    "14_extends_missing.yml": "AS-115",
    "15_data_rule_contradiction.yml": "AS-116",
    "16_unknown_key.yml": "AS-117",
    "17_duplicate_limit_key.yml": "AS-105",
}


def test_good_spec_is_clean() -> None:
    r = validate_spec(SPECS / "good.yml", today=TODAY)
    assert r.findings == [], [f"{f.rule_id} {f.detail}" for f in r.findings]


@pytest.mark.parametrize("name,rule_id", sorted(EXPECTED.items()))
def test_broken_spec_caught(name: str, rule_id: str) -> None:
    """Shape errors are spec issues (the spec is INVALID and no rule runs); the rest are findings."""
    r = validate_spec(SPECS / "broken" / name, today=TODAY)
    hits = [*(f for f in r.findings if f.rule_id == rule_id), *(i for i in r.spec_issues if i.rule_id == rule_id)]
    ids = {f.rule_id for f in r.findings} | {i.rule_id for i in r.spec_issues}
    assert hits, f"{name}: expected {rule_id}, got {sorted(ids)}"
    hit = hits[0]
    assert hit.spec_ref is not None and hit.spec_ref.line is not None
    assert getattr(hit, "evidence", None) or getattr(hit, "items", None)


def test_every_broken_fixture_has_expectation() -> None:
    assert {p.name for p in (SPECS / "broken").glob("*.yml")} == set(EXPECTED)
    assert len(EXPECTED) >= 15


def test_todo_line_numbers() -> None:
    r = validate_spec(SPECS / "broken" / "04_todo_left.yml", today=TODAY)
    f = next(f for f in r.findings if f.rule_id == "AS-101")
    src = (SPECS / "broken" / "04_todo_left.yml").read_text(encoding="utf-8").splitlines()
    assert src[f.spec_ref.line - 1].startswith("owner:")  # type: ignore[union-attr]


def test_extends_merge_and_union(tmp_path: Path) -> None:
    (tmp_path / "child.yml").write_text(
        "extends: profile:finance\nagent_id: a\nprohibited:\n  - {tool: wire_transfer, reason: r}\nrisk_level: critical\n"
    )
    loaded = load_spec(tmp_path / "child.yml")
    assert loaded.spec is not None
    assert loaded.spec.prohibited_tools() == {"issue_refund", "wire_transfer"}  # unioned, not replaced
    assert loaded.spec.risk_level == "critical"  # child wins on scalars
    assert loaded.spec.data_rules and loaded.spec.data_rules.never_store == ["card_number", "cvv"]
    assert "profiles/finance.yml" in str(loaded.ref("prohibited", 0)).replace("\\", "/")


def test_extends_cycle(tmp_path: Path) -> None:
    (tmp_path / "a.yml").write_text("extends: b.yml\nagent_id: a\n")
    (tmp_path / "b.yml").write_text("extends: a.yml\nrisk_level: low\n")
    r = validate_spec(tmp_path / "a.yml", today=TODAY)
    assert r.spec_status == "invalid" and not r.findings
    assert any(i.rule_id == "AS-115" and "cycle" in i.message for i in r.spec_issues)


def test_defaults_by_risk() -> None:
    assert tier2_defaults("critical")["dry_run_required"] and not tier2_defaults("low")["dry_run_required"]
    assert "bounds" not in tier2_defaults("critical")  # the library holds no number: bounds are the user's
    assert tier2_defaults(None) == tier2_defaults("medium")
    assert floor_violations("high", {"fail_closed": False})
    assert not floor_violations("low", {"fail_closed": False})


def test_builder_matches_yaml() -> None:
    spec = (
        SpecBuilder("po")
        .risk("high")
        .owner("o", "t")
        .prohibit("issue_refund", reason="x")
        .require_approval("pay_invoice", "finance")
        .limit("pay_invoice", "amount", max=5000)
        .build()
    )
    assert spec.prohibited_tools() == {"issue_refund"}
    assert spec.limits and spec.limits[0].limit.max == 5000


def test_short_forms() -> None:
    l = load_spec_text("agent_id: a\ngoal: do x\nprohibited: [rm_rf]\nuntrusted_inputs: [email]\n")  # noqa: E741
    assert l.spec and l.spec.goal and l.spec.goal.statement == "do x"
    assert l.spec.prohibited_tools() == {"rm_rf"} and l.spec.untrusted_sources() == {"email"}
