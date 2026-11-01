"""agenticreq never suggests numbers: every number in a spec comes from the user. The library compares the spec's
number with the code's, and shows a number found in the user's code as plain evidence."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq import thresholds as th
from agenticreq.cli import app
from agenticreq.defaults import RISK_LEVELS, tier2_defaults
from agenticreq.shapes import NUMBER, example, example_yaml, mask_numbers

FIX = Path(__file__).parent / "fixtures"
runner = CliRunner()
#: text the library must never print (the old recommendation wording, the old "not read" wording, the banned words)
FORBIDDEN = re.compile(
    r"recommend|suggest|typical|max=|NOT CHECKED|library gap|\bfakes?\b|sandbox(?!_policy)|attestation"
    r"|unknown(?! field)",
    re.IGNORECASE,
)
AGENT_DIRS = sorted(d for d in (FIX / "agents").iterdir() if d.is_dir() and any(d.glob("*.py")))
SPEC_DIRS = [
    d
    for d in [*(FIX / "agents").iterdir(), *(FIX / "agents_misc").iterdir(), *(FIX / "passfail").iterdir()]
    if (d / "spec.yml").is_file() and any(d.glob("*.py"))
]
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> set[str]:
    out = set()
    for n in NUMBER_RE.findall(text):
        out |= {n, str(float(n)), f"{float(n):g}", str(int(float(n)))}
    return out


# ============================================================================ the library holds no number
def _scalars(v: Any) -> list[Any]:
    if isinstance(v, dict):
        return [x for item in v.values() for x in _scalars(item)]
    if isinstance(v, list):
        return [x for item in v for x in _scalars(item)]
    return [v]


@pytest.mark.parametrize("risk", [*RISK_LEVELS, None])
def test_the_risk_level_defaults_hold_no_number(risk: str | None) -> None:
    numbers = [x for x in _scalars(tier2_defaults(risk)) if isinstance(x, (int, float)) and not isinstance(x, bool)]
    assert numbers == []


def test_there_is_no_recommendation_api_left() -> None:
    import agenticreq.defaults as defaults
    import agenticreq.scaffold as scaffold

    for mod, name in (
        (th, "recommendation"),
        (th, "rec_text"),
        (th, "recommended_by_field"),
        (defaults, "recommended"),
        (scaffold, "recommended_numbers"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name}"
    assert not hasattr(th.Numeric("k", "AS-1"), "has_recommendation")


def test_the_numbers_doc_exists_and_the_old_defaults_doc_is_gone() -> None:
    docs = Path(__file__).parent.parent / "docs"
    assert (docs / "numbers.md").is_file() and not (docs / "defaults-rationale.md").exists()
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert "agenticreq never suggests numbers. You write them; it checks your code matches." in text


# ============================================================================ gap logic: match, mismatch, missing
P = th.MAX_STEPS


def test_spec_and_code_match() -> None:
    d = th.judge(P, "high", spec=25, code=20)  # code at least as strict
    assert (d.row, d.status, d.reason) == ("match", "PASS", "Found in code: 20.")
    assert th.judge(P, "high", spec=25, code=25).status == "PASS"


def test_spec_and_code_mismatch_is_a_gap() -> None:
    d = th.judge(P, "high", spec=25, code=60)
    assert (d.row, d.status, d.line_status) == ("looser", "FAIL", "GAP")
    assert d.reason == "Code allows 60 steps; spec.yml says 25."


def test_a_spec_number_with_no_code_bound_is_a_gap() -> None:
    d = th.judge(P, "high", spec=25)
    assert (d.row, d.status, d.line_status) == ("no_limit", "FAIL", "GAP")
    assert d.reason == "spec.yml says 25; the code sets no limit."


def test_a_number_only_in_code_is_shown_as_found_in_code_and_passes() -> None:
    d = th.judge(P, "medium", code=30)
    assert (d.row, d.status, d.reason) == ("code_only", "PASS", "Found in code: 30.")


@pytest.mark.parametrize("p", th.NUMERIC_PARAMS, ids=lambda p: p.key)
@pytest.mark.parametrize("risk", [*RISK_LEVELS, None])
def test_a_number_the_spec_omits_names_no_number(p: th.Numeric, risk: str | None) -> None:
    d = th.judge(p, risk)
    assert d.row == "nowhere" and d.status == "FAIL"
    assert d.reason == f"No {p.name} set: write a value, or not_needed with a reason."
    assert not re.search(r"\d", d.reason.replace(p.name, ""))


def test_confidence_threshold_is_asked_for_at_critical_only_and_without_a_number(tmp_path: Path) -> None:
    from agenticreq.engine import run
    from agenticreq.loader import load_spec_text
    from agenticreq.models.verdict import results
    from agenticreq.observe import build_observed

    (tmp_path / "agent.py").write_text("def tool():\n    return 1\n", encoding="utf-8")
    base = yaml.safe_load((FIX / "agents_misc" / "bank_triage" / "spec.yml").read_text(encoding="utf-8"))
    for risk, asked in (("critical", True), ("high", False)):
        spec = {**base, "risk_level": risk}
        report = run(mode="check", loaded=load_spec_text(yaml.safe_dump(spec)), observed=build_observed(tmp_path))
        lines = [ln for ln in results(report) if ln.param == "confidence_threshold"]
        assert bool(lines) is asked, (risk, lines)
        assert all(not re.search(r"\d", ln.reason) for ln in lines)


# ============================================================================ examples, params, the schema
def test_examples_show_a_placeholder_where_a_number_goes() -> None:
    assert example("bounds") == {
        k: NUMBER for k in ("max_steps", "max_cost_usd", "max_duration_s", "max_retries", "max_tool_calls")
    }
    assert example("limits")[0]["limit"] == {"min": NUMBER, "max": NUMBER}
    assert example("approval_required")[0]["timeout"] == "<duration>"
    assert example("spec_version") == 1  # the format marker, not a threshold
    assert "<number>" in example_yaml("max_delegation_depth") and not re.search(r"\d", example_yaml("bounds"))
    assert mask_numbers({"a": [1, 2.5, True, "30m", "text", "2026-10-17"]}) == {
        "a": [NUMBER, NUMBER, True, "<duration>", "text", "2026-10-17"]
    }


def test_params_shows_types_and_meaning_only() -> None:
    console = runner.invoke(app, ["params"]).output
    data = json.loads(runner.invoke(app, ["params", "-f", "json"]).stdout)
    assert not FORBIDDEN.search(console) and not FORBIDDEN.search(json.dumps(data))
    assert not any("recommended" in row for row in data)
    assert "bounds.max_steps: you write it (a positive number), or not_needed with a reason" in " ".join(
        console.split()
    )
    assert "confidence_threshold: you write it (a number from zero to one)" in " ".join(console.split())


def test_the_json_schema_command_surface_holds_no_default_number() -> None:
    out = runner.invoke(app, ["params", "--field", "bounds", "-f", "json"]).stdout
    [row] = json.loads(out)
    assert row["example"]["max_steps"] == NUMBER and "recommended" not in row


# ============================================================================ init writes no number
def _numeric_values(text: str) -> list[tuple[str, Any]]:
    data = yaml.safe_load(text) or {}
    return [
        (k, v)
        for k, v in data.items()
        if k not in ("_derived", "spec_version")
        and any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in _scalars(v))
    ]


@pytest.mark.parametrize("agent", AGENT_DIRS, ids=lambda d: d.name)
def test_init_writes_no_number_and_none_of_the_old_words(agent: Path, tmp_path: Path) -> None:
    out = tmp_path / "spec.yml"
    r = runner.invoke(app, ["init", str(agent), "--out", str(out)])
    assert r.exit_code == 0, r.output
    text = out.read_text(encoding="utf-8")
    assert _numeric_values(text) == []
    assert not FORBIDDEN.search(text) and "# Numbers: write your own value" in text
    assert "timeout: 4h" not in text


# ============================================================================ the scan: every command, every fixture
def _scan(output: str, label: str) -> None:
    hit = FORBIDDEN.search(output)
    assert hit is None, f"{label}: {hit.group(0)!r} in: ...{output[max(0, hit.start() - 80) : hit.end() + 80]}..."


@pytest.mark.parametrize("agent", AGENT_DIRS, ids=lambda d: d.name)
def test_scan_init_validate_check_guide_mocks_on_every_fixture(
    agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "a"
    shutil.copytree(agent, work)
    monkeypatch.chdir(work)
    for cmd in (["init"], ["validate"], ["check"], ["check", "--coverage"], ["guide"], ["mocks"], ["audit"]):
        r = runner.invoke(app, cmd)
        _scan(r.output, f"{agent.name}: agenticreq {' '.join(cmd)}")
    _scan((work / "spec.yml").read_text(encoding="utf-8"), f"{agent.name}: spec.yml from init")
    _scan((work / "mocks.yml").read_text(encoding="utf-8"), f"{agent.name}: mocks.yml")


def test_scan_params_and_mock_run_on_the_multi_run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for cmd in (["params"], ["params", "--brief"], ["params", "--level", "2"]):
        _scan(runner.invoke(app, cmd).output, f"agenticreq {' '.join(cmd)}")
    work = tmp_path / "a"
    shutil.copytree(FIX / "agents_misc" / "multi_run_app", work)
    monkeypatch.chdir(work)
    r = runner.invoke(app, ["mock-run", "--python", sys.executable])
    assert r.exit_code == 0, r.output
    _scan(r.output, "agenticreq mock-run")


@pytest.mark.parametrize("agent", SPEC_DIRS, ids=lambda d: f"{d.parent.name}/{d.name}")
def test_every_number_in_a_numeric_result_is_copied_from_the_users_spec_or_code(
    agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "a"
    shutil.copytree(agent, work)
    monkeypatch.chdir(work)
    args = ["check", "-f", "json"]
    if (work / "captures.json").is_file():
        args += ["--captures", "captures.json"]
    r = runner.invoke(app, args)
    data = json.loads(r.stdout)
    source = (work / "spec.yml").read_text(encoding="utf-8") + "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in work.rglob("*.py")
    )
    allowed = _numbers(source)
    numeric = {p.key for p in th.NUMERIC_PARAMS} | {"limits", "escalation", "approval_required"}
    for row in data["results"]:
        if row.get("param") not in numeric:
            continue
        text = re.sub(r"AS-\d+|\b[\w./-]+\.(?:py|yml):\d+", "", row.get("reason", ""))
        # the code's own value ("Code allows 2", "Found in code: 25") is read from the analysis of the code; the
        # spec side ("spec.yml says 1") must be a number the user wrote
        text = re.sub(r"Code allows [\d.]+|Found in code: [\d.]+", "", text)
        stray = {n for n in NUMBER_RE.findall(text) if n not in allowed and f"{float(n):g}" not in allowed}
        assert not stray, f"{row['param']}: {stray} not in the spec or the code: {row['reason']}"
        assert not FORBIDDEN.search(row.get("reason", "")), row["reason"]
    _scan(r.stdout, f"{agent.name}: check -f json")


# ============================================================================ fix writes only the user's number
CODE = """from agents import Agent, Runner

agent = Agent(name="A", instructions="x", model="gpt-4.1-2025-04-14")


async def go():
    return await Runner.run(agent, "hi")
"""


def _fix_project(tmp_path: Path, steps: str) -> Path:
    (tmp_path / "agent.py").write_text(CODE, encoding="utf-8")
    spec = yaml.safe_load((FIX / "agents_misc" / "bank_triage" / "spec.yml").read_text(encoding="utf-8"))
    spec.update({"bounds": {"max_steps": int(steps)}} if steps else {})
    (tmp_path / "spec.yml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    return tmp_path


def test_fix_writes_the_users_max_steps_and_nothing_of_its_own(tmp_path: Path) -> None:
    root = _fix_project(tmp_path, "7")
    r = runner.invoke(app, ["fix", str(root), "--spec", str(root / "spec.yml"), "--only", "AS-208"])
    assert "max_turns=7" in r.output and "bounds.max_steps is not set" not in r.output


def test_fix_without_a_user_number_writes_no_step_cap(tmp_path: Path) -> None:
    root = _fix_project(tmp_path, "")
    r = runner.invoke(app, ["fix", str(root), "--spec", str(root / "spec.yml"), "--only", "AS-208"])
    assert "bounds.max_steps is not set in spec.yml: no step cap is written" in " ".join(r.output.split())
    assert "max_turns=" not in r.output and "nothing to fix" in r.output
