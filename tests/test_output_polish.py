"""Output and command polish: nothing is cut with "…", one vocabulary for waivers and confirmed:, audit's own gap
title, "." as every command's default path, `check --coverage`, and the wrong-environment message."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture.discover import WRONG_ENV, DiscoverResult, explain
from agenticreq.cli import app
from agenticreq.params import PARAMS

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
PROJECTS = [FIX / "passfail" / "clean", FIX / "passfail" / "two_agents", FIX / "agents_misc" / "mock_run_app"]


@pytest.mark.parametrize("width", [60, 120])
@pytest.mark.parametrize("project", PROJECTS, ids=[p.name for p in PROJECTS])
def test_no_ellipsis_at_any_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: Path, width: int) -> None:
    shutil.copytree(project, tmp_path / "p")
    monkeypatch.chdir(tmp_path / "p")
    monkeypatch.setenv("COLUMNS", str(width))
    for args in (["check", "--all", "--quiet"], ["audit", "--quiet"], ["check", "--all", "--verbose", "--quiet"]):
        if args[0] == "check" and not Path("spec.yml").is_file():
            continue
        out = runner.invoke(app, args, terminal_width=width).output
        assert "…" not in out and "..." not in out, (args, out)
        assert all(len(ln) <= width for ln in out.splitlines()), args


def _spec_with(extra: str) -> str:
    return (FIX / "passfail" / "clean" / "spec.yml").read_text(encoding="utf-8") + "\n" + extra + "\n"


def _check_json(tmp: Path, monkeypatch: pytest.MonkeyPatch, extra: str) -> dict:  # type: ignore[type-arg]
    shutil.copytree(FIX / "passfail" / "clean", tmp / "p")  # tmp may be new: copytree makes it
    (tmp / "p" / "spec.yml").write_text(_spec_with(extra), encoding="utf-8")
    monkeypatch.chdir(tmp / "p")
    return json.loads(runner.invoke(app, ["check", "-f", "json", "--quiet"]).stdout)  # type: ignore[no-any-return]


def test_a_waiver_by_parameter_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = json.loads(
        runner.invoke(
            app,
            [
                "check",
                str(FIX / "passfail" / "clean"),
                "--spec",
                str(FIX / "passfail" / "clean" / "spec.yml"),
                "-f",
                "json",
            ],
        ).stdout
    )
    gap = next(r for r in before["results"] if r["status"] == "GAP")
    data = _check_json(tmp_path, monkeypatch, f"waivers:\n  - {{rule_id: {gap['param']}, reason: accepted}}")
    line = next(r for r in data["results"] if r["param"] == gap["param"])
    assert line["status"] == "PASS" and "waived by you" in line["reason"]


def test_a_waiver_or_confirmed_entry_that_matches_nothing_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _check_json(
        tmp_path,
        monkeypatch,
        "waivers:\n  - {rule_id: no_such_thing, reason: x}\nconfirmed:\n"
        "  - {clause: approval_timeoutt, by: a, date: 2026-10-01, expires: "
        "2027-10-01, evidence: x}",
    )
    reasons = " | ".join(r["reason"] for r in data["results"] if r["status"] == "GAP").lower()
    assert "waiver matches no rule: no_such_thing" in reasons
    assert "matches no parameter: approval_timeoutt" in reasons


def test_audit_titles_its_first_group_gaps_found_in_code() -> None:
    out = runner.invoke(app, ["audit", str(FIX / "passfail" / "two_agents"), "--quiet"], terminal_width=120).output
    assert "GAPS FOUND IN CODE" in out and "SPEC VS CODE GAPS" not in out


@pytest.mark.parametrize("cmd", ["stats", "fix", "check", "validate", "export"])
def test_every_command_defaults_to_this_folder_and_says_when_it_needs_a_spec(
    cmd: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, [cmd])
    assert "Missing argument" not in r.output and "Traceback" not in r.output, r.output
    if cmd in ("check", "validate", "export"):
        assert r.exit_code == 2 and "this command needs one" in " ".join(r.output.split())


def test_coverage_lists_every_parameter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(FIX / "passfail" / "clean", tmp_path / "p")
    monkeypatch.chdir(tmp_path / "p")
    out = runner.invoke(app, ["check", "--coverage"]).stdout
    names = [
        ln.split()[0]
        for ln in out.splitlines()[2:]
        if ln.strip() and not ln[0].isdigit() and not ln.startswith("sub-fields")
    ]
    assert names == [p.name for p in PARAMS if p.level <= 3]
    assert "parameters checked;" in out and "not applicable;" in out and "confirmed by a person" in out
    assert "single agent" in out  # a Level 3 parameter on a single agent says why


def test_the_wrong_environment_message() -> None:
    res = DiscoverResult(entry="agent.py:main")
    text = explain(
        "run-0",
        {"type": "ModuleNotFoundError", "message": "No module named 'agenticreq'", "missing_module": "agenticreq"},
        "",
        0,
        res,
        r"C:\envs\agent\python.exe",
    )
    assert WRONG_ENV == "install agenticreq in the same environment as your agent, or pass --python <interpreter>"
    assert WRONG_ENV in text
