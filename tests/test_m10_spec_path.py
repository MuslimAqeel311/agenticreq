"""M10: --spec resolves against the folder being checked; when the agent folder and this folder both hold a
spec.yml and they differ, the line says which was used; with none, the command says so and checks nothing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app

runner = CliRunner()
CLEAN = Path(__file__).parent / "fixtures" / "passfail" / "clean"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    shutil.copytree(CLEAN, d, ignore=shutil.ignore_patterns(".agenticreq"))
    return d


def test_m10_the_agent_folders_spec_is_used_from_another_folder(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    r = runner.invoke(app, ["check", str(project), "--quiet", "-f", "json"])
    assert r.exit_code in (0, 1), r.output
    assert '"spec_file"' in r.stdout and str(project / "spec.yml") in r.stdout


def test_m10_two_different_specs_say_which_one(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    here = tmp_path / "here"
    here.mkdir()
    (here / "spec.yml").write_text("agent_id: other\n", encoding="utf-8")
    monkeypatch.chdir(here)
    r = runner.invoke(app, ["check", str(project), "--quiet"])
    out = " ".join(r.output.split())
    assert f"using {project / 'spec.yml'} (the agent folder's spec)" in out and "pass --spec to choose it" in out


def test_m10_no_spec_anywhere_checks_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "p"
    d.mkdir()
    (d / "agent.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["check", str(d)])
    out = " ".join(r.output.split())
    assert r.exit_code == 2 and f"no spec at {d / 'spec.yml'}" in out


def test_m10_explicit_spec_wins(project: Path, tmp_path: Path) -> None:
    other = tmp_path / "x.yml"
    other.write_text((project / "spec.yml").read_text(encoding="utf-8"), encoding="utf-8")
    r = runner.invoke(app, ["check", str(project), "--spec", str(other), "--quiet", "-f", "json"])
    assert str(other) in r.stdout
