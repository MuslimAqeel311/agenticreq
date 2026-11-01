"""`agenticreq guide`, `agenticreq` with no command, and the step sentence that starts every --help."""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.progress import GUIDE_TABLE, guide_text

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
TODAY = dt.date(2026, 10, 28)
COMMANDS = (
    "audit",
    "init",
    "check",
    "mocks",
    "mock-run",
    "guide",
    "validate",
    "params",
    "export",
    "export-spec",
    "fix",
    "stats",
    "rules",
    "version",
)


@pytest.fixture(autouse=True)
def _state_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTICREQ_NO_STATE", raising=False)


def _snapshot(name: str, text: str) -> None:
    path = GOLDEN / f"guide_{name}.txt"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(text + "\n", encoding="utf-8")
    assert text + "\n" == path.read_text(encoding="utf-8")


def test_guide_in_a_fresh_folder(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("def main():\n    pass\n", encoding="utf-8")
    text = guide_text(tmp_path, today=TODAY)
    assert text.startswith("You are at step 1 of 5: audit.")
    assert "Next: run a quick scan of this folder: agenticreq audit" in text
    assert "spec.yml is missing: create it with: agenticreq init" in text
    assert "mocks.yml is missing: create it with: agenticreq mocks" in text
    for name, what, cmd in GUIDE_TABLE:
        assert name in text and what in text and cmd in text
    _snapshot("fresh", text)


def test_guide_mid_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "app"
    shutil.copytree(FIX / "agents_misc" / "mock_run_app", d)
    monkeypatch.chdir(d)
    runner.invoke(app, ["check"])  # the spec is complete and has gaps: step 3
    text = guide_text(d, today=TODAY)
    assert text.startswith("You are at step 3 of 5: check.")
    assert "then run: agenticreq check" in text and "spec.yml is missing" not in text
    _snapshot("mid_way", text)
    out = runner.invoke(app, ["guide"])
    assert out.exit_code == 0 and "You are at step 3 of 5: check." in out.output


def test_no_command_prints_the_overview_and_exits_0() -> None:
    r = runner.invoke(app, [])
    assert r.exit_code == 0
    for cmd in ("audit", "init", "check", "mocks", "mock-run"):
        assert f"agenticreq {cmd}" in r.output
    assert "agenticreq guide" in r.output


@pytest.mark.parametrize("cmd", COMMANDS)
def test_every_help_starts_with_its_step(cmd: str) -> None:
    out = runner.invoke(app, [cmd, "--help"], terminal_width=200).output
    body = [ln.strip(" │|") for ln in out.splitlines() if ln.strip(" │|") and not ln.strip().startswith("Usage:")]
    first = body[0].strip()
    assert first.startswith(("Step ", "After step ", "Any step:", "All 5 steps:")), (cmd, first)
