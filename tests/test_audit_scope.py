"""Point 4: audit (no spec) degrades cleanly everywhere and says plainly, in the header, what it could and
could not check. Level 1 is UNKNOWN, never implied as passing. With no recognised framework the
framework-agnostic checks still run and the rest is UNKNOWN with the reason."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.models.finding import Bucket
from agenticreq.observe import build_observed
from agenticreq.params import NEEDS_SPEC
from agenticreq.report import render_console

AGENTS = Path(__file__).parent / "fixtures" / "agents"
runner = CliRunner()

UNKNOWN_FW = """
import requests

API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
MODEL = "gpt-4o"
BASE = "https://api.production.example.com/v1"


def fetch(url):
    return requests.get(url, timeout=5).text
"""


def _audit(path: Path):  # type: ignore[no-untyped-def]
    return run(mode="audit", observed=build_observed(path))


@pytest.mark.parametrize("agent", sorted(p.name for p in AGENTS.iterdir() if p.is_dir()))
def test_header_says_what_was_and_was_not_checked(agent: str) -> None:
    r = _audit(AGENTS / agent)
    scope = r.metadata["audit_scope"]
    assert scope["checked"] and set(scope["needs_spec"]) == set(NEEDS_SPEC)  # type: ignore[index]
    text = render_console(r)
    head = text.split("\n\n")[0] + text.split("\n\n")[1]
    assert "audit without a spec" in head and "needs a spec (Level 1)" in text
    assert any(f.rule_id == "AS-907" and f.bucket is Bucket.UNKNOWNS for f in r.findings)
    passed = {p.rule_id for p in r.passed}
    assert not passed & {"AS-203", "AS-205", "AS-206", "AS-207", "AS-401"}  # Level 1 never "passes" without a spec


def test_unknown_framework_runs_agnostic_checks(tmp_path: Path) -> None:
    (tmp_path / "bot.py").write_text(UNKNOWN_FW, encoding="utf-8")
    r = _audit(tmp_path)
    ids = {f.rule_id for f in r.findings}
    assert {"AS-214", "AS-210", "AS-223", "AS-216"} <= ids  # secret, pinning, env separation, unbounded call
    fw = next(f for f in r.findings if f.rule_id == "AS-904")
    assert fw.bucket is Bucket.UNKNOWNS and "bounds" in fw.detail and "approval gates" in fw.detail
    assert "AS-224" not in ids  # output typing is framework-specific: unknown, not a warning
    scope = r.metadata["audit_scope"]
    assert scope["framework"] == "not recognised" and "bounds" in scope["not_checked"]  # type: ignore[index,operator]
    assert "secrets_policy" in scope["checked"]  # type: ignore[index,operator]
    unbounded = next(f for f in r.findings if f.rule_id == "AS-216")
    assert "whether the model can call fetch is unknown" in unbounded.detail


@pytest.mark.parametrize(
    "files",
    [
        {},  # nothing to scan
        {"agent.py": "import a_package_that_is_not_installed\nfrom also_missing import thing\n"},
        {"agent.py": "def broken(:\n"},
    ],
)
def test_audit_never_crashes(tmp_path: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    r = runner.invoke(app, ["audit", str(tmp_path), "-f", "json"])
    assert r.exit_code in (0, 1, 3), r.output
    d = json.loads(r.stdout)
    assert d["metadata"]["audit_scope"]["needs_spec"]
    assert d["spec"] is None  # audit read no spec
    assert (r.exit_code == 3) is d["nothing_checked"]  # nothing to analyse is never reported as a pass


@pytest.mark.parametrize("files", [{}, {"agent.py": "def broken(:\n"}])
def test_nothing_to_check_is_not_a_pass(tmp_path: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    r = runner.invoke(app, ["audit", str(tmp_path)])
    assert r.exit_code == 3
    assert "Nothing was checked" in r.output and "NOT a pass" in r.output
    assert runner.invoke(app, ["audit", str(tmp_path), "--fail-on", "never"]).exit_code == 0


def test_audit_on_a_single_file(tmp_path: Path) -> None:
    (tmp_path / "bot.py").write_text(UNKNOWN_FW, encoding="utf-8")
    r = runner.invoke(app, ["audit", str(tmp_path / "bot.py"), "--verbose"])  # rule ids show with --verbose
    assert "audit without a spec" in r.output and "AS-214" in r.output


def test_unreadable_config_is_reported_not_dropped(tmp_path: Path) -> None:
    """A config file the agent code loops over that cannot be parsed must show up (AS-902), not vanish."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "agents.yaml").write_text("a: [unclosed\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text(
        "import yaml\nfrom crewai import Agent\ncfg = yaml.safe_load(open('config/agents.yaml'))\n"
        "for key, c in cfg.items():\n    Agent(role=c['role'], goal='g', backstory='b')\n",
        encoding="utf-8",
    )
    r = runner.invoke(app, ["audit", str(tmp_path), "-f", "json"])
    d = json.loads(r.stdout)
    found = [f for f in d["findings"] if f["rule_id"] == "AS-902"]
    assert any("config file could not be read" in f["detail"] for f in found), found
