"""The framework version is the AGENT's: its own environment (any OS layout), its lock file or an exact pin.
A range (`crewai>=0.100`) is never read as a version, and agenticreq's own Python never stands in for the
agent's when the project says otherwise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticreq import versions as V
from agenticreq.engine import run
from agenticreq.observe import build_observed

CREW_AGENT = "from crewai import Agent\nagent = Agent(role='r', goal='g', backstory='b', llm='gpt-4o-2024-08-06')\n"


@pytest.mark.parametrize(
    "version,spec,ok",
    [
        ("1.9.3", ">=1.9.3,<2", True),
        ("2.0.0", ">=1.9.3,<2", False),
        ("1.10", ">=1.9.3", True),
        ("1.4.7", "~=1.4.5", True),
        ("1.5.0", "~=1.4.5", False),
        ("1.2.9", "==1.2.*", True),
        ("1.3", "==1.2.*", False),
        ("1.0", "!=1.0", False),
        ("0.100.0", ">0.99", True),
        ("1.0", "", True),
        ("1.0", ">=1.0 ; python_version >= '3.10'", True),
    ],
)
def test_specifiers(version: str, spec: str, ok: bool) -> None:
    assert V.satisfies(version, spec) is ok


def test_unreadable_specs_are_unknown_not_false() -> None:
    assert V.clauses("@ git+https://github.com/x/y") is None
    assert V.satisfies("1.0", "@ git+https://github.com/x/y") is None


@pytest.mark.parametrize(
    "spec,allowed",
    [
        (">=0.100.0", True),
        (">=0.100,<1", False),
        ("<=1.9.2", False),
        ("~=1.10", True),
        (">=2", False),
        ("==1.12.*", True),
        ("!=1.9.3", True),
    ],
)
def test_range_against_validated(spec: str, allowed: bool) -> None:
    assert V.allows_any(spec, "1.9.3", "2", ["1.9.3", "1.15.28"]) is allowed


def test_exact_pins() -> None:
    assert V.exact("==1.9.3") == "1.9.3" and V.exact("===1.9.3") == "1.9.3"
    assert V.exact("==1.9.*") is None and V.exact(">=1.9") is None


@pytest.mark.parametrize("layout", ["lib/python3.12/site-packages", "Lib/site-packages"])  # macOS-Linux / Windows
def test_project_venv_wins(tmp_path: Path, layout: str) -> None:
    sp = tmp_path / ".venv" / layout
    (sp / "crewai-1.15.28.dist-info").mkdir(parents=True)
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    assert V.agent_version(tmp_path, ["crewai"], {"crewai": ">=0.100.0"}) == ("1.15.28", "installed in .venv")


def test_conda_env_in_project(tmp_path: Path) -> None:
    env = tmp_path / "envs" / "agent"
    (env / "conda-meta").mkdir(parents=True)
    (env / "lib" / "python3.11" / "site-packages" / "crewai-1.12.0.dist-info").mkdir(parents=True)
    assert V.agent_version(tmp_path, ["crewai"], {}) == ("1.12.0", "installed in agent")


@pytest.mark.parametrize(
    "name,text",
    [
        ("uv.lock", '[[package]]\nname = "crewai"\nversion = "1.14.0"\nsource = { registry = "x" }\n'),
        ("poetry.lock", '[[package]]\nname = "crewai"\nversion = "1.14.0"\ndescription = ""\n'),
        ("Pipfile.lock", json.dumps({"default": {"crewai": {"version": "==1.14.0"}}})),
    ],
)
def test_lock_files(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")
    assert V.agent_version(tmp_path, ["crewai"], {"crewai": ">=0.100.0"}) == ("1.14.0", name)


def test_a_range_is_never_read_as_a_version(tmp_path: Path) -> None:
    got = V.agent_version(tmp_path, ["agenticreq-no-such-dist"], {"agenticreq-no-such-dist": ">=0.100.0"})
    assert got == (">=0.100.0", "requirements (range)")


def test_agenticreqs_own_python_is_used_only_when_the_project_agrees(tmp_path: Path) -> None:
    import importlib.metadata

    have = importlib.metadata.version("pyyaml")  # installed next to agenticreq
    assert V.agent_version(tmp_path, ["pyyaml"], {}) == (have, "installed")
    other = f"=={int(have.split('.')[0]) + 50}.0"  # the project pins another version: a different environment
    assert V.agent_version(tmp_path, ["pyyaml"], {"pyyaml": ">" + other[2:]}) == (
        ">" + other[2:],
        "requirements (range)",
    )


@pytest.mark.parametrize(
    "req,flagged", [("crewai>=0.100.0", False), ("crewai>=0.100,<1", True), ("crewai~=1.10", False), ("crewai", False)]
)
def test_audit_reads_a_requirement_range(tmp_path: Path, req: str, flagged: bool) -> None:
    """The user's case: `crewai>=0.100.0` lets pip install a validated 1.x, so it is not 'version 0.100.0'."""
    (tmp_path / "agent.py").write_text(CREW_AGENT, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(req + "\n", encoding="utf-8")
    obs = build_observed(tmp_path)
    ids = {f.rule_id for f in run(mode="audit", observed=obs).findings}
    assert ("AS-908" in ids) is flagged
