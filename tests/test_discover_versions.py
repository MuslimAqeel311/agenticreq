"""P3: discover never hangs into a raw traceback. An unsupported framework version is refused up front with
the supported range and the `audit` fallback; a run that does not finish exits with a clear message."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture import DiscoverError, discover
from agenticreq.capture.discover import _version_tuple, check_framework_versions
from agenticreq.cli import app
from agenticreq.data import load_yaml

CREW_AGENT = "from crewai import Agent\nagent = Agent(role='r', goal='g', backstory='b', llm='gpt-4o-2024-08-06')\n"


def _fake_dist(site: Path, name: str, version: str) -> None:
    d = site / f"{name}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")


@pytest.mark.parametrize(
    "version,ok,noted",
    [
        ("0.86.0", False, False),
        ("1.9.3", True, False),
        ("1.15.28", True, False),
        ("2.0.0", True, True),
        ("1.5.0", True, True),
    ],
)
def test_crewai_version_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, ok: bool, noted: bool
) -> None:
    """Only a known-bad version is refused; any other version outside the range runs, with a note."""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text(CREW_AGENT, encoding="utf-8")
    site = tmp_path / "site"
    _fake_dist(site, "crewai", version)
    monkeypatch.setenv("PYTHONPATH", str(site))  # the probe sees this fake crewai first
    if ok:
        notes: list[str] = []
        assert check_framework_versions(tmp_path / "agent", sys.executable, notes)["crewai"] == version
        assert bool(notes) is noted
        if noted:
            assert "outside the validated range" in notes[0] and "exits 1" in notes[0]
        return
    with pytest.raises(DiscoverError) as e:
        check_framework_versions(tmp_path / "agent", sys.executable)
    msg = str(e.value)
    assert f"crewai {version} is not supported" in msg and ">=1.9.3,<2" in msg and "agenticreq audit" in msg
    if version == "0.86.0":
        assert "hangs" in msg  # the known problem is named


def test_unsupported_version_cli_exits_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "agent.py").write_text(CREW_AGENT, encoding="utf-8")
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    site = tmp_path / "site"
    _fake_dist(site, "crewai", "0.86.0")
    monkeypatch.setenv("PYTHONPATH", str(site))
    r = CliRunner().invoke(
        app,
        [
            "mock-run",
            "--entry",
            str(tmp_path / "agent.py"),
            "--mocks",
            str(tmp_path / "mocks.yml"),
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    assert r.exit_code == 2
    assert "Traceback" not in r.output and "not supported by agenticreq mock-run" in r.output.replace("\n", " ")
    assert not (tmp_path / "c.json").exists()  # refused before anything ran


def test_frameworks_without_a_range_are_not_gated(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("import smolagents\n", encoding="utf-8")
    assert check_framework_versions(tmp_path, sys.executable) == {}


def test_timeout_is_a_clear_error_not_a_traceback(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    res = discover(str(tmp_path / "agent.py"), mock_tools=str(tmp_path / "mocks.yml"), timeout=2)
    assert res.failures["run-0"]["type"] == "Timeout"  # reported with what was reached, not raised
    assert any("before the 2s limit stopped run-0 (last step: model call 0" in p for p in res.problems())
    r = CliRunner().invoke(
        app,
        [
            "mock-run",
            "--entry",
            str(tmp_path / "agent.py"),
            "--mocks",
            str(tmp_path / "mocks.yml"),
            "--timeout",
            "2",
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    out = " ".join(r.output.split())
    assert r.exit_code == 1 and "Traceback" not in out and "Raise --timeout" in out


def test_version_table_is_well_formed() -> None:
    table = load_yaml("supported_versions.yml")["frameworks"]
    for fw, row in table.items():
        assert _version_tuple(row["min"]) < _version_tuple(row["below"]), fw
        for v in row.get("validated") or []:
            assert _version_tuple(row["min"]) <= _version_tuple(v) < _version_tuple(row["below"]), (fw, v)


def test_readme_documents_the_ranges() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for row in load_yaml("supported_versions.yml")["frameworks"].values():
        assert f"{row['dist']} >={row['min']},<{row['below']}" in readme


@pytest.mark.parametrize("version,flagged", [("0.86.0", True), ("1.15.28", False), ("2.1.0", True)])
def test_static_audit_flags_an_unvalidated_framework_version(tmp_path: Path, version: str, flagged: bool) -> None:
    """audit never runs the agent, but its code reading is still validated per framework version."""
    (tmp_path / "agent.py").write_text(CREW_AGENT, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(f"crewai=={version}\n", encoding="utf-8")
    from agenticreq.engine import run
    from agenticreq.observe import build_observed

    obs = build_observed(tmp_path)
    # an exact pin is the agent's version, whatever crewai agenticreq's own Python may have installed
    assert (obs.framework_version, obs.framework_version_source) == (version, "requirements (pinned)")
    ids = {f.rule_id for f in run(mode="audit", observed=obs).findings}
    assert ("AS-908" in ids) is flagged
