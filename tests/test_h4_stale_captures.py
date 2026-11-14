"""H4: a mock run older than the code is never read as evidence, never dropped and never a pass. Its items go back
to "Check with a mock run" with one line, the counts include them, and the exit code is not a pass."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import STALE_REASON, app
from agenticreq.models.finding import Report
from agenticreq.models.verdict import exit_code, overall

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "issues" / "a1_gate"


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """a1_gate after one mock run (shared: the run is the slow part)."""
    from agenticreq.capture import discover, save_captures

    d = tmp_path_factory.mktemp("h4") / "a1_gate"
    shutil.copytree(FIXTURE, d)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        res = discover("agent.py:main", mock_tools="mocks.yml", call_args={"task": "write the report"})
        save_captures(res, d / ".agenticreq" / "captures.json")
    finally:
        os.chdir(cwd)
    return d


def _check(d: Path, *extra: str) -> tuple[int, str]:
    r = runner.invoke(app, ["check", str(d), "--spec", str(d / "spec.yml"), "--all", "--quiet", *extra])
    return r.exit_code, r.output


def _copy(ran: Path, tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    shutil.copytree(ran, d)
    return d


def test_h4_fresh_run_is_used(ran: Path, tmp_path: Path) -> None:
    d = _copy(ran, tmp_path)
    data = json.loads((d / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))
    assert data["source_hashes"]["agent.py"]
    _, out = _check(d)
    assert "The code changed after" not in out and "version_pinning" in out


def test_h4_an_edited_file_with_an_old_timestamp_makes_the_run_stale(ran: Path, tmp_path: Path) -> None:
    d = _copy(ran, tmp_path)
    f = d / "agent.py"
    st = f.stat()
    f.write_text(f.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    os.utime(f, (st.st_atime, st.st_mtime - 3600))  # a copy / checkout keeps an old timestamp
    code, out = _check(d)
    assert "The code changed after the last mock run" in out
    assert out.count(STALE_REASON) >= 2  # tool_schema_strictness, version_pinning come back as checks
    assert code == 1


def test_h4_stale_items_are_counted_and_never_pass(ran: Path, tmp_path: Path) -> None:
    d = _copy(ran, tmp_path)
    (d / "agent.py").write_text((d / "agent.py").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    code, out = _check(d, "-f", "json")
    data = json.loads(out)
    lines = [ln for ln in data["results"] if STALE_REASON in ln["reason"]]
    assert lines and all(ln["status"] == "CHECK" and ln["group"] == "mock_run" for ln in lines)
    assert not any(ln["param"] in {x["param"] for x in lines} for ln in data["results"] if ln["status"] == "PASS")
    assert data["counts"]["mock_run"] >= len(lines) and data["result"] == "FAIL"
    assert code == 1


def test_h4_audit_also_ignores_a_stale_run(ran: Path, tmp_path: Path) -> None:
    d = _copy(ran, tmp_path)
    (d / "agent.py").write_text((d / "agent.py").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    r = runner.invoke(app, ["audit", str(d), "--quiet"])
    assert "The code changed after the last mock run" in r.output
    assert r.exit_code == 1


def test_h4_a_removed_file_makes_the_run_stale(ran: Path, tmp_path: Path) -> None:
    d = _copy(ran, tmp_path)
    (d / "minifw.py").rename(d / "minifw_old.txt")
    _, out = _check(d)
    assert "The code changed after the last mock run" in out


def test_h4_a_stale_report_is_never_pass_even_with_no_gaps() -> None:
    report = Report(agent_id="a", command="check", findings=[], passed=[])
    report.metadata["mock_run"] = {"captured": 3, "total": 3, "stale": True}
    assert overall(report, fail_on="gaps") == "FAIL"
    assert exit_code(report, "gaps") == 1 and exit_code(report, "all") == 1
    assert exit_code(report, "never") == 0  # --fail-on never keeps its meaning
    report.metadata["mock_run"]["stale"] = False
    assert exit_code(report, "gaps") == 0
