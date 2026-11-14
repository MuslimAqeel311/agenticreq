"""M11: a folder holding several agents or scripts can be scoped with --entry, or include: / exclude: in spec.yml;
the report header names what was scanned. By default the whole folder is scanned, as before."""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.observe import build_observed

runner = CliRunner()
M11 = Path(__file__).parent / "fixtures" / "repro" / "m11"
SPEC = Path(__file__).parent / "fixtures" / "repro" / "h1_input" / "spec.yml"


def test_m11_default_scans_the_whole_folder() -> None:
    obs = build_observed(M11)
    assert {a.name for a in obs.declared_agents} == {"support", "batch"}
    assert "delete_old_rows" in obs.declared_tools and not obs.scope


def test_m11_entry_scans_the_entry_and_its_imports_only() -> None:
    obs = build_observed(M11, include=["agent.py"])
    assert sorted(obs.files) == ["agent.py", "helpers.py"]
    assert {a.name for a in obs.declared_agents} == {"support"} and "delete_old_rows" not in obs.declared_tools


def test_m11_exclude_folder_and_glob() -> None:
    for pattern in ("scripts/", "scripts/*", "**/batch_*.py"):
        obs = build_observed(M11, exclude=[pattern])
        assert "scripts/batch_summarise.py" not in obs.files, pattern


def test_m11_cli_entry_and_header() -> None:
    r = runner.invoke(app, ["audit", str(M11), "--entry", "agent.py", "--quiet"], terminal_width=250)
    out = " ".join(r.output.split())
    assert "entry agent.py; scanned 2 files: agent.py, helpers.py" in out
    assert "delete_old_rows" not in out


def test_m11_spec_exclude_and_r1_without_it_the_batch_script_is_still_reported(tmp_path: Path) -> None:
    d = tmp_path / "p"
    shutil.copytree(M11, d)
    spec = SPEC.read_text(encoding="utf-8").replace("approval_required:\n", "exclude: [scripts/]\napproval_required:\n")
    spec = spec.replace("tool: save_record", "tool: lookup")
    (d / "spec.yml").write_text(spec, encoding="utf-8")
    r = runner.invoke(app, ["check", str(d), "--all", "--quiet"], terminal_width=250)
    out = " ".join(r.output.split())
    assert "excluded scripts/" in out and "gpt-4o " not in out + " "
    # R1: without the scope, the batch script's unpinned model is still found
    a = runner.invoke(app, ["audit", str(d), "--quiet"], terminal_width=250)
    assert "batch" in a.output or "gpt-4o" in a.output
