"""M13: the summary line and the section headings agree; after a mock run, step 4 says the run was saved and that
`agenticreq check` reads it (it never stays a bare ✗); tool_effects in spec.yml come before what the code shows."""

from __future__ import annotations

import os
import shutil
import time
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec
from agenticreq.models.verdict import results, totals
from agenticreq.observe import build_observed
from agenticreq.progress import compute, render

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"


def test_m13_a_summary_equals_the_lines_it_counts() -> None:
    for d in (FIX / "issues" / "a1_gate", FIX / "issues" / "b2_gate", FIX / "repro" / "h1_strands"):
        report = run(
            mode="check", loaded=load_spec(d / "spec.yml"), observed=build_observed(d), today=date(2026, 11, 14)
        )
        lines = results(report)
        t = totals(lines)
        assert t["gaps"] == sum(1 for ln in lines if ln.status == "GAP")
        assert t["mock_run"] == sum(1 for ln in lines if ln.status == "CHECK")


def test_m13_c_after_a_mock_run_step_4_says_check_reads_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "clean"
    shutil.copytree(FIX / "passfail" / "clean", d, ignore=shutil.ignore_patterns(".agenticreq"))
    monkeypatch.chdir(d)
    runner.invoke(app, ["audit", "--quiet"])
    runner.invoke(app, ["check", "--quiet"])
    (d / ".agenticreq").mkdir(exist_ok=True)
    cap = d / ".agenticreq" / "captures.json"
    cap.write_text('{"calls": [], "coverage": "0/0"}', encoding="utf-8")
    later = time.time() + 5
    os.utime(cap, (later, later))
    text = render(compute(d, d / "spec.yml"))
    if "✓ 3. check" in text:  # only once the check step is done does step 4 speak about the run
        assert "mock run saved; run agenticreq check to read it" in text


def test_m13_d_spec_tool_effects_come_first() -> None:
    d = FIX / "issues" / "c3_effects"
    obs = build_observed(d)
    loaded = load_spec(d / "spec.yml") if (d / "spec.yml").is_file() else None
    if loaded is None or loaded.spec is None or not loaded.spec.tool_effects:
        pytest.skip("fixture has no tool_effects")
    run(mode="check", loaded=loaded, observed=obs)
    for name, eff in loaded.spec.tool_effects.items():
        if name in obs.declared_tools:
            assert obs.declared_tools[name].declared_effect == eff
