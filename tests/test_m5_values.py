"""M5: a limit's value is read from constants, imports, class attributes, settings defaults, one-hop variables,
environment defaults and helper call sites; several values or a run-time value go to sign-off, never "no limit";
a value that breaks the spec is still a gap (R1); per-agent values from a factory stay per agent."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from agenticreq.engine import run
from agenticreq.loader import load_spec
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

M5 = Path(__file__).parent / "fixtures" / "repro" / "m5"


_ORDER = ["GAP", "SIGN-OFF", "CHECK", "PASS"]


def _line(project: Path, param: str = "bounds.max_steps") -> tuple[str, str]:
    """The worst line for ``param`` (one per agent when agents have their own bound)."""
    report = run(
        mode="check", loaded=load_spec(project / "spec.yml"), observed=build_observed(project), today=date(2026, 11, 14)
    )
    lines = sorted((x for x in results(report) if x.param == param), key=lambda x: _ORDER.index(x.status))
    return lines[0].status, lines[0].reason


@pytest.mark.parametrize(
    "name", ["const", "anyname", "imported", "modattr", "dataclass", "classattr", "onehop", "helper", "factory"]
)
def test_m5_value_is_read(name: str) -> None:
    status, reason = _line(M5 / name)
    assert status == "PASS" and "Found in code: 6 (" in reason


@pytest.mark.parametrize("name", ["getenv", "pyd"])
def test_m5_a_default_is_labelled_and_never_a_pass(name: str) -> None:
    status, reason = _line(M5 / name)
    assert status == "CHECK" and "max_turns=6 (assumed: default in code of" in reason


@pytest.mark.parametrize("name,files", [("several", "a_cfg.py, b_cfg.py"), ("several_module", "agent.py")])
def test_m5_several_values_go_to_sign_off(name: str, files: str) -> None:
    status, reason = _line(M5 / name)
    assert status == "SIGN-OFF"
    assert f"Several values for max_turns found in {files}" in reason and "cannot pick" in reason
    assert "sets no limit" not in reason


def test_m5_a_run_time_value_goes_to_sign_off() -> None:
    status, reason = _line(M5 / "runtime")
    assert status == "SIGN-OFF" and "int(sys.argv[1])" in reason and "agent.py:" in reason


def test_m5_factory_bound_is_attributed_to_each_agent() -> None:
    bounds = build_observed(M5 / "factory").bounds
    assert sorted(b.agent or "" for b in bounds) == ["researcher", "writer"]


def _tweak(tmp_path: Path, name: str, old: str, new: str, file: str = "agent.py") -> Path:
    d = tmp_path / name
    shutil.copytree(M5 / name, d)
    p = d / file
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new), encoding="utf-8")
    return d


@pytest.mark.parametrize(
    "name,old,new,file",
    [
        ("classattr", "TURNS = 6", "TURNS = 40", "agent.py"),
        ("helper", "go(task, 6)", "go(task, 40)", "agent.py"),
        ("imported", "TURNS = 6", "TURNS = 40", "config.py"),
        ("factory", 'build_agent("writer", 6)', 'build_agent("writer", 40)', "agent.py"),
    ],
)
def test_m5_r1_a_value_above_the_spec_is_still_a_gap(tmp_path: Path, name: str, old: str, new: str, file: str) -> None:
    d = _tweak(tmp_path, name, old, new, file)
    status, reason = _line(d)
    assert status == "GAP" and "40" in reason


def test_m5_r1_a_helper_called_with_two_values_is_never_a_pass(tmp_path: Path) -> None:
    d = _tweak(tmp_path, "helper", "    return go(task, 6)\n", "    go(task, 6)\n    return go(task, 40)\n")
    status, reason = _line(d)
    assert status == "SIGN-OFF" and "Several values for max_turns" in reason


def test_m5_r1_agent_without_tools_and_without_bound_is_reported(tmp_path: Path) -> None:
    d = _tweak(tmp_path, "factory", "max_iter=steps, ", "")
    status, _ = _line(d)
    assert status in ("GAP", "PASS")  # crewai's framework default is judged against the spec; never absent
