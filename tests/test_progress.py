"""The five-step progress block, stage gating in check, and the yes / no / not_needed answers under confirmed:.

Most tests copy tests/fixtures/passfail/captured (an agent that passes every check, with all 14 sign-offs signed)
into a temporary folder, put its mock run where `agenticreq mock-run` saves one, and change one thing."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.progress import ALL_DONE, compute, render, sign_off_yaml

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
PASSING = FIX / "passfail" / "captured"
TODAY = dt.date(2026, 10, 28)


@pytest.fixture(autouse=True)
def _state_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTICREQ_NO_STATE", raising=False)  # these tests are about the state file


def _passing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "app"
    shutil.copytree(PASSING, d)
    (d / ".agenticreq").mkdir()
    shutil.move(str(d / "captures.json"), str(d / ".agenticreq" / "captures.json"))
    (d / "mocks.yml").write_text("entry: agent.py:main\ntools: {}\n", encoding="utf-8")
    old = (d / ".agenticreq" / "captures.json").stat().st_mtime - 3600
    os.utime(d / "agent.py", (old, old))  # the mock run is newer than the code
    monkeypatch.chdir(d)
    return d


def _edit_spec(d: Path, change) -> None:  # type: ignore[no-untyped-def]
    data = yaml.safe_load((d / "spec.yml").read_text(encoding="utf-8"))
    change(data)
    (d / "spec.yml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _block(out: str) -> str:
    """The progress block of a command's output, with + x as ✓ ✗ (the test stream may not print them)."""
    text = out[out.index("Your progress") :]
    return re.sub(r"^  \+ ", "  ✓ ", re.sub(r"^  x ", "  ✗ ", text, flags=re.M), flags=re.M)


def _marks(block: str) -> list[str]:
    return [ln[2] for ln in block.splitlines()[1:6]]


def _json(args: list[str]) -> dict:  # type: ignore[type-arg]
    r = runner.invoke(app, [*args, "-f", "json"])
    return {"exit": r.exit_code, **json.loads(r.stdout)}


def _line(data: dict, param: str) -> dict:  # type: ignore[type-arg]
    return next(r for r in data["results"] if r["param"] == param)


# ================================================================== one test per step state
def test_fresh_folder_points_at_audit(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("def main():\n    pass\n", encoding="utf-8")
    p = compute(tmp_path, today=TODAY)
    assert p.current == 0 and p.next_lines == ["Next: run a quick scan of this folder: agenticreq audit"]
    text = render(p)
    assert "✗ 1. audit      quick scan, no setup" in text and "    2. spec" in text


def test_audit_then_no_spec_points_at_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "agent.py").write_text("def main():\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = runner.invoke(app, ["audit"]).output
    block = _block(out)
    assert _marks(block) == ["✓", "✗", " ", " ", " "]
    assert "spec.yml not written yet" in block and "agenticreq init" in block
    assert json.loads((tmp_path / ".agenticreq" / "state.json").read_text(encoding="utf-8"))["audit_ran"] is True


def test_spec_with_todo_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(d, lambda s: s.update(owner="TODO", technical_owner="TODO"))
    block = _block(runner.invoke(app, ["check"]).output)
    assert _marks(block) == ["✓", "✗", " ", " ", " "]
    assert "2 fields still TODO in spec.yml" in block
    assert "Next: open spec.yml, fill the 2 TODO fields, then run: agenticreq check" in block


def test_gaps_point_at_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(d, lambda s: s.update(prohibited=[{"tool": "lookup_order", "reason": "test"}]))
    block = _block(runner.invoke(app, ["check"]).output)
    assert _marks(block) == ["✓", "✓", "✗", " ", " "]
    assert "3. check      spec vs project" in block and "then run: agenticreq check" in block


def test_stale_mock_run_points_at_mock_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    later = (d / ".agenticreq" / "captures.json").stat().st_mtime + 3600
    os.utime(d / "agent.py", (later, later))  # the code changed after the mock run
    block = _block(runner.invoke(app, ["check"]).output)
    assert _marks(block)[:4] == ["✓", "✓", "✓", "✗"]
    assert "Next: run your agent with mock tools: agenticreq mock-run" in block


def test_sign_offs_left_give_the_yaml_for_the_first_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(
        d, lambda s: s.update(confirmed=[c for c in s["confirmed"] if c["clause"] not in ("owner", "kill_switch")])
    )
    out = runner.invoke(app, ["check"]).output
    block = _block(out)
    assert _marks(block) == ["✓", "✓", "✓", "✓", "✗"]
    assert "5. sign-off   owner confirms 2 items" in block
    first = re.search(r"SIGN-OFF\s+(\w+)", out).group(1)  # type: ignore[union-attr]  # the first open line shown
    today = dt.date.today()
    assert f"  - clause: {first}\n    answer: yes\n    by: you@company.com\n    date: {today.isoformat()}" in block
    assert "Next: add this under confirmed: in spec.yml, with your own details:" in block
    assert "signoff" not in block.replace("sign-off", "")  # no command to run for this step


def test_everything_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _passing(tmp_path, monkeypatch)
    runner.invoke(app, ["audit"])
    r = runner.invoke(app, ["check"])
    block = _block(r.output)
    assert r.exit_code == 0 and _marks(block) == ["✓"] * 5
    assert ALL_DONE in block and "Next:" not in block
    # the other commands read the same state from .agenticreq/state.json
    assert ALL_DONE in _block(runner.invoke(app, ["mocks", "--force"]).output)


def test_quiet_and_json_hide_the_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _passing(tmp_path, monkeypatch)
    assert "Your progress" not in runner.invoke(app, ["check", "--quiet"]).output
    r = runner.invoke(app, ["check", "-f", "json"])
    assert "Your progress" not in r.output and json.loads(r.stdout)["result"] == "PASS"
    assert "Your progress" not in runner.invoke(app, ["check", "-f", "sarif"]).output


def test_block_words_and_ascii_fallback() -> None:
    p = compute(Path("does-not-exist"), today=TODAY)
    for text in (render(p), render(p, unicode=False)):
        assert "fake" not in text.lower() and " code" not in text.lower()
    assert "✓" not in render(p, unicode=False) and "x 1. audit" in render(p, unicode=False)


# ================================================================== stage gating
def test_todo_spec_hides_sign_offs_and_all_shows_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(d, lambda s: s.update(owner="TODO"))
    out = runner.invoke(app, ["check"]).output
    assert re.search(r"Human sign-off: \d+ items, shown after the spec is complete", out)
    assert "HUMAN SIGN-OFF (" not in out and "PASSED (" not in out
    full = runner.invoke(app, ["check", "--all"]).output
    assert "HUMAN SIGN-OFF (" in full and "shown after the spec is complete" not in full


def test_gaps_hide_mock_run_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(FIX / "agents_misc" / "mock_run_app", tmp_path / "app")
    monkeypatch.chdir(tmp_path / "app")
    out = runner.invoke(app, ["check"]).output
    assert "SPEC VS CODE GAPS" in out and "HUMAN SIGN-OFF (" in out  # the spec is complete: sign-offs show
    assert re.search(r"Check with a mock run: \d+ items?, shown after the spec vs project gaps are fixed", out)
    assert "CHECK WITH A MOCK RUN" in runner.invoke(app, ["check", "--all"]).output


# ================================================================== yes / no / not_needed
def _answer(d: Path, clause: str, **entry: object) -> None:
    def change(s: dict) -> None:  # type: ignore[type-arg]
        s["confirmed"] = [c for c in s["confirmed"] if c["clause"] != clause] + [{"clause": clause, **entry}]

    _edit_spec(d, change)


def test_yes_is_signed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _passing(tmp_path, monkeypatch)
    assert _line(_json(["check"]), "owner")["status"] == "SIGNED"


def test_no_is_a_gap_and_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _answer(d, "kill_switch", answer="no", by="lead@example.com", date="2026-10-01", reason="the flag is not built yet")
    data = _json(["check"])
    ln = _line(data, "kill_switch")
    assert ln["status"] == "GAP" and ln["reason"] == "Owner says this is not in place yet: the flag is not built yet."
    assert data["exit"] == 1 and data["result"] == "FAIL"
    out = runner.invoke(app, ["check"]).output
    assert re.search(
        r"x kill_switch\s+Owner says this is not in place yet: the flag is not built yet", out
    ) or re.search(r"✗ kill_switch\s+Owner says this is not in place yet: the flag is not built yet", out)


def test_no_written_as_a_yaml_bool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    text = (
        (d / "spec.yml")
        .read_text(encoding="utf-8")
        .replace("{clause: kill_switch, by:", "{clause: kill_switch, answer: no, reason: not built, by:")
    )
    (d / "spec.yml").write_text(text, encoding="utf-8")  # a bare `no` is YAML false
    assert _line(_json(["check"]), "kill_switch")["status"] == "GAP"


def test_not_needed_is_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)  # risk_level: low, so kill_switch may be not_needed
    _answer(
        d,
        "kill_switch",
        answer="not_needed",
        by="lead@example.com",
        date="2026-10-01",
        reason="internal demo, never deployed",
    )
    data = _json(["check", "--fail-on", "all"])
    assert _line(data, "kill_switch")["status"] == "NOT NEEDED" and data["exit"] == 0
    assert data["counts"]["not_needed"] == 1 and data["counts"]["sign_off_left"] == 0
    out = runner.invoke(app, ["check", "--verbose"]).output
    assert re.search(r"NOT NEEDED\s+kill_switch\s+Internal demo, never deployed\.", out)
    assert "1 not needed" in out


def test_not_needed_on_a_safety_item_at_high_risk_is_a_spec_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(d, lambda s: s.update(risk_level="high"))
    _answer(d, "kill_switch", answer="not_needed", by="lead@example.com", date="2026-10-01", reason="demo")
    r = runner.invoke(app, ["check"])
    assert r.exit_code == 2 and "kill_switch cannot be not_needed at high risk" in " ".join(r.output.split())


def test_yes_without_evidence_stays_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _answer(d, "owner", by="lead@example.com", date="2026-10-01", expires="2099-01-01")
    ln = _line(_json(["check"]), "owner")
    assert ln["status"] == "SIGN-OFF" and "evidence is missing" in ln["reason"]


def test_expired_sign_off_is_open_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _answer(d, "owner", by="lead@example.com", date="2020-01-01", expires="2021-01-01", evidence="org chart")
    ln = _line(_json(["check"]), "owner")
    assert ln["status"] == "SIGN-OFF" and "sign-off expired" in ln["reason"]


def test_changed_evidence_file_is_open_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _passing(tmp_path, monkeypatch)
    _answer(d, "owner", by="lead@example.com", date="2020-01-01", expires="2099-01-01", evidence="agent.py:1")
    ln = _line(_json(["check"]), "owner")
    assert ln["status"] == "SIGN-OFF" and "file changed since sign-off" in ln["reason"]


# ================================================================== the Next: YAML and snapshots
def test_sign_off_yaml_uses_today_and_six_months() -> None:
    lines = sign_off_yaml("owner", dt.date(2026, 8, 31))
    assert lines[1:] == [
        "  - clause: owner",
        "    answer: yes",
        "    by: you@company.com",
        "    date: 2026-08-31",
        "    expires: 2027-02-28",
        '    evidence: "Name of the person who owns this agent"',
    ]
    assert yaml.safe_load("\n".join(ln[2:] for ln in lines[1:]))[0]["clause"] == "owner"  # pastes as valid YAML


def _snapshot(name: str, text: str) -> None:
    path = GOLDEN / f"progress_{name}.txt"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(text + "\n", encoding="utf-8")
    assert text + "\n" == path.read_text(encoding="utf-8")


def test_progress_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "fresh").mkdir()
    _snapshot("fresh", render(compute(tmp_path / "fresh", today=TODAY)))
    d = _passing(tmp_path, monkeypatch)
    _edit_spec(d, lambda s: s.update(owner="TODO"))
    runner.invoke(app, ["check"])
    _snapshot("spec_todo", render(compute(d, today=TODAY)))
    _edit_spec(
        d,
        lambda s: s.update(owner="owner@example.com", confirmed=[c for c in s["confirmed"] if c["clause"] != "owner"]),
    )
    runner.invoke(app, ["check"])
    _snapshot("sign_off", render(compute(d, today=TODAY)))
