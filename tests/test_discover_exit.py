"""discover never reports an all-clear for a run that proves nothing: a crash, an import error from a broken
environment, or a run that never called a model all exit 1 with the reason (and, for an import error, the fix)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture.discover import DiscoverResult
from agenticreq.cli import app

AGENTS = {
    "import_error": "from collections import Sequence\n\ndef main():\n    pass\n",
    "raises": "def main():\n    raise RuntimeError('boom')\n",
    "no_model_call": "def main():\n    return 1\n",
}


@pytest.mark.parametrize("kind", sorted(AGENTS))
def test_a_run_that_proves_nothing_exits_1(tmp_path: Path, kind: str) -> None:
    (tmp_path / "agent.py").write_text(AGENTS[kind], encoding="utf-8")
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    r = CliRunner().invoke(
        app,
        [
            "mock-run",
            "--entry",
            f"{tmp_path / 'agent.py'}:main",
            "--mocks",
            str(tmp_path / "mocks.yml"),
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    out = " ".join(r.output.split())
    assert r.exit_code == 1, out
    assert "mock-run captured 0 model calls." in out and out.rstrip().count("then run: agenticreq mock-run")
    assert (tmp_path / "c.json").exists()  # the captures are still written, for debugging
    if kind == "import_error":  # the environment, and how to fix it
        assert "with ImportError" in out and "a dependency is missing" in out and "python: in mocks.yml" in out
        assert "--python <interpreter>" in out
    if kind == "raises":
        assert "stopped at agent.py:2 with RuntimeError: boom. Fix the error" in out and "dependency" not in out
    if kind == "no_model_call":
        assert "returned without calling a model" in out


def test_problems_are_empty_only_for_a_clean_run() -> None:
    from agenticreq.models.observed import CapturedCall

    call = CapturedCall.model_validate(
        {
            "index": 0,
            "provider": "openai_chat",
            "model": "gpt-4o-2024-08-06",
            "url": "https://api.openai.com/v1/chat/completions",
        }
    )
    assert DiscoverResult(entry="a", calls=[call]).problems() == []
    assert DiscoverResult(entry="a", calls=[call], llm_misses=["x"]).problems()
    assert DiscoverResult(entry="a", calls=[call], zero_touch=False, changed_files=["a.py"]).problems()
    assert DiscoverResult(entry="a", calls=[call], run_errors={"run-1": "Traceback\nValueError: x"}).problems()


def test_audit_warns_about_an_empty_captures_file(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "c.json").write_text('{"entry": "a", "calls": [], "run_errors": {"run-0": "boom"}}', encoding="utf-8")
    r = CliRunner().invoke(
        app, ["audit", str(tmp_path / "agent.py"), "--captures", str(tmp_path / "c.json"), "--fail-on", "never"]
    )
    assert "holds no captured LLM call" in " ".join(r.output.split())


def test_the_exception_line_wins_over_framework_notes() -> None:
    from agenticreq.capture.discover import _last_line

    tb = (
        'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: `temperature` is not supported\n'
        "During task with name 'agent' and id '62db'\n"
    )
    assert _last_line(tb) == "ValueError: `temperature` is not supported"
    assert _last_line("SystemExit(2)") == "SystemExit(2)"


def test_the_child_sees_only_the_agenticreq_package(tmp_path: Path) -> None:
    """Installed into a busy site-packages (conda base), agenticreq must not put that folder on the agent's path:
    its old packages would shadow the agent's own environment."""
    from agenticreq.capture.discover import _isolated_package

    root = Path(_isolated_package(str(tmp_path)))
    assert [p.name for p in root.iterdir()] == ["agenticreq"]
    assert (root / "agenticreq" / "capture" / "runner.py").is_file()


def test_mocks_env_reaches_the_agent_but_never_a_live_secret(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "import os\n\ndef main():\n    raise SystemExit(0 if os.environ.get('AGENT_MODEL') == 'm-1' else 3)\n",
        encoding="utf-8",
    )
    (tmp_path / "mocks.yml").write_text("env:\n  AGENT_MODEL: m-1\ntools: {}\n", encoding="utf-8")
    from agenticreq.capture import discover

    res = discover(f"{tmp_path / 'agent.py'}:main", mock_tools=str(tmp_path / "mocks.yml"))
    assert not res.run_errors  # SystemExit(3) would be recorded as a crash
    (tmp_path / "mocks.yml").write_text(
        "env:\n  OPENAI_API_KEY: sk-proj-abcdefghijklmnopqrstuvwx\ntools: {}\n", encoding="utf-8"
    )
    from agenticreq.capture import MockRunRefused

    with pytest.raises(MockRunRefused):
        discover(f"{tmp_path / 'agent.py'}:main", mock_tools=str(tmp_path / "mocks.yml"))
