"""mock-run keeps every protection after the environment warning was removed: credentials withheld (and said so),
the network blocked, other programs blocked, and a run that changes the project's files reported and failed."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.cli import app

runner = CliRunner()


def _mock_run(tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch, *mocks_lines: str) -> tuple[int, str]:
    (tmp_path / "agent.py").write_text(body, encoding="utf-8")
    (tmp_path / "mocks.yml").write_text(
        "\n".join(["entry: agent.py:main", *mocks_lines, "tools: {}", ""]), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_SERVICE_TOKEN", "real-secret-value")
    r = runner.invoke(app, ["mock-run"], terminal_width=200)
    return r.exit_code, " ".join(r.output.split())


def test_mock_run_output_has_no_environment_warning_and_still_withholds_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out = _mock_run(
        tmp_path, "import os\ndef main():\n    assert 'MY_SERVICE_TOKEN' not in os.environ\n", monkeypatch
    )
    assert "container" not in out.lower()
    assert "credentials withheld from the agent:" in out and "MY_SERVICE_TOKEN" in out
    assert "crashed" not in out  # the agent's own assert held: the credential never reached it
    assert code == 1  # no model call: never an all-clear


def test_network_and_programs_are_still_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "import socket, subprocess, sys\n"
        "def main():\n"
        "    try:\n        socket.create_connection(('example.com', 80), timeout=3)\n"
        "    except OSError:\n        pass\n"
        "    try:\n        subprocess.run([sys.executable, '-c', 'pass'])\n"
        "    except PermissionError:\n        pass\n"
    )
    _, out = _mock_run(tmp_path, body, monkeypatch)
    assert "blocked during the run" in out
    assert "example.com" in out or "connect:" in out
    assert "exec:" in out


def test_a_write_under_allow_write_is_listed_and_not_a_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "out").mkdir()
    body = "import pathlib\ndef main():\n    pathlib.Path('out/result.txt').write_text('done', encoding='utf-8')\n"
    _, out = _mock_run(tmp_path, body, monkeypatch, "allow_write: [out]")
    assert "wrote out/result.txt, 4 bytes" in out
    assert "changed the agent's files" not in out


def test_a_write_outside_allow_write_is_blocked_and_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "import pathlib\ndef main():\n    try:\n        pathlib.Path('report.md').write_text('x')\n"
        "    except PermissionError:\n        pass\n"
    )
    _, out = _mock_run(tmp_path, body, monkeypatch, "allow_write: [out]")
    assert "blocked write to report.md; to allow it, add it under allow_write: in mocks.yml" in out
    assert not (tmp_path / "report.md").exists() and "changed the agent's files" not in out


def test_the_old_option_is_gone() -> None:
    r = runner.invoke(app, ["mock-run", "--require-" + "container"])
    assert r.exit_code == 2 and "No such option" in r.output
    import agenticreq.capture.guard as guard

    assert not hasattr(guard, "in_" + "container")
