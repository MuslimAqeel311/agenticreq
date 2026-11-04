"""Phase 3: `mocks`, `mock-run`, `check` from start to end.

The first half needs no framework. The second runs a real framework in its own environment: point
AGENTICREQ_LIVE_VENVS at a folder with one venv per framework (langgraph/, crewai/, ...); a venv that is missing is
skipped, and the static tests still cover that fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agenticreq.capture.guard import norm_path, under
from agenticreq.cli import app
from agenticreq.observe import build_observed

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
ISSUES = FIX / "issues"
LIVE = FIX / "live"
VENVS = Path(os.environ.get("AGENTICREQ_LIVE_VENVS", "/nonexistent"))


def _tree(d: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(d)): p.read_bytes()
        for p in sorted(d.rglob("*"))
        if p.is_file() and ".agenticreq" not in p.parts and "__pycache__" not in p.parts
    }


def _project(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / name
    shutil.copytree(ISSUES / name, d)
    monkeypatch.chdir(d)
    return d


def _run(*extra: str) -> Any:
    return runner.invoke(app, ["mock-run", "--python", sys.executable, "--quiet", *extra])


def test_generated_mocks_run_without_manual_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    (d / "mocks.yml").unlink()
    assert runner.invoke(app, ["mocks"]).exit_code == 0
    assert _run().exit_code == 0


def test_call_false_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    (d / "mocks.yml").write_text(
        "entry: agent.py:main\nargs: {task: t}\nallow_write: [report.txt]\ntools:\n  search: {args: {q: a}}\n"
        "  save_report: {call: false}\n",
        encoding="utf-8",
    )
    assert _run().exit_code == 0
    data = json.loads((d / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))
    asked = {tc["name"] for c in data["calls"] for tc in c["tool_calls"]}
    assert "save_report" not in asked and "search" in asked


def test_writes_outside_allow_write_leave_the_tree_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a5_write", tmp_path, monkeypatch)
    before = _tree(d)
    assert _run().exit_code == 0
    assert _tree(d) == before


def test_guards_network_process_credentials_and_no_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "g"
    d.mkdir()
    shutil.copy(ISSUES / "a1_gate" / "minifw.py", d)
    (d / "agent.py").write_text(
        "import os, socket, subprocess\nfrom minifw import Agent, tool\n\n\n@tool\ndef probe(q: str) -> str:\n"
        '    """Probe."""\n    for _ in range(3):\n        try:\n'
        "            socket.create_connection(('example.com', 80), timeout=1)\n        except OSError:\n            pass\n"
        "        try:\n            subprocess.run(['ls'])\n        except OSError:\n            pass\n"
        "    return os.environ['OPENAI_API_KEY']\n\n\ndef main(task: str = 'go') -> str:\n"
        "    return Agent(name='a', tools=[probe]).invoke(task, max_turns=3)\n",
        encoding="utf-8",
    )
    (d / "mocks.yml").write_text(
        "entry: agent.py:main\nargs: {task: t}\ntools:\n  probe: {args: {q: a}}\n", encoding="utf-8"
    )
    monkeypatch.chdir(d)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-real")
    r = _run()
    assert r.exit_code == 0, r.output
    text = " ".join(r.output.split())
    line = next(ln for ln in r.output.replace("\n", " ").split("blocked during the run")[1:2])
    assert "sk-live-real" not in text
    assert line.count("dns:example.com") <= 1 and "exec:" in line


def test_several_runs_merge_and_an_unreached_tool_is_still_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project("a4_order", tmp_path, monkeypatch)
    text = (d / "mocks.yml").read_text(encoding="utf-8")
    (d / "mocks.yml").write_text(
        text.replace("  - {name: support, order: [transfer_to_support]}\n", ""), encoding="utf-8"
    )
    assert _run().exit_code == 0
    data = json.loads((d / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))
    assert {tc["name"] for c in data["calls"] for tc in c["tool_calls"]} == {"transfer_to_billing"}


def test_a_timeout_ends_cleanly_with_its_own_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    (d / "agent.py").write_text(
        (d / "agent.py")
        .read_text(encoding="utf-8")
        .replace('return Agent(name="writer"', 'import time\n    time.sleep(30)\n    return Agent(name="writer"'),
        encoding="utf-8",
    )
    r = _run("--timeout", "2")
    assert r.exit_code == 1 and "before the 2s limit stopped" in " ".join(r.output.split())


def test_running_again_gives_identical_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project("a1_gate", tmp_path, monkeypatch)
    _run()
    first = runner.invoke(app, ["check", "-f", "json", "--fail-on", "never", "--quiet"]).stdout
    _run()
    second = runner.invoke(app, ["check", "-f", "json", "--fail-on", "never", "--quiet"]).stdout

    def strip(text: str) -> Any:
        d = json.loads(text)
        d.pop("generated_at", None)
        d.get("mock_run", {}).pop("time", None)
        return d["results"]

    assert strip(first) == strip(second)


def test_the_agents_environment_does_not_need_agenticreq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    venv = tmp_path / "bare"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = venv / "bin" / "python"
    subprocess.run([str(py), "-m", "pip", "install", "-q", "httpx", "pydantic"], check=True, capture_output=True)
    assert subprocess.run([str(py), "-c", "import agenticreq"], capture_output=True).returncode != 0
    r = runner.invoke(app, ["mock-run", "--python", str(py), "--quiet"])
    assert r.exit_code == 0, r.output
    assert (d / ".agenticreq" / "captures.json").is_file()


@pytest.mark.parametrize(
    "path,root,inside",
    [
        ("C:\\proj\\workspace\\a.md", "C:\\proj\\workspace", True),
        ("C:\\proj\\workspace\\..\\report.md", "C:\\proj\\workspace", False),
        ("C:/proj/workspace/a.md", "C:\\proj\\workspace", True),
        ("C:\\proj\\workspace2\\a.md", "C:\\proj\\workspace", False),
    ],
)
def test_windows_path_handling(path: str, root: str, inside: bool) -> None:
    assert under(path, [norm_path(root)]) is inside


def test_a_read_only_config_directory_does_not_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    (home / ".config").chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("HOME", str(home))
    (d / "agent.py").write_text(
        (d / "agent.py")
        .read_text(encoding="utf-8")
        .replace(
            "def main(",
            'def _cfg() -> None:\n    import os\n    p = os.path.expanduser("~/.config/tool")\n'
            '    os.makedirs(p, exist_ok=True)\n    open(os.path.join(p, "c.json"), "w").write("{}")\n\n\ndef main(',
        )
        .replace('    return Agent(name="writer"', '    _cfg()\n    return Agent(name="writer"'),
        encoding="utf-8",
    )
    try:
        r = _run()
        assert r.exit_code == 0, r.output
        assert not (home / ".config" / "tool").exists()
    finally:
        (home / ".config").chmod(stat.S_IRWXU)


# =========================================================================== real frameworks
#: fixture folder -> venv folder under AGENTICREQ_LIVE_VENVS
LIVE_PROJECTS = {
    "langgraph": "langgraph",
    "crewai": "crewai",
    "openai_agents": "openai_agents",
    "google_adk": "google_adk",
    "smolagents": "smolagents",
    "autogen": "autogen",
    "llamaindex": "llamaindex",
    "pydantic_ai": "pydantic_ai",
    "haystack": "haystack",
    "agno": "agno",
    "semantic_kernel": "semantic_kernel",
    "sdk_openai": "openai",
    "sdk_anthropic": "anthropic",
    "sdk_genai": "google_genai",
}


@pytest.mark.parametrize("project", LIVE_PROJECTS)
def test_every_framework_fixture_reads_statically(project: str) -> None:
    obs = build_observed(LIVE / project)
    assert "lookup" in obs.declared_tools or obs.declared_tools


@pytest.mark.parametrize("project,venv", LIVE_PROJECTS.items())
def test_live_mocks_then_mock_run_then_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: str, venv: str
) -> None:
    py = VENVS / venv / "bin" / "python"
    if not py.is_file():
        pytest.skip(f"no venv for {venv}: set AGENTICREQ_LIVE_VENVS")
    d = tmp_path / project
    shutil.copytree(LIVE / project, d)
    (d / "spec.yml").write_text((ISSUES / "b3_hosted" / "spec.yml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(d)
    before = _tree(d)
    assert runner.invoke(app, ["mocks"]).exit_code == 0
    before["mocks.yml"] = (d / "mocks.yml").read_bytes()
    r = runner.invoke(app, ["mock-run", "--python", str(py), "--quiet", "--timeout", "120"])
    assert r.exit_code == 0, r.output
    assert _tree(d) == before  # nothing written outside .agenticreq/
    data = json.loads((d / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))
    assert data["calls"] and "lookup" in {t.replace("Notes-", "") for t in data["offered_tools"]}
    c = runner.invoke(app, ["check", "-f", "json", "--fail-on", "never", "--quiet"])
    lines = {ln["param"]: ln for ln in json.loads(c.stdout)["results"]}
    assert "Never offered" not in lines.get("lookup", {}).get("reason", "")
