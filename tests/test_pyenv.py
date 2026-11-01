"""Which Python `mock-run` uses, and the check that it can run the agent (agenticreq/pyenv.py)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq import pyenv
from agenticreq.cli import app

FIX = Path(__file__).parent / "fixtures" / "agents_misc"
runner = CliRunner()


def _exe(prefix: Path, windows: bool = False, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    exe = prefix / ("Scripts" if windows else "bin") / ("python.exe" if windows else "python")
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(body, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def _venv(folder: Path, name: str, windows: bool = False) -> Path:
    (folder / name).mkdir(parents=True, exist_ok=True)
    (folder / name / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return _exe(folder / name, windows)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYENV_ROOT", "ASDF_DATA_DIR"):
        monkeypatch.delenv(var, raising=False)


def _choose(folder: Path, **kw: object) -> pyenv.PythonChoice:
    kw.setdefault("environ", {"HOME": str(folder / "home")})
    return pyenv.choose(folder, **kw)  # type: ignore[arg-type]


# ============================================================================ project folders (pyvenv.cfg)
@pytest.mark.parametrize("name", [".venv", "venv", "env", "uv", "myenv", ".virtualenv", "poetry-env"])
def test_a_project_folder_with_pyvenv_cfg_is_found_whatever_its_name(tmp_path: Path, name: str) -> None:
    exe = _venv(tmp_path, name)
    got = _choose(tmp_path)
    assert got.path == str(exe) and got.source == f"./{name} (pyvenv.cfg)" and not got.warning


def test_dot_venv_beats_venv_beats_env_beats_other_names(tmp_path: Path) -> None:
    _venv(tmp_path, "zeta")
    _venv(tmp_path, "env")
    assert _choose(tmp_path).source == "./env (pyvenv.cfg)"
    _venv(tmp_path, "venv")
    assert _choose(tmp_path).source == "./venv (pyvenv.cfg)"
    _venv(tmp_path, ".venv")
    assert _choose(tmp_path).source == "./.venv (pyvenv.cfg)"


def test_a_folder_without_pyvenv_cfg_or_without_an_interpreter_is_skipped(tmp_path: Path) -> None:
    _exe(tmp_path / "notvenv")  # an interpreter but no pyvenv.cfg: not an environment
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")  # a venv with no interpreter in it
    good = _venv(tmp_path, "venv")
    assert _choose(tmp_path).path == str(good)


def test_nothing_found_falls_back_to_agenticreqs_own_interpreter_with_a_visible_warning(tmp_path: Path) -> None:
    got = _choose(tmp_path)
    assert got.path == sys.executable and got.source == "agenticreq's own interpreter"
    assert "--python" in got.warning and "mocks.yml" in got.warning
    assert got.line() == f"python: {sys.executable} (from agenticreq's own interpreter)"


# ============================================================================ precedence
def test_the_first_match_wins_in_the_documented_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = _venv(tmp_path, ".venv")
    active = _exe(tmp_path / "active")
    conda = _exe(tmp_path / "conda")
    env = {
        "HOME": str(tmp_path / "home"),
        "VIRTUAL_ENV": str(tmp_path / "active"),
        "CONDA_PREFIX": str(tmp_path / "conda"),
    }
    assert pyenv.choose(tmp_path, "cli-py", "mocks-py", env) == pyenv.PythonChoice("cli-py", "--python")  # 1
    assert pyenv.choose(tmp_path, None, "mocks-py", env) == pyenv.PythonChoice("mocks-py", "python: in mocks.yml")  # 2
    assert pyenv.choose(tmp_path, environ=env).path == str(active)  # 3: $VIRTUAL_ENV beats $CONDA_PREFIX and the folder
    assert pyenv.choose(tmp_path, environ=env).source == "$VIRTUAL_ENV"
    del env["VIRTUAL_ENV"]
    assert pyenv.choose(tmp_path, environ=env).path == str(conda)
    assert pyenv.choose(tmp_path, environ=env).source == "$CONDA_PREFIX"
    del env["CONDA_PREFIX"]
    assert pyenv.choose(tmp_path, environ=env).path == str(local)  # 4
    shutil.rmtree(tmp_path / ".venv")
    assert pyenv.choose(tmp_path, environ=env).path == sys.executable  # 5


def test_an_active_virtual_env_or_conda_prefix_without_an_interpreter_falls_through(tmp_path: Path) -> None:
    local = _venv(tmp_path, ".venv")
    env = {
        "HOME": str(tmp_path / "home"),
        "VIRTUAL_ENV": str(tmp_path / "gone"),
        "CONDA_PREFIX": str(tmp_path / "gone2"),
    }
    assert pyenv.choose(tmp_path, environ=env).path == str(local)


def test_an_active_conda_base_is_not_the_projects_environment(tmp_path: Path) -> None:
    base = _exe(tmp_path / "anaconda")
    local = _venv(tmp_path, ".venv")
    env = {"HOME": str(tmp_path / "home"), "CONDA_PREFIX": str(tmp_path / "anaconda"), "CONDA_DEFAULT_ENV": "base"}
    assert pyenv.choose(tmp_path, environ=env).path == str(local) != str(base)
    shutil.rmtree(tmp_path / ".venv")
    assert pyenv.choose(tmp_path, environ=env).path == sys.executable  # not the base either; --python picks it


# ============================================================================ Poetry, Pipenv, Conda, pyenv, asdf
def _tools(monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]) -> list[list[str]]:
    """Replace the external tools: ``answers`` maps the tool name to its stdout; every call is recorded."""
    calls: list[list[str]] = []

    def fake(cmd: list[str], cwd: Path) -> str:
        calls.append(cmd)
        return answers.get(cmd[0], "")

    monkeypatch.setattr(pyenv, "run_tool", fake)
    return calls


def test_poetry_is_asked_for_its_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _exe(tmp_path / "cache" / "poetry-env")
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'x'\n", encoding="utf-8")
    calls = _tools(monkeypatch, {"poetry": str(exe) + "\n"})
    got = _choose(tmp_path)
    assert got.path == str(exe) and got.source == "Poetry (poetry env info)"
    assert calls == [["poetry", "env", "info", "--executable"]]


def test_poetry_is_not_asked_when_the_project_does_not_use_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    calls = _tools(monkeypatch, {"poetry": "/nope"})
    assert _choose(tmp_path).path == sys.executable and calls == []


def test_pipenv_is_asked_for_its_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = tmp_path / "cache" / "proj-AbC"
    exe = _exe(prefix)
    (tmp_path / "Pipfile").write_text("[packages]\n", encoding="utf-8")
    calls = _tools(monkeypatch, {"pipenv": str(prefix)})
    got = _choose(tmp_path)
    assert got.path == str(exe) and got.source == "Pipenv (pipenv --venv)" and calls == [["pipenv", "--venv"]]


def test_conda_matches_the_environment_yml_name_against_conda_env_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other, mine = tmp_path / "envs" / "other", tmp_path / "envs" / "agentenv"
    _exe(other)
    exe = _exe(mine)
    (tmp_path / "environment.yml").write_text("name: agentenv\ndependencies: [python=3.11]\n", encoding="utf-8")
    calls = _tools(monkeypatch, {"conda": json.dumps({"envs": [str(other), str(mine)]})})
    got = _choose(tmp_path)
    assert got.path == str(exe) and got.source == "Conda environment agentenv (environment.yml)"
    assert calls == [["conda", "env", "list", "--json"]]


def test_conda_with_an_unknown_name_or_bad_output_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "environment.yml").write_text("name: nowhere\n", encoding="utf-8")
    _tools(monkeypatch, {"conda": json.dumps({"envs": [str(tmp_path / "envs" / "other")]})})
    assert _choose(tmp_path).path == sys.executable
    _tools(monkeypatch, {"conda": "not json"})
    assert _choose(tmp_path).path == sys.executable


def test_a_missing_tool_never_raises_and_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "Pipfile").write_text("[packages]\n", encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(pyenv.shutil, "which", lambda name: None)
    assert pyenv.run_tool(["poetry", "env", "info"], tmp_path) == ""
    assert _choose(tmp_path).path == sys.executable


def test_a_tool_that_fails_or_hangs_returns_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pyenv.shutil, "which", lambda name: "/usr/bin/" + name)

    def boom(*a: object, **k: object) -> None:
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(pyenv.subprocess, "run", boom)
    assert pyenv.run_tool(["conda", "env", "list"], tmp_path) == ""
    monkeypatch.setattr(
        pyenv.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="partial", stderr="x")
    )
    assert pyenv.run_tool(["conda", "env", "list"], tmp_path) == ""  # a non-zero exit is no answer


def test_the_tools_are_called_with_a_short_timeout_and_no_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(pyenv.shutil, "which", lambda name: "/usr/bin/" + name)

    def fake(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        seen.update(kw, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="x", stderr="")

    monkeypatch.setattr(pyenv.subprocess, "run", fake)
    pyenv.run_tool(["poetry", "env", "info", "--executable"], tmp_path)
    assert 0 < float(seen["timeout"]) <= 10 and "install" not in seen["cmd"] and "create" not in seen["cmd"]  # type: ignore[operator]


def test_pyenv_reads_python_version_from_disk(tmp_path: Path) -> None:
    root = tmp_path / "pyenvroot"
    exe = _exe(root / "versions" / "3.11.7")
    _exe(root / "versions" / "3.12.1")
    (tmp_path / ".python-version").write_text("3.11.7\n", encoding="utf-8")
    env = {"HOME": str(tmp_path / "home"), "PYENV_ROOT": str(root)}
    got = pyenv.choose(tmp_path, environ=env)
    assert got.path == str(exe) and got.source == "pyenv (.python-version)"
    (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")  # a partial version: the newest 3.12.*
    assert pyenv.choose(tmp_path, environ=env).path.endswith("3.12.1/bin/python")
    (tmp_path / ".python-version").write_text("system\n", encoding="utf-8")
    assert pyenv.choose(tmp_path, environ=env).path == sys.executable


def test_asdf_reads_tool_versions_from_disk(tmp_path: Path) -> None:
    root = tmp_path / "asdf"
    exe = _exe(root / "installs" / "python" / "3.10.14")
    (tmp_path / ".tool-versions").write_text("nodejs 20.1.0\npython 3.10.14  # the agent\n", encoding="utf-8")
    got = pyenv.choose(tmp_path, environ={"HOME": str(tmp_path / "home"), "ASDF_DATA_DIR": str(root)})
    assert got.path == str(exe) and got.source == "asdf (.tool-versions)"


def test_project_folder_beats_poetry_pipenv_conda_and_pyenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = _venv(tmp_path, "env")
    (tmp_path / "Pipfile").write_text("", encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")
    calls = _tools(monkeypatch, {"poetry": "/x", "pipenv": "/y"})
    assert _choose(tmp_path).path == str(local) and calls == []  # the external tools were never called


# ============================================================================ Windows
def test_windows_uses_scripts_python_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    win = _venv(tmp_path, ".venv", windows=True)
    assert pyenv.interpreter_in(tmp_path / ".venv", windows=True) == win
    assert pyenv.interpreter_in(tmp_path / ".venv", windows=False) is None  # no bin/python in this one
    monkeypatch.setattr(pyenv, "is_windows", lambda: True)
    got = _choose(tmp_path)
    assert got.path == str(win) and got.path.endswith(os.path.join("Scripts", "python.exe"))


def test_posix_uses_bin_python_and_conda_on_windows_keeps_python_exe_at_the_prefix(tmp_path: Path) -> None:
    posix = _exe(tmp_path / "p")
    assert pyenv.interpreter_in(tmp_path / "p", windows=False) == posix
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "python.exe").write_text("", encoding="utf-8")
    assert pyenv.interpreter_in(tmp_path / "c", windows=True) == tmp_path / "c" / "python.exe"


# ============================================================================ before the run
def _fake_run(
    version: str = "3.11.7",
    modules: dict[str, bool] | None = None,
    dists: dict[str, bool] | None = None,
    code: int = 0,
    raises: Exception | None = None,
):  # type: ignore[no-untyped-def]
    def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if raises:
            raise raises
        want = json.loads(cmd[-1])
        out = {
            "version": version,
            "modules": {m: (modules or {}).get(m, True) for m in want["modules"]},
            "dists": {d: (dists or {}).get(d, True) for d in want["dists"]},
        }
        return subprocess.CompletedProcess(cmd, code, stdout=json.dumps(out) + "\n", stderr="boom" if code else "")

    return run


def _project(tmp_path: Path, imports: str = "import agents\n", **files: str) -> Path:
    (tmp_path / "agent.py").write_text(imports + "def run():\n    pass\n", encoding="utf-8")
    for name, text in files.items():
        (tmp_path / name.replace("__", ".")).write_text(text, encoding="utf-8")
    return tmp_path


def test_a_working_interpreter_with_the_framework_has_no_problem(tmp_path: Path) -> None:
    proj = _project(tmp_path, requirements__txt="openai-agents>=0.2\n")
    got = pyenv.preflight("py", proj, "agent.py:run", "--python", run=_fake_run())
    assert got.problems == [] and got.warnings == [] and got.version == "3.11.7"


def test_an_interpreter_that_does_not_start_stops_the_run(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    for run in (_fake_run(raises=OSError("exec format error")), _fake_run(code=1)):
        got = pyenv.preflight("/x/bin/python", proj, "agent.py:run", "./.venv (pyvenv.cfg)", run=run)
        [msg] = got.problems
        assert "/x/bin/python (from ./.venv (pyvenv.cfg)) does not start" in msg and "--python" in msg


def test_a_missing_framework_names_the_python_the_packages_and_the_install_command(tmp_path: Path) -> None:
    proj = _project(tmp_path, requirements__txt="openai-agents\nrequests\n")
    got = pyenv.preflight(
        "/x/bin/python",
        proj,
        "agent.py:run",
        "$VIRTUAL_ENV",
        run=_fake_run(modules={"agents": False}, dists={"openai-agents": False}),
    )
    [msg] = got.problems
    assert "/x/bin/python (from $VIRTUAL_ENV)" in msg and "missing openai-agents" in msg
    assert "/x/bin/python -m pip install -r requirements.txt" in msg and "installs nothing" in msg


def test_a_missing_framework_without_requirements_txt_names_the_package(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    got = pyenv.preflight(
        "py", proj, "agent.py:run", run=_fake_run(modules={"agents": False}, dists={"openai-agents": False})
    )
    assert "py -m pip install openai-agents" in got.problems[0]


def test_the_framework_is_read_from_pyproject_and_pipfile_too(tmp_path: Path) -> None:
    proj = _project(
        tmp_path,
        imports="",
        pyproject__toml='[project]\nname="x"\ndependencies=["crewai>=1"]\n',
        Pipfile='[packages]\nlanggraph = "*"\n',
    )
    assert set(pyenv.declared_dists(proj)) == {"crewai", "langgraph"}
    got = pyenv.preflight("py", proj, "agent.py:run", run=_fake_run(dists={"crewai": False, "langgraph": False}))
    assert "missing crewai, langgraph" in got.problems[0]


def test_the_projects_own_agents_package_is_not_the_framework(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    (proj / "agents").mkdir()
    got = pyenv.preflight(
        "py", proj, "agent.py:run", run=_fake_run(modules={"agents": False}, dists={"openai-agents": False})
    )
    assert got.problems == []


def test_a_python_version_outside_requires_python_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    proj = _project(tmp_path, pyproject__toml='[project]\nname="x"\nrequires-python = ">=3.11"\n')
    got = pyenv.preflight("py", proj, "agent.py:run", run=_fake_run(version="3.9.2"))
    assert got.problems == [] and "3.9.2" in got.warnings[0] and ">=3.11" in got.warnings[0]
    assert pyenv.preflight("py", proj, "agent.py:run", run=_fake_run(version="3.12.0")).warnings == []


def test_a_python_version_file_that_differs_is_a_warning(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    (proj / ".python-version").write_text("3.11.4\n", encoding="utf-8")
    got = pyenv.preflight("py", proj, "agent.py:run", run=_fake_run(version="3.9.2"))
    assert got.problems == [] and ".python-version 3.11.4" in got.warnings[0]


@pytest.mark.parametrize(
    "version,spec,bad",
    [
        ("3.11.7", ">=3.10", False),
        ("3.9.0", ">=3.10", True),
        ("3.13.0", ">=3.10,<3.13", True),
        ("3.12.9", ">=3.10,<3.13", False),
        ("3.11.4", "~=3.11", False),
        ("3.12.0", "~=3.11.0", True),
        ("3.11.9", "==3.11.*", False),
        ("3.10.1", "==3.11.*", True),
        ("3.11.0", "!=3.11.*", True),
        ("3.8.0", "garbage", False),
    ],
)
def test_requires_python_comparison(version: str, spec: str, bad: bool) -> None:
    assert pyenv.version_mismatch(version, spec) is bad


# ============================================================================ the CLI
def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "app"
    shutil.copytree(FIX / "multi_run_app", dest)
    monkeypatch.chdir(dest)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return dest


def test_mock_run_prints_the_interpreter_and_where_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app(tmp_path, monkeypatch)
    out = " ".join(runner.invoke(app, ["mock-run", "--python", sys.executable, "--run", "billing"]).output.split())
    assert f"python: {sys.executable} (from --python)" in out and "captured 2/2 LLM calls" in out


def test_mocks_yml_python_is_source_two_and_the_flag_beats_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, monkeypatch)
    (root / "mocks.yml").write_text(
        (root / "mocks.yml").read_text(encoding="utf-8") + f"python: {sys.executable}\n", encoding="utf-8"
    )
    out = " ".join(runner.invoke(app, ["mock-run", "--run", "billing"]).output.split())
    assert f"python: {sys.executable} (from python: in mocks.yml)" in out
    out = " ".join(runner.invoke(app, ["mock-run", "--run", "billing", "--python", sys.executable]).output.split())
    assert "(from --python)" in out and "python: in mocks.yml" not in out


def test_a_broken_project_interpreter_stops_before_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, monkeypatch)
    _venv(root, ".venv", windows=False)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    r = runner.invoke(app, ["mock-run"])
    out = " ".join(r.output.split())
    assert r.exit_code == 2 and "(from ./.venv (pyvenv.cfg))" in out and "does not start" in out
    assert not (root / ".agenticreq" / "captures.json").exists()  # nothing ran, nothing saved


def test_no_environment_found_warns_visibly_and_still_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app(tmp_path, monkeypatch)
    monkeypatch.setattr(pyenv.shutil, "which", lambda name: None)
    r = runner.invoke(app, ["mock-run", "--run", "billing"])
    out = " ".join(r.output.split())
    assert r.exit_code == 0 and "(from agenticreq's own interpreter)" in out and "warning: no environment found" in out


def test_the_mock_run_child_needs_neither_pyyaml_nor_agenticreq_ast(tmp_path: Path) -> None:
    """The agent's environment is not ours: the child reads the run's settings as JSON and never imports pyyaml or
    the code scanner, so a bare agent environment (pydantic only) can run it."""
    from agenticreq.capture.responder import MockConfig

    mocks = tmp_path / "mocks.yml"
    mocks.write_text(
        "order: [a]\ntools:\n  a: {args: {x: 1}, call: false}\nruns:\n  - {name: r, order: [b]}\n", encoding="utf-8"
    )
    cfg = MockConfig.load(mocks).for_run("r")
    blob = tmp_path / "cfg.json"
    blob.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    code = (
        "import sys, json\n"
        "sys.modules['yaml'] = None\n"  # an import of yaml raises ImportError, as in an environment without it
        "import agenticreq.capture.runner as r\n"
        "from agenticreq.capture.responder import MockConfig\n"
        f"m = MockConfig.from_dict(json.load(open({str(blob)!r})))\n"
        "assert m.order == ['b'] and m.tools['a'].call is False and m.tools['a'].args == {'x': 1}, m\n"
        "assert 'agenticreq.ast' not in sys.modules\n"
    )
    src = str(Path(pyenv.__file__).resolve().parents[1])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": src},
    )
    assert proc.returncode == 0, proc.stderr


def test_a_framework_that_is_installed_but_does_not_import_is_left_to_the_version_gate(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    got = pyenv.preflight(
        "py", proj, "agent.py:run", run=_fake_run(modules={"agents": False}, dists={"openai-agents": True})
    )
    assert got.problems == []  # the distribution is there: the version check (known_bad) explains what is wrong
