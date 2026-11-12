"""H3: every folder a mock run redirects exists before the run, the write guard never blocks it or an allow_write:
folder (path spelling of all three OSes), env: in mocks.yml reaches the agent, and an import failure names the
package, the framework and the file."""

from __future__ import annotations

import ntpath
import os
import posixpath
from pathlib import Path
from typing import Any

import pytest

from agenticreq.capture.discover import DiscoverResult, _USER_DIRS, explain, scratch_config_dirs
from agenticreq.capture.guard import norm_path, under


def test_h3_a_every_redirected_folder_is_created(tmp_path: Path) -> None:
    dirs = scratch_config_dirs(str(tmp_path))
    for k in (*_USER_DIRS, "HOME", "USERPROFILE"):
        assert Path(dirs[k]).is_dir(), k
    home = Path(dirs["HOME"])
    for sub in (".config", ".cache", ".local/share", "Library/Application Support", "AppData/Roaming", "AppData/Local"):
        assert (home / sub).is_dir(), sub


def test_h3_end_to_end_folders_exist_writes_allowed_and_env_reaches_the_agent(tmp_path: Path) -> None:
    from agenticreq.capture import discover as mock_run

    (tmp_path / "agent.py").write_text(
        "import os\nfrom pathlib import Path\n\ndef main():\n"
        "    for k in ('APPDATA', 'LOCALAPPDATA', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME'):\n"
        "        assert Path(os.environ[k]).is_dir(), k\n"
        "        (Path(os.environ[k]) / 'state.json').write_text('{}')\n"
        "    lib = Path.home() / 'Library' / 'Application Support'\n"
        "    assert lib.is_dir()\n"
        "    (lib / 'x.db').write_text('')\n"
        "    assert os.environ['MY_FLAG'] == 'on'\n"
        "    assert os.environ['XDG_CONFIG_HOME'] == os.environ['MY_CONFIG']\n",
        encoding="utf-8",
    )
    own = tmp_path / "own config"
    own.mkdir()
    (tmp_path / "mocks.yml").write_text(
        f"tools: {{}}\nenv:\n  MY_FLAG: 'on'\n  XDG_CONFIG_HOME: '{own}'\n  MY_CONFIG: '{own}'\n", encoding="utf-8"
    )
    res = mock_run(f"{tmp_path / 'agent.py'}:main", mock_tools=str(tmp_path / "mocks.yml"))
    assert not res.run_errors, res.run_errors
    assert not [b for b in res.egress_blocked if b.startswith("write:")], res.egress_blocked


# ------------------------------------------------------------------ path comparison, per OS
@pytest.mark.parametrize(
    "path,root",
    [
        ("C:\\Users\\Ana\\AppData\\Local\\x.db", "c:\\users\\ana\\appdata\\local"),  # case
        ("c:/users/ana/appdata/local/x.db", "C:\\Users\\Ana\\AppData\\Local\\"),  # / and trailing separator
        ("C:\\My Project\\out dir\\a.txt", "C:\\My Project\\out dir"),  # spaces
        ("\\\\server\\share\\agents\\out\\a.txt", "\\\\server\\share\\agents\\out"),  # UNC
        ("\\\\?\\C:\\very\\long\\path\\a.txt", "C:\\very\\long"),  # long-path prefix on the path
        ("C:\\very\\long\\path\\a.txt", "\\\\?\\C:\\very\\long"),  # ... and on the root
        ("\\\\?\\UNC\\server\\share\\a.txt", "\\\\server\\share"),  # long UNC
        ("C:\\proj\\sub\\..\\out\\a.txt", "C:\\proj\\out"),  # ..
    ],
)
def test_h3_b_windows_paths_compare_equal(path: str, root: str) -> None:
    assert under(path, [norm_path(root, ntpath)], ntpath)


@pytest.mark.parametrize(
    "path,root",
    [
        ("C:\\proj\\out2\\a.txt", "C:\\proj\\out"),  # a sibling with the same prefix is not inside
        ("D:\\proj\\out\\a.txt", "C:\\proj\\out"),
        ("C:\\proj\\out\\..\\secret.txt", "C:\\proj\\out"),
    ],
)
def test_h3_b_windows_paths_outside_stay_outside(path: str, root: str) -> None:
    assert not under(path, [norm_path(root, ntpath)], ntpath)


@pytest.mark.parametrize(
    "path,root,inside",
    [
        ("/Users/ana/Library/Application Support/x/a.db", "/Users/ana/Library/Application Support/", True),  # macOS
        ("/home/ana/.cache/x", "/home/ana/.cache", True),  # Linux
        ("/home/ana/.cache2/x", "/home/ana/.cache", False),
        ("/home/ana/proj/../.ssh/id", "/home/ana/proj", False),
        ("/Users/Ana/x", "/users/ana", False),  # case matters on posix spelling
    ],
)
def test_h3_b_posix_paths(path: str, root: str, inside: bool) -> None:
    assert under(path, [norm_path(root, posixpath)], posixpath) is inside


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_h3_b_symlinks_are_resolved(tmp_path: Path) -> None:
    real = tmp_path / "real out"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert under(link / "a.txt", [norm_path(real)])
    assert not under(link / "a.txt", [norm_path(tmp_path / "elsewhere")])


def test_h3_b_a_redirected_folder_is_never_blocked(tmp_path: Path) -> None:
    from agenticreq.capture.guard import ProcessGuard

    dirs = scratch_config_dirs(str(tmp_path / "scratch"))
    g = ProcessGuard(list(dirs.values()), cache_dirs=False, project=str(tmp_path))
    for v in dirs.values():
        assert g._under_roots(str(Path(v) / "file.json"))
    assert not g._under_roots(str(tmp_path / "agent_output.txt"))


# ------------------------------------------------------------------ d: the failure names package, framework, file
def _res() -> DiscoverResult:
    return DiscoverResult(entry="agent:main", timeout=120)


def test_h3_d_an_import_failure_names_package_framework_and_variables() -> None:
    f: dict[str, Any] = {
        "type": "com_error",
        "message": "(-2147024893, 'The system cannot find the path specified.', None, None)",
        "where": "agent.py:3",
        "package": "win32com",
        "package_where": "win32com/shell/shell.py:12",
        "packages": ["crewai", "appdirs", "win32com"],
        "importing": True,
        "filename": "",
    }
    text = explain("run-0", f, "", 0, _res(), None)
    assert "win32com (crewai) at win32com/shell/shell.py:12" in text
    assert "while it was being imported" in text and "APPDATA" in text and "env: in mocks.yml" in text


def test_h3_d_a_permission_error_names_the_file() -> None:
    f: dict[str, Any] = {
        "type": "PermissionError",
        "message": "[Errno 13] Permission denied",
        "where": "agent.py:9",
        "package": "",
        "packages": [],
        "importing": False,
        "filename": "C:\\Users\\Ana\\AppData\\Local\\crewai\\db.sqlite",
    }
    text = explain("run-0", f, "", 0, _res(), None)
    assert "C:\\Users\\Ana\\AppData\\Local\\crewai\\db.sqlite" in text and "agent.py:9" in text


def test_h3_d_describe_failure_records_package_and_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agenticreq.capture.runner import describe_failure

    monkeypatch.chdir(tmp_path)
    try:
        open(tmp_path / "missing" / "f.txt")  # noqa: SIM115
    except OSError as e:
        f = describe_failure(e, "agent:main", "run")
    assert f["filename"].endswith("f.txt") and f["importing"] is False and f["packages"] == []
