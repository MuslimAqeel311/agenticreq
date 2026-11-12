"""H2: the platform lookups SDKs make for an OS header never start a program and are never listed as blocked; any
other program start is still blocked. Path logic for Windows, macOS and Linux is tested with each OS's own spelling."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections import namedtuple
from typing import Any

import pytest

from agenticreq.capture.guard import ProcessGuard, freeze_platform, is_platform_probe


@pytest.mark.parametrize(
    "event,args",
    [
        # Windows: platform._syscmd_ver with shell=True arrives as one string
        ("subprocess.Popen", ("C:\\Windows\\system32\\cmd.exe", 'C:\\Windows\\system32\\cmd.exe /c "ver"', None, None)),
        ("subprocess.Popen", (None, "cmd /c ver", None, None)),
        ("subprocess.Popen", (None, "ver", None, None)),
        ("subprocess.Popen", (None, "command /c ver", None, None)),
        # macOS
        ("subprocess.Popen", ("/usr/bin/uname", ["uname", "-p"], None, None)),
        ("subprocess.Popen", (None, ["sw_vers", "-productVersion"], None, None)),
        ("subprocess.Popen", (None, ["file", "-b", "/Library/Frameworks/Python 3.framework/python3"], None, None)),
        # Linux
        ("subprocess.Popen", (b"/bin/uname", [b"uname", b"-p"], None, None)),
        ("os.posix_spawn", ("/usr/bin/uname", ["uname", "-m"], {})),
        ("subprocess.Popen", (None, ["file", "-b", "/usr/bin/python3"], None, None)),
    ],
)
def test_h2_platform_probes_are_recognised_on_every_os(event: str, args: tuple[Any, ...]) -> None:
    assert is_platform_probe(event, args)


@pytest.mark.parametrize(
    "event,args",
    [
        ("subprocess.Popen", (None, "cmd /c del C:\\x", None, None)),
        ("subprocess.Popen", (None, 'cmd /c "ver & del x"', None, None)),
        ("subprocess.Popen", (None, "cmd", None, None)),
        ("subprocess.Popen", (None, ["uname", "-a", "; rm x"], None, None)),
        ("subprocess.Popen", (None, ["file", "-b", "a", "b"], None, None)),
        ("subprocess.Popen", (None, ["ls", "-la"], None, None)),
        ("subprocess.Popen", (None, ["/bin/sh", "-c", "uname -p"], None, None)),
        ("os.system", ("uname -p",)),
        ("subprocess.Popen", (None, None, None, None)),
    ],
)
def test_h2_anything_else_is_not_a_probe(event: str, args: tuple[Any, ...]) -> None:
    assert not is_platform_probe(event, args)


def test_h2_guard_lets_probes_through_and_still_blocks_other_programs(tmp_path: Any) -> None:
    g = ProcessGuard([], project=str(tmp_path))
    g.install()
    try:
        g._hook("subprocess.Popen", (None, ["uname", "-p"], None, None))  # no exception
        with pytest.raises(PermissionError):
            g._hook("subprocess.Popen", (None, ["ls"], None, None))
    finally:
        g.uninstall()
    assert g.blocked == ["exec:ls"]


@pytest.fixture
def restore_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_syscmd_ver", "_syscmd_file"):
        if hasattr(platform, name):
            monkeypatch.setattr(platform, name, getattr(platform, name))
    proc = getattr(platform, "_Processor", None)
    if proc is not None:
        monkeypatch.setattr(proc, "from_subprocess", proc.__dict__["from_subprocess"])


def test_h2_windows_version_comes_from_the_interpreter(monkeypatch: pytest.MonkeyPatch, restore_platform: None) -> None:
    WinVer = namedtuple("WinVer", "major minor build")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "getwindowsversion", lambda: WinVer(10, 0, 22631), raising=False)
    monkeypatch.setattr(platform, "_syscmd_ver", lambda *a, **k: pytest.fail("cmd /c ver was started"), raising=False)
    freeze_platform()
    assert platform._syscmd_ver() == ("Microsoft Windows", "10", "10.0.22631")  # type: ignore[attr-defined]
    assert platform._syscmd_ver("Windows", "11", "") == ("Windows", "11", "10.0.22631")  # type: ignore[attr-defined]


@pytest.mark.parametrize("os_name", ["darwin", "linux"])
def test_h2_posix_lookups_start_no_program_after_freezing(
    monkeypatch: pytest.MonkeyPatch, restore_platform: None, os_name: str
) -> None:
    freeze_platform()
    monkeypatch.setattr(sys, "platform", os_name)

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("a program was started")

    monkeypatch.setattr(subprocess, "check_output", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    platform.architecture()
    proc = getattr(platform, "_Processor", None)
    if proc is not None:
        proc.from_subprocess()


def test_h2_blocked_list_has_no_duplicates_across_kept_runs() -> None:
    from agenticreq.capture.discover import DiscoverResult, _merge_kept

    res = DiscoverResult(entry="a:b", run_ids=["two"], egress_blocked=["exec:ls", "write:/x"])
    res.by_run = {"two": {"egress_blocked": ["exec:ls", "write:/x"], "llm_misses": []}}
    old = {"run_ids": ["one"], "by_run": {"one": {"egress_blocked": ["exec:ls"], "llm_misses": []}}}
    _merge_kept(res, old)
    assert res.egress_blocked == ["exec:ls", "write:/x"]


def test_h2_a_mock_run_that_asks_for_the_platform_runs_and_lists_no_probe(tmp_path: Any) -> None:
    from agenticreq.capture import discover as mock_run

    (tmp_path / "agent.py").write_text(
        "import platform, subprocess\n\ndef main():\n"
        "    header = platform.platform() + platform.uname().processor + platform.architecture()[0]\n"
        "    if hasattr(platform, 'win32_ver'):\n        platform.win32_ver()\n"
        "    try:\n        subprocess.run(['uname', '-p'], capture_output=True)\n"
        "    except OSError:\n        pass\n"
        "    try:\n        subprocess.run(['ls'], capture_output=True)\n"
        "    except OSError:\n        pass\n"
        "    try:\n        subprocess.run(['ls'], capture_output=True)\n"
        "    except OSError:\n        pass\n"
        "    assert header\n",
        encoding="utf-8",
    )
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    res = mock_run(f"{tmp_path / 'agent.py'}:main", mock_tools=str(tmp_path / "mocks.yml"))
    assert not res.run_errors, res.run_errors
    execs = [b for b in res.egress_blocked if b.startswith("exec:")]
    assert execs == ["exec:ls"], res.egress_blocked  # the probe is not listed; ls is, once
