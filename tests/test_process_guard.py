"""The discover process / file guard: other programs and out-of-scratch writes are refused on every OS."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agenticreq.capture.guard import ProcessGuard


@pytest.fixture
def guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))
    g = ProcessGuard(cache_dirs=False)
    g.install()
    try:
        yield g
    finally:
        g.uninstall()


def test_starting_a_program_is_blocked(guard: ProcessGuard) -> None:
    with pytest.raises(PermissionError, match="agenticreq mock run"):
        subprocess.run([sys.executable, "-c", "pass"])
    with pytest.raises(PermissionError):
        os.system("echo hi")
    assert any(b.startswith("exec:") for b in guard.blocked)


def test_write_outside_scratch_is_blocked_and_inside_is_allowed(guard: ProcessGuard, tmp_path: Path) -> None:
    outside = tmp_path / "report.md"
    with pytest.raises(PermissionError, match="allow_write"):
        outside.write_text("x", encoding="utf-8")
    assert not outside.exists()
    assert guard.blocked == [f"write:{outside}"]
    inside = tmp_path / "scratch" / "ok.txt"
    inside.write_text("fine", encoding="utf-8")
    assert inside.read_text(encoding="utf-8") == "fine"


def test_reading_anywhere_is_still_allowed(guard: ProcessGuard, tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    guard.active = False
    f.write_text("seed", encoding="utf-8")
    guard.active = True
    assert f.read_text(encoding="utf-8") == "seed"


def test_delete_and_rename_outside_scratch_are_blocked(guard: ProcessGuard, tmp_path: Path) -> None:
    guard.active = False
    victim = tmp_path / "keep.txt"
    victim.write_text("data", encoding="utf-8")
    guard.active = True
    with pytest.raises(PermissionError):
        victim.unlink()
    with pytest.raises(PermissionError):
        victim.rename(tmp_path / "moved.txt")
    guard.active = False
    assert victim.exists()


def test_allow_write_folder_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))
    g = ProcessGuard([str(out)], cache_dirs=False)
    g.install()
    try:
        (out / "r.txt").write_text("ok", encoding="utf-8")
    finally:
        g.uninstall()
    assert (out / "r.txt").read_text(encoding="utf-8") == "ok"


def test_uninstall_stops_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "scratch"))
    g = ProcessGuard(cache_dirs=False)
    g.install()
    g.uninstall()
    (tmp_path / "free.txt").write_text("x", encoding="utf-8")
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
