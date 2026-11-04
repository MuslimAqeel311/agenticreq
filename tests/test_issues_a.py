"""Priority A: a false pass, a hang or a broken mock run, each from a project that showed it."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agenticreq.capture.guard import RETRY_CAP, ProcessGuard, WriteLoopStopped
from agenticreq.capture.responder import MockConfig
from agenticreq.capture.runner import _async_stub, _stub, _stub_in_place, apply_patches
from agenticreq.cli import app
from agenticreq.models.verdict import Verdict, settle_after_run

runner = CliRunner()
ISSUES = Path(__file__).parent / "fixtures" / "issues"


def _project(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / name
    shutil.copytree(ISSUES / name, d)
    monkeypatch.chdir(d)
    return d


def _mock_run(*extra: str) -> Any:
    return runner.invoke(app, ["mock-run", "--python", sys.executable, "--quiet", *extra])


def _lines(project: Path) -> dict[str, dict[str, Any]]:
    r = runner.invoke(app, ["check", "-f", "json", "--fail-on", "never", "--quiet"])
    assert r.exit_code == 0, r.output
    return {ln["param"]: ln for ln in json.loads(r.stdout)["results"]}


# =========================================================================== A1
def test_a1_a_missing_gate_stays_a_gap_after_a_mock_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a1_gate", tmp_path, monkeypatch)
    assert _mock_run().exit_code == 0
    line = _lines(d)["approval_required"]
    assert line["status"] == "GAP" and "no human-approval gate" in line["reason"]


def test_a1_the_mock_replaced_text_is_only_for_a_patched_tool() -> None:
    reason = "Code pattern only: save_report (approval_call at tools.py:9). Add a mock, then run: agenticreq mock-run"
    patched = settle_after_run(Verdict("CHECK", reason), ("AS-205",), "p", frozenset({"save_report"}))
    assert patched.status == "SIGN-OFF" and patched.reason.startswith("The mock replaced save_report")
    other = settle_after_run(Verdict("CHECK", reason), ("AS-205",), "p", frozenset({"another_tool"}))
    assert other.status == "SIGN-OFF" and "mock replaced" not in other.reason and "has a gate" in other.reason


# =========================================================================== A2
def test_a2_a_retry_loop_on_a_blocked_write_ends_with_its_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project("a2_loop", tmp_path, monkeypatch)
    r = _mock_run("--timeout", "60")
    text = " ".join(r.output.split())
    assert r.exit_code == 1 and "blocked write" in text and "allow_write:" in text and "state-" in text
    assert "limit stopped" not in text  # it never reached the timeout


def test_a2_the_guard_stops_answering_permission_error_after_the_cap(tmp_path: Path) -> None:
    g = ProcessGuard([], cache_dirs=False, project=str(tmp_path))
    g.active = True
    tries = 0
    with pytest.raises(WriteLoopStopped) as e:
        for i in range(10_000):  # tempfile on Windows: PermissionError means "try the next name"
            tries += 1
            try:
                g._hook("open", (str(tmp_path / f"t{i}"), "w", 0))
            except PermissionError:
                continue
    assert tries == RETRY_CAP + 1 and str(tmp_path) in str(e.value) and "allow_write" in str(e.value)
    assert not isinstance(e.value, PermissionError)


def test_a2_a_timeout_names_the_last_blocked_file() -> None:
    from agenticreq.capture.discover import DiscoverResult, explain

    res = DiscoverResult(entry="a.py:main", timeout=30)
    f = {"type": "Timeout", "message": "x", "step": {"calls": 0, "last_tools": [], "blocked": ["/home/u/.tool/c.json"]}}
    text = explain("run-0", f, "", 0, res, None)
    assert "/home/u/.tool/c.json" in text and "allow_write:" in text


def test_a2_config_folders_of_the_run_are_scratch_folders(tmp_path: Path) -> None:
    from agenticreq.capture.discover import scratch_config_dirs

    dirs = scratch_config_dirs(str(tmp_path))
    names = {"HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "APPDATA", "LOCALAPPDATA"}
    assert names <= set(dirs) and all(dirs[k].startswith(str(tmp_path)) for k in names)


# =========================================================================== A3
def test_a3_generated_stubs_keep_name_docstring_and_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project("a3_stub", tmp_path, monkeypatch)
    r = _mock_run()
    assert r.exit_code == 0, r.output


def test_a3_a_stub_has_the_face_of_the_real_function() -> None:
    def real(q: str, n: int = 2) -> str:
        """Find things."""
        return q

    import inspect

    for make in (_stub, _async_stub):
        s = make("mock", real)
        assert s.__name__ == "real" and s.__doc__ == "Find things."
        assert list(inspect.signature(s).parameters) == ["q", "n"]


def test_a3_a_tool_object_keeps_its_type() -> None:
    class FunctionTool:
        async def on_invoke_tool(self, ctx: Any, raw: str) -> str:
            raise AssertionError("the real tool ran")

    t = FunctionTool()
    assert _stub_in_place(t, "mock", (("on_invoke_tool",),), ("on_invoke_tool",))
    import asyncio

    assert asyncio.run(t.on_invoke_tool(None, "{}")) == "mock" and isinstance(t, FunctionTool)


def test_a3_only_the_innermost_function_is_stubbed() -> None:
    class Wrapped:
        def func(self, q: str) -> str:
            raise AssertionError("the real tool ran")

        def run(self, q: str) -> list[str]:
            return [self.func(q)]  # the framework's own wrapper, kept

    t = Wrapped()
    assert _stub_in_place(t, "mock", (("func",), ("run",)), ())
    assert t.run("x") == ["mock"]


def test_a3_a_tool_class_stays_a_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "toolmod.py").write_text(
        "class Save:\n    name = 'save'\n    def _run(self, x):\n        raise RuntimeError('real')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from agenticreq.capture.responder import ToolMock

    apply_patches(MockConfig(tools={"save": ToolMock(patch="toolmod:Save", returns="ok")}), (("_run",),), ())
    import toolmod  # type: ignore[import-not-found]

    assert isinstance(toolmod.Save, type) and toolmod.Save()._run("x") == "ok"
    del sys.modules["toolmod"]


# =========================================================================== A4
def test_a4_each_run_takes_its_own_path_and_the_output_says_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project("a4_order", tmp_path, monkeypatch)
    r = _mock_run()
    text = " ".join(r.output.split())
    assert r.exit_code == 0 and "followed every handoff" not in text
    assert "asked for: transfer_to_billing" in text and "asked for: transfer_to_support" in text


def test_a4_without_a_run_order_the_output_says_every_handoff_was_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project("a4_order", tmp_path, monkeypatch)
    text = (d / "mocks.yml").read_text(encoding="utf-8")
    (d / "mocks.yml").write_text(text[: text.index("runs:")], encoding="utf-8")
    r = _mock_run()
    assert "followed every handoff" in " ".join(r.output.split())


def test_a4_the_top_level_order_still_asks_for_the_other_tools() -> None:
    from conftest import chat_call

    from agenticreq.capture.responder import CannedResponder

    tools = [{"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}} for n in ("a", "b")]
    call = chat_call(0, tools)
    top = CannedResponder(MockConfig(order=["b"]))
    assert [t.name for t in top.plan(call, {"tools": []})] == ["b"]
    assert [t.name for t in top.plan(call, {"tools": []})] == ["a"]
    only = CannedResponder(MockConfig(order=["b"], only_order=True))
    assert [t.name for t in only.plan(call, {"tools": []})] == ["b"]
    assert only.plan(call, {"tools": []}) == []


# =========================================================================== A5
def test_a5_a_blocked_write_does_not_end_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project("a5_write", tmp_path, monkeypatch)
    r = _mock_run()
    text = " ".join(r.output.split())
    assert r.exit_code == 0 and "PermissionError" not in text
    assert "blocked write to out/report.txt; to allow it, add it under allow_write: in mocks.yml" in text
    assert not (d / "out").exists()  # the working tree is unchanged


def test_a5_the_scratch_copy_reads_back_and_the_real_path_is_untouched(tmp_path: Path) -> None:
    g = ProcessGuard([], cache_dirs=False, project=str(tmp_path))
    g.active = True
    g.divert_writes(str(tmp_path.parent / f"{tmp_path.name}-scratch"))
    try:
        target = tmp_path / "x.txt"
        target.write_text("hello", encoding="utf-8")
        assert open(target, encoding="utf-8").read() == "hello"
        assert not os.path.exists(target)
    finally:
        g.uninstall()
    assert g.blocked == [f"write:{target}"]
