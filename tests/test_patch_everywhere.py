"""A patch: in mocks.yml stubs the tool however the project holds it: a list built at import time, a wrapper
object made by the framework, a decorator (functools.wraps), an async function. The signature stays as written."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

from agenticreq.capture.responder import MockConfig, ToolMock
from agenticreq.capture.runner import apply_patches

MOD = """
import functools

def safe_tool(fn):
    @functools.wraps(fn)
    async def wrapper(*a, **k):
        return await fn(*a, **k)
    return wrapper

@safe_tool
async def pypi_info(package: str) -> str:
    raise RuntimeError("real network call")

TOOLS = [pypi_info]

class Holder:  # a framework's tool object, built before the patch
    def __init__(self, f):
        self._func = f

HELD = Holder(pypi_info)
"""


def test_a_patched_tool_never_runs_its_real_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "proj_tools.py").write_text(MOD, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    import proj_tools  # noqa: F401  - imported first, as the project would

    sig = str(inspect.signature(sys.modules["proj_tools"].pypi_info))
    cfg = MockConfig(tools={"pypi_info": ToolMock(patch="proj_tools:pypi_info", returns="mock result")})
    apply_patches(cfg)
    m = sys.modules["proj_tools"]
    for f in (m.TOOLS[0], m.HELD._func, m.pypi_info):
        assert asyncio.run(f("requests")) == "mock result"
    assert str(inspect.signature(m.HELD._func)) == sig
