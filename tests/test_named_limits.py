"""A limit the code names (MAX_NO_PROGRESS_STEPS = 5, MAX_CHARS = 6000, APPROVAL_TIMEOUT_S = 4 * 3600) is read, so it
is never reported as "the code sets no limit"; without such a constant it still is."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

from agenticreq.ast.constants import find
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

SPEC = """
agent_id: a
risk_level: medium
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: [{tool: save_file, approver: lead, timeout: 4h, on_timeout: safe_stop}]
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
progress_guard: {max_no_progress_steps: 5}
output_size_limits: {max_chars: 20000}
"""
CODE = """
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    {consts}

    @tool
    def save_file(path: str, text: str) -> str:
        '''Save a file.'''
        with open(path, "w") as fh:
            fh.write(text[:100])
        return "ok"

    agent = create_react_agent("openai:gpt-4o-2024-08-06", [save_file])
"""
CONSTS = "MAX_NO_PROGRESS_STEPS = 5\n    MAX_CHARS = 6000\n    APPROVAL_TIMEOUT_S = 4 * 3600"
PARAMS = ("approval_required[].timeout", "progress_guard.max_no_progress_steps", "output_size_limits.max_chars")


def _lines(tmp: Path, consts: str) -> dict[str, str]:
    (tmp / "agent.py").write_text(textwrap.dedent(CODE.format(consts=consts)), encoding="utf-8")
    report = run(mode="check", loaded=load_spec_text(SPEC), observed=build_observed(tmp), today=date(2026, 10, 30))
    report.metadata["mock_run"] = {"captured": 1, "total": 1, "stale": False}
    return {ln.param: ln.status for ln in results(report) if ln.param in PARAMS}


def test_named_limits_are_read_from_code(tmp_path: Path) -> None:
    assert _lines(tmp_path, CONSTS) == {p: "PASS" for p in PARAMS}


def test_without_them_the_code_sets_no_limit(tmp_path: Path) -> None:
    assert set(_lines(tmp_path, "").values()) == {"GAP"}


def test_units_and_ambiguity(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("APPROVAL_TIMEOUT_H = 4\nX_MAX_CHARS = 10\nY_MAX_CHARS = 20\n", encoding="utf-8")
    assert find(tmp_path, "approval", "timeout", seconds=True) == (14400.0, "APPROVAL_TIMEOUT_H", "a.py:1")
    assert find(tmp_path, "max", "chars") is None  # two different values: not read as one limit


def test_a_helper_that_keeps_head_and_tail_is_a_cap(tmp_path: Path) -> None:
    import ast as _ast

    from agenticreq.ast.limits import find_limits

    src = (
        "def clip_block(text: str, limit: int = 4000) -> str:\n    if len(text) <= limit:\n        return text\n"
        "    half = limit // 2\n    return text[:half] + '...' + text[-half:]\n"
    )
    fn = _ast.parse(src).body[0]
    assert find_limits(None, None, fn)  # type: ignore[arg-type]


def test_a_setting_read_from_env_with_a_default_is_read(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text(
        "import os\nfrom dataclasses import dataclass\n\ndef _num(k, d, t):\n    return t(os.getenv(k, d))\n\n"
        "@dataclass\nclass Settings:\n    max_messages: int\n\nsettings = Settings(max_messages=_num('MAX_MESSAGES', 120, int))\n",
        encoding="utf-8",
    )
    from agenticreq.ast.constants import setting_default

    assert setting_default(str(tmp_path), "max_messages") == (120.0, "default in code of env MAX_MESSAGES")
