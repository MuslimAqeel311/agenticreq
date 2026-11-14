"""M12: errors and messages: the full traceback is kept, a blocked program and a timeout each have their own
message, a shortened list says how many were hidden, and a YAML value split at a comma says to quote it."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from agenticreq.capture import runner as runner_mod
from agenticreq.capture.discover import DiscoverResult, explain
from agenticreq.capture.guard import ProcessGuard
from agenticreq.cli import _more
from agenticreq.loader import load_spec


def test_m12_a_the_runner_keeps_every_frame() -> None:
    src = inspect.getsource(runner_mod.main)
    assert "format_exc()" in src and "limit=8" not in src


def test_m12_b_a_blocked_program_has_its_own_message(tmp_path: Path) -> None:
    g = ProcessGuard([], cache_dirs=False, project=str(tmp_path))
    g.active = True
    with pytest.raises(PermissionError) as e:
        g._hook("subprocess.Popen", (None, ["ls"], None, None))
    msg = str(e.value)
    assert "starting a program is blocked (ls" in msg and "allow_write" not in msg
    f: dict[str, Any] = {"type": "PermissionError", "message": msg, "where": "agent.py:4"}
    text = explain("run-0", f, "", 0, DiscoverResult(entry="agent:main"), None)
    assert "when the agent started a program" in text and "blocked write" not in text


def test_m12_b_a_timeout_is_never_reported_as_a_crash() -> None:
    f: dict[str, Any] = {"type": "Timeout", "message": "did not finish within 5s", "step": {"calls": 0}}
    text = explain("run-0", f, "", 0, DiscoverResult(entry="agent:main", timeout=5), None)
    assert "before the 5s limit stopped run-0" in text and "crash" not in text


def test_m12_c_a_shortened_list_says_how_many_were_hidden() -> None:
    assert _more(8, 5) == " (+3 more)" and _more(5, 5) == ""


def test_m12_g_a_value_split_at_a_comma_says_to_quote_it(tmp_path: Path) -> None:
    spec = (Path(__file__).parent / "fixtures" / "repro" / "h1_input" / "spec.yml").read_text(encoding="utf-8")
    spec += "confirmed:\n  - {clause: kill_switch, answer: yes, by: a@b.com, reason: checked, see agent.py:42}\n"
    (tmp_path / "spec.yml").write_text(spec, encoding="utf-8")
    issues = load_spec(tmp_path / "spec.yml").issues
    msg = " ".join(i.message for i in issues)
    assert "must be quoted" in msg and "line" in msg
