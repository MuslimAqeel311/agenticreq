"""Fixes from a user's report: every result on a line is shown, a crashed mock run never passes, generated
arguments fit their parameter, and the version-pinning advice is one that works."""

from __future__ import annotations

from agenticreq.jsonschema_util import named_example
from agenticreq.models.finding import Report
from agenticreq.models.verdict import also, overall


def test_every_result_on_a_line_is_shown() -> None:
    assert also("First problem.", ["Second problem", "First problem."]) == "First problem. Also: Second problem."


def test_a_mock_run_that_captured_nothing_is_a_fail() -> None:
    r = Report(agent_id="a", command="check")
    r.metadata["mock_run"] = {"captured": 0, "total": 0, "stale": False}
    assert overall(r) == "FAIL"


def test_generated_arguments_fit_the_parameter() -> None:
    assert named_example("url", {"type": "string"}) == "https://example.com"
    assert named_example("order_id", {"type": "string"}) == "1"
    assert named_example("steps", {"type": "array", "items": {"type": "string"}}) == ["example"]
    assert named_example("topic", {"type": "string"}) == "example topic"
