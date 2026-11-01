import os
import sys
from pathlib import Path

# Rich reads these when a Console is built (at import), so set them first: CI runners force colour and use
# narrow terminals, which put ANSI codes and line wraps inside the phrases the CLI tests look for.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "200"
os.environ.pop("FORCE_COLOR", None)
# commands run inside fixture folders must not leave .agenticreq/state.json there (tests/test_progress.py clears it)
os.environ["AGENTICREQ_NO_STATE"] = "1"

import pytest  # noqa: E402

from agenticreq.models.observed import CapturedCall  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(Path(__file__).parent))


def chat_call(
    index: int,
    tools: list[dict],
    system: str = "sys",
    run_id: str = "run-0",
    tool_calls: list[tuple[str, dict]] | None = None,
    synthetic: bool = False,
    tool_results: list[tuple[str, str]] | None = None,
) -> CapturedCall:
    """Build a CapturedCall from an OpenAI chat-completions request body (as the hook would)."""
    from agenticreq.capture.wire import parse_request
    from agenticreq.models.observed import ToolCall

    msgs: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": "go"}]
    for cid, content in tool_results or []:
        msgs.append({"role": "tool", "tool_call_id": cid, "content": content})
    body = {"model": "gpt-4o", "messages": msgs, "tools": [{"type": "function", "function": t} for t in tools]}
    c = parse_request(index, "https://api.openai.com/v1/chat/completions", body, run_id=run_id)
    c.tool_calls = [ToolCall(name=n, arguments=a, id=f"c{index}{n}") for n, a in (tool_calls or [])]
    c.synthetic = synthetic
    return c


@pytest.fixture
def fix() -> Path:
    return FIX
