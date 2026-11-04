"""Limits that are not step limits: a tool-call cap, a retry loop. A real step loop for contrast."""

import anthropic
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits


def tool_a(q: str) -> str:
    """Tool."""
    return q


agent = Agent("openai:gpt-4o-2024-08-06", tools=[tool_a])
LIMITS = UsageLimits(tool_calls_limit=7)
client = anthropic.Anthropic()


def ask(q: str) -> str:
    for attempt in range(4):
        try:
            r = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=9, messages=[])
            return str(r)
        except Exception:
            continue
    return ""


def run(q: str) -> None:
    agent.run_sync(q, usage_limits=LIMITS)
