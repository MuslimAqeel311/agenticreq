"""The Claude Agent SDK starts its own CLI process: nothing to intercept at the HTTP client layer."""

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, query

OPTIONS = ClaudeAgentOptions(allowed_tools=["Read"], system_prompt="s", model="claude-sonnet-4-5-20250929")


def main(task: str = "go") -> None:
    async def go() -> None:
        async for _ in query(prompt=task, options=OPTIONS):
            pass

    asyncio.run(go())
