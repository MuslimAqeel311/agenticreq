"""A Claude Agent SDK project: built-in tools, allowed by name."""

from claude_agent_sdk import ClaudeAgentOptions, query

OPTIONS = ClaudeAgentOptions(
    allowed_tools=["Read", "Bash", "Write"],
    system_prompt="You maintain this repository.",
    model="claude-sonnet-4-5-20250929",
    max_turns=8,
)


async def run(task: str) -> None:
    async for _ in query(prompt=task, options=OPTIONS):
        pass
