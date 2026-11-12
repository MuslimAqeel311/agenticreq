"""Strands: a BeforeToolCallEvent hook interrupts save_record for a human."""
from pathlib import Path

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry


@tool
def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


class Approval(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.ask)

    def ask(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] == "save_record":
            approval = event.interrupt("approval", reason="save a record?")
            if approval != "y":
                event.cancel_tool = "not approved"


agent = Agent(model="us.anthropic.claude-sonnet-4-20250514-v1:0", tools=[save_record], hooks=[Approval()])
