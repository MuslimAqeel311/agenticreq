"""LlamaIndex workflow: the save step waits for a HumanResponseEvent before writing."""
from pathlib import Path

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context, HumanResponseEvent, InputRequiredEvent
from llama_index.llms.openai import OpenAI


async def save_record(ctx: Context, text: str) -> str:
    """Save a record to disk."""
    ctx.write_event_to_stream(InputRequiredEvent(prefix=f"Save {text}? yes/no"))
    answer = await ctx.wait_for_event(HumanResponseEvent)
    if answer.response != "yes":
        return "not approved"
    Path("record.txt").write_text(text)
    return "saved"


agent = FunctionAgent(tools=[FunctionTool.from_defaults(async_fn=save_record)], llm=OpenAI(model="gpt-4o-2024-08-06"))
