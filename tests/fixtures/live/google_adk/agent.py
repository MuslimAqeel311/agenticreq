"""Google ADK agent with two tools."""
import asyncio
from pathlib import Path

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


async def go(task: str) -> str:
    agent = Agent(name="a", model="gemini-2.0-flash-001", instruction="Help.", tools=[lookup, save_note])
    runner = InMemoryRunner(agent=agent, app_name="app")
    s = await runner.session_service.create_session(app_name="app", user_id="u")
    msg = types.Content(role="user", parts=[types.Part(text=task)])
    out = ""
    async for ev in runner.run_async(user_id="u", session_id=s.id, new_message=msg, run_config=RunConfig(max_llm_calls=6)):
        out = str(ev)
    return out


def main(task: str = "go") -> str:
    return asyncio.run(go(task))
