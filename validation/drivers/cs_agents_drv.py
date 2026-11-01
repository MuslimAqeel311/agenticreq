import asyncio
from types import SimpleNamespace
from agents import Runner
from airline.agents import triage_agent
from airline.context import create_initial_context
ctx = SimpleNamespace(state=create_initial_context())
asyncio.run(Runner.run(triage_agent, "I want to change my seat on flight FLT-123", context=ctx, max_turns=20))
