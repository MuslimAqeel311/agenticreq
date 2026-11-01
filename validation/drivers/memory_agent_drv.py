import asyncio
from langgraph.store.memory import InMemoryStore
from memory_agent.context import Context
from memory_agent.graph import builder
g = builder.compile(store=InMemoryStore())  # in-memory store: no persistence
asyncio.run(g.ainvoke({"messages": [("user", "I like hiking, remember that.")]}, context=Context()))
