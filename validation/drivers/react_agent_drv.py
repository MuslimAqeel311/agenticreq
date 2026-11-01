import asyncio
from react_agent.context import Context
from react_agent.graph import graph
asyncio.run(graph.ainvoke({"messages": [("user", "Who maintains LangGraph?")]}, context=Context()))
