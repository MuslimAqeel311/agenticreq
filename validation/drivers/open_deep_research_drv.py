import asyncio
from open_deep_research.deep_researcher import deep_researcher
cfg = {"configurable": {"allow_clarification": False, "max_researcher_iterations": 2, "max_react_tool_calls": 3,
                        "max_concurrent_research_units": 1}}
asyncio.run(deep_researcher.ainvoke({"messages": [{"role": "user", "content": "History of the Lahore Fort"}]}, cfg))
