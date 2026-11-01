import asyncio
from enrichment_agent.graph import graph
schema = {"type": "object", "properties": {"companies": {"type": "array", "items": {"type": "string"}}},
          "required": ["companies"]}
asyncio.run(graph.ainvoke({"topic": "Top bakeries in Paris", "extraction_schema": schema}))
