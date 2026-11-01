import asyncio
from examples.financial_research_agent.manager import FinancialResearchManager
asyncio.run(FinancialResearchManager().run("Snowflake's most recent quarter"))
