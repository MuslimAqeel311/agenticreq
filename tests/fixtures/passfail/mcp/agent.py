"""Support agent whose tools come from an MCP server at run time; no tool list is supplied."""
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

orders = MCPServerStdio(params={"command": "orders-mcp", "args": []})

support = Agent(name="support", instructions="Answer order questions.", mcp_servers=[orders],
                model="gpt-4o-2024-08-06")


def main(question: str) -> str:
    return Runner.run_sync(support, question, max_turns=10).final_output
