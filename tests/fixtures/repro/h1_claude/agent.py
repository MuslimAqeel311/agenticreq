"""Claude Agent SDK: a can_use_tool callback asks a person before save_record."""
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool


@tool("save_record", "Save a record to disk", {"text": str})
async def save_record(args):
    Path("record.txt").write_text(args["text"])
    return {"content": [{"type": "text", "text": "saved"}]}


async def ask(tool_name, tool_input, context):
    if input(f"allow {tool_name}? ") != "y":
        return {"behavior": "deny", "message": "not approved"}
    return {"behavior": "allow", "updated_input": tool_input}


server = create_sdk_mcp_server(name="records", tools=[save_record])
options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929", mcp_servers={"records": server},
                             allowed_tools=["mcp__records__save_record"], can_use_tool=ask)
client = ClaudeSDKClient(options=options)
