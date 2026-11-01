"""Minimal stdio MCP server for tests: answers initialize + paginated tools/list."""
import json
import sys

PAGES = [
    [{"name": "get_inventory", "description": "Look up stock.",
      "inputSchema": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}],
    [{"name": "delete_records", "description": "Delete records.",
      "inputSchema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}}],
]
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        res = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "0"}}
    elif msg["method"] == "tools/list":
        page = int((msg.get("params") or {}).get("cursor") or 0)
        res = {"tools": PAGES[page]}
        if page + 1 < len(PAGES):
            res["nextCursor"] = str(page + 1)
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}) + "\n")
    sys.stdout.flush()
