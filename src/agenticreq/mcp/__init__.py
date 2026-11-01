"""MCP `tools/list` client (stdio JSON-RPC, no SDK dependency) and file loader.

Runs the server command the user explicitly passes (--mcp "python server.py").
MCP was NOT part of the feasibility spike — see SCOPE.md.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agenticreq.jsonschema_util import tool_from_json_schema
from agenticreq.models.observed import ToolSchema

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


def tools_from_list_result(result: dict[str, Any]) -> list[ToolSchema]:
    return [
        tool_from_json_schema(t["name"], t.get("description"), t.get("inputSchema"), "mcp", "mcp:tools/list")
        for t in result.get("tools", [])
        if isinstance(t, dict) and t.get("name")
    ]


def load_tools_file(path: str | Path) -> list[ToolSchema]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    if isinstance(data, list):
        data = {"tools": data}
    return tools_from_list_result(data)


class _Stdio:
    def __init__(self, cmd: list[str], env: dict[str, str] | None, timeout: float) -> None:
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        self.timeout = timeout
        self._id = 0
        self._lines: list[str] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._cv:
                self._lines.append(line)
                self._cv.notify_all()

    def send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        rid = self._id
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout
        with self._cv:
            while True:
                for i, line in enumerate(self._lines):
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and msg.get("id") == rid:
                        del self._lines[i]
                        if "error" in msg:
                            raise MCPError(f"{method}: {msg['error']}")
                        return msg.get("result") or {}
                left = deadline - time.monotonic()
                if left <= 0 or self.proc.poll() is not None:
                    raise MCPError(f"{method}: no response (server exited={self.proc.poll() is not None})")
                self._cv.wait(timeout=left)

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def list_tools_stdio(
    command: str | list[str], env: dict[str, str] | None = None, timeout: float = 20.0
) -> list[ToolSchema]:
    cmd = shlex.split(command) if isinstance(command, str) else command
    s = _Stdio(cmd, env, timeout)
    try:
        s.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agenticreq", "version": "0.1"},
            },
        )
        s.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools: list[ToolSchema] = []
        cursor: str | None = None
        for _ in range(100):
            res = s.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += tools_from_list_result(res)
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools
    finally:
        s.close()
