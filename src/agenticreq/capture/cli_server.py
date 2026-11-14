"""A local model endpoint for agents that call the model from another process (the Claude Agent SDK runs the
`claude` CLI). The CLI is pointed at it with ANTHROPIC_BASE_URL; every request it makes is recorded and answered by
the same canned model as the in-process hooks. Standard library only: this runs in the agent's environment.

Honest limit: the CLI is a separate program, so the Python network guard does not reach it. Its telemetry and
updates are turned off through its own environment variables, and its only model endpoint is this one.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: what the CLI is told so that it reaches nothing but this endpoint
CLI_ENV = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_BUG_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    # offer the project's MCP tools directly, not behind a tool-search step the canned model does not take
    "ENABLE_TOOL_SEARCH": "false",
}


def start(session: Any) -> tuple[ThreadingHTTPServer, str]:
    """Serve POST /v1/messages on 127.0.0.1 (a free port); returns the server and its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:  # quiet: the run's own output stays readable
            return None

        def _reply(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            url = f"https://api.anthropic.com{self.path}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            answer = None
            if (
                isinstance(body, dict)
                and "messages" in body
                and "/messages" in self.path
                and "count_tokens" not in self.path
            ):
                answer = session.answer(url, body, "claude-cli")
            if answer is None:  # token counting and anything else: a minimal, valid answer
                payload = {"input_tokens": 1} if "count_tokens" in self.path else {}
                self._reply(200, {"content-type": "application/json"}, json.dumps(payload).encode())
                return
            self._reply(answer[0], answer[1], answer[2])

        def do_GET(self) -> None:  # noqa: N802
            self._reply(200, {"content-type": "application/json"}, b"{}")

        do_HEAD = do_GET

    class _Server(ThreadingHTTPServer):
        def server_bind(self) -> None:
            # HTTPServer.server_bind looks up socket.getfqdn(host): a reverse DNS query that can take many seconds
            # (GitHub's macOS runners). The name is never used here, so the address is bound with no lookup.
            import socketserver

            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    server = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
