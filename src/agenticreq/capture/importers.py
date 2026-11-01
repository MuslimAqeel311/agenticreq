"""Foreign capture formats → CapturedCall.

Any recorder that kept the raw LLM request body can be imported: the body is re-parsed with the
same wire parser ``discover`` uses, so imported and native captures carry identical evidence.

Supported:

* ``spike``  — the boundary-capture spike (``{"captures": [{"via", "raw_request", "reply"}], "versions"}``)
* ``raw``    — a list of ``{"url", "request"|"body", "response"?}`` exchanges (any HTTP recorder)
"""

from __future__ import annotations

from typing import Any

from agenticreq.capture.wire import parse_request, parse_response_tool_calls
from agenticreq.models.observed import CapturedCall, ToolCall


def detect_format(data: Any) -> str | None:
    if (
        isinstance(data, dict)
        and isinstance(data.get("captures"), list)
        and all(isinstance(c, dict) and "raw_request" in c and "via" in c for c in data["captures"])
    ):
        return "spike"
    items = data if isinstance(data, list) else None
    if items and all(isinstance(c, dict) and "url" in c and ("request" in c or "body" in c) for c in items):
        return "raw"
    return None


def _url_from_via(via: str) -> tuple[str, str]:
    """'httpx2: POST api.openai.com/v1/responses' -> ('https://api.openai.com/v1/responses', 'httpx2')."""
    lib, _, rest = via.partition(":")
    target = rest.strip().split(None, 1)[-1] if rest.strip() else ""
    if target and "://" not in target:
        target = "https://" + target
    return target, (lib.strip() or "httpx")


def from_spike(data: dict[str, Any]) -> tuple[list[CapturedCall], dict[str, str]]:
    run_id = f"{data.get('agent_id', 'spike')}:{data.get('strategy', '')}".rstrip(":")
    calls: list[CapturedCall] = []
    for i, c in enumerate(data["captures"]):
        url, lib = _url_from_via(str(c.get("via", "")))
        body = c.get("raw_request")
        if not isinstance(body, dict):
            continue
        call = parse_request(i, url, body, run_id=run_id, client_lib=lib)
        reply = c.get("reply") or {}
        if isinstance(reply, dict) and reply.get("kind") == "tool" and reply.get("name"):
            call.tool_calls = [ToolCall(name=str(reply["name"]), arguments=reply.get("args") or {})]
        call.synthetic = True  # the spike answered with a canned model: offered tools are evidence, calls are not
        call.raw_request = body
        calls.append(call)
    return calls, _dist_versions(data.get("versions") or {})


def from_raw(items: list[dict[str, Any]]) -> list[CapturedCall]:
    calls: list[CapturedCall] = []
    for i, c in enumerate(items):
        body = c.get("request") or c.get("body")
        if not isinstance(body, dict):
            continue
        call = parse_request(
            i, str(c["url"]), body, run_id=str(c.get("run_id", "run-0")), client_lib=str(c.get("client_lib", "httpx"))
        )
        if isinstance(c.get("response"), dict):
            call.tool_calls = parse_response_tool_calls(call.provider, c["response"])
        call.synthetic = bool(c.get("synthetic", False))
        call.raw_request = body
        calls.append(call)
    return calls


def _dist_versions(versions: dict[str, Any]) -> dict[str, str]:
    """Recorders often key versions by import name (``agents``); observe expects distribution names."""
    from agenticreq.ast.framework import FRAMEWORKS

    out = {str(k): str(v) for k, v in versions.items()}
    for mod, (_, dist) in FRAMEWORKS.items():
        if mod in out and dist not in out:
            out[dist] = out[mod]
    return out
