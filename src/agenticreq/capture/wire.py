"""Parse LLM HTTP request/response bodies into CapturedCall.

Supported wire formats (see SCOPE.md for what is not):
* OpenAI Chat Completions   POST …/chat/completions
* OpenAI Responses          POST …/responses
* Anthropic Messages        POST …/messages   (incl. beta.messages)
* Gemini generateContent    POST …/models/<m>:generateContent, :streamGenerateContent

Which format a request is comes from data/providers.yml. A mock run gets the table from its parent through
``configure``, so the agent's environment needs no YAML reader.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agenticreq.jsonschema_util import tool_from_json_schema
from agenticreq.models.observed import CapturedCall, Provider, ToolCall, ToolResult, ToolSchema


_table: dict[str, Any] | None = None


def configure(table: dict[str, Any]) -> None:
    """The contents of data/providers.yml, handed over by the parent of a mock run."""
    global _table
    _table = table


def provider_table() -> dict[str, Any]:
    global _table
    if _table is None:
        from agenticreq.data import load_yaml

        _table = load_yaml("providers.yml")
    return _table


def detect_provider(url: str, body: dict[str, Any] | None) -> Provider | None:
    """None = not an LLM call at all."""
    path = url.split("?", 1)[0].rstrip("/")
    for f in provider_table()["formats"]:
        hit = path.endswith(f["path_end"]) if "path_end" in f else bool(re.search(f["path_regex"], path))
        if hit and (not f.get("body_has") or (isinstance(body, dict) and f["body_has"] in body)):
            return f["name"]  # type: ignore[no-any-return]
    if isinstance(body, dict) and "model" in body and ("messages" in body or "input" in body):
        return "unknown"
    return None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(x for x in parts if x)
    return ""


def _loads(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {"_": v}
        except json.JSONDecodeError:
            return {"_raw": s}
    return {}


# ---------------------------------------------------------------- requests
def _parse_openai_chat(body: dict[str, Any], c: CapturedCall) -> None:
    sys_parts = []
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            sys_parts.append(_text_of(m.get("content")))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                c.prior_tool_calls.append(
                    ToolCall(name=fn.get("name", "?"), arguments=_loads(fn.get("arguments")), id=tc.get("id"))
                )
        elif role == "tool":
            c.tool_results.append(
                ToolResult(call_id=m.get("tool_call_id"), name=m.get("name"), content=_text_of(m.get("content")))
            )
    c.system_prompt = "\n".join(sys_parts) or None
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            c.tools.append(
                tool_from_json_schema(
                    fn["name"], fn.get("description"), fn.get("parameters"), "wire", "wire:openai_chat"
                )
            )
    c.message_count = len(body.get("messages") or [])


def _parse_openai_responses(body: dict[str, Any], c: CapturedCall) -> None:
    sys_parts = [body["instructions"]] if isinstance(body.get("instructions"), str) else []
    items = body.get("input")
    items = [{"role": "user", "content": items}] if isinstance(items, str) else (items or [])
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = it.get("type")
        if typ == "function_call":
            c.prior_tool_calls.append(
                ToolCall(name=it.get("name", "?"), arguments=_loads(it.get("arguments")), id=it.get("call_id"))
            )
        elif typ == "function_call_output":
            c.tool_results.append(ToolResult(call_id=it.get("call_id"), content=_text_of(it.get("output"))))
        elif it.get("role") in ("system", "developer"):
            sys_parts.append(_text_of(it.get("content")))
    c.system_prompt = "\n".join(p for p in sys_parts if p) or None
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and t.get("name"):
            c.tools.append(
                tool_from_json_schema(
                    t["name"], t.get("description"), t.get("parameters"), "wire", "wire:openai_responses"
                )
            )
        elif t.get("type"):
            # provider-hosted tools (web_search, file_search...) — outside SCOPE, recorded by type
            c.tools.append(ToolSchema(name=f"hosted:{t['type']}", source="wire", kind="wire:hosted"))
    c.message_count = len(items)


def _parse_anthropic(body: dict[str, Any], c: CapturedCall) -> None:
    c.system_prompt = _text_of(body.get("system")) or None
    for m in body.get("messages") or []:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                c.prior_tool_calls.append(
                    ToolCall(name=b.get("name", "?"), arguments=b.get("input") or {}, id=b.get("id"))
                )
            elif b.get("type") == "tool_result":
                c.tool_results.append(ToolResult(call_id=b.get("tool_use_id"), content=_text_of(b.get("content"))))
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        if "input_schema" in t:
            c.tools.append(
                tool_from_json_schema(t["name"], t.get("description"), t.get("input_schema"), "wire", "wire:anthropic")
            )
        else:  # server tools (web_search_20250305 etc.)
            c.tools.append(ToolSchema(name=f"hosted:{t['name']}", source="wire", kind="wire:hosted"))
    c.message_count = len(body.get("messages") or [])


def _parse_gemini(url: str, body: dict[str, Any], c: CapturedCall) -> None:
    found = re.search(r"/models/([^/:]+):", url)
    c.model = found.group(1) if found else c.model
    system = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(system, dict):
        c.system_prompt = _text_of(system.get("parts")) or None
    for m in body.get("contents") or []:
        for p in (m.get("parts") if isinstance(m, dict) else None) or []:
            if not isinstance(p, dict):
                continue
            call = p.get("functionCall") or p.get("function_call")
            result = p.get("functionResponse") or p.get("function_response")
            if isinstance(call, dict):
                c.prior_tool_calls.append(
                    ToolCall(name=call.get("name", "?"), arguments=call.get("args") or {}, id=call.get("id"))
                )
            elif isinstance(result, dict):
                c.tool_results.append(
                    ToolResult(
                        call_id=result.get("id"), name=result.get("name"), content=json.dumps(result.get("response"))
                    )
                )
    for t in body.get("tools") or []:
        for fn in (
            (t.get("functionDeclarations") or t.get("function_declarations") or []) if isinstance(t, dict) else []
        ):
            schema = fn.get("parametersJsonSchema") or fn.get("parameters")
            c.tools.append(tool_from_json_schema(fn["name"], fn.get("description"), schema, "wire", "wire:gemini"))
    c.message_count = len(body.get("contents") or [])


def parse_request(
    index: int, url: str, body: dict[str, Any], *, run_id: str = "run-0", client_lib: str = "httpx"
) -> CapturedCall:
    provider = detect_provider(url, body) or "unknown"
    c = CapturedCall(
        index=index,
        run_id=run_id,
        provider=provider,
        url=url,
        client_lib=client_lib,
        model=body.get("model") if isinstance(body.get("model"), str) else None,
    )
    if provider == "openai_chat":
        _parse_openai_chat(body, c)
    elif provider == "openai_responses":
        _parse_openai_responses(body, c)
    elif provider == "anthropic":
        _parse_anthropic(body, c)
    elif provider == "gemini":
        _parse_gemini(url, body, c)
    if c.system_prompt:
        c.system_prompt_sha256 = sha256(c.system_prompt)
    return c


# ---------------------------------------------------------------- responses
def parse_response_tool_calls(provider: Provider, body: dict[str, Any]) -> list[ToolCall]:
    out: list[ToolCall] = []
    if provider == "openai_chat":
        for ch in body.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                out.append(ToolCall(name=fn.get("name", "?"), arguments=_loads(fn.get("arguments")), id=tc.get("id")))
    elif provider == "openai_responses":
        for it in body.get("output") or []:
            if isinstance(it, dict) and it.get("type") == "function_call":
                out.append(
                    ToolCall(name=it.get("name", "?"), arguments=_loads(it.get("arguments")), id=it.get("call_id"))
                )
    elif provider == "anthropic":
        for b in body.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                out.append(ToolCall(name=b.get("name", "?"), arguments=b.get("input") or {}, id=b.get("id")))
    elif provider == "gemini":
        for cand in body.get("candidates") or []:
            for p in (cand.get("content") or {}).get("parts") or []:
                if isinstance(p, dict) and isinstance(p.get("functionCall"), dict):
                    fc = p["functionCall"]
                    out.append(ToolCall(name=fc.get("name", "?"), arguments=fc.get("args") or {}, id=fc.get("id")))
    return out
