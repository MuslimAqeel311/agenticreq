"""Canned responder: answers every intercepted LLM call locally, in the provider's wire shape.

Strategy (deterministic, no LLM): within a conversation, request each offered tool
once, in order, with placeholder arguments derived from its JSON schema (or the
mock file's ``args``); once all have been requested, return a final text answer.
This drives the agent through its tool loop so the boundary sees each tool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from agenticreq.jsonschema_util import example_arguments, example_instance
from agenticreq.models.observed import CapturedCall, ToolCall


@dataclass
class ToolMock:
    args: dict[str, Any] = field(default_factory=dict)
    returns: Any = None
    patch: str | None = None  # "module:attr" to replace with a stub returning `returns`
    call: bool = True  # False = never ask the model-side to call this tool


#: what one run under ``runs:`` in mocks.yml may set; everything else comes from the top level of the file
RUN_KEYS = ("name", "entry", "args", "order", "tools", "structured", "final_text", "human_input", "max_llm_calls")


@dataclass
class RunSpec:
    """One named mock run: its own task (entry / args) and the settings it changes (order, tool args, call:)."""

    name: str
    entry: str | None = None
    args: Any = None
    has_args: bool = False  # args: given (even null), so the top-level args: does not apply
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockConfig:
    tools: dict[str, ToolMock] = field(default_factory=dict)
    final_text: str = "Done. (agenticreq canned response)"
    max_llm_calls: int = 40
    order: list[str] = field(default_factory=list)  # preferred tool order
    #: field name -> value for structured-output answers (e.g. {is_relevant: true, plan: [...]})
    structured: dict[str, Any] = field(default_factory=dict)
    #: what the run answers at an input() prompt (a console approval or question): it never waits for a human
    human_input: str = "yes"
    #: non-secret settings the agent reads from its environment (a model name, a feature flag): discover passes
    #: only an allowlist of the caller's variables, so these are the way in. Live-looking credentials are refused.
    env: dict[str, str] = field(default_factory=dict)
    #: what `agenticreq mock-run` runs: the entry (module:function or file.py[:function]), its inputs and the
    #: interpreter (empty = ./.venv when it exists, else the current one). Command-line flags override them.
    entry: str | None = None
    args: Any = None
    python: str | None = None
    #: folders (relative to the agent's folder) the agent may write to during the run
    allow_write: list[str] = field(default_factory=list)
    #: runs: in mocks.yml, in file order; empty = one run from the top level (the single-task file)
    runs: list[RunSpec] = field(default_factory=list)
    source: str = "<none>"

    @classmethod
    def load(cls, path: str | Path) -> MockConfig:
        import yaml  # parent process only: the child reads the resolved settings as JSON (to_dict / from_dict)

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        if not isinstance(data, dict) or not isinstance(data.get("tools", {}), dict):
            raise ValueError(f"{path}: mock file must be a mapping with a `tools:` mapping")
        tools = {}
        for name, body in (data.get("tools") or {}).items():
            body = body if isinstance(body, dict) else {"returns": body}
            tools[str(name)] = ToolMock(
                args=dict(body.get("args") or {}),
                returns=body.get("returns"),
                patch=body.get("patch"),
                call=bool(body.get("call", True)),
            )
        return cls(
            runs=_load_runs(data.get("runs"), path),
            tools=tools,
            final_text=str(data.get("final_text", cls.final_text)),
            max_llm_calls=int(data.get("max_llm_calls", 40)),
            order=list(data.get("order") or []),
            structured=dict(data.get("structured") or {}),
            human_input=str(data.get("human_input", "yes")),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            entry=_filled(data.get("entry")),
            args=None if _is_todo(data.get("args")) else data.get("args"),
            python=_filled(data.get("python")),
            allow_write=[str(x) for x in data.get("allow_write") or []],
            source=str(path),
        )

    def to_dict(self) -> dict[str, Any]:
        """The settings of one run as plain JSON, for the child process (which needs no YAML reader)."""
        return asdict(replace(self, runs=[]))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockConfig:
        d = dict(data)
        d["tools"] = {k: ToolMock(**v) for k, v in (d.get("tools") or {}).items()}
        d["runs"] = []
        return cls(**d)

    def run_names(self) -> list[str]:
        return [r.name for r in self.runs]

    def for_run(self, name: str | None) -> MockConfig:
        """The settings of one run: this file's top level with that run's order, tools (args / call:), structured,
        final_text, human_input and max_llm_calls applied on top. No such run (or none named) = unchanged."""
        spec = next((r for r in self.runs if r.name == name), None)
        if spec is None or not spec.overrides:
            return self
        o = spec.overrides
        tools = {k: ToolMock(v.args, v.returns, v.patch, v.call) for k, v in self.tools.items()}
        for tname, body in (o.get("tools") or {}).items():
            base = tools.setdefault(tname, ToolMock())
            if not isinstance(body, dict):
                base.returns = body
                continue
            base.args = {**base.args, **dict(body.get("args") or {})}
            if "returns" in body:
                base.returns = body["returns"]
            if "patch" in body:
                base.patch = body["patch"]
            if "call" in body:
                base.call = bool(body["call"])
        return replace(
            self,
            tools=tools,
            runs=[],
            order=list(o["order"]) if "order" in o else self.order,
            structured={**self.structured, **dict(o.get("structured") or {})},
            final_text=str(o["final_text"]) if "final_text" in o else self.final_text,
            human_input=str(o["human_input"]) if "human_input" in o else self.human_input,
            max_llm_calls=int(o["max_llm_calls"]) if "max_llm_calls" in o else self.max_llm_calls,
        )


def _load_runs(raw: Any, path: str | Path) -> list[RunSpec]:
    """runs: in mocks.yml: a list of {name, entry, args, order, tools, structured, ...}. Errors name the run."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: runs: must be a list of runs, each with a name:")
    out: list[RunSpec] = []
    for i, body in enumerate(raw):
        if not isinstance(body, dict):
            raise ValueError(f"{path}: runs[{i}] must be a mapping (name:, args:, ...)")
        name = str(body.get("name") or f"run-{i}").strip()
        bad = sorted(set(body) - set(RUN_KEYS))
        if bad:
            raise ValueError(
                f"{path}: run {name!r} has unknown field {', '.join(bad)}; a run may set {', '.join(RUN_KEYS)}"
            )
        if any(r.name == name for r in out):
            raise ValueError(f"{path}: two runs are named {name!r}; give each run its own name:")
        if not isinstance(body.get("tools") or {}, dict):
            raise ValueError(f"{path}: run {name!r}: tools: must be a mapping of tool name to args:/call:")
        args = body.get("args")
        out.append(
            RunSpec(
                name=name,
                entry=_filled(body.get("entry")),
                args=None if _is_todo(args) else args,
                has_args="args" in body and not _is_todo(args),
                overrides={k: v for k, v in body.items() if k not in ("name", "entry", "args")},
            )
        )
    return out


def _is_todo(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper().startswith("TODO")


def _filled(v: Any) -> str | None:
    """A text value from mocks.yml, or None when it is empty or still TODO."""
    if v is None or _is_todo(v) or not str(v).strip():
        return None
    return str(v).strip()


def _tool_schema(body: dict[str, Any], provider: str, name: str) -> dict[str, Any] | None:
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if provider == "openai_chat":
            fn = t.get("function") or {}
            if fn.get("name") == name:
                return fn.get("parameters")
        elif provider == "openai_responses" and t.get("name") == name:
            return t.get("parameters")
        elif provider == "anthropic" and t.get("name") == name:
            return t.get("input_schema")
    return None


def output_schema(body: dict[str, Any], provider: str) -> dict[str, Any] | None:
    """The JSON schema a request asks the model's *text* answer to follow, if any.
    ``{}`` means JSON was requested without a schema (json_object mode)."""
    if provider == "openai_chat":
        rf = body.get("response_format") or {}
        if rf.get("type") == "json_schema":
            return (rf.get("json_schema") or {}).get("schema") or {}
        if rf.get("type") == "json_object":
            return {}
    elif provider == "openai_responses":
        fmt = (body.get("text") or {}).get("format") or {}
        if fmt.get("type") == "json_schema":
            return fmt.get("schema") or {}
        if fmt.get("type") == "json_object":
            return {}
    elif provider == "anthropic":
        of = body.get("output_format") or {}
        if of.get("type") == "json_schema":
            return of.get("schema") or {}
    return None


class CannedResponder:
    def __init__(self, mocks: MockConfig | None = None) -> None:
        self.mocks = mocks or MockConfig()
        self.calls = 0
        # Tools already requested per conversation. Needed because some frameworks
        # (smolagents ToolCallingAgent) do not replay prior tool calls in wire format.
        self._asked: dict[str, set[str]] = {}
        self._final = self.mocks.final_text

    @staticmethod
    def conversation_key(call: CapturedCall, body: dict[str, Any]) -> str:
        first_user = ""
        msgs = body.get("messages") or body.get("input") or []
        if isinstance(msgs, str):
            first_user = msgs
        else:
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    first_user = json.dumps(m.get("content"), sort_keys=True, default=str)
                    break
        return f"{call.system_prompt_sha256}|{hashlib.sha256(first_user.encode()).hexdigest()[:16]}"

    def plan(self, call: CapturedCall, body: dict[str, Any]) -> list[ToolCall]:
        self.calls += 1
        schema = output_schema(body, call.provider)
        # structured output (guardrail agents, planners): answer with JSON that validates, not prose
        self._final = (
            self.mocks.final_text
            if schema is None
            else json.dumps(
                example_instance(schema, self.mocks.structured) if schema else {"result": self.mocks.final_text}
            )
        )
        if self.calls > self.mocks.max_llm_calls:
            return []
        asked = self._asked.setdefault(self.conversation_key(call, body), set())
        already = {tc.name for tc in call.prior_tool_calls} | asked
        offered = [t.name for t in call.tools if not t.name.startswith("hosted:")]
        order = [n for n in self.mocks.order if n in offered] + [n for n in offered if n not in self.mocks.order]
        for name in order:
            mock = self.mocks.tools.get(name)
            if name in already or (mock is not None and not mock.call):
                continue
            schema = _tool_schema(body, call.provider, name)
            args = example_arguments(schema, mock.args if mock else None)
            asked.add(name)
            return [ToolCall(name=name, arguments=args, id=f"call_ac_{self.calls}_{name}"[:64])]
        return []

    # ------------------------------------------------------------ rendering
    def render(self, call: CapturedCall, tool_calls: list[ToolCall], stream: bool) -> tuple[bytes, str]:
        model = call.model or "agenticreq-canned"
        if call.provider == "anthropic":
            body = self._anthropic(model, tool_calls)
            return (self._anthropic_sse(body) if stream else json.dumps(body).encode()), (
                "text/event-stream" if stream else "application/json"
            )
        if call.provider == "openai_responses":
            body = self._responses(model, tool_calls)
            return (self._responses_sse(body) if stream else json.dumps(body).encode()), (
                "text/event-stream" if stream else "application/json"
            )
        body = self._chat(model, tool_calls)
        return (self._chat_sse(body) if stream else json.dumps(body).encode()), (
            "text/event-stream" if stream else "application/json"
        )

    _USAGE_CHAT = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    def _chat(self, model: str, tcs: list[ToolCall]) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": None if tcs else self._final}
        if tcs:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in tcs
            ]
        return {
            "id": f"chatcmpl-ac{self.calls}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "message": msg, "logprobs": None, "finish_reason": "tool_calls" if tcs else "stop"}
            ],
            "usage": self._USAGE_CHAT,
        }

    def _chat_sse(self, body: dict[str, Any]) -> bytes:
        msg = body["choices"][0]["message"]
        base = {k: body[k] for k in ("id", "object", "created", "model")}
        base["object"] = "chat.completion.chunk"
        delta: dict[str, Any] = {"role": "assistant"}
        if msg.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])]
        else:
            delta["content"] = msg["content"]
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": body["choices"][0]["finish_reason"]}],
                "usage": self._USAGE_CHAT,
            },
        ]
        return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"

    def _anthropic(self, model: str, tcs: list[ToolCall]) -> dict[str, Any]:
        content: list[dict[str, Any]] = (
            [
                {
                    "type": "tool_use",
                    "id": (tc.id or "toolu_ac").replace("call_", "toolu_"),
                    "name": tc.name,
                    "input": tc.arguments,
                }
                for tc in tcs
            ]
            if tcs
            else [{"type": "text", "text": self._final}]
        )
        return {
            "id": f"msg_ac{self.calls}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if tcs else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def _anthropic_sse(self, body: dict[str, Any]) -> bytes:
        ev: list[tuple[str, dict[str, Any]]] = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        **body,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
        ]
        for i, b in enumerate(body["content"]):
            if b["type"] == "text":
                ev.append(
                    (
                        "content_block_start",
                        {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                    )
                )
                ev.append(
                    (
                        "content_block_delta",
                        {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}},
                    )
                )
            else:
                ev.append(
                    (
                        "content_block_start",
                        {"type": "content_block_start", "index": i, "content_block": {**b, "input": {}}},
                    )
                )
                ev.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])},
                        },
                    )
                )
            ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        ev.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            )
        )
        ev.append(("message_stop", {"type": "message_stop"}))
        return b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in ev)

    def _responses(self, model: str, tcs: list[ToolCall]) -> dict[str, Any]:
        if tcs:
            output: list[dict[str, Any]] = [
                {
                    "type": "function_call",
                    "id": f"fc_ac{self.calls}_{i}",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                    "status": "completed",
                }
                for i, tc in enumerate(tcs)
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": f"msg_ac{self.calls}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": self._final, "annotations": []}],
                }
            ]
        return {
            "id": f"resp_ac{self.calls}",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": "completed",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "temperature": 1.0,
            "top_p": 1.0,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    def _responses_sse(self, body: dict[str, Any]) -> bytes:
        created = {**body, "status": "in_progress", "output": []}
        ev: list[dict[str, Any]] = [{"type": "response.created", "response": created, "sequence_number": 0}]
        for i, item in enumerate(body["output"]):
            ev.append(
                {"type": "response.output_item.added", "output_index": i, "item": item, "sequence_number": len(ev)}
            )
            ev.append(
                {"type": "response.output_item.done", "output_index": i, "item": item, "sequence_number": len(ev)}
            )
        ev.append({"type": "response.completed", "response": body, "sequence_number": len(ev)})
        return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in ev)
