"""OpenTelemetry GenAI spans reader + trace_coverage (scenario matrix derived from the spec).

Accepted inputs:
* OTLP/JSON export (``{"resourceSpans": [...]}``), one document or one per line
* JSONL / JSON list of flat spans: ``{"trace_id", "span_id", "parent_span_id", "name", "attributes": {...}}``

Attributes used (OTel GenAI semantic conventions): ``gen_ai.operation.name``,
``gen_ai.tool.name``, ``gen_ai.agent.name``, ``gen_ai.request.model``,
``gen_ai.tool.call.arguments``. Scenario tagging: ``agenticreq.scenario``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticreq.models.observed import TraceSpan


def _attr_value(v: dict[str, Any]) -> Any:
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _attr_value(kv["value"]) for kv in v["kvlistValue"].get("values", [])}
    return None


def _attrs(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {a["key"]: _attr_value(a.get("value", {})) for a in raw if isinstance(a, dict) and "key" in a}
    return {}


def _span(d: dict[str, Any]) -> TraceSpan:
    attrs = _attrs(d.get("attributes"))
    events = [e.get("name", "") for e in d.get("events") or [] if isinstance(e, dict)]
    return TraceSpan(
        trace_id=d.get("trace_id") or d.get("traceId"),
        span_id=d.get("span_id") or d.get("spanId"),
        parent_span_id=d.get("parent_span_id") or d.get("parentSpanId") or None,
        name=d.get("name", ""),
        operation=attrs.get("gen_ai.operation.name"),
        agent_name=attrs.get("gen_ai.agent.name"),
        tool_name=attrs.get("gen_ai.tool.name"),
        model=attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
        attributes=attrs,
        events=events,
        start_time=_start_time(d),
    )


def _start_time(d: dict[str, Any]) -> float | None:
    """OTLP startTimeUnixNano / start_time_unix_nano (ns), or an ISO-8601 start_time."""
    v = d.get("startTimeUnixNano") or d.get("start_time_unix_nano") or d.get("start_time")
    if v is None:
        return None
    try:
        n = float(v)
        return n / 1e9 if n > 1e12 else n
    except (TypeError, ValueError):
        pass
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _from_doc(doc: Any) -> list[TraceSpan]:
    out: list[TraceSpan] = []
    if isinstance(doc, list):
        for x in doc:
            out += _from_doc(x)
    elif isinstance(doc, dict) and "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                out += [_span(s) for s in ss.get("spans", [])]
    elif isinstance(doc, dict) and ("span_id" in doc or "spanId" in doc or "name" in doc):
        out.append(_span(doc))
    return out


def load_traces(path: str | Path) -> list[TraceSpan]:
    p = Path(path)
    files = sorted(p.glob("*.json*")) if p.is_dir() else [p]
    spans: list[TraceSpan] = []
    for f in files:
        text = f.read_text(encoding="utf-8-sig")
        try:
            spans += _from_doc(json.loads(text))
        except json.JSONDecodeError:
            for line in text.splitlines():
                if line.strip():
                    spans += _from_doc(json.loads(line))
    return spans
