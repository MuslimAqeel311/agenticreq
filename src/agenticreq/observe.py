"""Assemble one ObservedAgent from AST + captures + MCP + traces (+ recognizers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticreq.ast.framework import refine_with_wire
from agenticreq.ast.extractor import extract
from agenticreq.models.observed import CapturedCall, ObservedAgent, TraceSpan, ToolSchema


def _plugin_names(obs: ObservedAgent, calls: list[CapturedCall]) -> None:
    """``Notes-lookup`` on the wire is the code's ``lookup`` (data/tool_aliases.yml)."""
    from agenticreq.data import load_yaml

    sep = (load_yaml("tool_aliases.yml").get("plugin_separator") or {}).get(obs.framework or "")
    if not sep:
        return
    for c in calls:
        items: list[Any] = [*c.tools, *c.tool_calls, *c.prior_tool_calls]
        for item in items:
            tail = item.name.split(sep, 1)[-1]
            if sep in item.name and tail in obs.declared_tools:
                item.name = tail


def merge_captures(obs: ObservedAgent, calls: list[CapturedCall]) -> None:
    _plugin_names(obs, calls)
    obs.captures.extend(calls)
    for c in calls:
        if c.provider == "unknown":
            obs.unsupported_calls.append(c.url)
        for t in c.tools:
            obs.wire_tools.setdefault(t.name, t)
        if c.model:
            from agenticreq.ast.classify import is_pinned
            from agenticreq.models.observed import ModelRef

            if not any(m.name == c.model and m.source == "wire" for m in obs.models):
                from urllib.parse import urlparse

                from agenticreq.ast.classify import provider_from

                obs.models.append(
                    ModelRef(
                        name=c.model,
                        pinned=is_pinned(c.model),
                        source="wire",
                        provider=provider_from(host=urlparse(c.url).hostname or "", model=c.model),
                    )
                )
    refine_with_wire(obs)


def merge_mcp(obs: ObservedAgent, tools: list[ToolSchema]) -> None:
    for t in tools:
        obs.mcp_tools.setdefault(t.name, t)


def merge_traces(obs: ObservedAgent, spans: list[TraceSpan]) -> None:
    obs.trace_spans.extend(spans)


def _apply_runtime_frameworks(obs: ObservedAgent, runtime_versions: dict[str, str]) -> None:
    """A framework the captured run loaded is observed, whatever the static scan made of the imports; the
    version that produced the wire beats whatever the analyzer has installed."""
    from agenticreq.ast.framework import _PRIORITY, DISTS, FRAMEWORKS, GENERIC, classify_execution_model

    loaded = [
        FRAMEWORKS[m][0]
        for m in (k[len("module:") :] for k in runtime_versions if k.startswith("module:"))
        if m in FRAMEWORKS and FRAMEWORKS[m][0] not in GENERIC
    ]
    if loaded:
        obs.frameworks = sorted({*obs.frameworks, *loaded}, key=lambda f: _PRIORITY.index(f) if f in _PRIORITY else 99)
        if (
            obs.framework not in loaded and obs.framework_source != "import"
        ):  # nothing static was proven: the run decides
            obs.framework = next(f for f in obs.frameworks if f in loaded)
            obs.framework_version = obs.framework_version_source = None
            classify_execution_model(obs)
        if obs.framework in loaded:
            obs.framework_source = "runtime"
    if obs.framework:
        dist = next((d for d in DISTS.get(obs.framework, ()) if runtime_versions.get(d)), None)
        if dist:
            obs.framework_version = runtime_versions[dist]
            obs.framework_version_source = "runtime"


def build_observed(
    path: str | Path,
    *,
    captures: list[CapturedCall] | None = None,
    spans: list[TraceSpan] | None = None,
    mcp_tools: list[ToolSchema] | None = None,
    use_recognizers: bool = True,
    runtime_versions: dict[str, str] | None = None,
) -> ObservedAgent:
    obs = extract(path)
    if runtime_versions:
        _apply_runtime_frameworks(obs, runtime_versions)
    if captures:
        merge_captures(obs, captures)
    if mcp_tools:
        merge_mcp(obs, mcp_tools)
    if spans:
        merge_traces(obs, spans)
    if use_recognizers:
        from agenticreq.recognizers import run_recognizers

        run_recognizers(obs, Path(path))
    from agenticreq.graph.builder import build_delegation_graph

    build_delegation_graph(obs)
    return obs
