"""Static evidence for an end-to-end trace id, in the order the rule asks for it: where an id is created, where it is
passed on, and where log or audit records carry it. Table-driven by data/tracing.yml; a project's own correlation id
and OpenTelemetry count for every framework."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from agenticreq.ast.extract_support import _LOG_CALLS, _ref
from agenticreq.ast.schema import _name
from agenticreq.data import load_yaml
from agenticreq.models.finding import SourceRef
from agenticreq.models.observed import TraceEvidence

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor

_ID_NAME = re.compile(r"(^|_)(run|trace|correlation|request|session|invocation)_?id$", re.IGNORECASE)
_ID_MAKERS = {"uuid4", "uuid1", "token_hex", "token_urlsafe", "ulid", "gen_trace_id"}


def _makes_id(node: ast.AST) -> bool:
    return any(isinstance(c, ast.Call) and _name(c.func) in _ID_MAKERS for c in ast.walk(node))


def _is_log_call(c: ast.Call) -> bool:
    return isinstance(c.func, ast.Attribute) and c.func.attr in _LOG_CALLS


def _mentions(c: ast.Call, name: str) -> bool:
    return any(
        isinstance(x, ast.Name) and x.id == name for a in [*c.args, *(k.value for k in c.keywords)] for x in ast.walk(a)
    )


def _correlation_id(ex: ProjectExtractor) -> list[TraceEvidence]:
    out: dict[str, TraceEvidence] = {}
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            if isinstance(n, ast.Assign) and _makes_id(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name) and _ID_NAME.search(t.id):
                        ev = out.setdefault(t.id, TraceEvidence(mechanism="correlation id", name=t.id))
                        ev.created.append(_ref(fc.rel, n))
    for fc in ex.files:
        for c in ast.walk(fc.tree):
            if not isinstance(c, ast.Call):
                continue
            for name, ev in out.items():
                if _mentions(c, name):
                    (ev.logged if _is_log_call(c) else ev.used).append(_ref(fc.rel, c))
    return list(out.values())


def _otel(ex: ProjectExtractor, table: dict[str, list[str]]) -> list[TraceEvidence]:
    ev = TraceEvidence(mechanism="OpenTelemetry", name="trace id of the active span")
    for fc in ex.files:
        if "opentelemetry" not in fc.imports:
            continue
        for c in ast.walk(fc.tree):
            if isinstance(c, ast.Call):
                nm = _name(c.func)
                if nm in table["create"]:
                    ev.created.append(_ref(fc.rel, c))
                elif nm in table["carry"]:
                    ev.used.append(_ref(fc.rel, c))
                elif _is_log_call(c) and "trace_id" in ast.unparse(c):
                    ev.logged.append(_ref(fc.rel, c))
    return [ev] if ev.created else []


def _framework(ex: ProjectExtractor, table: dict[str, dict[str, object]]) -> list[TraceEvidence]:
    row = table.get(ex.obs.framework or "")
    if row is None:
        return []
    tokens = set(row["tokens"])  # type: ignore[call-overload]
    ev = TraceEvidence(mechanism=str(row["mechanism"]), name=str(row["id"]))
    for fc in ex.files:
        for n in ast.walk(fc.tree):
            word = n.attr if isinstance(n, ast.Attribute) else n.arg if isinstance(n, ast.keyword) else None
            word = word or (_name(n.func) if isinstance(n, ast.Call) else None)
            if word in tokens:
                ref = _ref(fc.rel, n) if not isinstance(n, ast.keyword) else _ref(fc.rel, n.value)
                ev.used.append(ref)
            if isinstance(n, ast.Call) and _is_log_call(n) and any(t in ast.unparse(n) for t in tokens):
                ev.logged.append(_ref(fc.rel, n))
    return [ev] if ev.used else []


def _unique(refs: list[SourceRef]) -> list[SourceRef]:
    """One place per file:line, however many expressions on that line mention the id."""
    return list({str(r): r for r in refs}.values())


def trace_evidence(ex: ProjectExtractor) -> list[TraceEvidence]:
    table = load_yaml("tracing.yml")
    found = [*_framework(ex, table["frameworks"]), *_otel(ex, table["otel"]), *_correlation_id(ex)]
    for ev in found:
        ev.created, ev.used, ev.logged = _unique(ev.created), _unique(ev.used), _unique(ev.logged)
    return found
