"""Python builder API — an alternative way to *produce* spec.yml.

YAML is canonical: `check` reads spec.yml, never Python. The spec must be readable where the agent's
code cannot run (CI without credentials, another Python environment, a compliance review), and
reading a Python spec means importing its module with all its side effects and dependencies. The
builder therefore always ends in YAML: ``SpecBuilder(...).write("spec.yml")`` or
``agenticreq export-spec my_spec.py:SPEC``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from agenticreq.models.spec import AgentSpec


class SpecBuilder:
    """Fluent builder producing the same raw shape as YAML, so validation is identical.

    >>> spec = (SpecBuilder("po-agent").risk("high").owner("ops@x")
    ...         .prohibit("issue_refund", reason="refunds need finance").build())
    """

    def __init__(self, agent_id: str) -> None:
        self._d: dict[str, Any] = {"agent_id": agent_id}

    def risk(self, level: str) -> SpecBuilder:
        self._d["risk_level"] = level
        return self

    def goal(self, statement: str, *done_criteria: str) -> SpecBuilder:
        self._d["goal"] = {"statement": statement, "done_criteria": list(done_criteria)}
        return self

    def owner(self, owner: str, technical_owner: str | None = None) -> SpecBuilder:
        self._d["owner"] = owner
        if technical_owner:
            self._d["technical_owner"] = technical_owner
        return self

    def out_of_scope(self, *items: str) -> SpecBuilder:
        self._d.setdefault("out_of_scope", []).extend(items)
        return self

    def allow(self, *tools: str) -> SpecBuilder:
        self._d.setdefault("role_boundary", {}).setdefault("allowed_tools", []).extend(tools)
        return self

    def prohibit(self, tool: str, *, reason: str | None = None) -> SpecBuilder:
        self._d.setdefault("prohibited", []).append({"tool": tool, "reason": reason})
        return self

    def require_approval(
        self, tool: str, approver: str, timeout: str = "4h", on_timeout: str = "safe_stop"
    ) -> SpecBuilder:
        self._d.setdefault("approval_required", []).append(
            {"tool": tool, "approver": approver, "timeout": timeout, "on_timeout": on_timeout}
        )
        return self

    def limit(self, tool: str, arg: str, **bounds: Any) -> SpecBuilder:
        self._d.setdefault("limits", []).append({"tool": tool, "arg": arg, "limit": bounds})
        return self

    def escalate(self, metric: str, op: str, threshold: float | int | str, to: str) -> SpecBuilder:
        self._d.setdefault("escalation", []).append({"metric": metric, "op": op, "threshold": threshold, "to": to})
        return self

    def untrusted(self, *sources: str) -> SpecBuilder:
        self._d.setdefault("untrusted_inputs", []).extend({"source": s} for s in sources)
        return self

    def data_rules(
        self, *, never_store: list[str] | None = None, retention: dict[str, str] | None = None
    ) -> SpecBuilder:
        self._d["data_rules"] = {"never_store": list(never_store or []), "retention": dict(retention or {})}
        return self

    def set(self, key: str, value: Any) -> SpecBuilder:
        self._d[key] = value
        return self

    def raw(self) -> dict[str, Any]:
        return dict(self._d)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self._d, sort_keys=False, allow_unicode=True)

    def write(self, path: str | Path, *, force: bool = False) -> Path:
        """Validate, then write spec.yml — the canonical form `check` reads. Refuses to overwrite."""
        return write_spec(self, path, force=force)

    def build(self) -> AgentSpec:
        from agenticreq.loader.yaml_loader import load_spec_text

        loaded = load_spec_text(self.to_yaml(), "<builder>.yml")
        if loaded.spec is None:
            msgs = "; ".join(f.detail for f in loaded.load_findings)
            raise ValueError(f"invalid spec: {msgs}")
        return loaded.spec


def spec_to_yaml(obj: SpecBuilder | AgentSpec | dict[str, Any]) -> str:
    """YAML text for a builder, a validated AgentSpec, or a raw dict."""
    if isinstance(obj, SpecBuilder):
        return obj.to_yaml()
    if isinstance(obj, AgentSpec):
        data = obj.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_defaults=True)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if isinstance(obj, dict):
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    raise ValueError(f"expected SpecBuilder, AgentSpec or dict, got {type(obj).__name__}")


def write_spec(obj: SpecBuilder | AgentSpec | dict[str, Any], path: str | Path, *, force: bool = False) -> Path:
    """Validate and write spec.yml. Raises FileExistsError unless force; ValueError if it does not load."""
    from agenticreq.loader.yaml_loader import load_spec_text

    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists; pass force=True to overwrite")
    text = spec_to_yaml(obj)
    loaded = load_spec_text(text, p.name)
    if loaded.spec is None:
        raise ValueError("invalid spec: " + "; ".join(f.detail for f in loaded.load_findings))
    p.write_text(text, encoding="utf-8")
    return p


def load_python_spec(target: str) -> SpecBuilder | AgentSpec | dict[str, Any]:
    """Import ``file.py:NAME`` / ``package.module:NAME`` and return NAME (called if it is a function)."""
    from agenticreq.capture.entry import split_entry

    mod_ref, attr = split_entry(target)
    if not attr:
        raise ValueError(f"{target!r}: expected file.py:NAME or package.module:NAME")
    if mod_ref.endswith(".py"):
        file = Path(mod_ref).resolve()
        if not file.exists():
            raise FileNotFoundError(str(file))
        spec = importlib.util.spec_from_file_location(f"_agenticreq_spec_{file.stem}", file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {file}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(file.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(file.parent))
    else:
        module = importlib.import_module(mod_ref)
    obj = getattr(module, attr)
    if callable(obj) and not isinstance(obj, (SpecBuilder, AgentSpec, dict)):
        obj = obj()
    if not isinstance(obj, (SpecBuilder, AgentSpec, dict)):
        raise ValueError(f"{target} is a {type(obj).__name__}, not a SpecBuilder / AgentSpec / dict")
    return obj
