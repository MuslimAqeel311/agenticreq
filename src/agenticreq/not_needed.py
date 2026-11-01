"""``not_needed``: the one way to say "this does not apply to my project", on any spec field at any level.

    conflict_resolution: not_needed          # any type: a string, a list, a mapping, a number, a boolean, an enum
    bounds: {max_cost_usd: not_needed}       # a nested key
    confirmed:
      - {clause: conflict_resolution, answer: not_needed, by: me@x.com, date: 2026-10-30, reason: "pipeline"}

Writing ``not_needed`` on the field is enough to be valid; the ``confirmed:`` entry adds the reason. The field counts
as answered (no TODO, required-field or shape error) and the report shows it as ``NOT NEEDED <parameter> <reason>``
under Human sign-off: done, never a gap and never a pass. Without a reason it shows ``(no reason given)`` and stays
in the "left" count.

``[]`` and ``{}`` keep their meaning: "answered: nothing".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticreq.models.finding import SourceRef, SpecIssue

if TYPE_CHECKING:
    from agenticreq.models.spec import AgentSpec

PathKey = tuple[str | int, ...]

VALUE = "not_needed"
NO_REASON = "(no reason given)"
#: every project has these: they cannot be not_needed at any risk level
ALWAYS: dict[str, str] = {
    "agent_id": "every project has an id",
    "risk_level": "every project has a risk level",
    "goal": "every project has a goal",
    "owner": "every project has an owner",
}
#: meta keys: never a parameter
_META = frozenset({"confirmed", "waivers", "_derived", "extends", "spec_version", "tool_effects"})
#: printed once at the top of `agenticreq params`
PARAMS_NOTE = (
    "Any field (or a key inside one) can be `not_needed`: it does not apply to this project. Give the reason "
    "under confirmed: (answer: not_needed). Not allowed for agent_id, risk_level, goal, owner, nor at high "
    "or critical risk for "
    + ", ".join(
        (
            "kill_switch",
            "fail_closed",
            "audit_log",
            "version_pinning",
            "self_modification",
            "log_immutable",
            "transitive_prohibition",
            "dry_run_required",
        )
    )
    + "."
)
#: how to write a value instead, the end of every shape error
SHAPE_HINT = "write a value as shown, remove the line to use the default, or write `not_needed`"


def is_value(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() == VALUE


def forbidden() -> tuple[str, ...]:
    from agenticreq.models.spec import NOT_NEEDED_FORBIDDEN

    return NOT_NEEDED_FORBIDDEN


def blocked_message(param: str, risk: Any) -> str:
    """Why ``param`` cannot be not_needed here (the spec error), or "" when it can."""
    if param in ALWAYS:
        return f"{param} cannot be not_needed: {ALWAYS[param]}"
    if risk in ("high", "critical") and param in forbidden():
        return (
            f"{param} cannot be not_needed at {risk} risk: a {risk}-risk agent must have it in place; answer yes "
            "or no under confirmed:"
        )
    return ""


def _label(path: PathKey) -> str:
    """bounds.max_cost_usd, approval_required[].timeout: the parameter name a person writes under confirmed:."""
    out = ""
    for p in path:
        out = out + "[]" if isinstance(p, int) else (f"{out}.{p}" if out else str(p))
    return out


def extract(
    raw: dict[str, Any], refs: dict[PathKey, SourceRef], file: str
) -> tuple[dict[str, Any], dict[str, SourceRef], list[SpecIssue]]:
    """``raw`` without its ``not_needed`` values, {parameter: where it was written}, and a spec error for each
    parameter that cannot be not_needed."""
    found: dict[str, SourceRef] = {}

    def walk(v: Any, path: PathKey) -> Any:
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if not path and k in _META:
                    out[k] = x
                elif is_value(x):
                    found.setdefault(_label((*path, k)), refs.get((*path, k)) or SourceRef(file=file, line=1))
                else:
                    out[k] = walk(x, (*path, k))
            return out
        if isinstance(v, list):
            return [walk(x, (*path, i)) if isinstance(x, (dict, list)) else x for i, x in enumerate(v)]
        return v

    cleaned = walk(raw, ())
    risk = raw.get("risk_level")
    issues = [blocked_issue(p, ref, msg) for p, ref in found.items() for msg in [blocked_message(p, risk)] if msg]
    return cleaned, found, issues


def blocked_issue(param: str, ref: SourceRef | None, message: str) -> SpecIssue:
    from agenticreq.shapes import example_yaml, spec_fields

    top = param.split(".")[0].split("[")[0]
    return SpecIssue(
        rule_id="AS-107",
        kind="invalid",
        field=param,
        spec_ref=ref,
        message=message,
        expected="a value",
        example=example_yaml(top) if top in spec_fields() else "",
        items=[param],
    )


@dataclass(frozen=True)
class Entry:
    """One not_needed parameter as the report shows it. ``open``: no reason (or an incomplete confirmed: entry),
    so it stays in the "left" count."""

    param: str
    reason: str
    open: bool
    line: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"param": self.param, "reason": self.reason, "open": self.open, "line": self.line}


def entries(
    written: dict[str, SourceRef], spec: AgentSpec | None, today: dt.date, root: str | Path | None = None
) -> list[Entry]:
    """Every not_needed parameter: written on the field, or answered not_needed under confirmed:."""
    from agenticreq.models.verdict import sentence
    from agenticreq.signoff import assess

    answers = {a.clause: a for a in (spec.confirmed if spec else []) if a.answer == VALUE}
    out: list[Entry] = []
    for param in dict.fromkeys([*written, *answers]):
        a = answers.get(param)
        ref = written.get(param)
        line = str(ref) if ref else ""
        if a is None:
            out.append(Entry(param, NO_REASON, True, line))
            continue
        st = assess(a, today, root)
        reason = sentence(a.reason) if a.reason else NO_REASON
        if st.state == VALUE:
            out.append(Entry(param, reason, not a.reason, line))
        else:  # a field the answer needs is missing (by, date, reason): it does not count yet
            out.append(
                Entry(
                    param,
                    f"{reason.rstrip('.')}; the entry under confirmed: does not count yet: {st.message}.",
                    True,
                    line,
                )
            )
    return out


def covers(params: set[str] | frozenset[str], param: str) -> bool:
    """``param`` is one of ``params`` or lies under one (bounds covers bounds.max_cost_usd)."""
    return any(param == p or param.startswith((p + ".", p + "[")) for p in params)
