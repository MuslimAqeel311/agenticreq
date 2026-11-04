"""Every number in a spec comes from the user. The library never proposes, defaults or recommends one.

It does two things with a number: it compares the number in spec.yml with the number in the code, and it shows a
number FOUND in the user's code as plain evidence ("Found in code: 25"). A number fails only when it is not defined,
or when code and spec.yml disagree. The user may write any positive value at any risk level.

``judge`` is the one decision table every numeric parameter follows:

========================================  ======  =====================================================================
situation                                 result  reason
========================================  ======  =====================================================================
spec and code, code at least as strict    PASS    Found in code: <value>.
code only                                 PASS    Found in code: <value>.
spec and code, code looser                GAP     Code allows <code>; spec.yml says <spec>.
spec only, library looked, code has none  GAP     spec.yml says <spec>; the code sets no limit.
nowhere                                   GAP     No <name> set: write a value, or not_needed with a reason.
spec is TODO                              GAP     <name> is still TODO in spec.yml.
spec only, library cannot read the limit  CHECK   spec.yml says <spec>; not readable in code (or sign it off ...
  ... signed under confirmed: / seen        SIGNED  Signed by <who>.  /  PASS  Seen when the agent ran.
========================================  ======  =====================================================================

"At least as strict" is lower for ``max_*`` values and higher for ``confidence_threshold``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck, Severity, SourceRef
from agenticreq.models.verdict import MOCK_RUN

if TYPE_CHECKING:
    from agenticreq.models.spec import Attestation
    from agenticreq.rules.base import CheckContext

Row = Literal["match", "code_only", "looser", "no_limit", "nowhere", "todo", "unreadable", "confirmed", "observed"]

_RPM_KEY = re.compile(r"^(llm_)?(requests?|calls?)_per_minute$|^rpm$", re.IGNORECASE)


@dataclass(frozen=True)
class Numeric:
    """One numeric parameter of a spec."""

    key: str  # where it lives in spec.yml; also the name of its report line
    rule_id: str  # rule a missing / not proven result is filed under
    differs_rule_id: str = ""  # rule a code-vs-spec disagreement is filed under (default: rule_id)
    name_: str = ""  # the name used in reasons (default: the last part of key)
    higher_is_stricter: bool = False
    #: True when the library looks for this limit in code, so "code sets no limit" is something it established.
    #: False for limits that live in a runtime, gateway or process (progress guard, confidence, approval timeout).
    looked: bool = True
    clauses: tuple[str, ...] = ()  # confirmed: names that confirm it, besides the key and the name
    unit: str = ""  # what the number counts, in a reason ("Code allows 60 steps; spec.yml says 25.")

    @property
    def name(self) -> str:
        return self.name_ or self.key.rsplit(".", 1)[-1]

    @property
    def differs(self) -> str:
        return self.differs_rule_id or self.rule_id

    @property
    def attest_names(self) -> tuple[str, ...]:
        parent = self.key.split(".")[0].replace("[]", "")
        return tuple(dict.fromkeys((self.key, self.name, parent, *self.clauses)))


MAX_STEPS = Numeric("bounds.max_steps", "AS-208", "AS-209", unit="steps")
# cost and duration are read from code in few frameworks (a UsageLimits / wait_for proves that a limit exists, not
# its value), so a spec number with no readable value in code is "not proven", never "code sets no limit"
MAX_COST = Numeric("bounds.max_cost_usd", "AS-221", looked=False)
MAX_DURATION = Numeric("bounds.max_duration_s", "AS-222", looked=False)
MAX_RETRIES = Numeric("bounds.max_retries", "AS-209", unit="retries")
MAX_TOOL_CALLS = Numeric("bounds.max_tool_calls", "AS-209", unit="tool calls")
MAX_DEPTH = Numeric("max_delegation_depth", "AS-302")
NO_PROGRESS = Numeric("progress_guard.max_no_progress_steps", "AS-228", looked=False)
MAX_CHARS = Numeric("output_size_limits.max_chars", "AS-213", looked=False)
CONFIDENCE = Numeric("confidence_threshold", "AS-501", higher_is_stricter=True, looked=False)
RATE_LIMITS = Numeric("rate_limits", "AS-231", looked=False, clauses=("requests_per_minute",))
AGG_COST = Numeric("aggregate_budget.max_cost_usd", "AS-306", name_="aggregate_budget.max_cost_usd", looked=False)
AGG_STEPS = Numeric("aggregate_budget.max_steps", "AS-306", name_="aggregate_budget.max_steps", unit="steps")
AGG_TOOLS = Numeric("aggregate_budget.max_tool_calls", "AS-306", name_="aggregate_budget.max_tool_calls", looked=False)
APPROVAL_TIMEOUT = Numeric(
    "approval_required[].timeout",
    "AS-205",
    name_="approval_required[].timeout",
    looked=False,
    clauses=("approval_timeout",),
)

#: every numeric parameter a user's agent is compared against (`agenticreq params` lists them by name)
NUMERIC_PARAMS: tuple[Numeric, ...] = (
    MAX_STEPS,
    MAX_COST,
    MAX_DURATION,
    MAX_RETRIES,
    MAX_TOOL_CALLS,
    MAX_DEPTH,
    NO_PROGRESS,
    MAX_CHARS,
    CONFIDENCE,
    RATE_LIMITS,
    AGG_COST,
    AGG_STEPS,
    AGG_TOOLS,
    APPROVAL_TIMEOUT,
)
BY_KEY: dict[str, Numeric] = {p.key: p for p in NUMERIC_PARAMS}
RISKS: tuple[str, ...] = ("low", "medium", "high", "critical")


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@dataclass(frozen=True)
class Decision:
    row: Row
    status: Literal["PASS", "FAIL"]
    kind: Literal["missing", "not_proven"] | None
    reason: str

    @property
    def line_status(self) -> str:
        """The report group status (models/verdict.py): a gap, a check a mock run can settle, signed, or PASS."""
        if self.row == "confirmed":
            return "SIGNED"
        if self.status == "PASS":
            return "PASS"
        return "CHECK" if self.row == "unreadable" else "GAP"


def judge(
    p: Numeric,
    risk: str | None,
    *,
    spec: Any = None,
    spec_todo: bool = False,
    code: float | None = None,
    code_present: bool = False,
    looked: bool | None = None,
    confirmed: bool = False,
    observed_ok: bool = False,
    several: str | None = None,
) -> Decision:
    """The decision table above. ``code`` is the value read from code (None = none readable), ``code_present`` says a
    limit of this kind exists in code but its value cannot be read, ``looked`` overrides ``p.looked``.
    A limit that exists in code with an unreadable value and no spec value is the caller's to report."""
    look = p.looked if looked is None else looked
    unit = f" {p.unit}" if p.unit else ""
    if spec_todo:
        return Decision("todo", "FAIL", "missing", f"{p.name} is still TODO in spec.yml.")
    if code is not None:
        if spec is None:
            return Decision("code_only", "PASS", None, f"Found in code: {fmt(code)}.")
        ok = code >= spec if p.higher_is_stricter else code <= spec
        if ok:
            return Decision("match", "PASS", None, f"Found in code: {fmt(code)}.")
        return Decision("looser", "FAIL", "missing", f"Code allows {fmt(code)}{unit}; spec.yml says {fmt(spec)}.")
    if spec is not None:
        if several and not confirmed:  # the code sets it, in more than one place with different values
            return Decision(
                "unreadable",
                "FAIL",
                "not_proven",
                f"spec.yml says {fmt(spec)}; {several} (or sign it off under confirmed:). Run: {MOCK_RUN}",
            )
        if look and not code_present:
            return Decision("no_limit", "FAIL", "missing", f"spec.yml says {fmt(spec)}; the code sets no limit.")
        if observed_ok:
            return Decision("observed", "PASS", None, "Seen when the agent ran.")
        if confirmed:
            return Decision("confirmed", "PASS", None, "Signed by you.")
        return Decision("unreadable", "FAIL", "not_proven", unreadable_reason(spec, code_present))
    if confirmed and not look:
        return Decision("confirmed", "PASS", None, "Signed by you.")
    return Decision("nowhere", "FAIL", "missing", f"No {p.name} set: write a value, or not_needed with a reason.")


def unreadable_reason(spec: Any, code_present: bool = False) -> str:
    """``code_present``: a limit of this kind is in code, but its value is not (an env var, a settings object)."""
    if code_present:
        return (
            f"spec.yml says {fmt(spec)}; the code sets this limit from a value agenticreq cannot read (or sign it "
            f"off under confirmed:). Run: {MOCK_RUN}"
        )
    return f"spec.yml says {fmt(spec)}; not readable in code (or sign it off under confirmed:). Run: {MOCK_RUN}"


# ---------------------------------------------------------------- reading the spec
def _number(v: Any) -> float | int | None:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def spec_value(ctx: CheckContext, p: Numeric) -> float | int | None:
    """The number the user wrote in spec.yml for this parameter, else None (audit has no spec)."""
    if not ctx.loaded:
        return None
    cur: Any = ctx.loaded.user_tier2
    if p is RATE_LIMITS:
        cur = cur.get("rate_limits") if isinstance(cur, dict) else None
        if not isinstance(cur, dict):
            return None
        return next((_number(v) for k, v in cur.items() if _RPM_KEY.match(str(k)) and _number(v) is not None), None)
    for part in p.key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return _number(cur)


def spec_todo(ctx: CheckContext, p: Numeric) -> bool:
    """The user left TODO in this field (or in the whole mapping it lives in)."""
    if not ctx.loaded:
        return False
    parts = tuple(p.key.replace("[]", "").split("."))
    for path in ctx.loaded.todos:
        names = tuple(str(x) for x in path if not isinstance(x, int))
        if names and (names == parts[: len(names)] or parts == names[: len(parts)]):
            return True
    return False


def attestation(ctx: CheckContext, p: Numeric) -> Attestation | None:
    """An unexpired ``confirmed:`` entry that confirms this number (by its key, its name or its parent field)."""
    if not ctx.spec:
        return None
    from agenticreq.signoff import counts

    names = set(p.attest_names)
    root = ctx.observed.root if ctx.observed is not None else None
    return next((a for a in ctx.spec.confirmed if a.clause in names and counts(a, ctx.today, root)), None)


def todo_key(path_text: str) -> str | None:
    """'bounds.max_steps' / 'approval_required.0.timeout' (a TODO path in AS-101's evidence) -> the numeric key."""
    norm = re.sub(r"\.\d+(?=\.|$)", "[]", path_text)
    if norm.split(".")[0] == "rate_limits":  # rate_limits.requests_per_minute / rate_limits itself
        return "rate_limits"
    return norm if norm in BY_KEY else None


def timeout_of(a: Any) -> str | None:
    t = getattr(a, "timeout", None)
    return t if isinstance(t, str) and t.strip() else None


# ---------------------------------------------------------------- turning a decision into a result
def _confidence(row: Row) -> Confidence:
    if row == "looser":
        return Confidence.proven()
    if row in ("no_limit", "nowhere"):
        return Confidence.heuristic()  # absence is inferred from what the extraction could read
    if row == "unreadable":
        return Confidence.attestation()
    return Confidence.unknown()


def emit(
    ctx: CheckContext,
    p: Numeric,
    d: Decision,
    *,
    risk: str | None,
    code: float | None = None,
    spec: Any = None,
    code_ref: SourceRef | None = None,
    spec_ref: SourceRef | None = None,
    agent: str | None = None,
    evidence: str = "",
    rule_id: str | None = None,
    attested: Attestation | None = None,
    where: str | None = None,
) -> list[Finding]:
    """PASS -> recorded on ``ctx``; FAIL -> returned as a finding carrying the exact report line. A number found in
    code is printed with the file:line it was read from (``where`` or ``code_ref``), or not at all."""
    from agenticreq.rules.registry import REGISTRY, make_finding

    ev = (
        evidence
        or f"{p.key}: spec={fmt(spec) if spec is not None else 'none'}, "
        f"code={fmt(code) if code is not None else 'none'}"
    )
    if d.row in ("match", "code_only"):
        at = where or (str(code_ref) if code_ref else None)
        ctx.pass_(
            PassedCheck(
                rule_id=rule_id or p.differs,
                subject=fmt(code),
                confidence=Confidence.proven(),
                param=p.key,
                agent=agent,
                note=at or "",
                reason=None if at else "Limit found in code.",
            )
        )
        return []
    if d.row == "confirmed":
        by = f" by {attested.by}" if attested else " by you"
        on = f" on {attested.date}" if attested and attested.date else ""
        return [
            make_finding(
                "AS-502",
                confidence=Confidence.attestation(),
                spec_ref=spec_ref,
                param=p.key,
                agent=agent,
                title=f"{p.name} signed{by}",
                evidence=ev,
                reason=f"Signed{by}{on}.",
                detail=f"{p.key} = {fmt(spec) if spec is not None else 'not set'}; signed{by}{on}. "
                "agenticreq records the sign-off; it does not check it.",
            )
        ]
    if d.row == "observed":
        return []  # recorded by the runtime rule that saw the runs
    text = d.reason.rstrip(".")
    rid = rule_id or (p.differs if d.row == "looser" else p.rule_id)
    conf = _confidence(d.row)
    severity: Severity | None = None
    if d.row == "unreadable":
        rid, severity = "AS-501", "attestation"  # a person has to confirm it: the path every unreadable clause takes
    elif REGISTRY.get(rid).severity == "attestation":
        severity = "warning"  # an attestation-class rule reporting something that is not an attestation
    return [
        make_finding(
            rid,
            confidence=conf,
            severity=severity,
            spec_ref=spec_ref,
            code_ref=code_ref,
            title=text.split(". ")[0],
            detail=f"{p.key}: {text}",
            evidence=ev,
            reason=d.reason,
            param=p.key,
            agent=agent,
            status_hint=d.line_status,
        )
    ]
