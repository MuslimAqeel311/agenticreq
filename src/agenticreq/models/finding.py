"""Finding, SourceRef, Report — the schema every other module depends on."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agenticreq.models.confidence import Confidence, ConfidenceLevel

Severity = Literal["blocker", "warning", "info", "attestation"]

RULE_ID_RE = re.compile(r"^AS-[1-9]\d{2}$")


class QualityAttribute(str, Enum):
    """The ten quality attributes findings are grouped under in reports."""

    GOAL_INTEGRITY = "Goal Integrity"
    BOUNDEDNESS = "Boundedness"
    LEAST_PRIVILEGE = "Least Privilege"
    AUTONOMY_SAFETY = "Autonomy Safety"
    HUMAN_OVERSIGHT = "Human Oversight"
    RECOVERABILITY = "Recoverability"
    TRUST_BOUNDARIES = "Trust Boundaries"
    DELEGATION_INTEGRITY = "Delegation Integrity"
    ACCOUNTABILITY = "Accountability"
    CHANGE_SAFETY = "Change Safety"


class Bucket(str, Enum):
    """Four internal buckets. Users see PASS / FAIL (models/verdict.py); the buckets stay for --verbose."""

    BLOCKERS = "BLOCKERS"
    WARNINGS = "WARNINGS"
    UNKNOWNS = "UNKNOWNS"
    ATTESTATIONS = "ATTESTATIONS"


class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        if self.line is None:
            return self.file
        return f"{self.file}:{self.line}"


class Finding(BaseModel):
    """One claim about the agent. rule_id, confidence and evidence are mandatory."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    severity: Severity
    confidence: Confidence
    title: str
    detail: str
    evidence: str = Field(min_length=1)
    spec_ref: SourceRef | None = None
    code_ref: SourceRef | None = None
    path: list[str] | None = None
    remediation: str | None = None
    fixable: bool = False
    #: numeric thresholds (agenticreq/thresholds.py): the exact PASS/FAIL line text ("missing: ..." / "not proven: ..."),
    #: the report line it belongs to (e.g. "bounds.max_steps") and the agent it is about (None = the whole run)
    reason: str | None = None
    param: str | None = None
    agent: str | None = None
    #: with ``reason``: the report group status that line takes (GAP | CHECK | SIGN-OFF), see models/verdict.py
    status_hint: str | None = None

    @field_validator("rule_id")
    @classmethod
    def _rule_id_format(cls, v: str) -> str:
        if not RULE_ID_RE.match(v):
            raise ValueError(f"rule_id must look like AS-NNN, got {v!r}")
        return v

    @field_validator("title")
    @classmethod
    def _title_one_line(cls, v: str) -> str:
        if not v.strip() or "\n" in v:
            raise ValueError("title must be a single non-empty line")
        return v

    @model_validator(mode="after")
    def _attestation_consistency(self) -> Finding:
        if self.severity == "attestation" and self.confidence.level is not ConfidenceLevel.ATTESTATION:
            raise ValueError("attestation severity requires attestation confidence")
        return self

    @property
    def bucket(self) -> Bucket:
        if self.severity == "attestation":
            return Bucket.ATTESTATIONS
        if not self.confidence.can_pass:
            # unknown / insufficient_evidence: not a pass, not a proven failure.
            return Bucket.UNKNOWNS
        if self.severity == "blocker":
            return Bucket.BLOCKERS
        return Bucket.WARNINGS

    @property
    def namespace(self) -> str:
        return self.rule_id[:4] + "xx"

    def location(self) -> SourceRef | None:
        return self.code_ref or self.spec_ref


class PassedCheck(BaseModel):
    """A check that was evaluated and did not fail.

    Constructing one with ``unknown`` / ``insufficient_evidence`` confidence raises:
    those levels can never become a PASS.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str
    subject: str
    confidence: Confidence
    note: str = ""
    #: numeric thresholds: the exact PASS line text, the report line and the agent it belongs to
    reason: str | None = None
    param: str | None = None
    agent: str | None = None

    @model_validator(mode="after")
    def _never_pass_without_evidence(self) -> PassedCheck:
        if not self.confidence.can_pass:
            raise ValueError(f"{self.rule_id} on {self.subject!r}: {self.confidence} can never be a PASS")
        return self

    @property
    def wording(self) -> str:
        # Traces prove violations, not safety.
        if self.confidence.level is ConfidenceLevel.OBSERVED:
            return f"no violation observed {self.confidence}"
        return f"pass ({self.confidence})"


class Coverage(BaseModel):
    """How much of the spec/agent was actually extracted. Counts are over this only."""

    total_clauses: int = 0
    extracted_clauses: int = 0

    @property
    def ratio(self) -> float:
        return self.extracted_clauses / self.total_clauses if self.total_clauses else 0.0


SpecIssueKind = Literal["invalid", "prose", "todo", "placeholder", "missing"]


class SpecIssue(BaseModel):
    """One problem with the spec file itself (not with the agent). Repeated instances of the same defect
    are one issue with ``items``. ``invalid`` stops `check` / `validate` before any rule runs; the other
    kinds leave the field undeclared (UNKNOWN), never satisfied."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    kind: SpecIssueKind
    field: str
    message: str
    expected: str = ""  # the exact shape
    example: str = ""  # a filled YAML example to paste
    items: list[str] = Field(default_factory=list)  # every offending path / value
    spec_ref: SourceRef | None = None
    #: an unknown field (AS-117): the real field(s) it is closest to; one entry = `validate --fix` may rename it
    did_you_mean: list[str] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return max(1, len(self.items))


SpecStatus = Literal["valid", "incomplete", "invalid"]


class Report(BaseModel):
    agent_id: str | None = None
    command: str = "check"
    findings: list[Finding] = Field(default_factory=list)
    passed: list[PassedCheck] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    metadata: dict[str, object] = Field(default_factory=dict)
    #: None when no spec was read (audit)
    spec_status: SpecStatus | None = None
    spec_issues: list[SpecIssue] = Field(default_factory=list)

    def by_bucket(self) -> dict[Bucket, list[Finding]]:
        out: dict[Bucket, list[Finding]] = {b: [] for b in Bucket}
        for f in self.findings:
            out[f.bucket].append(f)
        return out

    def counts(self) -> dict[str, int]:
        """Bucket counts plus pass count — all over the *extracted* portion only."""
        buckets = self.by_bucket()
        out = {b.value: len(v) for b, v in buckets.items()}
        out["PASSED"] = len(self.passed)
        return out

    @property
    def nothing_checked(self) -> bool:
        """True when no agent, tool or readable code was found and nothing else was reported: the run proved
        nothing, so it must not look like a pass."""
        a = self.metadata.get("analysis")
        if not isinstance(a, dict) or a.get("captures"):
            return False
        no_code = int(a.get("files") or 0) <= 0
        no_agent = not (a.get("framework") or a.get("tools") or a.get("agents"))
        real = [f for f in self.findings if not f.rule_id.startswith("AS-9")]
        return (no_code or no_agent) and not real and not self.passed

    @property
    def unknown_blockers(self) -> list[Finding]:
        """UNKNOWN findings on a rule whose severity is blocker: a rule that could have failed the run but
        could not be decided. `--fail-on unknown` treats them as failures."""
        return [f for f in self.by_bucket()[Bucket.UNKNOWNS] if f.severity == "blocker"]

    @property
    def exit_code(self) -> int:
        """2 = the spec is invalid (nothing was checked), 1 = any result is FAIL (missing or not proven),
        0 = every result is PASS. See :mod:`agenticreq.models.verdict`.

        `validate` lints the spec, not the agent, and keeps its own rule: 1 only for a proven spec blocker. A TODO
        or a prose field leaves a field undeclared, so a fresh `init` scaffold still validates; `check` fails on it."""
        from agenticreq.models.verdict import exit_code

        if self.spec_status == "invalid":
            return 2
        if self.command == "validate":
            return 1 if self.by_bucket()[Bucket.BLOCKERS] else 0
        return exit_code(self)
