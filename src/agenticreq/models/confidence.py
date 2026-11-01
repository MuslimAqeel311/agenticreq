"""Confidence model — the foundation of every claim agenticreq makes.

Levels (strongest → weakest evidence):

* ``proven``                — read directly from config / wire / AST
* ``heuristic``             — inferred from structure (delegation edges, logging hooks)
* ``observed``              — seen in traces, carries ``n/N``; NEVER promoted to proven
* ``insufficient_evidence`` — traces exist but did not cover the scenario
* ``unknown``               — could not be extracted at all
* ``attestation``           — not verifiable from code; the user declares it

Hard rule: ``unknown`` and ``insufficient_evidence`` can never back a PASS.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class ConfidenceLevel(str, Enum):
    PROVEN = "proven"
    HEURISTIC = "heuristic"
    OBSERVED = "observed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNKNOWN = "unknown"
    ATTESTATION = "attestation"


#: Levels that may never be used to claim that a check passed.
NON_PASSABLE: frozenset[ConfidenceLevel] = frozenset({ConfidenceLevel.UNKNOWN, ConfidenceLevel.INSUFFICIENT_EVIDENCE})


class Confidence(BaseModel):
    """A confidence level, with ``n/N`` counts when the level is ``observed``."""

    model_config = ConfigDict(frozen=True)

    level: ConfidenceLevel
    n: int | None = None
    total: int | None = None

    @model_validator(mode="after")
    def _counts_only_for_observed(self) -> Confidence:
        has_counts = self.n is not None or self.total is not None
        if self.level is ConfidenceLevel.OBSERVED:
            if self.n is None or self.total is None:
                raise ValueError("observed confidence requires n and total (n/N)")
            if self.n < 0 or self.total < 0 or self.n > self.total:
                raise ValueError(f"invalid observed counts {self.n}/{self.total}")
        elif has_counts:
            raise ValueError(f"n/N counts are only valid for 'observed', not {self.level.value}")
        return self

    # -- constructors -------------------------------------------------------
    @classmethod
    def proven(cls) -> Confidence:
        return cls(level=ConfidenceLevel.PROVEN)

    @classmethod
    def heuristic(cls) -> Confidence:
        return cls(level=ConfidenceLevel.HEURISTIC)

    @classmethod
    def observed(cls, n: int, total: int) -> Confidence:
        return cls(level=ConfidenceLevel.OBSERVED, n=n, total=total)

    @classmethod
    def insufficient(cls) -> Confidence:
        return cls(level=ConfidenceLevel.INSUFFICIENT_EVIDENCE)

    @classmethod
    def unknown(cls) -> Confidence:
        return cls(level=ConfidenceLevel.UNKNOWN)

    @classmethod
    def attestation(cls) -> Confidence:
        return cls(level=ConfidenceLevel.ATTESTATION)

    # -- predicates ---------------------------------------------------------
    @property
    def can_pass(self) -> bool:
        return self.level not in NON_PASSABLE

    def __str__(self) -> str:
        if self.level is ConfidenceLevel.OBSERVED:
            return f"observed ({self.n}/{self.total})"
        return str(self.level.value)
