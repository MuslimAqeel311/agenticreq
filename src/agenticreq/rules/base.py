"""Rule extension point and the context every rule receives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Literal

from agenticreq.defaults import effective_tier2
from agenticreq.loader.yaml_loader import LoadedSpec
from agenticreq.models.finding import Finding, PassedCheck, SourceRef
from agenticreq.models.observed import ObservedAgent
from agenticreq.models.spec import AgentSpec

Mode = Literal["audit", "check", "validate"]
Group = Literal["validate", "conformance", "fidelity", "graph", "taint", "attest", "meta"]


@dataclass
class CheckContext:
    loaded: LoadedSpec | None = None
    observed: ObservedAgent | None = None
    today: date = field(default_factory=date.today)
    mode: Mode = "check"
    passed: list[PassedCheck] = field(default_factory=list)
    clauses_total: int = 0
    clauses_extracted: int = 0
    injected_allowlist: list[str] = field(default_factory=list)

    @property
    def spec(self) -> AgentSpec | None:
        return self.loaded.spec if self.loaded else None

    @property
    def risk_level(self) -> str:
        if self.spec and self.spec.risk_level:
            return self.spec.risk_level
        return "medium"

    def effective(self) -> dict[str, Any]:
        user = self.loaded.user_tier2 if self.loaded else {}
        return effective_tier2(self.risk_level, user)

    def spec_ref(self, *path: str | int) -> SourceRef | None:
        return self.loaded.ref(*path) if self.loaded else None

    def pass_(self, check: PassedCheck) -> None:
        self.passed.append(check)

    def clause(self, extracted: bool) -> None:
        """Count a clause for coverage; aggregate numbers use only the extracted ones."""
        self.clauses_total += 1
        if extracted:
            self.clauses_extracted += 1


RuleFn = Callable[[CheckContext], Iterable[Finding]]


@dataclass(frozen=True)
class RuleImpl:
    group: Group
    rule_ids: tuple[str, ...]
    fn: RuleFn
    modes: tuple[Mode, ...]


_IMPLS: list[RuleImpl] = []


def rule(group: Group, *rule_ids: str, modes: tuple[Mode, ...] = ("check",)) -> Callable[[RuleFn], RuleFn]:
    """Register a rule implementation. A plugin registers the same way after `REGISTRY.register`."""

    def deco(fn: RuleFn) -> RuleFn:
        _IMPLS.append(RuleImpl(group, rule_ids, fn, modes))
        return fn

    return deco


def implementations(mode: Mode, groups: Iterable[Group] | None = None) -> list[RuleImpl]:
    gs = set(groups) if groups is not None else None
    return [r for r in _IMPLS if mode in r.modes and (gs is None or r.group in gs)]


def run_rules(ctx: CheckContext, groups: Iterable[Group] | None = None) -> list[Finding]:
    out: list[Finding] = []
    for impl in implementations(ctx.mode, groups):
        out.extend(impl.fn(ctx))
    return out
