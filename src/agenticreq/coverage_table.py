"""`agenticreq check --coverage`: every parameter (Level 1, 2, 3), its status and where its value came from.

Nothing is left out: a parameter with no result says why (not applicable: single agent; sign-off: a person
confirms it). The table ends with "N of M parameters checked; K not applicable; J confirmed by a person".
"""

from __future__ import annotations

from dataclasses import dataclass

from agenticreq.loader.yaml_loader import LoadedSpec
from agenticreq.models.finding import Report
from agenticreq.models.verdict import _RANK, ResultLine, results

STATUS_WORD = {
    "GAP": "gap",
    "SIGN-OFF": "sign-off",
    "SIGNED": "signed",
    "NOT NEEDED": "not-needed",
    "CHECK": "mock-run",
    "PASS": "passed",
}


@dataclass(frozen=True)
class Row:
    name: str
    level: int
    status: str  # gap | sign-off | signed | not-needed | mock-run | passed | not-applicable | not-checked
    source: str  # spec | default | code
    why: str = ""


def _mine(name: str, lines: list[ResultLine]) -> list[ResultLine]:
    return [ln for ln in lines if ln.param == name or ln.param.startswith((name + ".", name + "["))]


def rows(report: Report, loaded: LoadedSpec | None) -> list[Row]:
    from agenticreq.params import PARAMS

    lines = results(report)
    raw = loaded.raw if loaded is not None else {}
    written = set(raw) | set(loaded.not_needed if loaded is not None else {})
    agents = report.metadata.get("agents")
    multi = bool(loaded and loaded.spec and loaded.spec.is_multi_agent) or (
        isinstance(agents, list) and len(agents) > 1
    )
    out = []
    for p in PARAMS:
        if p.level > 3:
            continue
        source = "spec" if p.name in written else "default" if p.level in (2, 3) else "code"
        mine = _mine(p.name, lines)
        if mine:
            worst = min(mine, key=lambda ln: _RANK[ln.status])
            out.append(Row(p.name, p.level, STATUS_WORD[worst.status], source))
        elif not p.rules:
            out.append(Row(p.name, p.level, "not-checked", source, "a person confirms it under confirmed:"))
        elif p.level == 3 and not multi:
            out.append(Row(p.name, p.level, "not-applicable", source, "single agent"))
        else:
            why = (
                "not in spec.yml, and nothing in the code to check"
                if p.name not in written
                else "nothing in the code it applies to"
            )
            out.append(Row(p.name, p.level, "not-applicable", source, why))
    return out


def summary(table: list[Row]) -> str:
    na = sum(1 for r in table if r.status == "not-applicable")
    nc = sum(1 for r in table if r.status == "not-checked")
    checked = len(table) - na - nc
    return f"{checked} of {len(table)} parameters checked; {na} not applicable; {nc} confirmed by a person"


def render(table: list[Row]) -> str:
    from agenticreq.subfields import sign_off_only

    w = max(len(r.name) for r in table) + 2
    lines = [f"{'parameter'.ljust(w)}level  {'status'.ljust(15)}{'from'.ljust(9)}why", ""]
    for r in table:
        lines.append(f"{r.name.ljust(w)}{str(r.level).ljust(7)}{r.status.ljust(15)}{r.source.ljust(9)}{r.why}".rstrip())
    lines += ["", summary(table), "sub-fields a person confirms (no code check): " + ", ".join(sign_off_only())]
    return "\n".join(lines) + "\n"
