"""The four result groups — what the user sees. Every result line lands in exactly one of them:

==========================  ===================  ==========================================================================
group (``group`` in JSON)   status               what goes here
==========================  ===================  ==========================================================================
Spec vs code gaps (gap)     GAP                  the spec says X and the code does not do it, or a required spec field is
                                                 missing or TODO (a BLOCKER or WARNING finding, any spec check AS-1xx)
Human sign-off (sign_off)   SIGN-OFF / SIGNED    things code cannot show, which a person confirms under ``confirmed:`` in
                                                 spec.yml. SIGNED once confirmed; never counted as passed
Check with a mock run       CHECK                things a run of the agent could prove: a code-pattern guess, a tool that
(mock_run)                                       never reached the model, a number not readable from code, a rule that
                                                 held in only some runs
Passed (passed)             PASS                 proven in code, seen in every captured mock run, or waived by you
==========================  ===================  ==========================================================================

The internal confidence levels and buckets stay as they are; this module only maps them. Results are shown one line
per (agent, parameter), named from the params catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.models.finding import Finding, PassedCheck, Report, SourceRef

Status = Literal["GAP", "SIGN-OFF", "SIGNED", "NOT NEEDED", "CHECK", "PASS"]
Group = Literal["gap", "sign_off", "mock_run", "passed"]
FailOn = Literal["gaps", "all", "never"]

#: the groups in report order, with their exact titles
GROUPS: tuple[Group, ...] = ("gap", "sign_off", "mock_run", "passed")
GROUP_TITLE: dict[Group, str] = {
    "gap": "Spec vs code gaps",
    "sign_off": "Human sign-off",
    "mock_run": "Check with a mock run",
    "passed": "Passed",
}
GROUP_OF: dict[Status, Group] = {
    "GAP": "gap",
    "SIGN-OFF": "sign_off",
    "SIGNED": "sign_off",
    "NOT NEEDED": "sign_off",
    "CHECK": "mock_run",
    "PASS": "passed",
}
#: results that are evidence from the code itself: shown next to a sign-off item as what the person can check
_CODE_EVIDENCE = ("Code pattern only", "Found in code")
_STATUS: dict[str, Status] = {
    "GAP": "GAP",
    "SIGN-OFF": "SIGN-OFF",
    "SIGNED": "SIGNED",
    "NOT NEEDED": "NOT NEEDED",
    "CHECK": "CHECK",
    "PASS": "PASS",
}
#: which status a line shows when several results share it (lower wins). A sign-off outranks a code-pattern check on
#: the same parameter: code cannot prove a sign-off item, so what the code shows is supporting evidence for it.
_RANK: dict[Status, int] = {"GAP": 0, "SIGN-OFF": 1, "SIGNED": 2, "NOT NEEDED": 3, "CHECK": 4, "PASS": 5}

#: findings that say what evidence the run lacked, never that the code is wrong
_EVIDENCE_RULES = frozenset(
    {"AS-201", "AS-901", "AS-902", "AS-903", "AS-904", "AS-905", "AS-906", "AS-907", "AS-908", "AS-909"}
)

#: rules listed under several parameters in the catalog: the one the user should read the line as
_PREFERRED = {
    "AS-110": "on_approval_timeout",
    "AS-111": "risk_level",
    "AS-114": "declared_delegation",
    "AS-213": "output_size_limits",
    "AS-215": "sandbox_policy",
    "AS-216": "egress_policy",
    "AS-225": "audit_log",
    "AS-304": "transitive_prohibition",
    "AS-401": "untrusted_inputs",
    "AS-402": "untrusted_inputs",
    "AS-905": "trace_coverage",
}
#: rules no catalog parameter lists: spec checks and run-level evidence notes
_UNLISTED = {
    "AS-100": "spec",
    "AS-106": "spec",
    "AS-115": "spec",
    "AS-117": "spec",
    "AS-118": "spec",
    "AS-218": "mcp_tool_safety",
    "AS-608": "tool_schema_strictness",
    "AS-901": "execution_model",
    "AS-902": "extraction",
    "AS-903": "runtime_evidence",
    "AS-904": "framework",
    "AS-906": "wire_format",
    "AS-907": "business_rules",
    "AS-908": "framework_version",
    "AS-909": "agent_registry",
    "AS-910": "runtime_agents",
}

_MAX_REASON = 160
#: the next command at the end of every "Check with a mock run" line
MOCK_RUN = "agenticreq mock-run"
_CONFIRM = "Confirm under confirmed: in spec.yml."

#: what a person confirms for each sign-off item, as the question they answer
SIGN_OFF_QUESTION: dict[str, str] = {
    "owner": "Who owns this agent?",
    "escalation": "Who handles hard cases?",
    "data_rules": "Are retention and never-store rules applied in storage?",
    "out_of_scope": "Does the agent stay out of the listed topics?",
    "log_immutable": "Is the log store write-once?",
    "kill_switch": "Can the agent be stopped at once in production?",
    "rollback_path": "Can a release be rolled back?",
    "environment_separation": "Are test and production credentials kept apart?",
    "drift_monitors": "Do monitors watch for drift?",
    "regression_triggers": "Do changes trigger the regression suite?",
    "audit_log": "Are tool calls logged where you can read them later?",
    "secrets_policy": "Are secrets kept in a secret store?",
    "termination_authority": "Who may stop the agent?",
    "cross_agent_trust": "How far do the agents trust each other?",
    "context_isolation": "Is each agent's context kept apart?",
    "failure_isolation": "Does one failing agent leave the others running?",
    "conflict_resolution": "Who decides when agents disagree?",
    "concurrency_policy": "How many runs may happen at once?",
    "stop_conditions": "Does the agent stop when the stop conditions are met?",
    "progress_guard": "Does a run stop when it makes no progress?",
    "completion_verification": "Is the result checked against done_criteria?",
    "goal_drift_guard": "Is drift from the goal caught?",
    "tool_result_validation": "Are tool results validated before use?",
    "tool_error_contract": "Do tools return errors in the agreed shape?",
    "read_write_separation": "Are reads and writes done with separate credentials?",
    "context_policy": "Is the must-never-drop context always kept?",
    "memory_policy": "Are memory writes validated?",
    "fallback_policy": "Does the fallback behave as written?",
    "rate_limits": "Are the rate limits enforced by a gateway or provider?",
    "identity_propagation": "Does the end user's identity reach the tools?",
    "clarification_policy": "Does the agent ask when the request is unclear?",
    "confidence_threshold": "Is the confidence threshold applied?",
    "handoff_contract": "Is the handoff payload checked?",
}


@dataclass(frozen=True)
class Verdict:
    status: Status
    reason: str
    #: NOT NEEDED with no reason (or an incomplete confirmed: entry): still counted as left
    open: bool = False

    @property
    def group(self) -> Group:
        return GROUP_OF[self.status]


@dataclass(frozen=True)
class ResultLine:
    """One line of the report: one parameter of one agent."""

    agent: str
    param: str
    status: Status
    reason: str
    rule_ids: tuple[str, ...]
    locations: tuple[str, ...]
    #: what the code shows for a sign-off item (a code-pattern result on the same parameter), else ""
    support: str = ""
    #: NOT NEEDED with no reason given: still counted as left
    open: bool = False

    @property
    def group(self) -> Group:
        return GROUP_OF[self.status]


def _short(text: str, n: int = _MAX_REASON) -> str:
    """``text`` on one line. Never cut: a reason is shown whole (the console wraps it)."""
    return " ".join(text.split())


def _lower_first(text: str) -> str:
    # "Write tool x runs ..." -> "write tool x runs ...", but keep "MCP", "AS-203", "Model 'GPT'"-style words
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def sentence(text: str) -> str:
    """``text`` as a sentence: a capital first letter (unless it starts with a code name such as ``max_steps``)
    and a full stop."""
    text = " ".join(text.split()).rstrip()
    if not text:
        return text
    first = text.split(" ", 1)[0]
    if first.isalpha() and first.islower():
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!:" or text.endswith(MOCK_RUN) else text + "."


def _what(f: Finding) -> str:
    """The finding in a few words. A rule's generic title ("Prohibited tool is reachable in code") does not say
    which tool, so the first line of the detail is added."""
    from agenticreq.rules.registry import REGISTRY

    what = _lower_first(f.title)
    if f.rule_id in REGISTRY and f.title == REGISTRY.get(f.rule_id).title and f.detail.strip():
        what += ": " + f.detail.strip().splitlines()[0]
    return _untagged(what)


def _gap_reason(f: Finding) -> str:
    return sentence(_short(_what(f)))


_NEXT = re.compile(r"(?:,? then run|\. Run): agenticreq [\w-]+$")


def with_note(reason: str, note: str) -> str:
    """``reason`` with ``note`` added, keeping the next command (``... then run: agenticreq mock-run``) last."""
    m = _NEXT.search(reason)
    if not m:
        return f"{reason} {note}"
    head = reason[: m.start()]
    head = head[:-1] if head.endswith(".") and not head.endswith("...") else head  # keep a cut's "..."
    return f"{head} {note}{'.' if m.group(0).startswith('.') else ''}{reason[m.start() :].removeprefix('.')}"


def _then_run(text: str, step: str = "") -> str:
    """A mock-run line: what is open, what to do, and the exact next command last."""
    text = sentence(_short(text)).rstrip(".")
    return f"{text}. {step + ', then run' if step else 'Run'}: {MOCK_RUN}"


def _check_reason(f: Finding) -> str:
    """A result a mock run could decide, in the words the user acts on, ending with the next command."""
    text = f"{f.title} {f.detail} {f.evidence}".lower()
    what = _lower_first(f.title)
    if f.rule_id == "AS-201":
        return _then_run("Never reached the model", "Add it to order in mocks.yml")
    if f.rule_id == "AS-903":
        return _then_run("The agent has not run yet, so nothing was seen at run time")
    if f.rule_id == "AS-907":
        return "No spec given, so the Level 1 rules did not run. Run: agenticreq init"
    if f.rule_id in ("AS-905", "AS-228", "AS-229", "AS-230", "AS-231"):
        return _then_run(what, "Run the agent so its calls are seen")
    if f.rule_id == "AS-210" and ("run time" in text or "env" in text):
        return (
            f"{sentence(what).rstrip('.')} (the model name comes from an env var or config). Give it a pinned "
            f"default in code (os.getenv('MODEL', 'gpt-4o-2024-08-06')) or confirm the deployed value under "
            f"confirmed:."
        )
    if (
        "mcp" in text
        or "toolset" in text
        or "registry" in text
        or "could not be resolved" in text
        or "unresolved" in text
    ):
        return _then_run(
            f"{what} (the tools come from an MCP server or a toolset)",
            "Pass --mcp-tools tools.json, or let the mock run list them",
        )
    if f.confidence.level is ConfidenceLevel.INSUFFICIENT_EVIDENCE:
        return _then_run(f"The mock runs did not cover this: {what}", "Add it to order in mocks.yml")
    if f.confidence.level is ConfidenceLevel.HEURISTIC:
        return _then_run(f"Code pattern only: {what}", "Add a mock that exercises it")
    return _then_run(what)


def _sign_off_reason(param: str) -> str:
    q = SIGN_OFF_QUESTION.get(param.split(".")[0])
    return f"{q} {_CONFIRM}" if q else f"Code cannot show this. {_CONFIRM}"


def is_spec_gap(f: Finding) -> bool:
    """A spec check (AS-1xx): a required field missing or TODO, prose, a placeholder, a contradiction. The spec has to
    change, so it is a gap whatever the confidence."""
    return f.rule_id.startswith("AS-1")


def finding_verdict(f: Finding, param: str = "") -> Verdict:
    if f.rule_id == "AS-502":
        return Verdict("SIGNED", f.reason or "Signed by you.")
    if f.reason:  # a numeric threshold (thresholds.py) wrote its own line, and its group
        return Verdict(_STATUS.get(f.status_hint or "", "GAP"), f.reason)
    if f.rule_id == "AS-106" and f.param:  # a sign-off that expired: open again, on its own line
        return Verdict("SIGN-OFF", sentence(f"Signed before, but {_lower_first(f.detail)}") + " " + _CONFIRM)
    if is_spec_gap(f):
        return Verdict("GAP", _gap_reason(f) if f.confidence.can_pass else sentence(_short(_lower_first(f.title))))
    if f.severity == "attestation":
        again = ""
        m = re.search(r"Signed before, but (.+?)\.$", f.detail.strip())
        if m:
            again = f"Signed before, but {m.group(1)}. "
        return Verdict("SIGN-OFF", again + _sign_off_reason(param or (f.param or "")))
    if not f.confidence.can_pass or f.rule_id in _EVIDENCE_RULES:
        return Verdict("CHECK", _check_reason(f))
    return Verdict("GAP", _gap_reason(f))


#: the framework/version tag rules attach for the JSON evidence (ObservedAgent.framework_tag): "[langgraph 1.2.18
#: (runtime)]". Not a reason, so the report lines leave it out
_FRAMEWORK_TAG = re.compile(
    r"\s*\[[\w.-]+ (?:[\w.+!-]+ \([\w ]+\)|\(version unresolved\))(?: \[declared, not imported\])?\]"
)


def _untagged(text: str) -> str:
    return _FRAMEWORK_TAG.sub("", text)


def passed_verdict(p: PassedCheck) -> Verdict:
    if p.reason:
        return Verdict("PASS", p.reason)
    what = p.subject + (f" ({_untagged(p.note)})" if p.note else "")
    if p.confidence.level is ConfidenceLevel.PROVEN:
        return Verdict("PASS", sentence("Found in code: " + _short(what, _MAX_REASON - 15)))
    if p.confidence.level is ConfidenceLevel.OBSERVED:
        # observed (n/N) on a pass counts the runs that violated the rule: it held in N - n of N
        total = p.confidence.total or 0
        held = total - (p.confidence.n or 0)
        if total and held == total:
            return Verdict("PASS", sentence(f"Held in {held} of {total} runs: " + _short(what, _MAX_REASON - 50)))
        return Verdict(
            "CHECK",
            _then_run(
                f"Held in only {held} of {total} runs: " + _short(what, _MAX_REASON - 60),
                "Add mocks for the other path",
            ),
        )
    if p.confidence.level is ConfidenceLevel.ATTESTATION:
        return Verdict("SIGNED", "Signed by you.")
    # a code pattern suggests the rule holds, but nothing proves it
    if p.rule_id == "AS-219":  # error handling: only a tool that fails shows whether the error is surfaced
        return Verdict(
            "CHECK", _then_run(f"Code pattern only ({p.note or p.subject})", "Add a mock that returns an error")
        )
    return Verdict(
        "CHECK", _then_run("Code pattern only: " + _short(what, _MAX_REASON - 40), "Add a mock that exercises it")
    )


# --------------------------------------------------------------------------- parameter names
_ATTEST_TITLE = re.compile(r"^'?([a-z_]+)'? (?:is declared|signed by)")
_NEVER_REACHED = re.compile(r"^(\S+) never reached the model")
_DEFAULTS_LIST = re.compile(r"any of them: (.+)\.$")


def _catalog() -> dict[str, list[str]]:
    from agenticreq.params import PARAMS

    out: dict[str, list[str]] = {}
    for p in PARAMS:
        for r in p.rules:
            out.setdefault(r, []).append(p.name)
    return out


def _param_order() -> dict[str, int]:
    from agenticreq.params import PARAMS
    from agenticreq.thresholds import NUMERIC_PARAMS

    order = {p.name: i for i, p in enumerate(PARAMS)}
    for n in NUMERIC_PARAMS:  # bounds.max_cost_usd sorts where `bounds` does, in the order the thresholds list them
        parent = n.key.split(".")[0].replace("[]", "")
        order.setdefault(n.key, order.get(parent, len(order)))
    return order


_TODO_FIELD = re.compile(r"([A-Za-z_][\w.\[\]]*) \(line \d+\)")


def _todo_fields(f: Finding) -> list[str]:
    """AS-101 "N fields still TODO": the fields, from its evidence ("owner (line 16); goal (line 3)"). A TODO number
    keeps its full name (``bounds.max_steps``) so it reads as that parameter's own line."""
    from agenticreq.thresholds import todo_key

    if f.rule_id != "AS-101" or "TODO" not in f.title:
        return []
    out = []
    for m in _TODO_FIELD.finditer(f.evidence):
        out.append(todo_key(m.group(1)) or re.split(r"[.\[]", m.group(1))[0])
    return list(dict.fromkeys(out))


def finding_parts(f: Finding, command: str = "check") -> list[tuple[str, Verdict]]:
    """(parameter, verdict) for each line a finding contributes. Most findings are one line; the AS-501 summary of
    built-in safeguards and an AS-101 list of TODO fields name several parameters, and each gets its own line."""
    todo = _todo_fields(f)
    if todo:
        from agenticreq.thresholds import BY_KEY

        return [
            (
                t,
                Verdict(
                    "GAP",
                    f"{BY_KEY[t].name} is still TODO in spec.yml."
                    if t in BY_KEY
                    else f"{t} is still TODO in spec.yml.",
                ),
            )
            for t in todo
        ]
    return [(p, finding_verdict(f, p)) for p in params_of_finding(f, command)]


def params_of_finding(f: Finding, command: str = "check") -> list[str]:
    """The catalog parameter(s) a finding is about."""
    if f.param:
        return [f.param]
    if f.rule_id == "AS-201":  # a tool that never reached the model: the tool's own line
        m = _NEVER_REACHED.match(f.title)
        if m:
            return [m.group(1)]
    if f.rule_id in ("AS-501", "AS-502"):
        m = _ATTEST_TITLE.match(f.title)
        if m:
            return [m.group(1)]
        lst = _DEFAULTS_LIST.search(f.detail.strip())
        if lst:
            return [c.strip() for c in lst.group(1).split(",") if c.strip()]
        return ["confirmed"]
    return [_param_of(f.rule_id, f"{f.title} {f.detail} {f.evidence}", command)]


#: `audit` reads no spec, so a result is never named after a Level-1 field only a spec declares
_AUDIT_NAMES = {"AS-401": "trust_boundary", "AS-402": "trust_boundary"}


def _param_of(rule_id: str, text: str, command: str = "check") -> str:
    if command == "audit" and rule_id in _AUDIT_NAMES:
        return _AUDIT_NAMES[rule_id]
    if rule_id in _PREFERRED:
        return _PREFERRED[rule_id]
    if rule_id in _UNLISTED:
        return _UNLISTED[rule_id]
    names = _catalog().get(rule_id, [])
    if len(names) == 1:
        return names[0]
    for n in names:  # AS-102 "Required Tier-1 field is missing": the field it names
        if re.search(rf"\b{re.escape(n)}\b", text):
            return n
    return "spec" if rule_id.startswith("AS-1") else (names[0] if names else rule_id)


# --------------------------------------------------------------------------- agents
def _agents(report: Report) -> list[tuple[str, list[str]]]:
    raw = report.metadata.get("agents")
    out: list[tuple[str, list[str]]] = []
    if isinstance(raw, list):
        for a in raw:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and all(a["name"] != n for n, _ in out):
                tools = a.get("tools")
                out.append((a["name"], [t for t in tools if isinstance(t, str)] if isinstance(tools, list) else []))
    return out


ALL_AGENTS = "all agents"


def agent_tools(report: Report) -> dict[str, list[str]]:
    """{agent: the tools the code gives it}, for the report's per-agent headings."""
    return dict(_agents(report))


def _owners(text: str, agents: list[tuple[str, list[str]]]) -> list[str]:
    """Agents a result names directly, or through a tool only that agent holds."""
    hits = []
    for name, tools in agents:
        words = [name, *tools]
        if any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text) for w in words if w):
            hits.append(name)
    return hits


def agent_names(report: Report) -> list[str]:
    """The report's groups, in order. One agent (or none found): a single group named after the spec's
    agent_id. Several: one group per agent, plus "all agents" for results about the whole project."""
    agents = _agents(report)
    if len(agents) <= 1:
        return [report.agent_id or (agents[0][0] if agents else "agent")]
    return [a for a, _ in agents] + [ALL_AGENTS]


def _assign(text: str, report: Report, agent: str | None = None) -> list[str]:
    agents = _agents(report)
    if len(agents) <= 1:
        return agent_names(report)
    if agent is not None and any(agent == n for n, _ in agents):  # the result names its agent (a per-agent bound)
        return [agent]
    return _owners(text, agents) or [ALL_AGENTS]


# --------------------------------------------------------------------------- lines
def _loc(ref: SourceRef | None) -> str:
    return str(ref) if ref else ""


def _waived(report: Report) -> list[dict[str, object]]:
    raw = report.metadata.get("waived")
    return [w for w in raw if isinstance(w, dict)] if isinstance(raw, list) else []


_BARE_RUN = f". Run: {MOCK_RUN}"
#: after a mock run: what is left for a line the run did not settle
AFTER_RUN = "Show it in code, or confirm under confirmed: in spec.yml."


def mock_run_done(report: Report) -> bool:
    """A mock run with captured model calls, newer than the code: running it again would show nothing new."""
    run = report.metadata.get("mock_run")
    return isinstance(run, dict) and bool(run.get("captured")) and not run.get("stale")


def after_mock_run(reason: str) -> str:
    """A check line whose only next step is "Run: agenticreq mock-run", once that run has happened: the step left
    is the code or a sign-off. A line with its own step (give args that reach it, add a mock) keeps it."""
    if not reason.endswith(_BARE_RUN):
        return reason
    head = reason[: -len(_BARE_RUN)].replace(" (or sign it off under confirmed:)", "")
    return f"{head}; the mock run did not settle it. {AFTER_RUN}"


def _not_needed(report: Report) -> dict[str, tuple[Verdict, str]]:
    """{parameter: (its NOT NEEDED verdict, where it was written)} from the spec's not_needed answers."""
    raw = report.metadata.get("not_needed")
    out: dict[str, tuple[Verdict, str]] = {}
    for e in raw if isinstance(raw, list) else []:
        if isinstance(e, dict) and isinstance(e.get("param"), str):
            out[e["param"]] = (
                Verdict("NOT NEEDED", str(e.get("reason") or ""), bool(e.get("open"))),
                str(e.get("line") or ""),
            )
    return out


#: parameters whose result depends on the full tool list: none may pass while a tool built at runtime is unknown
TOOL_LIST_PARAMS = frozenset(
    {
        "role_boundary",
        "prohibited",
        "approval_required",
        "limits",
        "untrusted_inputs",
        "tool_inventory_report",
        "read_write_separation",
        "output_size_limits",
    }
)


def unresolved_tools(report: Report) -> int:
    u = report.metadata.get("unresolved")
    return int(u.get("tools", 0)) if isinstance(u, dict) else 0


def unresolved_line(report: Report) -> str:
    """ "3 agents and 1 tool could not be identified; see _derived.unresolved", or ""."""
    u = report.metadata.get("unresolved")
    if not isinstance(u, dict):
        return ""
    parts = [
        f"{n} {w}{'s' if n != 1 else ''}"
        for w, n in (("agent", int(u.get("agents", 0))), ("tool", int(u.get("tools", 0))))
        if n
    ]
    return f"{' and '.join(parts)} could not be identified; see _derived.unresolved" if parts else ""


_UNREADABLE = re.compile(r"^spec\.yml says (.+?); not readable in code")
_FROM_UNREADABLE = re.compile(r"^spec\.yml says (.+?); the code sets this limit from a value agenticreq cannot read")
#: a mock run is synthetic: it never shows what only a real model's run shows
_REAL_RUN_RULES = frozenset({"AS-905", "AS-228", "AS-229", "AS-230", "AS-231"})


def _head(reason: str) -> tuple[str, str]:
    """``reason`` without its "..., then run: agenticreq mock-run" step, and its "(+N more)" note."""
    note = ""
    m = re.search(r" ?\(\+\d+ more\)", reason)
    if m:
        note, reason = m.group(0).strip(), reason[: m.start()] + reason[m.end() :]
    at = reason.rfind(MOCK_RUN)
    if at >= 0:  # drop the sentence that ends in the command: "... . Add a mock ..., then run: agenticreq mock-run"
        starts = [m.end() for m in re.finditer(r"\. (?=[A-Z])", reason[:at])]
        reason = reason[: starts[-1]] if starts else reason[:at]
    head = reason.replace(" (or sign it off under confirmed:)", "").strip().rstrip(".,;")
    return head, note


def settle_after_run(v: Verdict, rule_ids: tuple[str, ...], param: str) -> Verdict:
    """A mock-run check after a mock run that is newer than the code: running it again changes nothing, so the
    line stays only when editing mocks.yml would settle it (a tool to add to order:, args: that reach an agent, a
    mock for the other path), and says so. Every other line moves to where it can be settled:

    * a number in spec.yml with no limit in the code -> a gap (add it in code, or confirm where it is enforced);
    * something agenticreq cannot read (a value from env, a library it cannot follow, checkpointing) -> sign-off;
    * an approval gate inside a tool the mock replaced -> sign-off: the gate did not run.
    """
    if (
        "AS-201" in rule_ids
        or v.reason.startswith("Held in only")
        or "mocks.yml" in v.reason
        and not v.reason.startswith(("Code pattern only", "Cannot tell"))
    ):
        return v  # editing mocks.yml settles it: the line says what to change
    head, note = _head(v.reason)
    tail = f" {note}" if note else ""
    m = _UNREADABLE.match(head)
    if m:
        return Verdict(
            "GAP",
            f"spec.yml says {m.group(1)}; the code sets no limit. Add it in code, or confirm where "
            f"it is enforced under confirmed:.{tail}",
        )
    m = _FROM_UNREADABLE.match(head)
    if m:
        return Verdict(
            "SIGN-OFF",
            f"spec.yml says {m.group(1)}; the code sets this limit from a value agenticreq "
            f"cannot read. Confirm it under confirmed:.{tail}",
        )
    if "AS-205" in rule_ids or "AS-212" in rule_ids and "gate" in head:
        tool = re.search(r"(?:Code pattern only: )?([^()]+?) \(", head)
        name = tool.group(1).strip() if tool else "the tool"
        return Verdict(
            "SIGN-OFF", f"The mock replaced {name}, so the gate did not run; confirm it under confirmed:.{tail}"
        )
    if set(rule_ids) & _REAL_RUN_RULES:
        return Verdict(
            "SIGN-OFF",
            f"{sentence(head).rstrip('.')}: a mock run does not use a real model, so it cannot "
            f"show this. Confirm it under confirmed:, or pass real traces with --traces.{tail}",
        )
    if head.startswith("Cannot tell whether") or "cannot follow" in head:  # a tool's body agenticreq cannot read
        return Verdict(
            "SIGN-OFF",
            f"{sentence(head).rstrip('.')}. Declare it under tool_effects: or confirm it under confirmed:.{tail}",
        )
    return Verdict(
        "SIGN-OFF", f"{sentence(head).rstrip('.')}; a mock run cannot show more. Confirm it under confirmed:.{tail}"
    )


def first_sentence(text: str) -> str:
    """``text`` up to its first ". " outside parentheses ("hook (a. b) here. Next" -> "hook (a. b) here")."""
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if ch == "." and depth <= 0 and text[i + 1 : i + 2] == " ":
            return text[:i]
    return text.rstrip(".")


def also(first: str, rest: list[str]) -> str:
    """Several results on one line: every one is shown ("A. Also: B. Also: C"), never "(+2 more)"."""
    extra = [r for r in dict.fromkeys(rest) if r != first]
    if not extra:
        return first
    head = first if first.endswith((".", ":")) else first + "."
    return head + "".join(f" Also: {sentence(r)}" for r in extra)


def _signed(report: Report) -> dict[str, dict[str, object]]:
    raw = report.metadata.get("confirmed")
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}


def results(report: Report) -> list[ResultLine]:
    """One line per (agent, parameter), in the group of its most pressing result: a gap first, then an open
    sign-off, a signed sign-off, a mock-run check, and PASS only when every result for it passed. Sorted per agent,
    then by group."""
    entries: list[tuple[str, str, Verdict, str, str]] = []  # agent, param, verdict, rule_id, location
    for f in report.findings:
        who = " ".join([_what(f), *(f.path or [])])
        for param, v in finding_parts(f, report.command):
            for agent in _assign(who, report, f.agent):
                entries.append((agent, param, v, f.rule_id, _loc(f.location())))
    for p in report.passed:
        v = passed_verdict(p)
        for agent in _assign(p.subject, report, p.agent):
            entries.append(
                (agent, p.param or _param_of(p.rule_id, f"{p.subject} {p.note}", report.command), v, p.rule_id, "")
            )
    for w in _waived(report):  # an unexpired waiver in spec.yml: the person accepted this result
        detail = str(w.get("detail", ""))
        rid = str(w.get("rule_id", ""))
        why = str(w.get("reason") or "")
        v = Verdict("PASS", "waived by you" + (f": {_short(why, 80)}" if why else ""))
        for agent in _assign(detail, report):
            entries.append((agent, str(w.get("param") or "") or _param_of(rid, detail, report.command), v, rid, ""))

    nn = _not_needed(report)
    if nn:  # `not_needed` answers a parameter: its one line replaces every result for it, or under it
        from agenticreq.not_needed import covers

        names = frozenset(nn)
        entries = [e for e in entries if not covers(names, e[1])]
        for param, (v, loc) in nn.items():
            for agent in _assign(param, report):
                entries.append((agent, param, v, "AS-504", loc))

    grouped: dict[tuple[str, str], list[tuple[Verdict, str, str]]] = {}
    for agent, param, v, rid, loc in entries:
        grouped.setdefault((agent, param), []).append((v, rid, loc))
    lines = []
    ran = mock_run_done(report)
    signed = _signed(report)
    for (agent, param), items in grouped.items():
        worst = min(items, key=lambda x: _RANK[x[0].status])[0]
        same = [v for v, _, _ in items if v.status == worst.status]
        reason = also(worst.reason, [v.reason for v in same[1:]])
        if ran and worst.status == "CHECK":
            worst = settle_after_run(Verdict("CHECK", reason), tuple(r for _, r, _ in items), param)
            reason = worst.reason
        waived = worst.reason.startswith("waived by you")  # a waiver is the person's own decision
        if worst.status == "PASS" and param in TOOL_LIST_PARAMS and unresolved_tools(report) and not waived:
            # a tool the code builds at run time may be the one this clause is about: never a silent pass
            n = unresolved_tools(report)
            what = f"{n} tool{'s' if n > 1 else ''} built at runtime could not be identified"
            worst = (
                Verdict(
                    "SIGN-OFF",
                    f"{what}, so this is not shown for {'them' if n > 1 else 'it'}. Declare "
                    "it under tool_effects: or confirm it under confirmed:.",
                )
                if ran
                else Verdict(
                    "CHECK",
                    _then_run(
                        f"{what}, so this is not shown for {'them' if n > 1 else 'it'}",
                        "A mock run lists the tools the model is offered",
                    ),
                )
            )
            reason = worst.reason
        if worst.status == "SIGN-OFF":
            entry = signed.get(param) or signed.get(param.split(".")[0].split("[")[0])
            if entry is not None:  # a person confirmed this parameter under confirmed:
                state = entry.get("state")
                if state == "signed":
                    worst = Verdict("SIGNED", f"Signed by {entry.get('by')}.")
                elif state == "no":
                    worst = Verdict("GAP", sentence(f"Owner says this is not in place yet: {entry.get('reason')}"))
                reason = worst.reason
        rids = tuple(dict.fromkeys(r for _, r, _ in items))
        locs = tuple(dict.fromkeys(loc for _, _, loc in items if loc))
        support = ""
        if worst.group == "sign_off":  # what the code shows about a sign-off item: evidence for the person confirming
            seen = next(
                (
                    (v, loc)
                    for v, _, loc in items
                    if v.reason.startswith(_CODE_EVIDENCE) and not loc.split(":")[0].endswith((".yml", ".yaml"))
                ),
                None,
            )
            if seen:
                what = first_sentence(seen[0].reason).removeprefix("Code pattern only").lstrip(": ")
                if what.startswith("(") and what.endswith(")"):
                    what = what[1:-1]
                support = what.removeprefix("Found in code: ") + (f" at {seen[1]}" if seen[1] else "")
        lines.append(ResultLine(agent, param, worst.status, reason, rids, locs, support, worst.open))

    order = {a: i for i, a in enumerate(agent_names(report))}
    porder = _param_order()
    gorder = {g: i for i, g in enumerate(GROUPS)}
    lines.sort(
        key=lambda ln: (
            order.get(ln.agent, len(order)),
            gorder[ln.group],
            _RANK[ln.status],
            porder.get(ln.param, len(porder)),
            ln.param,
        )
    )
    return lines


def totals(lines: list[ResultLine]) -> dict[str, int]:
    """Counts per group: gaps, sign-offs still open, sign-offs signed, sign-offs the owner marked not needed,
    mock-run checks, passed."""
    lines = list({(ln.param, ln.status, ln.reason): ln for ln in lines}.values())  # one result, many agents
    return {
        "gaps": sum(1 for ln in lines if ln.status == "GAP"),
        "sign_off_left": sum(1 for ln in lines if ln.status == "SIGN-OFF" or ln.status == "NOT NEEDED" and ln.open),
        "signed": sum(1 for ln in lines if ln.status == "SIGNED"),
        "not_needed": sum(1 for ln in lines if ln.status == "NOT NEEDED" and not ln.open),
        "mock_run": sum(1 for ln in lines if ln.status == "CHECK"),
        "passed": sum(1 for ln in lines if ln.status == "PASS"),
    }


def _fails(t: dict[str, int], fail_on: FailOn) -> bool:
    if fail_on == "all":
        return bool(t["gaps"] or t["sign_off_left"] or t["mock_run"])
    return bool(t["gaps"])  # "gaps", and "never" (which only changes the exit code)


def overall(report: Report, lines: list[ResultLine] | None = None, fail_on: FailOn = "gaps") -> Literal["PASS", "FAIL"]:
    """PASS when the --fail-on condition is met: no gaps (``gaps``, ``never``), or no gaps, no open sign-off and no
    pending mock-run check (``all``). An invalid spec, or nothing checked, is never a PASS."""
    if report.spec_status == "invalid" or report.nothing_checked:
        return "FAIL"
    run = report.metadata.get("mock_run")
    if isinstance(run, dict) and not run.get("captured") and not run.get("stale"):
        return "FAIL"  # the latest mock run crashed or made no model call: nothing ran, so nothing passed
    lines = results(report) if lines is None else lines
    return "FAIL" if _fails(totals(lines), fail_on) else "PASS"


def exit_code(report: Report, fail_on: FailOn = "gaps") -> int:
    """0 = PASS, 1 = the --fail-on condition failed, 2 = the spec is invalid, 3 = nothing was checked.
    ``gaps`` (default) exits 1 on a spec vs code gap; ``all`` also on an open sign-off or a pending mock-run check;
    ``never`` always exits 0."""
    if fail_on == "never":
        return 0
    if report.nothing_checked:
        return 3  # no agent / readable code found: a pass here would be a false "all clear"
    if report.spec_status == "invalid":
        return 2
    return 1 if _fails(totals(results(report)), fail_on) else 0
