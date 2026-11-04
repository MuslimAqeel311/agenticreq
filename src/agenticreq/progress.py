"""The five steps from a first scan to a signed-off agent, and where this folder stands.

    1. audit     quick scan, no setup
    2. spec      spec.yml exists and has no TODO fields
    3. check     no "Spec vs code gaps"
    4. mock-run  a mock run newer than every scanned source file, and nothing left in "Check with a mock run"
    5. sign-off  nothing left in "Human sign-off"

Everything is read from the folder itself, never the network: spec.yml, ``.agenticreq/captures.json`` and a small
``.agenticreq/state.json`` that records that audit ran and what the latest check found. ``check`` passes its own
report, so its block is always current; the other commands read the latest check from the state file, and ignore it
once spec.yml changed after that check.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenticreq.models.finding import Report

STATE_FILE = Path(".agenticreq") / "state.json"
CAPTURES_FILE = Path(".agenticreq") / "captures.json"
STEPS: tuple[str, ...] = ("audit", "spec", "check", "mock-run", "sign-off")
GUIDE = "Help: agenticreq guide"
ALL_DONE = "All 5 steps done"

#: what to write as evidence when signing off an item (the Next: hint for step 5)
EVIDENCE_HINT: dict[str, str] = {
    "owner": "Name of the person who owns this agent",
    "technical_owner": "Name of the engineer who maintains this agent",
    "escalation": "Who handles hard cases, and how they are reached",
    "data_rules": "Where retention and never-store rules are applied",
    "out_of_scope": "How the agent is kept out of the listed topics",
    "log_immutable": "The write-once log store and its setting",
    "kill_switch": "The flag or command that stops the agent in production",
    "rollback_path": "How a release is rolled back",
    "environment_separation": "Where test and production credentials are kept apart",
    "drift_monitors": "The monitor that watches for drift",
    "regression_triggers": "The CI job that runs the regression suite",
    "audit_log": "Where tool calls are logged",
    "secrets_policy": "The secret store the agent reads from",
    "termination_authority": "Who may stop the agent",
}


def _hint(clause: str) -> str:
    return EVIDENCE_HINT.get(clause, f"Where {clause.replace('_', ' ')} is handled: a link, a document or file:line")


@dataclass
class Progress:
    """``done`` per step; ``current`` is the first step not done (None when all are)."""

    done: list[bool]
    labels: list[str]
    next_lines: list[str] = field(default_factory=list)

    @property
    def current(self) -> int | None:
        return next((i for i, d in enumerate(self.done) if not d), None)

    def step_done(self, name: str) -> bool:
        return self.done[STEPS.index(name)]


# ------------------------------------------------------------------ the state file
def folder_of(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _may_write(root: Path) -> bool:
    """Only the folder the user works in (or one agenticreq already wrote to) gets a state file.
    AGENTICREQ_NO_STATE=1 turns it off (the test suite runs commands inside its fixture folders)."""
    if os.environ.get("AGENTICREQ_NO_STATE"):
        return False
    try:
        return root.resolve() == Path.cwd().resolve() or (root / ".agenticreq").is_dir()
    except OSError:
        return False


def load_state(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(root: Path, data: dict[str, Any]) -> None:
    if not _may_write(root):
        return
    try:
        (root / STATE_FILE).parent.mkdir(exist_ok=True)
        (root / STATE_FILE).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # a read-only folder: the block still shows, it just is not remembered


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def record_audit(root: Path) -> None:
    state = load_state(root)
    state["audit_ran"] = True
    _save_state(root, state)


def check_summary(report: Report, spec_path: Path) -> dict[str, Any]:
    """What the progress block needs from one check."""
    from agenticreq.models.verdict import results, totals

    lines = results(report)
    t = totals(lines)
    run = report.metadata.get("mock_run")
    run_file = Path(str(run["file"])) if isinstance(run, dict) and run.get("file") else None
    first = next((ln.param for ln in lines if ln.status == "SIGN-OFF"), None)
    sign_off_items = sum(1 for ln in lines if ln.group == "sign_off")
    return {
        "spec_sha": _sha(spec_path),
        "spec_invalid": report.spec_status == "invalid",
        "gaps": t["gaps"],
        "mock_run": t["mock_run"],
        "sign_off_left": t["sign_off_left"],
        "sign_off_items": sign_off_items,
        "passed": t["passed"],
        "first_sign_off": first,
        "captures_mtime": _mtime(run_file) if run_file else None,
        "mock_run_stale": bool(isinstance(run, dict) and run.get("stale")),
    }


def record_check(root: Path, summary: dict[str, Any]) -> None:
    state = load_state(root)
    state["check"] = summary
    _save_state(root, state)


# ------------------------------------------------------------------ where the folder stands
def _source_newer_than(root: Path, stamp: float) -> bool:
    from agenticreq.ast.symbols import iter_py_files

    for p in iter_py_files(root):
        m = _mtime(p)
        if m is not None and m > stamp + 1:
            return True
    return False


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _add_months(d: dt.date, months: int) -> dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    for day in (d.day, 30, 29, 28):
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    return dt.date(year, month, 28)


def _clause_of(param: str) -> str:
    """The confirmed: clause name for a report line (a number's line signs off under its own name)."""
    from agenticreq.thresholds import BY_KEY

    base = param.split(" [")[0]
    p = BY_KEY.get(base)
    return p.attest_names[0] if p is not None and p.attest_names else base.split(".")[0].split("[")[0]


def sign_off_yaml(clause: str, today: dt.date) -> list[str]:
    """The ready-to-paste entry for one sign-off item."""
    return [
        "Next: add this under confirmed: in spec.yml, with your own details:",
        f"  - clause: {clause}",
        "    answer: yes",
        "    by: you@company.com",
        f"    date: {today.isoformat()}",
        f"    expires: {_add_months(today, 6).isoformat()}",
        f'    evidence: "{_hint(clause)}"',
    ]


def compute(
    root: Path,
    spec_path: Path | None = None,
    *,
    summary: dict[str, Any] | None = None,
    audit_ran: bool = False,
    today: dt.date | None = None,
) -> Progress:
    """Where ``root`` stands. ``summary``: the check just run (else the latest recorded, if spec.yml has not
    changed since)."""
    from agenticreq.loader import load_spec

    today = today or dt.date.today()
    state = load_state(root)
    spec_path = spec_path or root / "spec.yml"
    has_spec = spec_path.is_file()
    if summary is None:
        stored = state.get("check")
        if isinstance(stored, dict) and has_spec and stored.get("spec_sha") == _sha(spec_path):
            summary = stored
    todo, invalid = 0, False
    if has_spec:
        loaded = load_spec(spec_path)
        todo, invalid = len(loaded.todos), loaded.status == "invalid"
    captures = root / CAPTURES_FILE
    run_at = _mtime(captures)
    fresh_run = run_at is not None and not _source_newer_than(root, run_at)
    check_saw_run = bool(
        summary
        and run_at is not None
        and summary.get("captures_mtime") is not None
        and abs(float(summary["captures_mtime"]) - run_at) < 1
    )

    done = [
        bool(audit_ran or state.get("audit_ran") or has_spec),
        has_spec and not invalid and todo == 0,
        bool(summary) and not (summary or {}).get("spec_invalid") and (summary or {}).get("gaps") == 0,
        fresh_run and check_saw_run and (summary or {}).get("mock_run") == 0,
        bool(summary) and (summary or {}).get("sign_off_left") == 0,
    ]
    if not done[0]:  # a later step cannot be done before the first scan
        done = [False] * 5
    for i in range(1, 5):  # the steps are an order: a step after an open one is not done yet
        if not done[i - 1]:
            done[i:] = [False] * (5 - i)
            break

    left = (summary or {}).get("sign_off_left")
    labels = [
        "done" if done[0] else "quick scan, no setup",
        "done"
        if done[1]
        else (
            "spec.yml not written yet"
            if not has_spec
            else "spec.yml has errors"
            if invalid
            else f"{_plural(todo, 'field')} still TODO in spec.yml"
        ),
        "spec matches" if done[2] else "spec vs project",
        "test run with mock tools",
        "owner confirmed every item"
        if done[4]
        else (
            f"owner confirms {_plural(left, 'item')}"
            if isinstance(left, int) and left > 0
            else "owner confirms what the scan cannot see"
        ),
    ]
    p = Progress(done=done, labels=labels)
    p.next_lines = _next(p, root, has_spec, invalid, todo, summary, run_at, fresh_run, check_saw_run, today)
    return p


def _next(
    p: Progress,
    root: Path,
    has_spec: bool,
    invalid: bool,
    todo: int,
    summary: dict[str, Any] | None,
    run_at: float | None,
    fresh_run: bool,
    check_saw_run: bool,
    today: dt.date,
) -> list[str]:
    step = p.current
    if step is None:
        return []
    s = summary or {}
    if step == 0:
        return ["Next: run a quick scan of this folder: agenticreq audit"]
    if step == 1:
        if not has_spec:
            return ["Next: write spec.yml from your project, then fill its TODO fields: agenticreq init"]
        if invalid:
            return ["Next: fix the errors listed for spec.yml, then run: agenticreq validate spec.yml"]
        return [f"Next: open spec.yml, fill the {_plural(todo, 'TODO field')}, then run: agenticreq check"]
    if step == 2:
        if not summary:
            return ["Next: compare spec.yml with your project: agenticreq check"]
        return [
            f'Next: fix the {_plural(int(s.get("gaps") or 0), "gap")} under "Spec vs code gaps" (in the '
            "project or in spec.yml), then run: agenticreq check"
        ]
    if step == 3:
        if not (root / "mocks.yml").is_file():
            return ["Next: write mocks.yml for a test run with mock tools: agenticreq mocks"]
        if run_at is None or not fresh_run:
            return ["Next: run your agent with mock tools: agenticreq mock-run"]
        if not check_saw_run:
            return ["Next: read the new mock run: agenticreq check"]
        return [
            f'Next: settle the {_plural(int(s.get("mock_run") or 0), "item")} under "Check with a mock run" '
            "as each line says, then run: agenticreq check"
        ]
    first = s.get("first_sign_off")
    return (
        sign_off_yaml(_clause_of(str(first)), today)
        if first
        else ["Next: add the open items under confirmed: in spec.yml, then run: agenticreq check"]
    )


# ------------------------------------------------------------------ the block
def render(p: Progress, *, unicode: bool = True) -> str:
    done_mark, here = ("✓", "✗") if unicode else ("+", "x")
    lines = ["Your progress"]
    for i, (name, label) in enumerate(zip(STEPS, p.labels)):
        mark = done_mark if p.done[i] else here if i == p.current else " "
        lines.append(f"  {mark} {i + 1}. {name.ljust(10)} {label}")
    lines.append("")
    lines += p.next_lines if p.current is not None else [ALL_DONE]
    lines += ["", GUIDE]
    return "\n".join(lines)


def can_print_unicode(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "✓✗".encode(enc)
    except (LookupError, UnicodeEncodeError):
        return False
    return os.environ.get("TERM") != "dumb"


def held_back(p: Progress, lines: list[Any]) -> dict[str, str]:
    """The result groups check does not show yet at this step, each with the line that stands in for it."""
    from agenticreq.models.verdict import distinct

    names = {
        g: list(dict.fromkeys(ln.param for ln in distinct(lines) if ln.group == g))
        for g in ("sign_off", "mock_run", "passed")
    }
    n = {g: sum(1 for ln in distinct(lines) if ln.group == g) for g in names}

    def listed(g: str) -> str:  # every counted item is named, even while its group waits
        return f": {', '.join(names[g])}" if names[g] else ""

    out: dict[str, str] = {}
    if not p.step_done("spec"):
        out["sign_off"] = (
            f"Human sign-off: {_plural(n['sign_off'], 'item')}, shown after the spec is complete{listed('sign_off')}"
        )
        out["passed"] = f"Passed: {_plural(n['passed'], 'item')}, shown after the spec is complete{listed('passed')}"
    if not p.step_done("check"):
        out["mock_run"] = (
            f"Check with a mock run: {_plural(n['mock_run'], 'item')}, shown after the spec vs project gaps are "
            f"fixed{listed('mock_run')}"
        )
    return out


# ------------------------------------------------------------------ agenticreq guide
#: the five steps: name, what it means, the command
GUIDE_TABLE: tuple[tuple[str, str, str], ...] = (
    ("audit", "quick scan, no setup", "agenticreq audit"),
    ("spec", "write what the agent must and must not do", "agenticreq init"),
    ("check", "compare spec.yml with the project", "agenticreq check"),
    ("mock-run", "test run with mock tools", "agenticreq mocks, then agenticreq mock-run"),
    ("sign-off", "owner confirms what the scan cannot see", "add entries under confirmed: in spec.yml"),
)
#: what the current step is about, in one or two plain sentences
_ABOUT: tuple[str, ...] = (
    "Start with a quick scan. It reads the project and needs no setup.",
    "Write down what the agent must and must not do. agenticreq init writes spec.yml from your project; you fill "
    "its TODO fields.",
    "Compare spec.yml with the project. Each gap is something the spec asks for that the project does not do yet.",
    "Run the agent once with mock tools, so the scan sees what the agent sends to the model. No real tool or "
    "model is called.",
    "The owner confirms what the scan cannot see, one entry per item under confirmed: in spec.yml.",
)


def guide_text(root: Path, spec_path: Path | None = None, *, unicode: bool = True, today: dt.date | None = None) -> str:
    p = compute(root, spec_path, today=today)
    step = p.current
    if step is None:
        lines = [f"{ALL_DONE}.", "Run agenticreq check again after every change to the project or spec.yml."]
    else:
        lines = [f"You are at step {step + 1} of 5: {STEPS[step]}.", _ABOUT[step], "", *p.next_lines]
    lines += ["", "The 5 steps:"]
    done_mark, here = ("✓", "✗") if unicode else ("+", "x")
    for i, (name, what, cmd) in enumerate(GUIDE_TABLE):
        mark = done_mark if p.done[i] else here if i == step else " "
        lines.append(f"  {mark} {i + 1}. {name.ljust(9)} {what.ljust(42)} {cmd}")
    missing = [
        (f, c)
        for f, c in (((spec_path or root / "spec.yml"), "agenticreq init"), (root / "mocks.yml", "agenticreq mocks"))
        if not f.is_file()
    ]
    if missing:
        lines.append("")
        lines += [f"{f.name} is missing: create it with: {c}" for f, c in missing]
    return "\n".join(lines)
