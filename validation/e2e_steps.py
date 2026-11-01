"""End-to-end: walk a LangGraph multi-agent project through the five steps with the BUILT WHEEL in a fresh venv.

    python validation/e2e_steps.py <venv python> <work dir>

The venv must hold the agenticreq wheel plus langgraph and langchain-openai (the agent's own dependencies). The
script copies tests/fixtures/e2e/langgraph_team into <work dir>, then runs, from inside it:

    guide -> audit -> init -> (fill spec.yml) -> check -> mocks -> (fill args:) -> mock-run -> check
          -> (confirm the numbers the mock run cannot read) -> check
          -> (confirmed: entries, answer yes for all, one no, one not_needed) -> check -> (fix the no) -> check

It prints the progress block after each command and checks that it moves one step at a time, that the `no` entry
is a gap (exit 1), and that it ends at "All 5 steps done".
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "e2e" / "langgraph_team"
TODAY = dt.date.today()

#: what the person writes into spec.yml (each replaces a `key: TODO ...` line, so comments stay)
FILLED = {
    "agent_id": "support-team",
    "risk_level": "low",
    "goal": "{statement: Answer order and customer questions, done_criteria: [answer returned to the user]}",
    "out_of_scope": "[refunds]",
    "role_boundary": "{allowed_tools: [lookup_order, lookup_customer]}",
    "prohibited": "[]",
    "approval_required": "[]",
    "limits": "[]",
    "escalation": '[{metric: risk_score, op: ">=", threshold: 8, to: support-lead}]',
    "untrusted_inputs": "[]",
    "owner": "owner@example.com",
    "technical_owner": "eng@example.com",
    "data_rules": "{}",
}
#: low-risk opt-outs the person chooses (allowed at risk_level low), and the multi-agent shape
EXTRA = """
# Level-2 checks this low-risk team opts out of (allowed at risk_level low)
environment_separation: false
audit_log: false
trace_policy: ""
topology: pipeline
agent_registry:
  - {id: orders, tools: [lookup_order]}
  - {id: customers, tools: [lookup_customer]}
"""


def run(py: Path, cwd: Path, *args: str) -> tuple[int, str]:
    exe = py.parent / ("agenticreq.exe" if sys.platform == "win32" else "agenticreq")
    p = subprocess.run([str(exe), *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "COLUMNS": "110"})
    return p.returncode, p.stdout + p.stderr


def block(out: str) -> str:
    return out[out.index("Your progress"):].strip() if "Your progress" in out else "(no progress block)"


def marks(out: str) -> str:
    """The five marks: d = done, c = current, . = later."""
    b = block(out).splitlines()[1:6]
    return "".join("d" if ln[2] in "✓+" else "c" if ln[2] in "✗x" else "." for ln in b)


def fill_spec(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in FILLED.items():
        text, n = re.subn(rf"^{key}:(\s+)TODO\b[^\n]*", rf"{key}:\g<1>{value}", text, count=1, flags=re.M)
        assert n == 1, key
    # init wrote every recommended number; this team does not cap tool calls, so the person deletes that line
    text, n = re.subn(r"^  max_tool_calls: .*\n", "", text, count=1, flags=re.M)
    assert n == 1, "max_tool_calls"
    path.write_text(text.rstrip("\n") + "\n" + EXTRA, encoding="utf-8")


def open_items(py: Path, cwd: Path, status: str) -> list[str]:
    code, out = run(py, cwd, "check", "-f", "json")
    data = json.loads(out[out.index("{"):])
    return [r["param"] for r in data["results"] if r["status"] == status]


def confirm(path: Path, items: list[str], no: str | None = None, not_needed: str | None = None) -> None:
    lines = ["confirmed:"]
    for c in items:
        if c == no:
            lines.append(f"  - {{clause: {c}, answer: no, by: lead@example.com, date: {TODAY}, "
                         "reason: the stop flag is not built yet}")
        elif c == not_needed:
            lines.append(f"  - {{clause: {c}, answer: not_needed, by: lead@example.com, date: {TODAY}, "
                         "reason: one team runs this agent; nobody else decides}")
        else:
            lines.append(f"  - {{clause: {c}, answer: yes, by: lead@example.com, date: {TODAY}, "
                         f"expires: {TODAY.replace(year=TODAY.year + 1)}, evidence: TICKET-7}}")
    text = re.sub(r"\nconfirmed:\n(?:  - .*\n)*", "\n", path.read_text(encoding="utf-8"))
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    py, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    app = work / "langgraph_team"
    shutil.rmtree(app, ignore_errors=True)
    shutil.copytree(FIXTURE, app, ignore=shutil.ignore_patterns("__pycache__"))
    seen: list[str] = []
    problems: list[str] = []

    def step(title: str, *args: str, expect_marks: str | None = None, expect_code: int | None = None) -> str:
        code, out = run(py, app, *args)
        print(f"\n$ agenticreq {' '.join(args)}    (exit {code})  -- {title}")
        print(block(out) if args[0] != "guide" else out.strip())
        m = marks(out) if args[0] != "guide" else ""
        seen.append(m)
        if expect_marks and m != expect_marks:
            problems.append(f"{title}: progress {m}, expected {expect_marks}")
        if expect_code is not None and code != expect_code:
            problems.append(f"{title}: exit {code}, expected {expect_code}\n{out[-1500:]}")
        return out

    step("fresh folder", "guide")
    step("quick scan", "audit", expect_marks="dc...")
    step("write spec.yml", "init", expect_marks="dc...")
    fill_spec(app / "spec.yml")
    step("spec filled", "check", expect_marks="dddc.", expect_code=0)
    step("mocks.yml", "mocks", expect_marks="dddc.")
    mocks = app / "mocks.yml"
    mocks.write_text(re.sub(r"^args:.*$", "args: {question: 'Where is order A1 for customer C1?'}",
                            mocks.read_text(encoding="utf-8"), count=1, flags=re.M), encoding="utf-8")
    step("mock run", "mock-run", expect_marks="dddc.")
    step("read the mock run", "check", expect_marks="dddc.", expect_code=0)
    numbers = open_items(py, app, "CHECK")  # numbers no code shows: the person confirms them
    print(f"\n(mock-run items the person confirms: {', '.join(numbers)})")
    confirm(app / "spec.yml", numbers)
    step("mock-run items settled", "check", expect_marks="ddddc", expect_code=0)
    items = numbers + open_items(py, app, "SIGN-OFF")
    print(f"\n(open sign-off items: {', '.join(items[len(numbers):])})")
    no, not_needed = "kill_switch", "termination_authority"
    for c in (no, not_needed):
        if c not in items:
            problems.append(f"{c} is not an open sign-off item here: {items}")
    confirm(app / "spec.yml", items, no=no, not_needed=not_needed)
    out = step("signed: all yes, one no, one not_needed", "check", expect_marks="ddc..", expect_code=1)
    if not re.search(r"[✗x] kill_switch\s+Owner says this is not in place yet: the stop flag is not built yet", out):
        problems.append("the `no` entry is not shown as a gap")
    if not re.search(r"NOT NEEDED\s+termination_authority", run(py, app, "check", "--verbose")[1]):
        problems.append("the not_needed entry is not shown as NOT NEEDED")
    confirm(app / "spec.yml", items, not_needed=not_needed)  # the flag was built: the owner answers yes
    out = step("the no is fixed", "check", expect_marks="ddddd", expect_code=0)
    if "All 5 steps done" not in out:
        problems.append("did not end at All 5 steps done")
    step("guide at the end", "guide")
    print("\nPROBLEMS:" if problems else "\nPROBLEMS: none")
    for p in problems:
        print(" ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
