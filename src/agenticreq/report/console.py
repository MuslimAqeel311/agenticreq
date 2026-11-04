"""Rich console renderer: a header line, the Result line, then the four groups (models/verdict.py).

    agenticreq check · finance-multi-agent-system · spec: spec.yml · mock run: 2026-10-27 11:56 (6/6 calls)

    Result: FAIL    10 gaps · 11 sign-offs left (10 signed) · 9 to check · 6 passed

    SPEC VS CODE GAPS (10)
      ✗ bounds.max_steps        Code allows 60 steps; spec.yml says 25.

One result per parameter. Long text is never cut: it wraps onto the next line, indented to the text column. A
terminal that cannot print ✓ ✗ · gets + x - instead. --verbose adds rule ids, file:line and every signed item. No
score."""

from __future__ import annotations

import codecs
import io
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from agenticreq.models.finding import Report, SpecIssue
from agenticreq.models.verdict import (
    ALL_AGENTS,
    GROUP_TITLE,
    GROUPS,
    FailOn,
    Group,
    ResultLine,
    agent_names,
    agent_tools,
    overall,
    results,
    totals,
    unresolved_line,
    with_note,
)
from agenticreq.report.term import AsciiFallback, panel_box

_STYLE: dict[Group, str] = {"gap": "red", "sign_off": "yellow", "mock_run": "cyan", "passed": "green"}
_UNICODE = {"gap": "✗", "mock_run": "?", "passed": "✓", "dot": "·"}
_ASCII = {"gap": "x", "mock_run": "?", "passed": "+", "dot": "-"}
#: the parameter column is at least this wide when a label needs it (and up to 2/5 of the terminal)
_MAX_COL = 30


def _symbols(con: Console) -> dict[str, str]:
    """✓ ✗ · when the console can encode them, else + x - (a Windows cp1252 console, LANG=C)."""
    f = con.file
    if isinstance(f, AsciiFallback):
        return _ASCII
    enc = getattr(f, "encoding", None)
    if isinstance(f, io.StringIO) or not enc:  # a buffer is written out as UTF-8 later
        return _UNICODE
    try:
        "".join(_UNICODE.values()).encode(codecs.lookup(enc).name)
    except (LookupError, UnicodeEncodeError):
        return _ASCII
    return _UNICODE


def _wrap(text: str, width: int, indent: int = 0, first: int | None = None) -> list[str]:
    """``text`` in lines of at most ``width`` characters: never cut, never "…". Lines after the first start
    ``indent`` spaces in; the first has ``first`` characters of room (default: ``width - indent``)."""
    import textwrap

    text = " ".join(text.split())
    room = max(width - indent, 20)
    # a word longer than a line (a path) continues on the next line; a hyphenated name is never split
    lines = textwrap.wrap(text, width=first if first is not None else room, break_on_hyphens=False) or [""]
    if len(lines) > 1:
        rest = " ".join(lines[1:])
        lines = lines[:1] + textwrap.wrap(rest, width=room, break_on_hyphens=False)
    return [lines[0]] + [" " * indent + ln for ln in lines[1:]]


@dataclass(frozen=True)
class _Row:
    """One printed line: results with the same parameter, status and text for several agents are one row."""

    label: str
    status: str
    group: Group
    text: str
    rule_ids: tuple[str, ...]
    locations: tuple[str, ...]


def _rows(report: Report, lines: list[ResultLine]) -> list[_Row]:
    multi = ALL_AGENTS in agent_names(report)
    merged: dict[tuple[str, str, str], list[ResultLine]] = {}
    for ln in lines:
        merged.setdefault((ln.param, ln.status, ln.reason), []).append(ln)
    out = []
    for (param, status, reason), same in merged.items():
        agents = [ln.agent for ln in same if ln.agent != ALL_AGENTS]
        label = param + (f" [{agents[0]}]" if multi and len(same) == 1 and agents else "")
        text = reason
        if same[0].support:
            # what the code shows about a sign-off item replaces the generic question
            text = f"Found: {same[0].support}. Confirm it works under confirmed: in spec.yml."
        if len(agents) > 1:
            text = with_note(text, f"({len(agents)} agents)")
        out.append(
            _Row(
                label,
                status,
                same[0].group,
                text,
                tuple(dict.fromkeys(r for ln in same for r in ln.rule_ids)),
                tuple(dict.fromkeys(loc for ln in same for loc in ln.locations)),
            )
        )
    return out


def _mock_run_text(report: Report) -> str:
    m = report.metadata.get("mock_run")
    if not isinstance(m, dict):
        return "no mock run yet"
    when = str(m.get("time") or "")
    return f"mock run: {when} ({m.get('captured', 0)}/{m.get('total', 0)} calls)".replace(" ()", "")


def _header(report: Report, sym: dict[str, str]) -> str:
    dot = f" {sym['dot']} "
    parts = [f"agenticreq {report.command}"]
    if report.agent_id:
        parts.append(report.agent_id)
    spec = report.metadata.get("spec_file")
    parts.append(f"spec: {spec}" if spec else "no spec")
    if report.command != "validate":  # validate reads the spec on its own: no code, no run
        parts.append(_mock_run_text(report))
    return dot.join(parts)


def _counts_text(t: dict[str, int], sym: dict[str, str], short: bool = False) -> str:
    dot = f" {sym['dot']} "
    signed = f" ({t['signed']} signed)" if t["signed"] else ""
    sign_offs = f"{t['sign_off_left']} sign-off{'s' if t['sign_off_left'] != 1 else ''}"
    return dot.join(
        [
            f"{t['gaps']} gap{'s' if t['gaps'] != 1 else ''}",
            sign_offs if short else f"{sign_offs} left{signed}",
            f"{t['mock_run']} to check",
            f"{t['passed']} passed",
        ]
    )


def _heading(group: Group, n: int, t: dict[str, int], sym: dict[str, str], command: str = "check") -> str:
    # audit has no spec: what it finds are gaps in the code, not between a spec and the code
    title = "GAPS FOUND IN CODE" if group == "gap" and command == "audit" else GROUP_TITLE[group].upper()
    if group == "sign_off":
        not_needed = f" {sym['dot']} {t['not_needed']} not needed" if t.get("not_needed") else ""
        return f"{title} ({t['sign_off_left']} left {sym['dot']} {t['signed']} signed{not_needed})"
    return f"{title} ({n})"


def render_console(
    report: Report,
    *,
    console: Console | None = None,
    show_passed: bool = False,
    verbose: bool = False,
    fail_on: FailOn = "gaps",
    held_back: dict[Group, str] | None = None,
) -> str:
    """The four groups. ``verbose`` adds rule ids, file:line and every signed item. ``show_passed`` is kept for
    compatibility: passed lines always show. ``held_back``: groups not shown yet at this step (progress.py), each
    with the one line that replaces it."""
    buf = io.StringIO()
    con = console or Console(file=buf, force_terminal=False, width=110, highlight=False)
    sym = _symbols(con)
    width = max(con.width, 40)
    con.print(escape(chr(10).join(_wrap(_header(report, sym), width, 2))), soft_wrap=True)
    if report.spec_status == "invalid":
        con.print("\n[bold red]Result: FAIL[/bold red]  the spec is invalid; nothing was checked", soft_wrap=True)
        _render_spec_invalid(con, report)
        return "" if console is not None else buf.getvalue()
    if report.nothing_checked:  # no groups: empty groups would read as "nothing failed"
        con.print("\n[bold red]Result: FAIL[/bold red]  nothing was checked", soft_wrap=True)
        con.print(_nothing_checked(report), soft_wrap=True)
        return "" if console is not None else buf.getvalue()

    lines = results(report)
    t = totals(lines)
    verdict = overall(report, lines, fail_on)
    style = "bold green" if verdict == "PASS" else "bold red"
    counts, gap = _counts_text(t, sym), "    "
    if len(f"Result: {verdict}{gap}{counts}") > width:  # a narrow terminal: the short form, wrapped
        gap = "  "
        counts = _counts_text(t, sym, short=True)
    lead = len(f"Result: {verdict}{gap}")
    counts = "\n".join(_wrap(counts, width, lead, first=width - lead))
    con.print(f"\n[{style}]Result: {verdict}[/{style}]{gap}{escape(counts)}", soft_wrap=True)
    unknown = unresolved_line(report)
    if unknown:
        con.print(f"[yellow]{escape(chr(10).join(_wrap(unknown, width, 2)))}[/yellow]", soft_wrap=True)
    m = report.metadata.get("mock_run")
    if isinstance(m, dict) and m.get("stale"):
        con.print(
            "[yellow]"
            + escape(
                chr(10).join(
                    _wrap(
                        "The code changed after the last mock run, so its results are left out of this report. Run: "
                        "agenticreq mock-run",
                        width,
                        2,
                    )
                )
            )
            + "[/yellow]",
            soft_wrap=True,
        )
    if verbose and isinstance(m, dict) and m.get("file"):
        con.print(f"[dim]mock run file: {escape(str(m['file']))}[/dim]", soft_wrap=True)
    scope = report.metadata.get("audit_scope")
    if isinstance(scope, dict):  # audit has no spec: say what it could and could not check
        _render_scope(con, scope, width, sym)
    if verbose and ALL_AGENTS in agent_names(report):  # which tools the code gives each agent
        for name, tools in agent_tools(report).items():
            con.print(
                f"[dim]agent {escape(name)}: {escape(', '.join(tools)) if tools else 'no tools'}[/dim]", soft_wrap=True
            )

    rows = _rows(report, lines)
    for group in GROUPS:
        mine = [r for r in rows if r.group == group]
        # one column per group, as wide as its longest label up to 2/5 of the terminal (at least _MAX_COL);
        # a longer label is printed on its own line, so every reason starts in the same column
        col = min(max([len(r.label) for r in mine] + [12]), max(_MAX_COL, width * 2 // 5))
        n = len(mine)
        if not n:
            continue
        if held_back and group in held_back:
            con.print(f"\n[dim]{escape(chr(10).join(_wrap(held_back[group], width, 2)))}[/dim]", soft_wrap=True)
            continue
        gstyle = _STYLE[group]
        con.print(
            f"\n[bold {gstyle}]{escape(_heading(group, n, t, sym, report.command))}[/bold {gstyle}]", soft_wrap=True
        )
        shown = [r for r in mine if verbose or r.status != "SIGNED"]
        mark_w = max([len(r.status) for r in shown] + [8])  # SIGN-OFF / NOT NEEDED in one column
        for r in shown:
            mark = r.status.ljust(mark_w) if group == "sign_off" else sym[group]
            head = f"  {mark} "
            text_col = len(head) + col + 2
            if len(r.label) > col:  # a long label gets its own line; the text starts in the text column below it
                con.print(f"[{gstyle}]{escape(head + r.label)}[/{gstyle}]", soft_wrap=True)
                prefix = " " * text_col
            else:
                prefix = head + r.label.ljust(col) + "  "
            body = _wrap(r.text, width, text_col, first=width - text_col)
            if verbose:  # the rule ids and every file:line, on lines of their own in the text column
                more = _wrap(" ".join([*r.rule_ids, *r.locations]), width, text_col, first=width - text_col)
                body += [" " * text_col + more[0], *more[1:]]
            con.print(f"[{gstyle}]{escape(prefix)}[/{gstyle}]{escape(chr(10).join(body))}", soft_wrap=True)
        hidden = len(mine) - len(shown)
        if hidden:
            con.print(f"  [dim]{t['signed']} signed, use --verbose to list[/dim]", soft_wrap=True)
    if report.spec_status == "incomplete":
        con.print(
            "\n[dim]spec.yml still has TODO fields: write them, then run `agenticreq check` again.[/dim]",
            soft_wrap=True,
        )
    return "" if console is not None else buf.getvalue()


def _nothing_checked(report: Report) -> str:
    a = report.metadata.get("analysis")
    info: dict[str, object] = a if isinstance(a, dict) else {}
    if info.get("parse_errors"):
        why = "the Python file(s) could not be parsed (syntax error)"
    elif not info.get("files"):
        why = "no Python file was found"
    else:
        why = "no agent, tool or supported framework was found in the code"
    return f"Nothing was checked: {why}. This is NOT a pass. Point agenticreq at the folder that holds your agent code."


_KIND = {
    "invalid": "wrong shape",
    "prose": "written as prose (kept as a note; a gap until written as structure)",
    "placeholder": "placeholder, not expanded (a gap until written)",
    "todo": "TODO",
    "missing": "missing",
}


def _issue_lines(n: int, i: SpecIssue) -> list[str]:
    where = f"  [dim]{escape(str(i.spec_ref))}[/dim]" if i.spec_ref else ""
    style = "bold red" if i.kind == "invalid" else "red"
    kind = "unknown field" if i.rule_id == "AS-117" else _KIND[i.kind]
    out = [f"[{style}]{n}. {escape(i.field)}[/{style}] - {kind}{where}", f"   {escape(i.message)}"]
    if i.items and (len(i.items) > 1 or i.kind == "invalid"):
        out.append(f"   [dim]found ({len(i.items)}):[/dim]")
        out += [f"     - {escape(x)}" for x in i.items]
    if i.expected:
        out.append(f"   [dim]expected:[/dim] {escape(i.expected)}")
    if i.example:
        out.append("   [green]write it like:[/green]")
        out += [f"     {escape(x)}" for x in i.example.splitlines()]
    return out


def _render_spec_invalid(con: Console, report: Report) -> None:
    bad = [i for i in report.spec_issues if i.kind == "invalid"]
    rest = [i for i in report.spec_issues if i.kind != "invalid"]
    head = (
        f"[bold red]SPEC INVALID ({len(bad)})[/bold red] - {len(bad)} field{'s' if len(bad) != 1 else ''} with "
        "the wrong shape; no rule ran, nothing about the agent was checked"
    )
    if rest:
        head += f"\n[red]ALSO TO WRITE ({len(rest)})[/red] - fields that are TODO, prose or a placeholder"
    con.print(Panel(head, box=panel_box(con)))
    sections = [
        (f"SPEC INVALID ({len(bad)})", "blocks everything; no rule ran", "bold red", bad),
        (f"ALSO TO WRITE ({len(rest)})", "gaps until written as structure", "red", rest),
    ]
    for name, why, style, items in sections:
        if not items:
            continue
        con.print(f"\n[{style}]{name}[/{style}]  [dim]{why}[/dim]")
        for n, i in enumerate(items, 1):  # numbering restarts in each section
            con.print("")
            for line in _issue_lines(n, i):
                con.print("  " + line)
    con.print(
        "\n[dim]`agenticreq validate --fix <spec>` rewrites the unambiguous ones (dry-run); "
        "`agenticreq params` shows every field's shape.[/dim]"
    )


def _render_scope(con: Console, scope: dict[str, object], width: int, sym: dict[str, str]) -> None:
    """audit has no spec: say plainly what was and was not read, before any result. Each line fits the width."""
    fw = scope.get("framework") or "not recognised"

    def line(text: str) -> None:
        indent = len(text) - len(text.lstrip()) + 2
        con.print(f"[dim]{escape(chr(10).join(_wrap(text, width, indent)))}[/dim]", soft_wrap=True)

    line(f"audit without a spec (framework: {fw}): built-in defaults, not rules you wrote.")
    line("To check YOUR rules: `agenticreq init`, fill spec.yml, then `agenticreq check`.")
    for key, label in (
        ("checked", "checked"),
        ("not_checked", "not read from code"),
        ("needs_spec", "needs a spec (Level 1)"),
    ):
        vals = scope.get(key) or []
        if vals:
            line(f"  {label}: {', '.join(map(str, vals))}")  # type: ignore[call-overload]
    why = scope.get("framework_note")
    if why:
        line(f"  {why}")
