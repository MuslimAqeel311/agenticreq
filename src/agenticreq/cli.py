"""agenticreq CLI: audit · init · mocks · mock-run · check · export · fix · stats (+ validate, rules, params,
export-spec)."""

from __future__ import annotations

import errno
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
from rich.markup import escape

from agenticreq import __version__
from agenticreq.models.finding import Report
from agenticreq.models.observed import ObservedAgent
from agenticreq.report.term import make_console

app = typer.Typer(
    add_completion=False, help="agenticreq — is what the agent's spec declares actually enforced in its code? (no LLM)"
)

#: `agenticreq` with no command
OVERVIEW = """agenticreq checks an AI agent against a written spec, in 5 steps:

  agenticreq audit      1. quick scan, no setup
  agenticreq init       2. write spec.yml: what the agent must and must not do
  agenticreq check      3. compare spec.yml with the project
  agenticreq mocks      4. write mocks.yml, then: agenticreq mock-run (test run with mock tools)
  confirmed: entries    5. the owner confirms what the scan cannot see, in spec.yml

Where this folder stands and what to run next: agenticreq guide
Every command: agenticreq --help"""


@app.callback(invoke_without_command=True)
def _overview(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(OVERVIEW)
        raise typer.Exit(0)


err = make_console(stderr=True)


class Fmt(str, Enum):
    console = "console"
    json = "json"
    sarif = "sarif"


class ExportFmt(str, Enum):
    opa = "opa"
    cedar = "cedar"
    schema = "schema"


CapturesOpt = typer.Option(
    None,
    "--captures",
    "-c",
    help="a mock run file (default: the latest `agenticreq mock-run`, .agenticreq/captures.json)",
)
TracesOpt = typer.Option(None, "--traces", "-t", help="OTel GenAI spans (OTLP JSON / JSONL, file or dir)")
McpOpt = typer.Option(None, "--mcp", help="MCP server command; agenticreq runs tools/list against it")
McpToolsOpt = typer.Option(None, "--mcp-tools", help="JSON file with a saved MCP tools/list result")
FormatOpt = typer.Option(Fmt.console, "--format", "-f")
OutputOpt = typer.Option(None, "--output", "-o", help="write the report here instead of stdout")


def _run_observed(run_file: Path, captured: int) -> Optional[ObservedAgent]:
    """The code plus the run just saved, for the mock-run summary; None when there is nothing to read."""
    if not captured:
        return None
    try:
        return _observe(Path.cwd(), run_file, None, None, None)
    except Exception:  # noqa: BLE001 - the summary must not hide the run's own result
        return None


def _commented_out(path: Path, key: str) -> bool:
    """A top-level ``# key: ...`` line in mocks.yml that names a value, and no live one: why a key looks missing."""
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(rf"^#\s*{key}:\s*\S", text, re.M)) and not re.search(rf"^{key}:", text, re.M)


def _asked_in(res: Any, run_id: str) -> list[str]:
    """The tools the canned model asked for in one run, in order."""
    return list(dict.fromkeys(tc.name for c in res.calls if c.run_id == run_id for tc in c.tool_calls))


def _size(n: int) -> str:
    return f"{n} bytes" if n < 1024 else f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def _rel(path: str) -> str:
    """A path relative to the project folder when it is inside it, with / separators."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path


#: added to the reason when a tool that no run reached is also call: false (the two are separate facts)
_CALL_FALSE = (
    "It is also call: false in mocks.yml: once a run reaches it the model is offered it but never asked to call it."
)


def _never_offered(
    tools: list[str], obs: Optional[ObservedAgent], captured: int, skipped: Optional[list[str]] = None
) -> list[str]:
    """One line per reason a mocks.yml tool never reached the model, from the code and the runs just saved
    ("analyze_csv, calculator: the run never reached analyst. Add a run under runs: ..."). A tool that is also
    call: false keeps its own reason: call: false never hides a tool no run reached."""
    if not tools:
        return []
    from agenticreq.names import contains

    def tag(t: str) -> str:
        return " " + _CALL_FALSE if skipped and contains(skipped, t) else ""

    if captured and obs is not None:
        from agenticreq.rules.conformance import _never_reached_reason

        def why(t: str) -> str:
            return _never_reached_reason(obs, t).removeprefix("Never offered to the model: ")
    elif captured:

        def why(t: str) -> str:
            return "add a run under runs: in mocks.yml with an input that reaches it"
    else:  # no model call at all: the run's own failure says why

        def why(t: str) -> str:
            return "the agent made no model call"

    by_reason: dict[str, list[str]] = {}
    for t in tools:
        by_reason.setdefault(why(t) + tag(t), []).append(t)
    return [f"{', '.join(ts)}: {r}" for r, ts in by_reason.items()]


def _observe(
    path: Path, captures: Optional[Path], traces: Optional[Path], mcp: Optional[str], mcp_tools: Optional[Path]
) -> ObservedAgent:
    from agenticreq.capture.discover import load_capture_bundle, load_patched_tools, load_unsupported
    from agenticreq.observe import build_observed

    calls, versions = load_capture_bundle(captures) if captures else ([], {})
    if captures and not calls:
        err.print(
            "[yellow]warning:[/yellow] the mock run holds no captured LLM call (it crashed or never called a "
            "model), so it adds no runtime evidence. Fix the run and run `agenticreq mock-run` again."
        )
    spans = None
    if traces:
        from agenticreq.traces import load_traces

        spans = load_traces(traces)
    tools = []
    if mcp_tools:
        from agenticreq.mcp import load_tools_file

        tools += load_tools_file(mcp_tools)
    if mcp:
        from agenticreq.mcp import list_tools_stdio

        tools += list_tools_stdio(mcp)
    obs = build_observed(path, captures=calls, spans=spans, mcp_tools=tools, runtime_versions=versions)
    if captures:
        obs.patched_tools = load_patched_tools(captures)
        obs.mock_unsupported = load_unsupported(captures)
    return obs


class FailOn(str, Enum):
    gaps = "gaps"
    all = "all"
    never = "never"


FailOnOpt = typer.Option(
    FailOn.gaps,
    "--fail-on",
    help="exit 1 on: gaps (default: a spec vs code gap) | all (also an open sign-off or a check "
    "still waiting for a mock run) | never (report only, always exit 0)",
)
VerboseOpt = typer.Option(False, "--verbose", "-v", help="show rule ids, file:line and the internal bucket view")
QuietOpt = typer.Option(False, "--quiet", help="do not print the progress block (Your progress / Next:)")


def _show_progress(
    root: Path, spec_path: Optional[Path] = None, summary: Optional[dict[str, Any]] = None, audit_ran: bool = False
) -> None:
    """The five-step progress block, on stderr after the command's own output."""
    from agenticreq.progress import compute, render
    from agenticreq.report.console import _symbols

    p = compute(root, spec_path, summary=summary, audit_ran=audit_ran)
    err.print("\n" + escape(render(p, unicode=_symbols(err)["gap"] != "x")), soft_wrap=True, highlight=False)


def _exit_code(report: Report, fail_on: FailOn) -> int:
    """0 = PASS, 1 = the --fail-on condition failed, 2 = invalid spec, 3 = nothing checked."""
    from agenticreq.models.verdict import exit_code

    return exit_code(report, fail_on.value)


def _emit(
    report: Report,
    fmt: Fmt,
    output: Optional[Path],
    show_passed: bool = False,
    scanned: Optional[Path] = None,
    verbose: bool = False,
    fail_on: FailOn = FailOn.gaps,
    held_back: Optional[dict[Any, str]] = None,
) -> None:
    from agenticreq.report import render_console, render_json, render_sarif

    if fmt is Fmt.console and output is None:
        try:
            render_console(
                report,
                console=make_console(),
                show_passed=show_passed,
                verbose=verbose,
                fail_on=fail_on.value,
                held_back=held_back,
            )
        except OSError as e:  # the reader went away (`| head`): Windows says EINVAL, POSIX EPIPE
            if e.errno not in (errno.EPIPE, errno.EINVAL):
                raise
            import os

            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # no second error at exit
        return
    base = scanned if scanned is None or scanned.is_dir() else scanned.parent
    if fmt is Fmt.sarif:
        text = render_sarif(report, code_root=base, fail_on=fail_on.value)
    elif fmt is Fmt.json:
        text = render_json(report, fail_on=fail_on.value)
    else:
        text = render_console(
            report, show_passed=show_passed, verbose=verbose, fail_on=fail_on.value, held_back=held_back
        )
    if output:
        output.write_text(text, encoding="utf-8")
        err.print(f"wrote {output}")
    else:
        sys.stdout.write(text + "\n")


def _load_call_args(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        text = Path(raw[1:]).read_text(encoding="utf-8-sig") if raw.startswith("@") else raw
        val = json.loads(text)
    except (OSError, ValueError) as e:
        err.print(f"[red]--args:[/red] {e}")
        raise typer.Exit(2) from None
    if not isinstance(val, (list, dict)):
        err.print("[red]--args:[/red] must be a JSON object or list")
        raise typer.Exit(2)
    return val


MocksOpt = typer.Option(Path("mocks.yml"), "--mocks", help="the mocks file (`agenticreq mocks` writes it)")
#: where `agenticreq mock-run` saves the latest run, and where `check` looks for it
CAPTURES = Path(".agenticreq") / "captures.json"


def _latest_mock_run(path: Path) -> Optional[Path]:
    """The latest mock run for the agent at ``path``: <folder>/.agenticreq/captures.json, else ./.agenticreq/."""
    base = path if path.is_dir() else path.parent
    for cand in (base / CAPTURES, Path.cwd() / CAPTURES):
        if cand.is_file():
            return cand
    return None


def _mock_run_info(captures: Path, obs: ObservedAgent) -> dict[str, object]:
    """What the report header says about a mock run: when, how many calls, and whether the code changed since."""
    import datetime as dt
    import os

    saved = dt.datetime.fromtimestamp(captures.stat().st_mtime)
    captured = total = len(obs.captures)
    try:
        data = json.loads(captures.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        if data.get("saved_at"):
            try:
                saved = dt.datetime.fromisoformat(str(data["saved_at"]))
            except ValueError:
                pass
        cov = str(data.get("coverage") or "")
        if "/" in cov:
            a, _, b = cov.partition("/")
            if a.isdigit() and b.isdigit():
                captured, total = int(a), int(b)
    run_at = captures.stat().st_mtime
    root = Path(obs.root)
    stale = False
    for f in obs.files:
        p = Path(f) if Path(f).is_absolute() else root / f
        try:
            if os.path.getmtime(p) > run_at + 1:
                stale = True
                break
        except OSError:
            continue
    return {
        "time": saved.strftime("%Y-%m-%d %H:%M"),
        "captured": captured,
        "total": total,
        "stale": stale,
        "file": str(captures),
        "patched_tools": list(obs.patched_tools),
        "unsupported": list(obs.mock_unsupported),
    }


@app.command("guide", rich_help_panel="The 5 steps")
def guide(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder (default: this one)"),
    spec: Optional[Path] = typer.Option(None, "--spec", "-s", help="the spec file (default: spec.yml)"),
) -> None:
    """All 5 steps: where this folder stands and the exact next command.

    Reads the same state as the progress block (spec.yml, mocks.yml, .agenticreq/), never the network.
    """
    from agenticreq.progress import folder_of, guide_text
    from agenticreq.report.console import _symbols

    root = folder_of(path)
    typer.echo(guide_text(root, spec, unicode=_symbols(make_console())["gap"] != "x"))


@app.command("mock-run", rich_help_panel="The 5 steps")
def mock_run(
    mocks: Path = MocksOpt,
    entry: Optional[str] = typer.Option(
        None, "--entry", help="overrides entry: in mocks.yml (module:function or file.py[:function])"
    ),
    args: Optional[str] = typer.Option(
        None,
        "--args",
        help="overrides args: in mocks.yml: a JSON object (keywords), a JSON list (positional) or @file.json",
    ),
    python: Optional[str] = typer.Option(None, "--python", help="overrides python: in mocks.yml"),
    runs: int = typer.Option(1, "--runs", min=1, help="repeat each mock run this many times"),
    run: Optional[list[str]] = typer.Option(
        None, "--run", help="run only this mock run from runs: in mocks.yml (repeat the flag for several; default: all)"
    ),
    keep: bool = typer.Option(
        False,
        "--keep",
        help="keep the runs already saved in --out that are not run now (a run run again replaces its earlier capture)",
    ),
    timeout: float = typer.Option(120.0, "--timeout"),
    out: Path = typer.Option(CAPTURES, "--out", help="where to save the run (`agenticreq check` reads it from here)"),
    quiet: bool = QuietOpt,
) -> None:
    """Step 4 (mock-run): run the agent with the mock tools in mocks.yml: every run under runs:, or once.

    Run the agent with a canned model and the tools in mocks.yml, and save what it sent to the model.
    One run per entry under runs: (each with its own task); a file without runs: is one run. `check` counts a
    tool as offered when any saved run offered it.

    Reads entry, args and python from mocks.yml (`agenticreq mocks` writes it), so from the agent's folder
    `agenticreq mock-run` needs no flags. python: empty means ./.venv when it exists, else this interpreter.

    SAFETY: this executes the agent's code. Real credentials are withheld, and network connects, other programs
    and file writes outside a scratch area are blocked at the Python level. Run it on projects you trust; for a
    project from someone else, use a separate machine. Refuses without a mocks file (no bypass flag).
    """
    from agenticreq.capture import DiscoverError, MockRunRefused, save_captures
    from agenticreq.capture import discover as run_mock
    from agenticreq.capture.guard import NO_MOCKS
    from agenticreq.capture.responder import MockConfig
    from agenticreq.pyenv import choose, preflight

    if not mocks.is_file():
        err.print(f"[red]{escape(NO_MOCKS)}[/red]")
        raise typer.Exit(2)
    try:
        cfg = MockConfig.load(mocks)
    except (OSError, ValueError) as e:
        err.print(f"[red]{escape(str(mocks))} cannot be read:[/red] {escape(str(e).splitlines()[0])}")
        raise typer.Exit(2) from None
    unknown_runs = [n for n in run or [] if n not in cfg.run_names()]
    if unknown_runs:
        have = ", ".join(cfg.run_names()) or "none: add runs: to mocks.yml"
        err.print(
            f"[red]no mock run named {escape(', '.join(unknown_runs))} in {escape(str(mocks))}[/red] "
            f"(its runs: {escape(have)})"
        )
        raise typer.Exit(2)
    chosen = [r for r in cfg.runs if not run or r.name in run]
    entry_arg = entry
    entry = entry or cfg.entry
    lacks = not entry and (not chosen or any(not r.entry for r in chosen))  # a run with no entry: has nothing to run
    entry = entry or next((r.entry for r in chosen if r.entry), None)
    if lacks or not entry:
        hidden = " (the entry: line is commented out)" if _commented_out(mocks, "entry") else ""
        err.print(
            f"[red]{escape(str(mocks))} has no entry:[/red]{hidden} write the function that runs your agent under "
            "entry: (e.g. my_package.agent:run), or pass --entry"
        )
        raise typer.Exit(2)
    call_args = _load_call_args(args) if args is not None else cfg.args
    pick = choose(Path.cwd(), python, cfg.python)
    py = pick.path
    err.print(escape(pick.line()))
    if pick.warning:
        err.print(f"[yellow]warning:[/yellow] {escape(pick.warning)}")
    try:
        from agenticreq.capture.discover import resolve_python

        py = resolve_python(py)
    except DiscoverError as e:
        err.print(f"[red]mock-run stopped:[/red] {escape(str(e))}")
        raise typer.Exit(2) from None
    ready = preflight(py, Path.cwd(), entry, pick.source)
    for w in ready.warnings:
        err.print(f"[yellow]warning:[/yellow] {escape(w)}")
    if ready.problems:  # stop before the run: nothing ran, so nothing is saved
        for n in ready.problems:
            err.print(f"[red]mock-run stopped:[/red] {escape(n)}")
        raise typer.Exit(2)
    try:
        res = run_mock(
            entry,
            mock_tools=str(mocks),
            runs=runs,
            python=py,
            timeout=timeout,
            call_args=call_args,
            run_names=list(run) if run else None,
            use_run_inputs=entry_arg is None and args is None,
        )
    except MockRunRefused as e:
        err.print(f"[red]refused:[/red] {escape(str(e))}")
        raise typer.Exit(2) from None
    except DiscoverError as e:
        err.print(f"[red]mock-run stopped:[/red] {escape(str(e))}")
        raise typer.Exit(2) from None
    p = save_captures(res, out, keep=keep)
    cap, tot = res.coverage
    from agenticreq.names import not_in

    offered = res.offered_tools()
    listed = list(dict.fromkeys([*cfg.order, *cfg.tools]))
    skipped = [t for t in listed if t in cfg.tools and not cfg.tools[t].call]
    # mocks.yml may spell a tool as the code shows it ("Web Search") and the wire as the model sees it (web_search):
    # one tool, so it is never both offered and never offered
    # call: false only means "do not ask the model to call it": a call: false tool no run reached is still listed
    never = list(not_in(listed, offered))
    obs = _run_observed(p, cap)
    schemas = [t for t in offered if obs is not None and obs.is_output_schema(t)]  # sent as tools, not tools
    if cfg.runs:  # per-run lines first, then the combined summary
        for rid, names in res.offered_by_run().items():
            n_calls = sum(1 for c in res.calls if c.run_id == rid)
            names_shown = [t for t in names if not (obs is not None and obs.is_output_schema(t))]
            asked = _asked_in(res, rid)
            tail = f": {', '.join(names_shown) if names_shown else 'no tools offered'}"
            tail += f"; asked for: {', '.join(asked)}" if asked else ""
            err.print(
                f"mock run {escape(rid)}{' (kept)' if rid in res.kept else ''}: "
                f"{n_calls} LLM call{'' if n_calls == 1 else 's'}{escape(tail)}"
            )
    from agenticreq.ast.classify import is_handoff_tool

    for name in res.unsupported:
        err.print(f"mock run is not supported for {escape(name)}")
    for rid in res.run_ids:
        handoffs = [t for t in _asked_in(res, rid) if is_handoff_tool(t)]
        if len(handoffs) > 1:
            err.print(
                f"the canned model followed every handoff in {escape(rid)} ({escape(', '.join(handoffs))}). To take "
                "one path, give a run its own order: under runs: in mocks.yml"
            )
    err.print(f"captured {cap}/{tot} LLM calls in {res.runs} run(s)")
    tools_seen = [t for t in offered if t not in schemas]
    err.print(f"tools offered to the model: {escape(', '.join(tools_seen)) if tools_seen else 'none'}")
    if schemas:
        err.print(f"structured-output schemas (sent as tools): {escape(', '.join(schemas))}")
    if skipped:
        err.print(f"not asked for (call: false in mocks.yml): {escape(', '.join(skipped))}")
    for line in _never_offered(never, obs, cap, skipped):
        err.print(f"never offered to the model: {escape(line)}")
    if not res.zero_touch:
        err.print(f"[red]the run changed the agent's files:[/red] {escape(', '.join(res.changed_files[:5]))}")
    for rel, size in res.wrote:
        err.print(f"wrote {escape(rel)}, {_size(size)} (allowed by allow_write: in mocks.yml)")
    for path in res.blocked_writes()[:5]:
        rel_path = _rel(path)
        err.print(f"blocked write to {escape(rel_path)}; to allow it, add it under allow_write: in mocks.yml")
    other = list(dict.fromkeys(b for b in res.egress_blocked if not b.startswith("write:")))
    if other:
        err.print(f"blocked during the run (network / other programs): {other[:5]}")
    for n in res.notes:
        err.print(f"[yellow]note:[/yellow] {escape(n)}")
    if res.scrubbed_env:
        err.print(f"credentials withheld from the agent: {res.scrubbed_env}")
    err.print(f"saved: {p} (`agenticreq check` reads it)")
    problems = res.problems(py)
    for n in problems:  # each is a full sentence; the ones about the run itself start with "mock-run captured"
        err.print(
            f"[red]{escape(n)}[/red]" if n.startswith("mock-run ") else f"[red]mock-run failed:[/red] {escape(n)}"
        )
    if not quiet:
        _show_progress(Path.cwd())
    if problems:  # crashed, captured nothing, missed calls or touched files: never an all-clear
        raise typer.Exit(1)


@app.command("mocks", rich_help_panel="The 5 steps")
def mocks_cmd(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder (default: this one)"),
    out: Path = typer.Option(Path("mocks.yml"), "--out"),
    spec: Optional[Path] = typer.Option(
        None, "--spec", "-s", help="spec.yml whose prohibited: tools get call: false (default: ./spec.yml)"
    ),
    force: bool = typer.Option(False, "--force", help="overwrite an existing mocks file"),
    quiet: bool = QuietOpt,
) -> None:
    """Step 4 (mock-run), first part: write mocks.yml, the mock tools for a test run.

    Write mocks.yml for `agenticreq mock-run`: the entry that runs the agent, its inputs, and one entry per tool.

    Read-only tools get sample args; tools that write, pay, send or delete get patch + returns so the real function
    never runs; tools listed under prohibited: in spec.yml get call: false. Refuses to overwrite unless --force.
    """
    from agenticreq.ast import extract
    from agenticreq.mockgen import render_mocks

    if out.exists() and not force:
        err.print(f"[red]{escape(str(out))} exists[/red] - refusing to overwrite (use --force)")
        raise typer.Exit(2)
    prohibited: set[str] = set()
    effects: dict[str, str] = {}
    spec_path = spec or (Path("spec.yml") if Path("spec.yml").is_file() else None)
    if spec_path is not None:
        from agenticreq.loader import load_spec

        loaded = load_spec(spec_path)
        if loaded.spec is not None:
            prohibited = loaded.spec.prohibited_tools()
            effects = dict(loaded.spec.tool_effects or {})
    root = (path if path.is_dir() else path.parent).resolve()
    text = render_mocks(root, extract(path), prohibited, effects)
    out.write_text(text, encoding="utf-8")
    import yaml

    data = yaml.safe_load(text) or {}
    tools = data.get("tools") or {}
    entry = data.get("entry")
    err.print(
        f"wrote {escape(str(out))}: {len(tools)} tool(s), entry {escape(str(entry))}"
        + ("" if entry and entry != "TODO" else " - fill entry: and args:, then run `agenticreq mock-run`")
    )
    if not quiet:
        _show_progress(root, spec_path)


@app.command(rich_help_panel="The 5 steps")
def init(
    path: Path = typer.Argument(Path("."), exists=True),
    captures: Optional[Path] = CapturesOpt,
    entry: Optional[str] = typer.Option(None, "--entry", help="run a mock run of this entry first (uses mocks.yml)"),
    mocks: Path = MocksOpt,
    agent_id: Optional[str] = typer.Option(None, "--agent-id"),
    out: Path = typer.Option(Path("spec.yml"), "--out"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing spec file"),
    from_code: bool = typer.Option(
        False,
        "--from-code",
        help="copy the agent and tool names the code has into role_boundary and "
        "agent_registry (a spec copied from the code cannot find mismatches)",
    ),
    quiet: bool = QuietOpt,
) -> None:
    """Step 2 (spec): write spec.yml, what the agent must and must not do, from your project.

    Scaffold spec.yml: _derived filled from the code, TODO for the 12 Level-1 fields.
    Agent and tool names are your requirement: role_boundary and agent_registry are TODO, and the names the code
    has are listed in _derived only (--from-code copies them in).

    The ONLY command that creates spec.yml. Refuses to overwrite an existing file unless --force.
    Business intent (prohibited, limits, ...) is never auto-filled; `check` reports every TODO as a gap.
    It never needs the agent's own dependencies: without --entry it only parses the code, and when
    --entry cannot run the agent (a mock run, with mocks.yml) it falls back to that and says so. The file
    it writes always loads with zero shape errors (only TODOs).
    """
    import tempfile

    from agenticreq.loader import load_spec_text
    from agenticreq.models.observed import ObservedAgent
    from agenticreq.scaffold import render_init

    if out.exists() and not force:
        err.print(f"[red]{out} exists[/red] - refusing to overwrite (use --force)")
        raise typer.Exit(2)
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="agenticreq-init-") as tmp:
        if entry:  # captures stay in a temp dir: init writes nothing but the spec
            from agenticreq.capture import DiscoverError, MockRunRefused, save_captures
            from agenticreq.capture import discover as run_mock
            from agenticreq.capture.responder import MockConfig
            from agenticreq.pyenv import choose

            try:
                cfg_args = MockConfig.load(mocks).args if mocks.is_file() else None
                res = run_mock(entry, mock_tools=str(mocks), python=choose(Path.cwd()).path, call_args=cfg_args)
                captures = save_captures(res, Path(tmp) / "captures.json")
                if res.run_errors and not res.calls:
                    notes.append(
                        f"--entry {entry}: the mock run captured nothing ({next(iter(res.run_errors.values())).strip().splitlines()[-1][:120]}); "
                        "_derived is from the code only"
                    )
            except (MockRunRefused, DiscoverError, OSError, ValueError) as e:
                notes.append(
                    f"--entry {entry}: the mock run could not start ({str(e).splitlines()[0][:160]}); "
                    "_derived is from the code only"
                )
                captures = None
        try:
            obs = _observe(path, captures, None, None, None)
        except Exception as e:  # noqa: BLE001 - init must still write a spec; say what could not be read
            notes.append(f"code extraction failed ({type(e).__name__}: {str(e)[:120]}); _derived is empty")
            obs = ObservedAgent()
    for n in notes:
        err.print(f"[yellow]note:[/yellow] {n}")
    text = render_init(obs, agent_id, notes, from_code=from_code)
    if load_spec_text(text, str(out)).status == "invalid":  # a bug in init, never the user's fault
        notes.append("the derived section could not be written safely and was left out")
        text = render_init(ObservedAgent(), agent_id, notes)
    if from_code:
        from agenticreq.scaffold import FROM_CODE_WARNING

        err.print(f"[yellow]{FROM_CODE_WARNING}[/yellow]")
    out.write_text(text, encoding="utf-8")
    from agenticreq.schema import SCHEMA_FILE, spec_json_schema

    schema_file = out.parent / SCHEMA_FILE
    if not schema_file.exists():  # the file the first line of spec.yml points the editor at
        schema_file.write_text(json.dumps(spec_json_schema(), indent=2) + "\n", encoding="utf-8")
    err.print(
        f"wrote {out} — fill every TODO (`agenticreq params` shows each shape), then run "
        f"`agenticreq check`"
        + ("" if path == Path(".") and out == Path("spec.yml") else f" (`agenticreq check {path} --spec {out}`)")
    )
    if not quiet:
        from agenticreq.progress import folder_of

        _show_progress(folder_of(path), out)


@app.command(rich_help_panel="The 5 steps")
def audit(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder or file (default: this folder)"),
    captures: Optional[Path] = CapturesOpt,
    traces: Optional[Path] = TracesOpt,
    mcp: Optional[str] = McpOpt,
    mcp_tools: Optional[Path] = McpToolsOpt,
    format: Fmt = FormatOpt,
    output: Optional[Path] = OutputOpt,
    show_passed: bool = typer.Option(
        False, "--show-passed", hidden=True, help="kept for compatibility: passed lines always show"
    ),
    spec: Optional[Path] = typer.Option(
        None, "--spec", "-s", exists=True, help="optional: apply ONLY this spec's `waivers:` (nothing else is read)"
    ),
    fail_on: FailOn = FailOnOpt,
    verbose: bool = VerboseOpt,
    quiet: bool = QuietOpt,
) -> None:
    """Step 1 (audit): a quick scan of the project, with no setup.

    Zero-config findings: no spec, no setup, AST only (the agent's dependencies need not be installed).


    CHECKED without a spec (Level 2, built-in defaults for risk_level=medium):
      model version pinned, write tools without an approval gate, missing idempotency key on
      writes, hard-coded secrets, step / cost / duration bounds, unbounded tools (http/sql/shell),
      fail-open error handling, self-modification tools, environment separation, structured
      output, audit / trace hooks.

    NEEDS A SPEC (Level 1 and Level 3 - only you know them):
      prohibited, limits, approval_required, escalation, untrusted_inputs, role_boundary,
      declared_delegation. The report says they need a spec (AS-907); they never count as passed.
      Run `agenticreq init`, fill the TODOs, then `agenticreq check`.

    --spec spec.yml applies only that file's unexpired `waivers:`; audit still checks Level 2 only.
    """
    from agenticreq.engine import run

    waivers = None
    if spec:
        from agenticreq.loader import load_spec

        loaded = load_spec(spec)
        if loaded.spec is None:
            err.print("[red]spec does not load; run `agenticreq validate` first[/red]")
            raise typer.Exit(2)
        waivers = loaded.spec.waivers
    run_file = captures or _latest_mock_run(path)
    obs = _observe(path, run_file, traces, mcp, mcp_tools)
    report = run(mode="audit", observed=obs, waivers=waivers)
    if run_file is not None:
        report.metadata["mock_run"] = _mock_run_info(run_file, obs)
    _emit(report, format, output, show_passed, scanned=path, verbose=verbose, fail_on=fail_on)
    from agenticreq.progress import folder_of, record_audit

    root = folder_of(path)
    record_audit(root)
    if format is Fmt.console and not quiet:
        _show_progress(root, audit_ran=True)
    raise typer.Exit(_exit_code(report, fail_on))


@app.command(rich_help_panel="The 5 steps")
def check(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder or file (default: this folder)"),
    spec: Path = typer.Option(Path("spec.yml"), "--spec", "-s"),
    captures: Optional[Path] = CapturesOpt,
    traces: Optional[Path] = TracesOpt,
    mcp: Optional[str] = McpOpt,
    mcp_tools: Optional[Path] = McpToolsOpt,
    format: Fmt = FormatOpt,
    output: Optional[Path] = OutputOpt,
    show_passed: bool = typer.Option(
        False, "--show-passed", hidden=True, help="kept for compatibility: passed lines always show"
    ),
    fail_on: FailOn = FailOnOpt,
    verbose: bool = VerboseOpt,
    show_all: bool = typer.Option(False, "--all", help="show every group now, not only the ones for this step"),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        help="a table of every parameter (Level 1, 2, 3): its status "
        "and where its value came from (spec, default, code)",
    ),
    quiet: bool = QuietOpt,
) -> None:
    """Step 3 (check): compare spec.yml with the project, and read the latest mock run.

    Level 1+2+3: spec vs code, plus the latest mock run. Every TODO in spec.yml is a gap (AS-101).

    The latest `agenticreq mock-run` (.agenticreq/captures.json) is read automatically; --captures reads another
    run instead.
    """
    from agenticreq.engine import run
    from agenticreq.loader import load_spec

    _need_spec(spec)
    run_file = captures or _latest_mock_run(path)
    obs = _observe(path, run_file, traces, mcp, mcp_tools)
    loaded = load_spec(spec)
    info = _mock_run_info(run_file, obs) if run_file is not None else None
    if info is not None and info.get("stale"):  # the code changed since: its run results would be out of date
        obs = _observe(path, None, traces, mcp, mcp_tools)
    report = run(mode="check", loaded=loaded, observed=obs)
    report.metadata["spec_file"] = str(spec)
    if info is not None:
        report.metadata["mock_run"] = info
    from agenticreq.models.verdict import results
    from agenticreq.progress import check_summary, compute, folder_of, held_back, record_check

    root = folder_of(path)
    summary = check_summary(report, spec)
    record_check(root, summary)
    if coverage:
        from agenticreq.coverage_table import render as coverage_text
        from agenticreq.coverage_table import rows as coverage_rows

        sys.stdout.write(coverage_text(coverage_rows(report, loaded)))
        raise typer.Exit(_exit_code(report, fail_on))
    hold = None
    if not show_all and report.spec_status != "invalid":  # show the groups that matter at this step
        hold = held_back(compute(root, spec, summary=summary), results(report))
    _emit(report, format, output, show_passed, scanned=path, verbose=verbose, fail_on=fail_on, held_back=hold)
    if format is Fmt.console and not quiet:
        _show_progress(root, spec, summary)
    code = _exit_code(report, fail_on)
    if report.spec_status == "incomplete" and fail_on is not FailOn.never:
        code = max(code, 1)  # check refuses an unfinished spec: undeclared rules were not checked
    raise typer.Exit(code)


@app.command("export-schema", rich_help_panel="More")
def export_schema() -> None:
    """Step 2 (spec): print the JSON Schema of spec.yml, so an editor completes field names and flags a wrong one.

    agenticreq export-schema > spec.schema.json. `agenticreq init` writes it next to spec.yml and points the first
    line of spec.yml at it (# yaml-language-server: $schema=./spec.schema.json).
    """
    from agenticreq.schema import spec_json_schema

    sys.stdout.write(json.dumps(spec_json_schema(), indent=2) + "\n")


@app.command("export-spec", rich_help_panel="More")
def export_spec(
    target: str = typer.Argument(
        ..., help="file.py:NAME or package.module:NAME: a SpecBuilder, AgentSpec, dict, or a function returning one"
    ),
    out: Path = typer.Option(Path("spec.yml"), "--out", "-o"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing file"),
) -> None:
    """Step 2 (spec): write spec.yml from a spec built in Python.

    Write a spec built with the Python API out as YAML. spec.yml stays the source of truth.

    This imports TARGET's module (running its top-level code) - exactly why the canonical form is
    YAML: CI, reviewers and other environments must read the spec without importing the agent.
    """
    from agenticreq.loader.builder import load_python_spec, write_spec

    if out.exists() and not force:
        err.print(f"[red]{out} exists[/red] - refusing to overwrite (use --force)")
        raise typer.Exit(2)
    try:
        obj = load_python_spec(target)
        write_spec(obj, out, force=True)
    except (ValueError, ImportError, AttributeError, FileNotFoundError) as e:
        err.print(f"[red]cannot export:[/red] {e}")
        raise typer.Exit(2) from None
    err.print(f"wrote {out} - commit it; `agenticreq check` reads the YAML, not the Python")


@app.command(rich_help_panel="More")
def params(
    level: Optional[int] = typer.Option(None, "--level", "-l", min=1, max=4),
    field: Optional[str] = typer.Option(None, "--field", help="show one field only, e.g. --field limits"),
    brief: bool = typer.Option(False, "--brief", help="one line per field, without shapes and examples"),
    format: Fmt = typer.Option(Fmt.console, "--format", "-f"),
) -> None:
    """Step 2 (spec): list every field spec.yml can hold, with its shape and an example.

    Every spec field: level, how agenticreq verifies it, its exact shape and a filled example."""
    from agenticreq.params import BY_NAME, PARAMS
    from agenticreq.shapes import example, example_yaml, shape, spec_fields
    from agenticreq.thresholds import BY_KEY

    known = set(spec_fields())
    rows = [p for p in PARAMS if (level is None or p.level == level) and (field is None or p.name == field)]
    meta = [
        k
        for k in ("spec_version", "extends", "confirmed", "tool_effects", "waivers", "_derived")
        if level is None and (field is None or k == field)
    ]
    if field and not rows and not meta:
        import difflib

        from agenticreq.loader.issues import suggest

        near = suggest(field) or difflib.get_close_matches(field, sorted(known), n=3)
        err.print(f"[red]no field {field!r}[/red]" + (f" - did you mean {', '.join(near)}?" if near else ""))
        raise typer.Exit(2)
    if format is not Fmt.console:
        out = [
            {
                "name": p.name,
                "level": p.level,
                "group": p.group,
                "verified_by": p.how,
                "rules": list(p.rules),
                "audit_without_spec": p.audit,
                "note": p.note,
                "also_called": {full: list(words) for full, words in p.also_called()},
                **(
                    {"shape": shape(p.name), "example": example(p.name), "example_yaml": example_yaml(p.name)}
                    if p.name in known
                    else {}
                ),
            }
            for p in rows
        ]
        out += [
            {"name": k, "level": "meta", "shape": shape(k), "example": example(k), "example_yaml": example_yaml(k)}
            for k in meta
        ]
        sys.stdout.write(json.dumps(out, indent=2, default=str) + "\n")
        return
    from agenticreq.report.term import make_console

    con = make_console()
    names = {
        1: "Level 1 - user writes",
        2: "Level 2 - settings by risk_level (yes/no and choices; every number is yours)",
        3: "Level 3 - multi-agent",
        4: "Level 4 - derived (never written)",
    }
    from agenticreq.not_needed import PARAMS_NOTE

    con.print(f"[dim]{escape(PARAMS_NOTE)}[/dim]")

    def show(name: str) -> None:
        p = BY_NAME.get(name)
        for full, words in p.also_called() if p else []:  # the names a wrong spec.yml key is matched against
            con.print(
                f"      [dim]also called:[/dim] {escape(', '.join(words))}"
                + ("" if full == name else f"  [dim]({escape(full)})[/dim]")
            )
        if name in known:  # a number is yours to write: only what is valid is shown, never a value
            for key in (k for k in BY_KEY if k.split(".")[0] == name):
                valid = "a number from zero to one" if key == "confidence_threshold" else "a positive number"
                con.print(f"      [dim]{escape(key)}:[/dim] you write it ({valid}), or not_needed with a reason")
        if brief or name not in known:
            return
        con.print(f"      [dim]shape:[/dim]   {escape(shape(name))}")
        con.print("      [dim]example:[/dim]")
        for line in example_yaml(name).splitlines():
            con.print(f"        {escape(line)}")

    for lv in sorted({p.level for p in rows}):
        con.print(f"\n[bold]{names[lv]}[/bold]")
        for p in (x for x in rows if x.level == lv):
            tag = " [green]audit[/green]" if p.audit else ""
            con.print(
                f"  [bold]{p.name:<26}[/bold] {p.how:<12} {' '.join(p.rules):<34}{tag}"
                + (f"  [dim]{escape(p.note)}[/dim]" if p.note else "")
            )
            show(p.name)
    if meta:
        con.print("\n[bold]Meta[/bold]")
        for k in meta:
            con.print(f"  [bold]{k}[/bold]")
            show(k)


def _need_spec(spec: Path) -> None:
    """A command that needs a spec says so in one plain line, never a traceback."""
    if not spec.is_file():
        err.print(
            f"[red]no spec at {escape(str(spec))}[/red]: this command needs one. Write it with `agenticreq init`, "
            "or pass its path"
        )
        raise typer.Exit(2)


@app.command(rich_help_panel="More")
def validate(
    spec: Path = typer.Argument(Path("spec.yml"), help="the spec (default: spec.yml)"),
    format: Fmt = FormatOpt,
    output: Optional[Path] = OutputOpt,
    fix: bool = typer.Option(
        False,
        "--fix",
        help="rename wrong field names that have one clear match and "
        "rewrite unambiguous prose; shows a diff, then asks",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="with --fix: write without asking"),
    apply: bool = typer.Option(False, "--apply", hidden=True, help="kept for compatibility: same as --yes"),
) -> None:
    """Step 2 (spec): check spec.yml on its own, before comparing it with the project.

    Validate a spec on its own (AS-1xx). Exit 2 = invalid shape (nothing checked), 1 = blockers, 0 = loads.

    TODOs, prose and placeholders leave fields undeclared: `check` reports each one as a spec vs code gap
    until it is written. --fix renames a wrong field name only when exactly one field is clearly meant (never two
    equally close ones, never a guess), moves prose verbatim under `note:` and marks the structure TODO; it never
    guesses what a sentence means. It shows the diff and asks before writing (--yes writes without asking).
    """
    from agenticreq.engine import validate_spec

    _need_spec(spec)
    if fix:
        from agenticreq.loader import load_spec_text
        from agenticreq.loader.fix_spec import fix_spec, write

        res = fix_spec(spec)
        if not (res.rewrites or res.renames):
            err.print("validate --fix: nothing unambiguous to rewrite")
        for rn in res.renames:
            err.print(f"  rename `{rn.old}` -> `{rn.new}` (line {rn.line})")
        for u in res.unrenamed:
            why = (
                f"`{u.near[0]}` is already there: merge the two by hand"
                if u.taken
                else f"equally close: {', '.join(f'`{n}`' for n in u.near)}"
                if len(u.near) > 1
                else f"it belongs in another mapping: move it to `{u.near[0]}` by hand"
                if u.near
                else "no close field"
            )
            err.print(f"  not renamed `{u.name}`" + (f" (line {u.line})" if u.line else "") + f": {why}")
        for r in res.rewrites:
            err.print(f"  {r.field}: {r.why}")
        after = load_spec_text(res.after.lstrip("\ufeff"), str(spec))
        left = [i for i in after.issues if i.kind == "invalid"]
        if res.rewrites or res.renames:
            sys.stdout.write(res.diff())
            sys.stdout.flush()  # the diff, then the question
            go = yes or apply
            if not go:
                try:
                    go = typer.confirm(f"Write these changes to {spec}?", default=False, err=True)
                except typer.Abort:  # no terminal to answer from
                    go = False
            if go:
                write(res)
                err.print(f"wrote {spec}")
            else:
                err.print("[dim]nothing written. Re-run with --fix --yes to write without asking.[/dim]")
        err.print(
            f"after the rewrite the spec is {after.status}"
            + (
                f"; still wrong shape (not unambiguous, fix by hand): {', '.join(i.field for i in left)}"
                if left
                else ""
            )
            + ("; fill the TODOs, then run `agenticreq check`" if after.todos else "")
        )
        raise typer.Exit(2 if left else 0)
    report = validate_spec(spec)
    report.metadata["spec_file"] = str(spec)
    _emit(report, format, output)
    raise typer.Exit(report.exit_code)


@app.command(rich_help_panel="More")
def export(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder (default: this one)"),
    spec: Path = typer.Option(Path("spec.yml"), "--spec", "-s"),
    format: ExportFmt = typer.Option(ExportFmt.opa, "--format", "-f"),
    output: Optional[Path] = OutputOpt,
) -> None:
    """After step 3 (check): turn spec.yml into a policy for OPA or Cedar.

    Export OPA (Rego) / Cedar policy or tool-schema constraints. Enforcement happens there, not here."""
    from agenticreq.ast import extract
    from agenticreq.export import to_cedar, to_rego, to_schema_constraints
    from agenticreq.loader import load_spec

    _need_spec(spec)
    loaded = load_spec(spec)
    if loaded.spec is None:
        err.print("[red]spec does not load; run `agenticreq validate` first[/red]")
        raise typer.Exit(2)
    obs = extract(path)
    if format is ExportFmt.schema:
        text = json.dumps(to_schema_constraints(loaded.spec, obs), indent=2)
    else:
        text = (to_rego if format is ExportFmt.opa else to_cedar)(loaded.spec, obs)
    if output:
        output.write_text(text, encoding="utf-8")
        err.print(f"wrote {output}")
    else:
        sys.stdout.write(text)


@app.command(rich_help_panel="More")
def fix(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder (default: this one)"),
    spec: Optional[Path] = typer.Option(None, "--spec", "-s", exists=True),
    apply: bool = typer.Option(False, "--apply", help="write the patches (default: dry-run, print a diff)"),
    only: Optional[str] = typer.Option(None, "--only", help="comma-separated rule ids, e.g. AS-210,AS-208"),
) -> None:
    """Step 3 (check): patch the project for gaps that have a mechanical fix.

    Structural autofix: step cap, idempotency key, model pinning. Dry-run unless --apply."""
    from agenticreq.ast import extract
    from agenticreq.fix import apply as apply_patches
    from agenticreq.fix import plan, render_diff
    from agenticreq.loader import load_spec

    risk, user = None, {}
    if spec:
        loaded = load_spec(spec)
        risk = loaded.spec.risk_level if loaded.spec else None  # noqa: F841
        user = loaded.user_tier2
    steps = (user.get("bounds") or {}).get("max_steps") if isinstance(user.get("bounds"), dict) else None
    max_steps = int(steps) if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps > 0 else None
    if max_steps is None:
        err.print(
            "bounds.max_steps is not set in spec.yml: no step cap is written (write your own number there, "
            "then run fix again)"
        )
    patches = plan(extract(path), path, max_steps)
    if only:
        keep = {x.strip() for x in only.split(",")}
        patches = [p for p in patches if p.rule_id in keep]
    if not patches:
        err.print("nothing to fix")
        return
    for p in patches:
        err.print(f"  {p.rule_id}  {p.description}")
    base = path if path.is_dir() else path.parent
    if apply:
        files = apply_patches(patches)
        err.print(f"patched {len(files)} file(s)")
    else:
        sys.stdout.write(render_diff(patches, base))
        err.print("[dim]dry-run: nothing written. Re-run with --apply to write.[/dim]")


@app.command(rich_help_panel="More")
def stats(
    path: Path = typer.Argument(Path("."), exists=True, help="the agent's folder (default: this one)"),
    spec: Optional[Path] = typer.Option(None, "--spec", "-s", exists=True),
    captures: Optional[Path] = CapturesOpt,
    traces: Optional[Path] = TracesOpt,
    format: Fmt = typer.Option(Fmt.console, "--format", "-f"),
) -> None:
    """Step 3 (check): the numbers behind a check, for tracking over time.

    Agent registry + aggregate numbers (+ Tier-4 derived values when a spec is given)."""
    from agenticreq.derived import stats as compute
    from agenticreq.engine import run
    from agenticreq.loader import load_spec

    loaded = load_spec(spec) if spec else None
    s = loaded.spec if loaded else None
    obs = _observe(path, captures, traces, None, None)
    # enforceability is read from the rule results, so the rules run once here too
    passed = run(mode="check", loaded=loaded, observed=obs).passed if loaded is not None and s is not None else None
    data = compute(obs, s, passed)
    if format is Fmt.console:
        con = make_console()
        for k, v in data.items():
            con.print(f"[bold]{k}[/bold]: {json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}")
    else:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


@app.command(rich_help_panel="More")
def rules() -> None:
    """Any step: list every rule agenticreq checks.

    List every rule id, severity and quality attribute."""
    from agenticreq.engine import _load_rule_modules
    from agenticreq.report.json_report import SEVERITY
    from agenticreq.rules.registry import REGISTRY

    _load_rule_modules()
    con = make_console()
    for m in REGISTRY.all():  # severity in the words of the result groups ("sign-off"), as in the JSON report
        con.print(
            f"{m.rule_id}  {SEVERITY.get(m.severity, m.severity):<11} {m.attribute.value:<20} {m.title}"
            + ("  [fixable]" if m.fixable else "")
        )


@app.command(rich_help_panel="More")
def version() -> None:
    """Any step: print the agenticreq version.

    Print the version."""
    print(__version__)


if __name__ == "__main__":
    app()
