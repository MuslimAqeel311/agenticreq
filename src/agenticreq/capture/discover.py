"""The mock run (`agenticreq mock-run`): run the agent in a guarded subprocess, with a canned model and the tools
from mocks.yml, and capture its LLM boundary. The module keeps its old internal name; no user sees it."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenticreq.capture.entry import split_entry
from agenticreq.capture.guard import build_child_env, gate
from agenticreq.models.observed import CapturedCall

_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".agenticreq", "node_modules", ".mypy_cache", ".pytest_cache"}


def tree_hashes(root: Path) -> dict[str, str]:
    """sha256 of every file under root (zero-touch proof)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (_SKIP_DIRS & set(p.relative_to(root).parts)):
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@dataclass
class DiscoverResult:
    entry: str
    calls: list[CapturedCall] = field(default_factory=list)
    runs: int = 0
    run_errors: dict[str, str] = field(default_factory=dict)
    llm_misses: list[str] = field(default_factory=list)
    egress_blocked: list[str] = field(default_factory=list)
    libs_hooked: list[str] = field(default_factory=list)
    scrubbed_env: list[str] = field(default_factory=list)
    zero_touch: bool = True
    changed_files: list[str] = field(default_factory=list)
    #: files the run wrote inside allow_write: paths (relative path, size in bytes): allowed, listed, not a change
    wrote: list[tuple[str, int]] = field(default_factory=list)
    #: run id -> what stopped it (runner.describe_failure, or a timeout with the step reached)
    failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: requests the mock model could not answer
    rejections: list[str] = field(default_factory=list)
    timeout: float = 0.0
    versions: dict[str, str] = field(default_factory=dict)
    mocks: str = ""
    notes: list[str] = field(default_factory=list)
    saved_at: str = ""  # local time the run finished, ISO 8601 (the report header shows it)
    #: every mock run in this result, in order (the ones kept from an earlier save included)
    run_ids: list[str] = field(default_factory=list)
    #: run id -> its own llm_misses / egress_blocked, so --keep can replace one run and keep the rest
    by_run: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    #: run ids kept from the file that was already saved (--keep), not run now
    kept: list[str] = field(default_factory=list)
    #: tools mocks.yml replaced with `patch:` in these runs: only their real body did not run
    patched_tools: list[str] = field(default_factory=list)
    #: providers or transports the run cannot intercept (data/providers.yml): their items go to Human sign-off
    unsupported: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> tuple[int, int]:
        """(captured, captured + missed). Missed = LLM connects that reached the socket guard."""
        n = len(self.calls)
        return n, n + len(self.llm_misses)

    def problems(self, python: str | None = None) -> list[str]:
        """Why these captures are NOT usable proof of the run (empty = usable), each in one plain sentence with
        where it happened and the next step. A run that crashed, timed out, captured nothing, missed LLM calls or
        changed the agent's files must never look like an all-clear."""
        cap, tot = self.coverage
        if self.unsupported:  # nothing a run can show: not a failed run, and the items become sign-offs
            return []
        out = [explain(rid, self.failures.get(rid), msg, cap, self, python) for rid, msg in self.run_errors.items()]
        if self.llm_misses:
            hosts = ", ".join(sorted({m.split(" ")[-1] for m in self.llm_misses})[:3])
            out.append(
                f"mock-run captured {cap} of {tot} model calls. {tot - cap} call(s) went to {hosts} through an "
                "HTTP client the mock run cannot capture (only httpx, httpx2, requests and aiohttp are captured, "
                "which the OpenAI, Anthropic and LangChain SDKs use). Use such a client, or replace that call with patch: in "
                f"mocks.yml, then run: {_NEXT}"
            )
        if cap == 0 and not self.run_errors and not self.llm_misses:
            out.append(
                f"mock-run captured 0 model calls. The entry {self.entry} returned without calling a model (a "
                "cached answer, or not the function that runs the agent). Check entry: and args: in "
                f"mocks.yml, then run: {_NEXT}"
            )
        if not self.zero_touch:
            out.append(f"the run changed the agent's files: {', '.join(self.changed_files[:5])}")
        return out

    def blocked_writes(self) -> list[str]:
        """Files the guard refused to let the run write (outside allow_write:)."""
        return list(dict.fromkeys(b.removeprefix("write:") for b in self.egress_blocked if b.startswith("write:")))

    def offered_tools(self) -> list[str]:
        """Every tool the model was offered in the captured runs, in first-seen order."""
        return list(dict.fromkeys(t.name for c in self.calls for t in c.tools))

    def offered_by_run(self) -> dict[str, list[str]]:
        """run id -> the tools the model was offered in that run (a run that made no model call lists none)."""
        out: dict[str, list[str]] = {rid: [] for rid in self.run_ids}
        for c in self.calls:
            row = out.setdefault(c.run_id, [])
            row += [t.name for t in c.tools if t.name not in row]
        return out

    def to_dict(self) -> dict[str, Any]:
        cap, tot = self.coverage
        return {
            "entry": self.entry,
            "runs": self.runs,
            "coverage": f"{cap}/{tot}",
            "usable": not self.problems(),
            "zero_touch": self.zero_touch,
            "changed_files": self.changed_files,
            "wrote": self.wrote,
            "libs_hooked": self.libs_hooked,
            "versions": self.versions,
            "llm_misses": self.llm_misses,
            "egress_blocked": self.egress_blocked,
            "scrubbed_env": self.scrubbed_env,
            "run_errors": self.run_errors,
            "mocks": self.mocks,
            "notes": self.notes,
            "saved_at": self.saved_at,
            "offered_tools": self.offered_tools(),
            "run_ids": self.run_ids,
            "by_run": self.by_run,
            "patched_tools": self.patched_tools,
            "unsupported": self.unsupported,
            "calls": [c.model_dump(mode="json", exclude={"raw_request"}) for c in self.calls],
        }


_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)\b(?::|$)")


def _last_line(msg: str) -> str:
    """The exception line of a traceback. Frameworks append notes after it (LangGraph: "During task with name
    ..."), so the last line that names an exception wins over the very last line."""
    lines = [ln.strip() for ln in msg.strip().splitlines() if ln.strip()]
    return next((ln for ln in reversed(lines) if _EXC_LINE.match(ln)), lines[-1] if lines else msg)


_IMPORT_ERRORS = ("ImportError", "ModuleNotFoundError")


def _crash_hint(msg: str, python: str | None) -> str:
    """A crash on import is the agent's environment, not its code: say so and how to point the mock run at the right
    one."""
    if not any(e in _last_line(msg) for e in _IMPORT_ERRORS):
        return ""
    where = python or sys.executable
    return (
        f". The agent's dependencies are missing or broken in {where}. Install them in a clean environment "
        "(python -m venv .venv-agent, then .venv-agent/bin/pip install -r requirements.txt, or "
        ".venv-agent\\Scripts\\pip on Windows) and set python: in mocks.yml (or pass --python) to that "
        "environment's interpreter"
    )


_NEXT = "agenticreq mock-run"
#: where frameworks keep per-user config, cache and data (Linux and macOS / Windows)
_USER_DIRS = ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "APPDATA", "LOCALAPPDATA")


def _calls(n: int) -> str:
    return f"{n} model call{'' if n == 1 else 's'}"


#: the step when agenticreq and the agent are in different environments
WRONG_ENV = "install agenticreq in the same environment as your agent, or pass --python <interpreter>"


def explain(
    rid: str, f: dict[str, Any] | None, raw: str, captured: int, res: DiscoverResult, python: str | None
) -> str:
    """One run's failure as one sentence: what was captured, where the run stopped and why, and the next step."""
    lead = f"mock-run captured {_calls(captured)}" + ("" if not captured else ", then stopped")
    if f is None:  # the child died before it could write its result (an interpreter that cannot start, say)
        last = _last_line(raw)
        return f"{lead}. {rid} ended with {last}{_crash_hint(raw, python)}. Fix it, then run: {_NEXT}"
    kind, msg, where = f.get("type", ""), str(f.get("message", "")).rstrip(". "), f.get("where", "")
    err = f"{kind}: {msg}" if msg else kind
    at = f" at {where}" if where else ""
    first = f.get("first")
    hidden = (
        f" The first error was {first['type']}: {first['message']} (a retry raised the one above)."
        if first and (first.get("type"), first.get("message")) != (kind, msg)
        else ""
    )
    if kind == "Timeout":
        step = f.get("step") or {}
        tools = ", ".join(step.get("last_tools") or []) or "none"
        head = (
            f"mock-run captured {_calls(int(step.get('calls') or 0))} before the {res.timeout:g}s limit stopped "
            f"{rid} (last step: model call {int(step.get('calls') or 0)}, tools offered: {tools})."
        )
        blocked = step.get("blocked") or []
        if blocked:
            return (
                f"{head} The last blocked write was {blocked[-1]}: the agent may be retrying it. List that folder "
                f"under allow_write: in mocks.yml, then run: {_NEXT}"
            )
        return (
            f"{head} The agent may wait for input or loop. Raise --timeout, or fix what it waits for, then run: {_NEXT}"
        )
    if "agenticreq mock run:" in msg:  # the write guard stopped it: say which file, and the key that allows it
        return f"{lead}. The run stopped{at} at a blocked write. {msg.removeprefix('agenticreq mock run: ')}. Then run: {_NEXT}"
    if f.get("entry_missing"):
        return (
            f"{lead}. The entry point {res.entry} was not found ({err}). Fix entry: in mocks.yml (module:function "
            f"or file.py:function), then run: {_NEXT}"
        )
    if kind in ("ImportError", "ModuleNotFoundError"):
        mod = f.get("missing_module") or "the missing package"
        where_py = python or sys.executable
        if str(mod).split(".")[0] == "agenticreq":  # the agent's environment is not the one agenticreq is in
            return (
                f"{lead}. agenticreq is not installed in {where_py}, the environment that runs your agent "
                f"(agenticreq itself runs from {sys.executable}, started as {sys.argv[0]}): "
                f"{WRONG_ENV}, then run: {_NEXT}"
            )
        return (
            f"{lead}. The run stopped{at} with {err}: a dependency is missing in {where_py}. Install {mod} there, "
            f"or {WRONG_ENV} (python: in mocks.yml does the same), then run: {_NEXT}"
        )
    if res.rejections:
        return (
            f"{lead}. The run stopped{at} with {err}, because {res.rejections[0]}. Change that request (or the "
            f"model it names), then run: {_NEXT}"
        )
    if f.get("sdk") and not captured:
        return (
            f"{lead}. The run stopped{at} with {err}: the {f['sdk']} SDK rejected the request before sending it."
            f"{hidden} Fix the value or the model, then run: {_NEXT}"
        )
    return f"{lead}. The run stopped{at} with {err}.{hidden} Fix the error, then run: {_NEXT}"


def _isolated_package(tmp: str) -> str:
    """A folder holding ONLY the agenticreq package, for the child's PYTHONPATH. Installed normally, agenticreq
    lives in a site-packages full of other packages (conda base, a global Python): putting that folder on the
    path would shadow the agent's own environment with them (an old package there breaks the run)."""
    import shutil

    dest = Path(tmp) / "agenticreq-path"
    shutil.copytree(
        Path(__file__).resolve().parents[1], dest / "agenticreq", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return str(dest)


class DiscoverError(Exception):
    """The mock run cannot run this agent; the message says why and what to do instead."""


def resolve_python(python: str | None) -> str:
    """The agent's interpreter as an absolute path, on any OS: `.venv/bin/python`, `.venv\\Scripts\\python.exe`,
    `C:/envs/a/python.exe`, `python3.12` (looked up on PATH, with PATHEXT on Windows). A venv's python is a
    symlink: made absolute, never resolve()d, or the venv is lost."""
    import shutil

    if not python:
        return sys.executable
    looks_like_path = "/" in python or "\\" in python or Path(python).exists()
    if looks_like_path:
        p = Path(python).expanduser()
        p = p if p.is_absolute() else Path.cwd() / p
        candidates = [p, p.with_suffix(".exe")] if os.name == "nt" and not p.suffix else [p]
        for c in candidates:
            if c.is_file():
                return str(c)
        posix, win = Path(".venv") / "bin" / "python", Path(".venv") / "Scripts" / "python.exe"
        raise DiscoverError(
            f"interpreter {python!r} not found. A venv's interpreter is {posix} on macOS / Linux and "
            f"{win} on Windows; pass --python with the one for this OS."
        )
    found = shutil.which(python)
    if found is None:
        raise DiscoverError(f"interpreter {python!r} is not on PATH; pass its full path with --python")
    return found


def _version_tuple(v: str) -> tuple[int, ...]:
    from agenticreq.versions import parse

    return parse(v)


def check_framework_versions(root: Path, python: str | None, notes: list[str] | None = None) -> dict[str, str]:
    """{dist: version} of the gated frameworks the agent's environment has installed.

    Refuses (DiscoverError) only a version known to break a mock run (``known_bad`` in supported_versions.yml).
    Any other version outside the validated range runs, with a note in ``notes``: the capture works at the
    HTTP-client boundary, not on framework internals, and a run that captures nothing exits 1 anyway."""
    from agenticreq.ast import extract
    from agenticreq.data import load_yaml

    table = (load_yaml("supported_versions.yml") or {}).get("frameworks") or {}
    python = resolve_python(python)
    used = [fw for fw in extract(root).frameworks if fw in table]
    if not used:
        return {}
    dists = [table[fw]["dist"] for fw in used]
    probe = (
        "import json, importlib.metadata as m\nout = {}\nfor d in %r:\n    try:\n        out[d] = m.version(d)\n"
        "    except m.PackageNotFoundError:\n        pass\nprint(json.dumps(out))" % dists
    )
    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        found: dict[str, str] = json.loads(proc.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        raise DiscoverError(f"could not read framework versions with {python or sys.executable}: {e}") from None
    for fw in used:
        row = table[fw]
        have = found.get(row["dist"])
        if have is None:
            continue  # imported but not installed: the run itself reports the ImportError
        v = _version_tuple(have)
        supported = f"{row['dist']} >={row['min']},<{row['below']}"
        bad = next(
            (
                why
                for k, why in (row.get("known_bad") or {}).items()
                if v[: len(_version_tuple(k))] == _version_tuple(k)
            ),
            None,
        )
        if bad:
            raise DiscoverError(
                f"{row['dist']} {have} is not supported by agenticreq mock-run (supported: {supported}; validated on "
                f"{', '.join(row.get('validated') or [])}). Known problem with {have}: {bad}. Upgrade {row['dist']} "
                f"in the agent's environment, or run `agenticreq audit {root}` (static checks, nothing is run)."
            )
        if not _version_tuple(str(row["min"])) <= v < _version_tuple(str(row["below"])) and notes is not None:
            notes.append(
                f"{row['dist']} {have} is outside the validated range ({supported}); the mock run runs "
                "it anyway and checks what it captured. If it captures nothing it exits 1."
            )
    return found


def unsupported_transports(res: DiscoverResult, table: dict[str, Any]) -> list[str]:
    """The providers and transports this run met that cannot be intercepted at the HTTP client layer."""
    rows = table.get("unsupported") or {}
    out = [
        h["name"]
        for h in rows.get("hosts") or []
        if any(h["match"] in m for m in [*res.llm_misses, *res.egress_blocked])
    ]
    out += [name for mod, name in (rows.get("frameworks") or {}).items() if f"module:{mod}" in res.versions]
    return list(dict.fromkeys(out))


def scratch_config_dirs(tmp: str, python: str | None = None) -> dict[str, str]:
    """Per-user folders for the run, inside its scratch folder: frameworks keep config, caches and databases under
    the home folder, and write there freely (also when the real folders are read-only), while the user's own files
    stay as they are. Packages installed with ``pip --user`` stay importable."""
    home = Path(tmp) / "user" / "home"
    home.mkdir(parents=True, exist_ok=True)
    out = {k: str(Path(tmp) / "user" / k.lower()) for k in _USER_DIRS}
    out.update({"HOME": str(home), "USERPROFILE": str(home)})
    base = _user_base(python)
    if base:
        out["PYTHONUSERBASE"] = base
    pyenv = Path.home() / ".pyenv"
    if pyenv.is_dir():  # a shim finds its interpreters under the real home
        out.setdefault("PYENV_ROOT", os.environ.get("PYENV_ROOT", str(pyenv)))
    return out


def _user_base(python: str | None) -> str:
    """Where ``pip --user`` puts packages for this interpreter (it is worked out from the home folder)."""
    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", "import site; print(site.getuserbase())"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def discover(
    entry: str,
    *,
    mock_tools: str | None,
    runs: int = 1,
    timeout: float = 120.0,
    python: str | None = None,
    cwd: str | None = None,
    call_args: list[Any] | dict[str, Any] | None = None,
    run_names: list[str] | None = None,
    use_run_inputs: bool = True,
) -> DiscoverResult:
    """Run the agent once per mock run. ``runs:`` in mocks.yml lists named runs, each with its own entry / args and
    settings; ``run_names`` picks some of them (default: all); ``runs`` repeats each one. Without ``runs:`` in the
    file it is the single-task run. ``use_run_inputs=False`` makes ``entry`` / ``call_args`` win over a run's own."""
    mocks = gate(mock_tools)
    plan = plan_runs(mocks, runs, run_names)  # raises MockRunRefused: no mocks file, no run, no bypass
    python = resolve_python(python)
    target, func = split_entry(entry)
    root = Path(cwd).resolve() if cwd else (Path(target).resolve().parent if target.endswith(".py") else Path.cwd())
    env, scrubbed = build_child_env(dict(os.environ), root, mocks.env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"  # the child prints UTF-8 whatever the console code page (CrewAI box drawing)
    env["PYTHONIOENCODING"] = "utf-8"
    # libraries that cache under ~ (matplotlib) would trip the write guard: point them at the scratch area
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agenticreq-mpl"))

    from agenticreq.ast.framework import DISTS, FRAMEWORKS

    probe_dists = sorted({d for ds in DISTS.values() for d in ds})
    probe_modules = list(FRAMEWORKS)
    version_notes: list[str] = []
    check_framework_versions(root, python, version_notes)  # raises DiscoverError before anything runs
    res = DiscoverResult(entry=entry, scrubbed_env=scrubbed, notes=version_notes, mocks=mocks.source)
    before = tree_hashes(root)
    from agenticreq.capture.guard import norm_path, under

    # allow_write: paths are relative to the project folder; / and \\ alike, resolved before any comparison
    allowed = [norm_path(root / w.replace("\\", "/").strip()) for w in mocks.allow_write if str(w).strip()]
    abs_entry = entry  # noqa: F841
    if target.endswith(".py"):
        abs_entry = str(Path(target).resolve()) + (":" + func if func else "")  # noqa: F841

    from agenticreq.data import load_yaml

    with tempfile.TemporaryDirectory(prefix="agenticreq-") as tmp:
        env.update(scratch_config_dirs(tmp, python))
        wrappers = load_yaml("tool_wrappers.yml")
        env["PYTHONPATH"] = os.pathsep.join(x for x in (_isolated_package(tmp), env.get("PYTHONPATH", "")) if x)
        for run_id, spec in plan:
            run_entry, run_args = entry, call_args
            if spec is not None and use_run_inputs:
                run_entry = spec.entry or entry
                run_args = spec.args if spec.has_args else call_args
            run_target, run_func = split_entry(run_entry)
            abs_run = run_entry
            if run_target.endswith(".py"):
                abs_run = str(Path(run_target).resolve()) + (":" + run_func if run_func else "")
            fname = re.sub(r"[^\w.-]", "_", run_id)
            cfg = {
                "entry": abs_run,
                "run_id": run_id,
                "run_name": spec.name if spec is not None else None,
                "out": str(Path(tmp) / f"{fname}.json"),
                "mocks": str(Path(str(mock_tools)).resolve()),
                "probe_dists": probe_dists,
                "probe_modules": probe_modules,
                "mocks_data": mocks.for_run(spec.name if spec is not None else None).to_dict(),
                "allow_write": allowed,
                "call_args": run_args,
                "providers": load_yaml("providers.yml"),
                "wrapper_levels": wrappers["levels"],
                "async_attrs": list(wrappers["async_attrs"]),
            }
            cfg_path = Path(tmp) / f"{fname}.cfg.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [python or sys.executable, "-B", "-m", "agenticreq.capture.runner", str(cfg_path)],
                    env=env,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                # what the run reached is kept: the summary says where it was, and the next step
                progress = Path(str(cfg["out"])).with_suffix(".progress.json")
                try:
                    step = json.loads(progress.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    step = {}
                res.runs += 1
                res.run_ids.append(run_id)
                res.timeout = timeout
                res.run_errors[run_id] = f"Timeout: did not finish within {timeout:g}s"
                res.failures[run_id] = {
                    "type": "Timeout",
                    "message": f"did not finish within {timeout:g}s",
                    "step": step,
                }
                continue
            out_path = Path(str(cfg["out"]))
            res.runs += 1
            res.run_ids.append(run_id)
            if not out_path.exists():
                res.run_errors[run_id] = (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-2000:]
                continue
            data = json.loads(out_path.read_text(encoding="utf-8-sig"))
            offset = len(res.calls)
            for c in data["calls"]:
                call = CapturedCall.model_validate(c)
                call.index += offset
                res.calls.append(call)
            res.llm_misses += data["llm_misses"]
            res.egress_blocked += data["egress_blocked"]
            res.by_run[run_id] = {
                "llm_misses": list(data["llm_misses"]),
                "egress_blocked": list(data["egress_blocked"]),
            }
            res.notes += [n for n in data.get("notes") or [] if n not in res.notes]
            res.libs_hooked = sorted(set(res.libs_hooked) | set(data["libs_hooked"]))
            res.patched_tools += [t for t in data.get("patched_tools") or [] if t not in res.patched_tools]
            res.versions.update(data.get("versions") or {})
            if data.get("error"):
                res.run_errors[run_id] = data["error"]
            if data.get("failure"):
                res.failures[run_id] = data["failure"]
            res.rejections += [r for r in data.get("rejections") or [] if r not in res.rejections]

    res.unsupported = unsupported_transports(res, load_yaml("providers.yml"))
    after = tree_hashes(root)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    inside = [k for k in changed if allowed and under(root / k, allowed)]
    res.wrote = [(k.replace("\\", "/"), (root / k).stat().st_size if (root / k).is_file() else 0) for k in inside]
    res.changed_files = [k for k in changed if k not in inside]
    res.zero_touch = not res.changed_files
    import datetime as dt

    res.saved_at = dt.datetime.now().isoformat(timespec="seconds")
    return res


def plan_runs(mocks: Any, repeat: int = 1, only: list[str] | None = None) -> list[tuple[str, Any]]:
    """(run id, its RunSpec or None) for every run to execute. No ``runs:`` in mocks.yml = run-0 .. run-<repeat-1>
    (the single-task file). With ``runs:``, a run's id is its name (``name#2`` ... when repeated)."""
    if not mocks.runs:
        if only:
            raise DiscoverError(f"{mocks.source} has no runs: to pick from ({', '.join(only)}); add runs: to mocks.yml")
        return [(f"run-{i}", None) for i in range(repeat)]
    names = mocks.run_names()
    missing = [n for n in only or [] if n not in names]
    if missing:
        raise DiscoverError(
            f"no mock run named {', '.join(missing)} in {mocks.source}; its runs are: {', '.join(names)}"
        )
    chosen = [r for r in mocks.runs if not only or r.name in only]
    return [(r.name if i == 0 else f"{r.name}#{i + 1}", r) for r in chosen for i in range(repeat)]


def _merge_kept(res: DiscoverResult, old: dict[str, Any]) -> None:
    """--keep: put the runs of an earlier saved file that were NOT run now into ``res``. A run run again replaces
    its earlier capture (keyed by run id); every other run is kept as saved."""
    fresh = set(res.run_ids)
    old_calls = [CapturedCall.model_validate(c) for c in old.get("calls") or []]
    old_ids = [str(r) for r in old.get("run_ids") or dict.fromkeys(c.run_id for c in old_calls)]
    keep_ids = [r for r in old_ids if r not in fresh]
    if not keep_ids:
        return
    keep = set(keep_ids)
    by_run = {k: v for k, v in (old.get("by_run") or {}).items() if k in keep}
    legacy = not old.get("by_run")  # a file saved before runs were kept apart: its misses cannot be told apart
    kept_calls = [c for c in old_calls if c.run_id in keep]
    res.calls = kept_calls + res.calls
    for i, c in enumerate(res.calls):
        c.index = i
    res.run_ids = keep_ids + res.run_ids
    res.runs = len(res.run_ids)
    res.kept = keep_ids
    res.by_run = {**by_run, **res.by_run}
    if legacy:
        res.llm_misses = [*(old.get("llm_misses") or []), *res.llm_misses]
        res.egress_blocked = [*(old.get("egress_blocked") or []), *res.egress_blocked]
    else:
        res.llm_misses = [m for v in by_run.values() for m in v.get("llm_misses", [])] + res.llm_misses
        res.egress_blocked = [m for v in by_run.values() for m in v.get("egress_blocked", [])] + res.egress_blocked
    res.run_errors = {**{k: v for k, v in (old.get("run_errors") or {}).items() if k in keep}, **res.run_errors}
    res.libs_hooked = sorted(set(res.libs_hooked) | set(old.get("libs_hooked") or []))
    res.patched_tools += [str(t) for t in old.get("patched_tools") or [] if t not in res.patched_tools]
    res.unsupported += [str(t) for t in old.get("unsupported") or [] if t not in res.unsupported]
    res.versions = {**{str(k): str(v) for k, v in (old.get("versions") or {}).items()}, **res.versions}
    res.notes = [*[n for n in old.get("notes") or [] if n not in res.notes], *res.notes]
    seen = {r for r, _ in res.wrote}
    res.wrote += [(str(r), int(n)) for r, n in old.get("wrote") or [] if r not in seen]


def keep_earlier(res: DiscoverResult, path: str | Path) -> list[str]:
    """--keep: merge the runs saved at ``path`` that were not run now into ``res``; the ids kept (none when there
    is no readable earlier file)."""
    try:
        old = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    if not isinstance(old, dict) or not isinstance(old.get("calls"), list):
        return []
    try:
        _merge_kept(res, old)
    except (ValueError, TypeError):  # an unreadable earlier file is replaced, not carried forward
        return []
    return res.kept


def save_captures(res: DiscoverResult, out: str | Path, keep: bool = False) -> Path:
    """Write the run(s) to ``out``. A plain save replaces the file; ``keep=True`` first merges in the earlier
    file's runs that were not run now."""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if keep and p.is_file():
        keep_earlier(res, p)
    p.write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    return p


def load_capture_bundle(path: str | Path) -> tuple[list[CapturedCall], dict[str, str]]:
    """(calls, versions recorded inside the guarded mock run)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return load_captures(path), {}
    from agenticreq.capture.importers import detect_format, from_spike

    if detect_format(data) == "spike":
        return from_spike(data)
    versions = data.get("versions", {}) if isinstance(data, dict) else {}
    return load_captures(path), {str(k): str(v) for k, v in versions.items()}


def _saved_list(path: str | Path, key: str) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    names = data.get(key) if isinstance(data, dict) else None
    return [str(t) for t in names] if isinstance(names, list) else []


def load_patched_tools(path: str | Path) -> list[str]:
    """The tools a saved mock run replaced with ``patch:`` (a file from another tool lists none)."""
    return _saved_list(path, "patched_tools")


def load_unsupported(path: str | Path) -> list[str]:
    """What a saved mock run could not intercept."""
    return _saved_list(path, "unsupported")


def load_captures(path: str | Path) -> list[CapturedCall]:
    """Load captures written by save_captures (JSON) or a JSONL of CapturedCall objects."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [CapturedCall.model_validate(json.loads(line)) for line in text.splitlines() if line.strip()]
    from agenticreq.capture.importers import detect_format, from_raw, from_spike

    fmt = detect_format(data)
    if fmt == "spike":
        return from_spike(data)[0]
    if fmt == "raw":
        return from_raw(data)
    items = data["calls"] if isinstance(data, dict) else data
    return [CapturedCall.model_validate(c) for c in items]
