"""Which Python runs the agent in `agenticreq mock-run`, and whether it can run it.

The interpreter is the one that has the agent's own dependencies, whatever tool made it. First match wins, and the
output always says where it came from (``python: <path> (from <source>)``):

1. ``--python``
2. ``python:`` in mocks.yml
3. the active ``$VIRTUAL_ENV``, then ``$CONDA_PREFIX`` (an active conda ``base`` is skipped: it is on for almost every
   conda user and is never the project's own environment; pass ``--python`` to use it)
4. the project itself: a top-level folder holding ``pyvenv.cfg`` (.venv, venv, env, uv, virtualenv, Poetry in-project,
   any name), Poetry, Pipenv, Conda (``environment.yml`` name), pyenv / asdf (``.python-version``,
   ``.tool-versions``)
5. agenticreq's own interpreter, with a visible warning

An interpreter is ``bin/python``, or ``Scripts/python.exe`` on Windows. Nothing is installed or created: poetry,
pipenv and conda are called only when present, read-only, with a short timeout, and any failure falls through to
the next source. Before a run, :func:`preflight` checks that the interpreter starts and can import the agent's main
framework, so a wrong environment stops with the packages that are missing instead of a stack trace.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

#: seconds an external tool (poetry / pipenv / conda) or the interpreter probe may take before it is skipped
TOOL_TIMEOUT = 8.0
#: folder names tried first when several project folders hold a pyvenv.cfg
_PREFERRED = (".venv", "venv", "env")
_SKIP_DIRS = {
    ".git",
    ".agenticreq",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "site-packages",
}


@dataclass(frozen=True)
class PythonChoice:
    """The interpreter picked, where it came from, and a warning when it is only a fallback."""

    path: str
    source: str
    warning: str = ""

    def line(self) -> str:
        return f"python: {self.path} (from {self.source})"


def is_windows() -> bool:
    return os.name == "nt"


def interpreter_in(prefix: Path, windows: bool | None = None) -> Path | None:
    """The interpreter of an environment folder (``bin/python``; ``Scripts/python.exe`` on Windows, and a conda
    environment's own ``python.exe``), or None when the folder holds no working file."""
    win = is_windows() if windows is None else windows
    names = (
        (prefix / "Scripts" / "python.exe", prefix / "python.exe")
        if win
        else (prefix / "bin" / "python", prefix / "bin" / "python3")
    )
    for exe in names:
        if exe.is_file():  # whether it starts is checked by preflight(), which stops the run with the reason
            return exe  # a venv's python is a symlink: never resolve() it, or the venv is lost
    return None


def run_tool(cmd: list[str], cwd: Path) -> str:
    """stdout of a read-only external command, or "" when the tool is missing, slow or fails. Never raises."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


# ---------------------------------------------------------------- the sources
def project_envs(folder: Path) -> list[tuple[str, Path, bool]]:
    """(folder name, interpreter, has pyvenv.cfg) for every top-level folder holding ``pyvenv.cfg`` and a working
    interpreter, ``.venv`` / ``venv`` / ``env`` first, the rest by name. ``.venv`` / ``venv`` / ``env`` count with an
    interpreter alone (a copied or hand-made environment has no pyvenv.cfg)."""
    try:
        dirs = [d for d in folder.iterdir() if d.is_dir() and d.name not in _SKIP_DIRS]
    except OSError:
        return []
    dirs.sort(key=lambda d: (_PREFERRED.index(d.name) if d.name in _PREFERRED else len(_PREFERRED), d.name))
    out = []
    for d in dirs:
        cfg = (d / "pyvenv.cfg").is_file()
        if cfg or d.name in _PREFERRED:
            exe = interpreter_in(d)
            if exe is not None:  # a folder without a working interpreter is skipped
                out.append((d.name, exe, cfg))
    return out


def from_poetry(folder: Path) -> Path | None:
    pyproject = folder / "pyproject.toml"
    try:
        uses = (folder / "poetry.lock").is_file() or (
            pyproject.is_file() and "[tool.poetry]" in pyproject.read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        uses = False
    if not uses:
        return None
    exe = Path(_first_line(run_tool(["poetry", "env", "info", "--executable"], folder)) or os.devnull)
    return exe if exe.is_file() else None


def from_pipenv(folder: Path) -> Path | None:
    if not (folder / "Pipfile").is_file():
        return None
    venv = _first_line(run_tool(["pipenv", "--venv"], folder))
    return interpreter_in(Path(venv)) if venv else None


def _conda_env_file(folder: Path) -> Path | None:
    return next((p for p in (folder / "environment.yml", folder / "environment.yaml") if p.is_file()), None)


def from_conda(folder: Path) -> tuple[str, Path] | None:
    """(environment name, interpreter) for the ``name:`` (or ``prefix:``) in environment.yml, found in the
    environments ``conda env list`` knows."""
    path = _conda_env_file(folder)
    if path is None:
        return None
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    prefix, name = data.get("prefix"), data.get("name")
    if isinstance(prefix, str) and prefix:
        exe = interpreter_in(Path(prefix).expanduser())
        if exe is not None:
            return Path(prefix).name, exe
    if not isinstance(name, str) or not name:
        return None
    raw = run_tool(["conda", "env", "list", "--json"], folder)
    try:
        envs = json.loads(raw).get("envs") or []
    except (ValueError, AttributeError):
        return None
    for env in envs:
        if isinstance(env, str) and Path(env).name == name:
            exe = interpreter_in(Path(env))
            if exe is not None:
                return name, exe
    return None


def _version_dir(root: Path, version: str) -> Path | None:
    """``root/<version>``, or for a partial version such as 3.11 the highest ``root/3.11.*``."""
    exact = root / version
    if exact.is_dir():
        return exact
    try:
        near = sorted(
            (d for d in root.iterdir() if d.is_dir() and d.name.startswith(version + ".")),
            key=lambda d: [int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", d.name)],
        )
    except OSError:
        return None
    return near[-1] if near else None


def from_version_files(folder: Path, environ: dict[str, str]) -> tuple[str, Path] | None:
    """(source, interpreter) from ``.python-version`` (pyenv) or ``.tool-versions`` (asdf), read from disk only."""
    pv, tv = folder / ".python-version", folder / ".tool-versions"
    if not (pv.is_file() or tv.is_file()):
        return None  # nothing to read: build no paths at all
    home = Path(environ.get("HOME") or environ.get("USERPROFILE") or "~").expanduser()
    if pv.is_file():
        try:
            versions = [
                ln.strip()
                for ln in pv.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        except OSError:
            versions = []
        root = Path(environ.get("PYENV_ROOT") or home / ".pyenv") / "versions"
        for v in versions:
            d = _version_dir(root, v) if v != "system" else None
            exe = interpreter_in(d) if d is not None else None
            if exe is not None:
                return "pyenv (.python-version)", exe
    if tv.is_file():
        try:
            lines = tv.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        root = Path(environ.get("ASDF_DATA_DIR") or home / ".asdf") / "installs" / "python"
        for ln in lines:
            parts = ln.split("#")[0].split()
            if len(parts) >= 2 and parts[0] == "python":
                d = _version_dir(root, parts[1])
                exe = interpreter_in(d) if d is not None else None
                if exe is not None:
                    return "asdf (.tool-versions)", exe
    return None


def choose(
    folder: Path, cli: str | None = None, mocks: str | None = None, environ: dict[str, str] | None = None
) -> PythonChoice:
    """The interpreter for a mock run of the agent in ``folder``. See the module docstring for the order."""
    env = dict(os.environ) if environ is None else environ
    if cli:
        return PythonChoice(cli, "--python")
    if mocks:
        return PythonChoice(mocks, "python: in mocks.yml")
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = env.get(var)
        if not prefix or (var == "CONDA_PREFIX" and env.get("CONDA_DEFAULT_ENV") == "base"):
            continue  # an active conda base is not the agent's environment: the project is looked at first
        exe = interpreter_in(Path(prefix))
        if exe is not None:
            return PythonChoice(str(exe), f"${var}")
    for name, exe, cfg in project_envs(folder):
        return PythonChoice(str(exe), f"./{name} (pyvenv.cfg)" if cfg else f"./{name}")
    exe = from_poetry(folder)
    if exe is not None:
        return PythonChoice(str(exe), "Poetry (poetry env info)")
    exe = from_pipenv(folder)
    if exe is not None:
        return PythonChoice(str(exe), "Pipenv (pipenv --venv)")
    conda = from_conda(folder)
    if conda is not None:
        return PythonChoice(str(conda[1]), f"Conda environment {conda[0]} (environment.yml)")
    ver = from_version_files(folder, env)
    if ver is not None:
        return PythonChoice(str(ver[1]), ver[0])
    return PythonChoice(
        sys.executable,
        "agenticreq's own interpreter",
        warning="no environment found for this project, so the agent runs with agenticreq's own "
        "Python and needs its dependencies installed there; pass --python <interpreter> or "
        "set python: in mocks.yml to use the project's environment",
    )


# ---------------------------------------------------------------- before the run
def _entry_file(folder: Path, entry: str) -> Path | None:
    """The file an ``entry:`` (``file.py:func`` or ``pkg.module:func``) points at, when it can be found."""
    target = entry.rsplit(":", 1)[0] if ":" in entry and not re.match(r"^[A-Za-z]:[\\/]", entry) else entry
    if target.endswith(".py"):
        p = Path(target)
        p = p if p.is_absolute() else folder / p
        return p if p.is_file() else None
    rel = Path(*target.split("."))
    for base in (folder, folder / "src"):
        for cand in (base / rel.with_suffix(".py"), base / rel / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def imported_modules(path: Path) -> list[str]:
    """Top-level names a file imports (``google.adk`` kept dotted), in order, no duplicates."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for n in names:
            out.append("google.adk" if n.startswith("google.adk") else n.split(".")[0])
    return list(dict.fromkeys(out))


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def declared_dists(folder: Path) -> list[str]:
    """Distribution names the project declares: requirements.txt, pyproject.toml (PEP 621 and Poetry), Pipfile."""
    names: list[str] = []
    req = folder / "requirements.txt"
    if req.is_file():
        for ln in req.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _REQ.match(ln.split("#")[0])
            if m and not ln.strip().startswith("-"):
                names.append(m.group(1))
    for fname, tables in (
        ("pyproject.toml", (("project", "dependencies"), ("tool", "poetry", "dependencies"))),
        ("Pipfile", (("packages",),)),
    ):
        f = folder / fname
        if not f.is_file():
            continue
        try:
            import tomllib  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:  # Python 3.10
            import tomli as tomllib  # type: ignore[no-redef,import-not-found,import-untyped,unused-ignore]
        try:
            data = tomllib.loads(f.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        for path in tables:
            cur: object = data
            for part in path:
                cur = cur.get(part) if isinstance(cur, dict) else None
            if isinstance(cur, list):
                names += [m.group(1) for x in cur if isinstance(x, str) and (m := _REQ.match(x))]
            elif isinstance(cur, dict):
                names += [k for k in cur if k != "python"]
    return list(dict.fromkeys(names))


@dataclass
class Preflight:
    """What was found before the run. ``problems`` stop it; ``warnings`` do not."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    version: str = ""


_PROBE = """
import importlib.util, importlib.metadata as md, json, sys
want = json.loads(sys.argv[1])
def has(m):
    try:
        return importlib.util.find_spec(m) is not None
    except (ImportError, ValueError, AttributeError):
        return False
def dist(d):
    try:
        md.version(d)
        return True
    except md.PackageNotFoundError:
        return False
print(json.dumps({"version": "%d.%d.%d" % sys.version_info[:3],
                  "modules": {m: has(m) for m in want["modules"]},
                  "dists": {d: dist(d) for d in want["dists"]}}))
"""
#: what the capture hook itself imports inside the agent's interpreter (agenticreq is copied in, not installed)
HOOK_IMPORTS = {"pydantic": "pydantic"}


def _requires_python(folder: Path) -> str:
    f = folder / "pyproject.toml"
    if not f.is_file():
        return ""
    m = re.search(
        r'^\s*requires-python\s*=\s*["\']([^"\']+)["\']', f.read_text(encoding="utf-8", errors="replace"), re.MULTILINE
    )
    return m.group(1).strip() if m else ""


def version_mismatch(version: str, spec: str) -> bool:
    """True when ``version`` (3.11.4) breaks a requires-python such as ``>=3.10,<3.13`` or ``~=3.11``."""
    have = tuple(int(x) for x in version.split(".")[:3] if x.isdigit())
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        m = re.match(r"^(>=|<=|==|!=|~=|>|<)\s*(\d+(?:\.\d+)*)(\.\*)?$", clause)
        if not m:
            continue
        op, want_s, star = m.group(1), m.group(2), m.group(3)
        want = tuple(int(x) for x in want_s.split("."))
        cut = have[: len(want)]
        ok = {
            ">=": have >= want,
            "<=": cut <= want,
            ">": have > want,
            "<": have < want,
            "==": (cut == want) if star or len(want) < 3 else have == want,
            "!=": cut != want,
            "~=": have >= want and have[: max(len(want) - 1, 1)] == want[: max(len(want) - 1, 1)],
        }[op]
        if not ok:
            return True
    return False


def preflight(
    py: str,
    folder: Path,
    entry: str,
    source: str = "",
    run: Callable[[list[str], float], subprocess.CompletedProcess[str]] | None = None,
) -> Preflight:
    """Check that ``py`` starts, can import the agent's main framework (from the entry file's imports and the
    project's requirements.txt / pyproject.toml / Pipfile) and the library the capture hook needs. A Python
    version outside ``requires-python`` is a warning, never a failure. Nothing is installed."""
    from agenticreq.ast.framework import FRAMEWORKS, GENERIC, dist_label

    out = Preflight()
    where = f"{py} (from {source})" if source else py
    entry_path = _entry_file(folder, entry)
    # agent frameworks only: a plain SDK client (openai, anthropic, mcp) may be put on sys.path by the agent itself
    mods = [
        m
        for m in (imported_modules(entry_path) if entry_path else [])
        if m in FRAMEWORKS and FRAMEWORKS[m][0] not in GENERIC
    ]
    local = {p.stem for p in folder.glob("*.py")} | {d.name for d in folder.iterdir() if d.is_dir()}
    mods = [m for m in mods if m not in local]  # the project's own `agents` package is not the framework
    dists = [d for d in declared_dists(folder) if dist_label(d) and dist_label(d) not in GENERIC]
    mod_dist = {m: FRAMEWORKS[m][1] for m in mods}
    want = {
        "modules": [*dict.fromkeys([*mods, *HOOK_IMPORTS])],
        "dists": [*dict.fromkeys([*mod_dist.values(), *dists])],
    }
    runner = run or (
        lambda cmd, t: subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=t,
            encoding="utf-8",
            errors="replace",
            cwd=str(folder),
            stdin=subprocess.DEVNULL,
        )
    )
    try:
        proc = runner([py, "-c", _PROBE, json.dumps(want)], TOOL_TIMEOUT * 4)
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        if proc.returncode != 0:
            raise ValueError(proc.stderr)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, TypeError) as e:
        why = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        out.problems.append(
            f"python {where} does not start ({why}). Point agenticreq at a working interpreter with "
            "--python <interpreter> or python: in mocks.yml (a venv's is .venv/bin/python, or "
            ".venv\\Scripts\\python.exe on Windows)"
        )
        return out
    out.version = str(data.get("version", ""))
    have_mods, have_dists = data.get("modules") or {}, data.get("dists") or {}
    # a framework is there when its module imports or its distribution is installed (the version gate then speaks)
    missing = [mod_dist[m] for m in mods if not have_mods.get(m) and not have_dists.get(mod_dist[m])]
    missing += [d for d in dists if not have_dists.get(d)]
    missing = list(dict.fromkeys(missing))
    hook = [pkg for mod, pkg in HOOK_IMPORTS.items() if not have_mods.get(mod)]
    req = "requirements.txt" if (folder / "requirements.txt").is_file() else ""
    install = f"{py} -m pip install -r requirements.txt" if req else f"{py} -m pip install {' '.join(missing or hook)}"
    if missing:
        out.problems.append(
            f"python {where} cannot import the agent's framework: missing {', '.join(missing)}. The agent's "
            f"dependencies are not installed in that environment. Install them there with: {install} "
            "(agenticreq installs nothing), or choose the right interpreter with --python / python: in mocks.yml"
        )
    elif hook:
        out.problems.append(
            f"python {where} lacks {', '.join(hook)}, which the mock run's capture hook needs (agenticreq itself does "
            f"not have to be installed there). Install with: {py} -m pip install {' '.join(hook)}"
        )
    spec = _requires_python(folder)
    if spec and out.version and version_mismatch(out.version, spec):
        out.warnings.append(
            f"python {out.version} ({py}) is outside requires-python {spec} in pyproject.toml; "
            "the run goes on, but dependencies may differ from the ones the project targets"
        )
    pv = folder / ".python-version"
    if pv.is_file() and out.version and not spec:
        want_v = _first_line(pv.read_text(encoding="utf-8", errors="replace"))
        if re.match(r"^\d+\.\d+", want_v) and not out.version.startswith(".".join(want_v.split(".")[:2])):
            out.warnings.append(f"python {out.version} ({py}) differs from .python-version {want_v}; the run goes on")
    return out
