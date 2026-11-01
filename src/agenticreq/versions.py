"""Which version of a package the AGENT runs, and whether a version requirement allows a range.

agenticreq often runs from another Python than the agent (a global install, conda base), so the version
installed next to agenticreq says nothing about the agent unless the project's own declarations agree with it.
The agent's version is read, best first, from:

1. a virtual environment inside the project (``.venv``, ``venv``, ``env``, any folder holding ``pyvenv.cfg`` or
   ``conda-meta``), in the macOS / Linux layout (``lib/pythonX.Y/site-packages``) or the Windows one
   (``Lib/site-packages``);
2. a lock file: ``uv.lock``, ``poetry.lock``, ``pdm.lock``, ``Pipfile.lock``;
3. an exact pin in requirements / pyproject (``crewai==1.9.3``);
4. agenticreq's own environment, only when it satisfies what the project declares;
5. the declared range itself (``crewai>=0.100``): no single version, so callers check the range, never a bound.

Specifiers follow PEP 440 for the operators projects use (``== != >= <= > < ~= ===``, ``==1.2.*``), with no
dependency on ``packaging``.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

Version = tuple[int, ...]

_CLAUSE = re.compile(r"^\s*(===|==|!=|~=|>=|<=|>|<)\s*([0-9][0-9A-Za-z.*+!-]*)\s*$")
_SKIP = {".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", ".nox"}


def parse(v: str) -> Version:
    """Leading numeric release segments: "1.15.28" -> (1, 15, 28), "2.0.0rc1" -> (2, 0, 0), "v1.2" -> (1, 2)."""
    out = []
    for part in v.strip().lstrip("vV").split("."):
        digits = re.match(r"\d+", part)
        if not digits:
            break
        out.append(int(digits.group(0)))
        if digits.group(0) != part:
            break
    return tuple(out)


def _cmp(a: Version, b: Version) -> int:
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def clauses(spec: str | None) -> list[tuple[str, str]] | None:
    """``">=1.2, <2"`` -> [(">=", "1.2"), ("<", "2")]. Environment markers (``; python_version...``) and extras
    are dropped. None when the text is not a version specifier (a URL, a path, ``@ git+...``)."""
    if not spec:
        return []
    spec = spec.split(";", 1)[0].strip()
    if not spec:
        return []
    if spec.startswith("@") or "://" in spec:
        return None
    out = []
    for part in spec.split(","):
        if not part.strip():
            continue
        m = _CLAUSE.match(part)
        if not m:
            return None
        out.append((m.group(1), m.group(2)))
    return out


def satisfies(version: str | Version, spec: str | None) -> bool | None:
    """Does ``version`` meet ``spec``? None when the spec cannot be read."""
    cl = clauses(spec)
    if cl is None:
        return None
    v = parse(version) if isinstance(version, str) else version
    for op, raw in cl:
        wild = raw.endswith(".*")
        t = parse(raw[:-2] if wild else raw)
        if op in ("==", "===", "!=") and wild:
            hit = v[: len(t)] == t
            ok = hit if op != "!=" else not hit
        elif op == "~=":  # ~=1.4.5 means >=1.4.5, ==1.4.*
            ok = _cmp(v, t) >= 0 and v[: max(len(t) - 1, 1)] == t[: max(len(t) - 1, 1)]
        else:
            c = _cmp(v, t)
            ok = {"==": c == 0, "===": c == 0, "!=": c != 0, ">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0}[op]
        if not ok:
            return False
    return True


def exact(spec: str | None) -> str | None:
    """The one version an exact pin allows (``==1.9.3``, ``===1.9.3``), else None."""
    cl = clauses(spec)
    if cl and len(cl) == 1 and cl[0][0] in ("==", "===") and not cl[0][1].endswith(".*"):
        return cl[0][1]
    return None


def allows_any(spec: str | None, lo: str, below: str, known: list[str] | None = None) -> bool | None:
    """Can ``spec`` resolve to a version in ``[lo, below)``? Checked on the candidates that decide it: the range
    ends, the known versions and every bound the spec names (and the version just past it)."""
    cl = clauses(spec)
    if cl is None:
        return None
    lo_t, hi_t = parse(lo), parse(below)
    cands = {lo_t, *(parse(k) for k in known or [])}
    for _, raw in cl:
        t = parse(raw.rstrip(".*"))
        if t:
            cands |= {t, t[:-1] + (t[-1] + 1,), t + (0, 1)}
    return any(_cmp(lo_t, c) <= 0 and _cmp(c, hi_t) < 0 and satisfies(c, spec) for c in cands)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ----------------------------------------------------------------------------- the agent's environment
def _site_packages(env: Path) -> list[Path]:
    return [
        p
        for p in [
            *env.glob("lib/python*/site-packages"),
            *env.glob("lib64/python*/site-packages"),
            env / "Lib" / "site-packages",
        ]
        if p.is_dir()
    ]


def project_envs(root: Path, depth: int = 2) -> list[Path]:
    """Virtual / conda environments inside the project (not the project's own sources)."""
    found: list[Path] = []

    def walk(d: Path, level: int) -> None:
        try:
            kids = sorted(x for x in d.iterdir() if x.is_dir() and x.name not in _SKIP)
        except OSError:
            return
        for k in kids:
            if (k / "pyvenv.cfg").is_file() or (k / "conda-meta").is_dir():
                found.append(k)
            elif level < depth:
                walk(k, level + 1)

    walk(root, 1)
    return found


def installed_in_project(root: Path, dist: str) -> tuple[str, str] | None:
    """(version, where) of ``dist`` installed in a project environment, read from its ``*.dist-info`` folder."""
    want = norm(dist)
    for env in project_envs(root):
        for sp in _site_packages(env):
            for info in sp.glob("*.dist-info"):
                name, _, version = info.name[: -len(".dist-info")].rpartition("-")
                if norm(name) == want and version:
                    return version, f"installed in {env.name}"
    return None


def locked(root: Path, dist: str) -> tuple[str, str] | None:
    """(version, lock file) of ``dist`` in uv.lock / poetry.lock / pdm.lock / Pipfile.lock."""
    want = norm(dist)
    for name in ("uv.lock", "poetry.lock", "pdm.lock"):
        f = root / name
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for block in text.split("[[package]]")[1:]:
            n = re.search(r'^name\s*=\s*"([^"]+)"', block, re.M)
            v = re.search(r'^version\s*=\s*"([^"]+)"', block, re.M)
            if n and v and norm(n.group(1)) == want:
                return v.group(1), name
    f = root / "Pipfile.lock"
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for section in ("default", "develop"):
            for n, body in (data.get(section) or {}).items():
                if norm(n) == want and isinstance(body, dict) and isinstance(body.get("version"), str):
                    return body["version"].lstrip("="), "Pipfile.lock"
    return None


def agent_version(
    root: Path | None, dists: tuple[str, ...] | list[str], declared: dict[str, str | None]
) -> tuple[str | None, str | None]:
    """(version or declared range, source) of the first of ``dists`` the agent uses. See the module docstring."""
    for dist in dists:
        if root is not None:
            got = installed_in_project(root, dist) or locked(root, dist)
            if got:
                return got
    for dist in dists:
        pin = exact(declared.get(norm(dist)))
        if pin:
            return pin, "requirements (pinned)"
    for dist in dists:
        try:
            have = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        spec = declared.get(norm(dist))
        if spec is None or satisfies(have, spec) is not False:  # same environment as far as the project says
            return have, "installed"
    for dist in dists:
        spec = declared.get(norm(dist))
        if spec and clauses(spec):
            return spec.split(";", 1)[0].strip(), "requirements (range)"
    return None, None
