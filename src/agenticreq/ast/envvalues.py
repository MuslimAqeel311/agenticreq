"""Values that come from the environment, read without running anything.

``MAX = int(os.getenv("MAX_MESSAGES", "10"))`` and ``.env.example: MAX_MESSAGES=10`` both tell what the code
runs with unless the deployment overrides it. Such a value is *assumed*: it is used (a bound that exists in
code is a bound) and always labelled, never reported as proven.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from agenticreq.ast.schema import _name, literal

if TYPE_CHECKING:
    from agenticreq.ast.symbols import FileCtx, SymbolTable

#: files that document the environment a project expects; the real `.env` wins over a default in code
DOTENV_FILES = (
    ".env",
    ".env.local",
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
    "example.env",
    "env.example",
    "sample.env",
)
_ENV_GETTERS = {"getenv", "get", "config", "env", "int", "float", "str", "bool"}
_CASTS = {"int": int, "float": float, "str": str}
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class Assumed(NamedTuple):
    value: str | int | float
    var: str
    where: str  # "default in code" | ".env" | ".env.example" | ...

    @property
    def label(self) -> str:
        return f"{self.where} of env {self.var}"


def load_dotenv(*roots: Path) -> dict[str, tuple[str, str]]:
    """{VAR: (value, file name)} from the `.env`-style files under the roots (first file listed wins).
    Only used to resolve a bound or a model id the code reads from the environment."""
    out: dict[str, tuple[str, str]] = {}
    for name in DOTENV_FILES:
        for root in dict.fromkeys(roots):
            f = root / name
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                m = _LINE.match(line)
                if m:
                    value = m.group(2).split(" #")[0].strip().strip("'\"")
                    out.setdefault(m.group(1), (value, name))
    return out


def _env_key(call: ast.Call) -> str | None:
    """VAR of os.getenv("VAR") / os.environ.get("VAR") / config("VAR") / env.int("VAR"), else None."""
    f = call.func
    fn = _name(f)
    if fn not in _ENV_GETTERS or not call.args:
        return None
    owner = _name(f.value) if isinstance(f, ast.Attribute) else ""
    reads_env = (
        fn in ("getenv", "config")
        or owner in ("environ", "env", "os")
        or (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.value.attr == "environ")
    )
    if not reads_env:
        return None
    ok, key = literal(call.args[0])
    return key if ok and isinstance(key, str) else None


def _num(v: str) -> int | float | str:
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def env_default(
    symbols: SymbolTable, fc: FileCtx, node: ast.AST | None, dotenv: dict[str, tuple[str, str]], depth: int = 0
) -> Assumed | None:
    """What an environment read evaluates to when the deployment sets nothing: the real `.env`, else the
    default in the code, else `.env.example`. Looks through int()/float()/str() and one-assignment names."""
    if node is None or depth > 4:
        return None
    if isinstance(node, ast.Name):
        r = symbols.resolve(fc, node)
        if r is not None and r.kind == "const" and isinstance(r.node, ast.expr):
            return env_default(symbols, r.fc, r.node, dotenv, depth + 1)
        return None
    if isinstance(node, ast.Subscript) and _name(node.value).endswith("environ"):
        ok, key = literal(node.slice)
        hit = dotenv.get(key) if ok and isinstance(key, str) else None
        return Assumed(_num(hit[0]), key, hit[1]) if hit else None
    if not isinstance(node, ast.Call):
        return None
    fn = _name(node.func)
    if fn in _CASTS and node.args and not _env_key(node):
        inner = env_default(symbols, fc, node.args[0], dotenv, depth + 1)
        if inner is None:
            return None
        try:
            return Assumed(_CASTS[fn](inner.value), inner.var, inner.where)
        except (TypeError, ValueError):
            return None
    key = _env_key(node)
    if key is None:
        return None
    kws = {k.arg: k.value for k in node.keywords if k.arg}
    default = node.args[1] if len(node.args) > 1 else kws.get("default")
    cast = kws.get("cast")
    ok, dv = literal(default)
    hit = dotenv.get(key)
    value: str | int | float
    if hit and hit[1] in (".env", ".env.local"):
        value, where = _num(hit[0]), hit[1]
    elif ok and isinstance(dv, (str, int, float)) and not isinstance(dv, bool):
        value, where = dv, "default in code"
    elif hit:
        value, where = _num(hit[0]), hit[1]
    else:
        return None
    if cast is not None and _name(cast) in _CASTS:
        try:
            value = _CASTS[_name(cast)](value)
        except (TypeError, ValueError):
            return None
    return Assumed(value, key, where)
