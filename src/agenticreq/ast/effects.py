"""What a tool does, read from its body: READ, WRITE or UNCLEAR, with the reason and the line.

The body (and the project functions it calls, up to ``_DEPTH`` deep) is the evidence; the tool's name is only used
when there is no body to read. Rules that need "is this a write tool" ask ``classify.is_write_tool``, which uses
this result first.

WRITE evidence: ``open(..., "w" / "a" / "x" / "+")``, ``Path.write_text / write_bytes / touch / unlink / rename``,
``os.remove / rename / makedirs``, ``shutil`` copy / move / rmtree, a program started (``subprocess``,
``os.system``), HTTP POST / PUT / PATCH / DELETE (requests, httpx, aiohttp, urllib with data), SQL INSERT / UPDATE /
DELETE / DROP ..., ORM ``commit / save / delete / add``, sending mail or messages, cloud writes (``put_object``,
``upload_file`` ...), and a change to module-level state (``ORDERS.append(...)``).
READ evidence: ``open(...)`` with no mode or ``"r"``, ``read_text``, ``os.walk / listdir / stat``, HTTP GET, SQL
SELECT. A body with no I/O at all is READ ("no side effects found").
UNCLEAR: the body calls something agenticreq cannot follow (a function of a library it does not know, a call on
what such a function returned, a dynamic call); the reason names that call. ``mkdir(..., exist_ok=True)`` only makes
sure a folder exists: it is not counted as a write.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agenticreq.ast.extractor import ProjectExtractor
    from agenticreq.ast.symbols import FileCtx

Effect = Literal["read", "write", "unclear"]
_DEPTH = 4

#: modules whose functions do no I/O (math, text, parsing, data structures, number crunching)
_PURE_MODULES = {
    "math",
    "cmath",
    "statistics",
    "decimal",
    "fractions",
    "random",
    "re",
    "string",
    "textwrap",
    "unicodedata",
    "json",
    "ast",
    "operator",
    "functools",
    "itertools",
    "collections",
    "dataclasses",
    "enum",
    "typing",
    "copy",
    "datetime",
    "time",
    "calendar",
    "zoneinfo",
    "uuid",
    "hashlib",
    "hmac",
    "base64",
    "binascii",
    "struct",
    "html",
    "difflib",
    "heapq",
    "bisect",
    "numbers",
    "pprint",
    "inspect",
    "types",
    "abc",
    "contextlib",
    "warnings",
    "numpy",
    "np",
    "pandas",
    "pd",
    "scipy",
    "pydantic",
    "annotated_types",
    "csv",
    "io",
    "ipaddress",
    "urllib.parse",
    "html.parser",
    "logging",
    "shlex",
    "fnmatch",
    "locale",
    "secrets",
}
#: modules that only read: the network for GET, the file system for listing / stat
_READ_MODULES = {"socket", "glob", "platform", "getpass", "mimetypes", "dotenv", "pathlib"}
#: database drivers: connecting is neither; the SQL text they run decides (execute("SELECT ...") / ("UPDATE ..."))
_DB_MODULES = {"sqlite3", "psycopg2", "psycopg", "pymysql", "mysql", "sqlalchemy", "duckdb", "asyncpg", "aiosqlite"}
#: built-in functions with no side effect
_PURE_BUILTINS = {
    "len",
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "any",
    "all",
    "isinstance",
    "issubclass",
    "type",
    "repr",
    "format",
    "chr",
    "ord",
    "hash",
    "id",
    "divmod",
    "pow",
    "iter",
    "next",
    "getattr",
    "hasattr",
    "vars",
    "callable",
    "super",
    "object",
    "print",
    "slice",
    "hex",
    "oct",
    "bin",
    "ascii",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "IndexError",
    "ZeroDivisionError",
    "PermissionError",
    "NotImplementedError",
    "OSError",
    "FileNotFoundError",
    "property",
    "staticmethod",
    "classmethod",
    "complex",
}
_FS_WRITE_ATTRS = {
    "write_text",
    "write_bytes",
    "touch",
    "unlink",
    "rmdir",
    "rename",
    "replace",
    "symlink_to",
    "hardlink_to",
    "chmod",
    "write",
}
_FS_READ_ATTRS = {
    "read_text",
    "read_bytes",
    "exists",
    "is_file",
    "is_dir",
    "stat",
    "iterdir",
    "glob",
    "rglob",
    "resolve",
    "absolute",
    "relative_to",
    "with_suffix",
    "with_name",
    "joinpath",
    "read",
    "readline",
    "readlines",
    "open_text",
    "lstat",
    "samefile",
    "expanduser",
    "as_posix",
}
_OS_WRITE = {
    "remove",
    "unlink",
    "rename",
    "replace",
    "rmdir",
    "removedirs",
    "makedirs",
    "mkdir",
    "system",
    "popen",
    "kill",
    "chmod",
    "chown",
    "symlink",
    "link",
    "truncate",
    "putenv",
    "startfile",
    "execv",
    "execve",
    "execvp",
    "spawnl",
    "spawnv",
    "fork",
}
_OS_READ = {
    "walk",
    "listdir",
    "scandir",
    "stat",
    "getcwd",
    "getenv",
    "path",
    "environ",
    "fspath",
    "cpu_count",
    "getpid",
    "sep",
    "linesep",
    "lstat",
    "access",
}
_SHUTIL_WRITE = {
    "copy",
    "copy2",
    "copyfile",
    "copytree",
    "move",
    "rmtree",
    "make_archive",
    "unpack_archive",
    "copymode",
    "copystat",
}
#: names a str or bytes also has: they write only on a path (Path(...).replace(...), not "a,b".replace(",", ""))
_AMBIGUOUS_FS = {"rename", "replace", "write"}
_HTTP_WRITE = {"post", "put", "patch", "delete"}
_HTTP_READ = {"get", "head", "options", "urlopen", "build_opener", "Request", "open", "request"}
_HTTP_MODULES = {"requests", "httpx", "aiohttp", "urllib", "urllib.request", "urllib3", "http.client"}
#: methods that send or store, whatever object they are called on
_SEND_WRITE = {
    "send_message",
    "sendmail",
    "send_mail",
    "send_email",
    "post_message",
    "chat_postMessage",
    "put_object",
    "upload_file",
    "upload_fileobj",
    "delete_object",
    "delete_objects",
    "put_item",
    "update_item",
    "delete_item",
    "batch_write_item",
    "upload_blob",
    "upload_from_filename",
    "upload_from_string",
    "insert_one",
    "insert_many",
    "update_one",
    "update_many",
    "delete_one",
    "delete_many",
    "replace_one",
    "bulk_write",
    "commit",
    "executemany",
    "bulk_create",
    "bulk_update",
    "save",
    "flush",
}
_SEND_WRITE |= {"a" + m for m in _SEND_WRITE}  # asave(), acommit(), asend_message() ...
_SQL_WRITE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE|REPLACE|UPSERT|GRANT)\b", re.IGNORECASE
)
_SQL_READ = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE|EXPLAIN|PRAGMA)\b", re.IGNORECASE)
#: receivers that are a database, a session or a store: add / delete / update on them write
_STORE_NAME = re.compile(
    r"(session|db|database|conn|connection|cursor|cur|collection|table|repo|repository|store|"
    r"bucket|index|client|queue|cache|redis|orm)s?$",
    re.IGNORECASE,
)
_STORE_WRITE = {
    "add",
    "add_all",
    "delete",
    "merge",
    "update",
    "insert",
    "upsert",
    "set",
    "hset",
    "lpush",
    "rpush",
    "publish",
    "enqueue",
    "put",
    "remove",
    "drop",
    "create",
    "destroy",
}
_STORE_WRITE |= {"a" + m for m in _STORE_WRITE}  # the async forms: store.aput(...), session.adelete(...)


@dataclass
class _Found:
    writes: list[tuple[str, int | None]] = field(default_factory=list)
    reads: list[tuple[str, int | None]] = field(default_factory=list)
    unknown: list[tuple[str, int | None]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolEffect:
    effect: Effect
    reason: str
    line: int | None = None
    file: str = ""


def _external_imports(tree: ast.Module) -> dict[str, str]:
    """local name -> the module it comes from, for every import in the file (``import numpy as np`` -> np: numpy,
    ``from shutil import rmtree`` -> rmtree: shutil.rmtree)."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def _root(node: ast.AST) -> ast.AST:
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


def _mode(call: ast.Call, pos: int) -> str | None:
    m = call.args[pos] if len(call.args) > pos else next((k.value for k in call.keywords if k.arg == "mode"), None)
    return m.value if isinstance(m, ast.Constant) and isinstance(m.value, str) else None


def _writes_mode(mode: str | None) -> bool:
    return bool(mode) and any(ch in (mode or "") for ch in "wax+")


def _exist_ok(call: ast.Call) -> bool:
    return any(
        k.arg == "exist_ok" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords
    )


def _sql_text(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(str(v.value) for v in node.values if isinstance(v, ast.Constant))
    if isinstance(node, ast.Call) and node.args:  # text("UPDATE ..."), sql.SQL("DELETE ...")
        return _sql_text(node.args[0])
    return ""


class _Scan:
    def __init__(self, ex: ProjectExtractor | None, fc: FileCtx | None, fn: ast.AST) -> None:
        self.ex, self.fc, self.fn = ex, fc, fn
        self.ext = _external_imports(fc.tree) if fc is not None else {}
        self.module_names = set(fc.consts) if fc is not None else set()
        self.locals: set[str] = set()
        self.from_unknown: dict[str, str] = {}  # local -> the unknown call it came from
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.locals |= {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
            for x in (fn.args.vararg, fn.args.kwarg):
                if x is not None:
                    self.locals.add(x.arg)
        for n in ast.walk(fn):
            for t in (
                n.targets
                if isinstance(n, ast.Assign)
                else [n.target]
                if isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor))
                else []
            ):
                self.locals |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            if isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        self.locals |= {x.id for x in ast.walk(item.optional_vars) if isinstance(x, ast.Name)}
            if isinstance(n, ast.comprehension):
                self.locals |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
            if isinstance(n, ast.ExceptHandler) and n.name:
                self.locals.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn:
                if not isinstance(n, ast.Lambda):
                    self.locals.add(n.name)

    def module_of(self, name: str) -> str | None:
        return self.ext.get(name)


def _body_nodes(fn: ast.AST) -> list[ast.AST]:
    body = getattr(fn, "body", None)
    if not isinstance(body, list):
        return list(ast.walk(fn))
    return [x for st in body for x in ast.walk(st)]


def _call_name(f: ast.AST) -> str:
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _module_root(mod: str) -> str:
    return mod.split(".")[0]


def _scan(ex: ProjectExtractor | None, fc: FileCtx | None, fn: ast.AST, depth: int, seen: set[int]) -> _Found:
    s = _Scan(ex, fc, fn)
    found = _Found()

    def where(node: ast.AST) -> int | None:
        return getattr(node, "lineno", None)

    # memory inside the process (a cache dict, a global "loaded" flag) is not an effect on the world: not counted.
    # Setting an attribute on a parameter (state.seat = seat) changes the state object the caller keeps: a write
    params = set()
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        params = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]} - {"self", "cls"}
    for n in _body_nodes(fn):  # the body only: a decorator (@tool("name")) is not what the tool does
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for tgt in n.targets if isinstance(n, ast.Assign) else [n.target]:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id in params:
                    found.writes.append((f"changes the caller's {tgt.value.id}.{tgt.attr}", where(n)))
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = _call_name(f)
        root = _root(f)
        rname = root.id if isinstance(root, ast.Name) else ""
        mod = s.module_of(rname) if rname and rname not in s.locals else None
        full = (
            (mod + "." + ".".join(ast.unparse(f).split(".")[1:]))
            if mod and isinstance(f, ast.Attribute)
            else (mod or "")
        )
        line = where(n)
        # -------- files
        if name == "open" and isinstance(f, ast.Name) and rname not in s.locals:
            (found.writes if _writes_mode(_mode(n, 1)) else found.reads).append(
                (f"open(..., {_mode(n, 1)!r})" if _writes_mode(_mode(n, 1)) else "opens a file to read", line)
            )
            continue
        if name == "open" and isinstance(f, ast.Attribute):  # Path(...).open(mode), opener.open(request)
            (found.writes if _writes_mode(_mode(n, 0)) else found.reads).append(
                (f".open({_mode(n, 0)!r})" if _writes_mode(_mode(n, 0)) else "opens to read", line)
            )
            continue
        if name in ("NamedTemporaryFile", "TemporaryFile", "mkstemp") and _module_root(mod or "") == "tempfile":
            mode = _mode(n, 0)
            (found.writes if mode is None or _writes_mode(mode) else found.reads).append((f"tempfile.{name}", line))
            continue
        if name in ("mkdir", "makedirs"):
            if not _exist_ok(n):
                found.writes.append((f"{name}() creates a folder", line))
            continue  # mkdir(exist_ok=True) only makes sure a folder exists
        if mod and _module_root(mod) == "os":
            if name in _OS_WRITE:
                found.writes.append((f"os.{name}", line))
            elif name in _OS_READ or full.startswith("os.path"):
                found.reads.append((f"os.{name}", line))
            else:
                found.unknown.append((ast.unparse(f), line))
            continue
        if mod and _module_root(mod) == "shutil":
            (found.writes if name in _SHUTIL_WRITE else found.reads).append((f"shutil.{name}", line))
            continue
        if mod and _module_root(mod) in ("subprocess", "pty", "multiprocessing"):
            found.writes.append((f"{_module_root(mod)}.{name} starts another program", line))
            continue
        if name in ("exec", "eval") and isinstance(f, ast.Name):
            found.writes.append((f"{name}() runs dynamic code", line))
            continue
        if (
            isinstance(f, ast.Attribute)
            and name in _FS_WRITE_ATTRS
            and (name not in _AMBIGUOUS_FS or _pathish(f.value))
        ):
            found.writes.append((f".{name}()", line))  # write_text / unlink always; rename / replace on a path
            continue
        if isinstance(f, ast.Attribute) and name in _FS_READ_ATTRS and _pathish(f.value):
            found.reads.append((f".{name}()", line))  # path.read_text(), file_path.exists()
            continue
        # -------- SQL
        if name in ("execute", "executescript", "executemany", "exec_driver_sql", "raw", "query", "sql"):
            text = _sql_text(n.args[0] if n.args else None)
            if _SQL_WRITE.match(text):
                found.writes.append((f"SQL {text.split()[0].upper()}", line))
                continue
            if _SQL_READ.match(text):
                found.reads.append((f"SQL {text.split()[0].upper()}", line))
                continue
        # -------- HTTP
        if (
            mod
            and _module_root(mod) in {m.split(".")[0] for m in _HTTP_MODULES}
            or (
                isinstance(root, ast.Name)
                and root.id in s.locals
                and name in _HTTP_WRITE | {"get", "head"}
                and _looks_http_client(s, root.id, fn)
            )
        ):
            if (
                name in _HTTP_WRITE
                or _http_method(n) in _HTTP_WRITE
                or (name in ("Request", "urlopen") and _kw_or_pos(n, "data", 1) is not None)
            ):
                found.writes.append((f"HTTP {(_http_method(n) or name).upper()}", line))
            else:
                found.reads.append((f"HTTP {(_http_method(n) or 'GET').upper()}", line))
            continue
        # -------- sending, cloud, ORM, stores
        if isinstance(f, ast.Attribute) and name in _SEND_WRITE:
            found.writes.append((f".{name}()", line))
            continue
        if (
            isinstance(f, ast.Attribute)
            and name in _STORE_WRITE
            and isinstance(f.value, (ast.Name, ast.Attribute))
            and _STORE_NAME.search(_call_name(f.value) or (f.value.id if isinstance(f.value, ast.Name) else ""))
        ):
            found.writes.append((f"{ast.unparse(f.value)}.{name}()", line))
            continue
        # -------- what is left: pure, a project function to follow, or a call agenticreq cannot follow
        if isinstance(f, ast.Name) and f.id in _PURE_BUILTINS and f.id not in s.locals:
            continue
        if mod is not None:
            top = _module_root(mod)
            if mod in _PURE_MODULES or top in _PURE_MODULES or mod.rsplit(".", 1)[0] in _PURE_MODULES:
                continue
            if top in _READ_MODULES:
                found.reads.append((mod, line))
                continue
            if top in _DB_MODULES:
                continue  # connect(), cursor(): the SQL they run is judged above
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            continue  # a method of what another call returned (x.execute(...).fetchall()): that call decides
        callee = _project_function(ex, fc, n, s)
        if callee is not None:
            cfc, cfn = callee
            if id(cfn) in seen or depth >= _DEPTH:
                continue
            seen.add(id(cfn))
            sub = _scan(ex, cfc, cfn, depth + 1, seen)
            found.writes += [(f"{w} (in {getattr(cfn, 'name', '?')})", ln) for w, ln in sub.writes]
            found.reads += [(f"{r} (in {getattr(cfn, 'name', '?')})", ln) for r, ln in sub.reads]
            found.unknown += [(f"{u} (in {getattr(cfn, 'name', '?')})", ln) for u, ln in sub.unknown]
            continue
        if isinstance(root, ast.Name) and (root.id in s.locals or root.id in ("self", "cls")):
            continue  # a method of a local value (str / list / dict / a parsed object): no I/O of its own
        if not isinstance(root, ast.Name):
            continue  # a method of a computed value: "sep".join(...), (1 + rets).cumprod(), (root / name).resolve()
        if root.id == "open" and "open" not in s.locals:
            continue  # open(f).read(): the open itself was judged by its mode above
        if root.id in _PURE_BUILTINS:
            continue  # super().__init__(), str(x).strip(), dict(a).items()
        if _project_class(ex, fc, root):
            continue  # building a project object; its methods called on it are locals above
        if isinstance(root, ast.Name) and root.id in s.module_names:
            continue  # a module-level object of this project (a compiled regex, a dict of functions)
        found.unknown.append((ast.unparse(f)[:60], line))
    return found


def _pathish(node: ast.AST) -> bool:
    """A value that is a file system path: Path(...), a name or attribute that says path / file / dir."""
    if isinstance(node, ast.Call) and _call_name(node.func) in ("Path", "PurePath", "WindowsPath", "PosixPath"):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):  # root / "name"
        return True
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    return bool(re.search(r"(path|file|dir|folder|dest|target|src)", name, re.IGNORECASE))


def _kw_or_pos(n: ast.Call, kw: str, pos: int) -> ast.AST | None:
    v = next((k.value for k in n.keywords if k.arg == kw), None)
    return v if v is not None else (n.args[pos] if len(n.args) > pos else None)


def _http_method(n: ast.Call) -> str:
    """requests.request("POST", url) / Request(url, method="PUT")."""
    if _call_name(n.func) == "request" and n.args and isinstance(n.args[0], ast.Constant):
        return str(n.args[0].value).lower()
    m = next((k.value for k in n.keywords if k.arg == "method"), None)
    return str(m.value).lower() if isinstance(m, ast.Constant) else ""


def _looks_http_client(s: _Scan, name: str, fn: ast.AST) -> bool:
    """A local built from an HTTP client: client = httpx.Client(), session = requests.Session()."""
    for n in ast.walk(fn):
        val = (
            n.value
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            else None
        )
        if val is None and isinstance(n, (ast.With, ast.AsyncWith)):
            val = next(
                (
                    i.context_expr
                    for i in n.items
                    if isinstance(i.optional_vars, ast.Name) and i.optional_vars.id == name
                ),
                None,
            )
        if isinstance(val, (ast.Call, ast.Await)):
            r = _root(val.value if isinstance(val, ast.Await) else val)
            if isinstance(r, ast.Name) and _module_root(s.module_of(r.id) or "") in {"requests", "httpx", "aiohttp"}:
                return True
    return False


def _project_function(
    ex: ProjectExtractor | None, fc: FileCtx | None, call: ast.Call, s: _Scan
) -> tuple[FileCtx, ast.AST] | None:
    if ex is None or fc is None:
        return None
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls"):
        for cls in fc.classes.values():
            m = next(
                (x for x in cls.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == f.attr),
                None,
            )
            if m is not None:
                return fc, m
        return None
    if isinstance(f, ast.Name) and f.id in s.locals:
        return None
    r = ex.symbols.resolve(fc, f)
    if r is not None and r.kind == "function" and isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return r.fc, r.node
    if r is not None and r.kind == "class" and isinstance(r.node, ast.ClassDef):
        init = next((x for x in r.node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
        return (r.fc, init) if init is not None else (r.fc, ast.Module(body=[], type_ignores=[]))
    return None


def _project_class(ex: ProjectExtractor | None, fc: FileCtx | None, root: ast.AST) -> bool:
    if ex is None or fc is None or not isinstance(root, ast.Name):
        return False
    r = ex.symbols.resolve(fc, root)
    return r is not None and r.kind in ("class", "function", "const")


def _placeholder(fn: ast.AST) -> bool:
    """The body only raises, passes or is ``...``: the real implementation is not in the code."""
    body = list(getattr(fn, "body", []))
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # the docstring
    return bool(body) and all(
        isinstance(st, (ast.Raise, ast.Pass))
        or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and st.value.value is Ellipsis)
        for st in body
    )


def classify_body(ex: ProjectExtractor | None, fc: FileCtx | None, fn: ast.AST) -> ToolEffect:
    file = fc.rel if fc is not None else ""
    if _placeholder(fn):
        return ToolEffect(
            "unclear",
            "has only a placeholder body (raise / pass), so what it does is not in the code",
            getattr(fn, "lineno", None),
            file,
        )
    found = _scan(ex, fc, fn, 0, {id(fn)})

    def own_first(items: list[tuple[str, int | None]]) -> tuple[str, int | None]:
        return next((x for x in items if " (in " not in x[0]), items[0])  # the tool's own line before a helper's

    if found.writes:
        why, line = own_first(found.writes)
        return ToolEffect("write", f"writes: {why}", line, file)
    if found.unknown:
        call, line = own_first(found.unknown)
        return ToolEffect("unclear", f"calls {call}, which agenticreq cannot follow", line, file)
    if found.reads:
        why, line = own_first(found.reads)
        return ToolEffect("read", f"reads only: {why}", line, file)
    return ToolEffect("read", "no side effects found", getattr(fn, "lineno", None), file)


def tool_effect(obs: object, t: object) -> ToolEffect:
    """The effect of a tool as stored by the extractor (``ToolSchema.effect``), else from its name."""
    from agenticreq.ast.classify import is_write_tool, verb_class

    eff = getattr(t, "effect", None)
    if eff:
        return ToolEffect(eff, getattr(t, "effect_reason", "") or "", getattr(t, "effect_line", None))
    if is_write_tool(t):  # type: ignore[arg-type]
        return ToolEffect("write", "the name says it writes (no body to read)")
    unclear = verb_class(getattr(t, "name", "")) == "unknown"
    return ToolEffect("unclear" if unclear else "read", "the name only (no body to read)")
