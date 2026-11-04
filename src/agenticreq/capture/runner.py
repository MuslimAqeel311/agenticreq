"""Child-process entry for a mock run: install guards + hook, run the agent, dump captures.

Invoked as ``python -B -m agenticreq.capture.runner <config.json>`` by discover.py, inside a
scrubbed environment. Never imported by the parent.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any

from agenticreq.capture.entry import split_entry
from agenticreq.capture.gates import GateProbe
from agenticreq.capture.hook import CaptureSession
from agenticreq.capture.responder import CannedResponder, MockConfig
from agenticreq.capture.guard import LLM_HOSTS, ProcessGuard, SocketGuard
from agenticreq.capture.wire import configure


#: data/tool_wrappers.yml, resolved by the parent (the agent's environment needs no YAML reader); this copy is only
#: for a child started without it
_WRAPPER_LEVELS: tuple[tuple[str, ...], ...] = (("func", "coroutine"),)
_ASYNC_ATTRS = ("coroutine",)


def _same_face(stub: Any, real: Any) -> Any:
    """The stub keeps the real function's name, docstring and signature: frameworks build the tool's schema from
    them, and some refuse two tools with one name."""
    if real is None:
        return stub
    for attr in ("__name__", "__qualname__", "__doc__", "__module__", "__annotations__"):
        try:
            setattr(stub, attr, getattr(real, attr))
        except (AttributeError, TypeError):
            pass
    try:
        stub.__signature__ = inspect.signature(real)
    except (TypeError, ValueError):
        pass
    return stub


def _stub(value: Any, real: Any = None) -> Any:
    def stub(*_a: Any, **_k: Any) -> Any:
        return value

    return _same_face(stub, real)


def _async_stub(value: Any, real: Any = None) -> Any:
    async def stub(*_a: Any, **_k: Any) -> Any:
        return value

    return _same_face(stub, real)


def _stub_for(attr: str, real: Any, value: Any, async_attrs: tuple[str, ...]) -> Any:
    is_async = attr in async_attrs or inspect.iscoroutinefunction(real)
    return _async_stub(value, real) if is_async else _stub(value, real)


def _stub_in_place(target: Any, value: Any, levels: Any, async_attrs: tuple[str, ...]) -> bool:
    """Stub the function a tool object or tool class holds, leaving the wrapper as it is. Only the innermost level
    it has: the framework's own ``run()`` keeps working around the stub. False: it holds none."""
    held = next((found for attrs in levels if (found := [a for a in attrs if callable(getattr(target, a, None))])), [])
    for a in held:
        new = _stub_for(a, getattr(target, a), value, async_attrs)
        if inspect.isclass(target):
            setattr(target, a, new)
        else:
            object.__setattr__(target, a, new)
    return bool(held)


def apply_patches(
    mocks: MockConfig, levels: Any = _WRAPPER_LEVELS, async_attrs: tuple[str, ...] = _ASYNC_ATTRS
) -> list[str]:
    """Replace each ``patch:`` target with a stub returning ``returns``. A tool object or tool class (anything
    holding the real function under a name in data/tool_wrappers.yml) is stubbed in place, so it keeps its type, the
    lists the agent built from it at import time get the stub too, and the real function never runs."""
    applied = []
    for name, m in mocks.tools.items():
        if not m.patch:
            continue
        mod_name, _, attr = m.patch.partition(":")
        mod = importlib.import_module(mod_name)
        value = m.returns if m.returns is not None else f"mock result for {name}"
        current = getattr(mod, attr, None)
        plain = current is None or inspect.isfunction(current) or inspect.ismethod(current)
        if plain or not _stub_in_place(current, value, levels, async_attrs):
            make = _async_stub if inspect.iscoroutinefunction(current) else _stub  # async stays async
            stub = make(value, current)
            setattr(mod, attr, stub)
            if current is not None:
                _replace_everywhere(current, stub)
                f = current
                for _ in range(10):  # @safe_tool / functools.wraps: the real body is the innermost __wrapped__
                    _swap_code(f, value)
                    if not callable(getattr(f, "__wrapped__", None)):
                        break
                    f = f.__wrapped__
        applied.append(m.patch)
    return applied


def _swap_code(fn: Any, value: Any) -> None:
    """The function object itself returns the mock value: a framework that wrapped it at import time (AutoGen
    FunctionTool, a registry) holds this same object, so its real body never runs either."""
    import types

    if not isinstance(fn, types.FunctionType):
        return
    head = "async def" if inspect.iscoroutinefunction(fn) else "def"
    ns: dict[str, Any] = {}
    exec(f"{head} _mock(*args, **kwargs):\n    return {value!r}\n", ns)  # noqa: S102 - a literal, from mocks.yml
    try:
        sig = inspect.signature(fn)  # a framework reads the tool's parameters from it: they stay as written
        fn.__code__ = ns["_mock"].__code__
        fn.__signature__ = sig  # type: ignore[attr-defined]
    except ValueError:  # a closure: its free variables do not fit; the name and list replacement above still apply
        pass


def _replace_everywhere(old: Any, new: Any) -> None:
    """``tools/__init__.py`` imported the function before the patch, and put it in TOOLS = [f, ...]: replace it in
    every loaded project module's names and in the lists and dicts they hold, so the real function never runs."""
    root = Path.cwd().resolve()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not _inside(f, root) or _package_of(f):
            continue
        for k, v in list(vars(mod).items()):
            if v is old:
                setattr(mod, k, new)
            elif isinstance(v, list):
                v[:] = [new if x is old else x for x in v]
            elif isinstance(v, dict):
                for dk, dv in list(v.items()):
                    if dv is old:
                        v[dk] = new


def _call(fn: Any, args: Any) -> Any:
    """Call the entry with the arguments from --args (a JSON list = positional, a JSON object = keyword)."""
    pos, kw = (list(args), {}) if isinstance(args, list) else ([], dict(args or {}))
    try:
        inspect.signature(fn).bind(*pos, **kw)
    except TypeError as e:
        raise TypeError(
            f"{e}. The mock run calls the entry itself: give its inputs under args: in mocks.yml, e.g. "
            "args: {request: analyse AAPL} (or pass --args with a JSON object)"
        ) from None
    except ValueError:
        pass  # builtins without a signature
    return fn(*pos, **kw)


#: packages whose own code checks a model request before it is sent (an SDK, or the framework's model class)
_SDK_PACKAGES = (
    "anthropic",
    "openai",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_google_genai",
    "langchain_mistralai",
    "langchain_groq",
    "langchain_aws",
    "langchain_core",
    "google",
    "mistralai",
    "cohere",
    "groq",
    "litellm",
    "pydantic_ai",
    "smolagents",
    "crewai",
    "agents",
    "autogen_ext",
    "llama_index",
)


def _norm(filename: str) -> str:
    return filename.replace("\\", "/")


def _package_of(filename: str) -> str:
    """The installed package a frame's file belongs to ("anthropic"), or "" for project / standard-library code."""
    norm = _norm(filename)
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in norm:
            return norm.split(marker, 1)[1].split("/", 1)[0].removesuffix(".py")
    return ""


def _inside(filename: str, root: Path) -> bool:
    if filename.startswith("<"):  # <frozen importlib._bootstrap>, <string>: not a file of the project
        return False
    try:
        Path(filename).resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def describe_failure(exc: BaseException, entry: str, phase: str) -> dict[str, Any]:
    """What stopped the run, in parts the summary turns into one sentence: the exception, the project file:line it
    stopped at, the SDK that raised it (if one did), and the first error of the chain (a retry hides it)."""
    root = Path.cwd().resolve()
    frames = traceback.extract_tb(exc.__traceback__)
    own = [f for f in frames if not _package_of(f.filename) and _inside(f.filename, root)]
    where = own[-1] if own else None
    raised_in = _package_of(frames[-1].filename) if frames else ""
    first: BaseException = exc
    seen = {id(exc)}
    while True:  # the first error of a chain: a retry loop raises the last one "during handling of" the first
        nxt = first.__cause__ or first.__context__
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        first = nxt
    rel = ""
    if where is not None:
        try:
            rel = f"{Path(where.filename).resolve().relative_to(root).as_posix()}:{where.lineno}"
        except ValueError:
            rel = f"{Path(where.filename).name}:{where.lineno}"
    target, func = split_entry(entry)  # C:\x\agent.py:run: the drive letter is not the function
    func = func or ""
    mod = Path(target).stem if target.endswith(".py") else target
    entry_missing = (
        phase == "run"
        and not own
        and (
            (isinstance(exc, ModuleNotFoundError) and exc.name in (mod, mod.split(".")[0], target))
            or (isinstance(exc, (AttributeError, KeyError)) and bool(func) and func in str(exc))
            or (isinstance(exc, FileNotFoundError) and target.endswith(".py"))
        )
    )

    def text(e: BaseException) -> str:
        s = str(e).strip()
        return s.splitlines()[0][:300] if s else ""

    return {
        "type": type(exc).__name__,
        "message": text(exc),
        "where": rel,
        "phase": phase,
        "entry_missing": entry_missing,
        "sdk": raised_in if raised_in.split(".")[0] in _SDK_PACKAGES else "",
        "missing_module": exc.name if isinstance(exc, ImportError) else None,
        "first": None if first is exc else {"type": type(first).__name__, "message": text(first)},
    }


class _Progress:
    """Writes how far the run got (model calls so far) every half second, so a run stopped by the timeout can
    still say which step it reached."""

    def __init__(self, path: Path, session: Any, procs: Any = None) -> None:
        import threading

        self.path, self.session, self.procs = path, session, procs
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _write(self) -> None:
        calls = list(self.session.calls)
        last = calls[-1] if calls else None
        data = {
            "calls": len(calls),
            "last_tools": [t.name for t in last.tools][:8] if last else [],
            "last_asked": [tc.name for tc in last.tool_calls] if last else [],
            "blocked": [b.removeprefix("write:") for b in (self.procs.blocked if self.procs else [])][-3:],
        }
        try:
            self.path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            self._write()

    def start(self) -> None:
        self._write()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._write()


def run_entry(entry: str, args: Any = None) -> None:
    target, func = split_entry(entry)
    if target.endswith(".py"):
        path = Path(target).resolve()
        sys.path.insert(0, str(path.parent))
        sys.argv = [str(path)]
        if func:
            ns = runpy.run_path(str(path), run_name="agenticreq_entry")
            result = _call(ns[func], args)
        else:
            runpy.run_path(str(path), run_name="__main__")
            return
    else:
        sys.path.insert(0, str(Path.cwd()))
        if func:
            result = _call(getattr(importlib.import_module(target), func), args)
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
            return
    if inspect.iscoroutine(result):
        asyncio.run(result)


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    if cfg.get("mocks_data") is not None:  # the parent resolved this run's settings: no YAML reader needed here
        mocks = MockConfig.from_dict(cfg["mocks_data"])
    else:
        mocks = MockConfig.load(cfg["mocks"]).for_run(cfg.get("run_name")) if cfg.get("mocks") else MockConfig()
    if cfg.get("providers"):
        configure(cfg["providers"])
    hosts = [h for h in (cfg.get("providers") or {}).get("openai_compatible_hosts") or []]
    guard = SocketGuard([], (*LLM_HOSTS, *hosts))  # a mock run has no network: its model is canned, its tools are local
    guard.install()
    # writes are allowed in the temp folder, the project's .agenticreq/ and the paths under allow_write: in
    # mocks.yml; nothing else (no library cache folders either)
    procs = ProcessGuard(
        [*(cfg.get("allow_write") or []), str(Path(cfg["out"]).resolve().parent), str(Path.cwd() / ".agenticreq")],
        cache_dirs=False,
        project=str(Path.cwd()),
    )
    procs.install()
    procs.divert_writes(str(Path(cfg["out"]).resolve().parent / "blocked-writes"))
    session = CaptureSession(CannedResponder(mocks), run_id=cfg["run_id"], egress_allow=[])
    session.install()
    gates = GateProbe(session, answer=mocks.human_input)
    gates.install()
    error = None
    failure: dict[str, Any] | None = None
    patched: list[str] = []
    progress = _Progress(Path(cfg["out"]).with_suffix(".progress.json"), session, procs)
    progress.start()
    phase = "patch"
    try:
        patched = apply_patches(
            mocks,
            cfg.get("wrapper_levels") or _WRAPPER_LEVELS,
            tuple(cfg.get("async_attrs") or _ASYNC_ATTRS),
        )
        phase = "run"
        run_entry(cfg["entry"], cfg.get("call_args"))
    except SystemExit as e:
        if e.code not in (0, None):
            error = f"SystemExit({e.code})"
            failure = {
                "type": "SystemExit",
                "message": f"exit code {e.code}",
                "where": "",
                "sdk": "",
                "first": None,
                "phase": phase,
                "entry_missing": False,
                "missing_module": None,
            }
    except BaseException as e:  # the agent may raise anything; record it and keep the captures
        error = traceback.format_exc(limit=8)
        failure = describe_failure(e, cfg["entry"], phase)
    finally:
        progress.stop()
        gates.uninstall()
        session.uninstall()
        guard.uninstall()
        procs.uninstall()
    versions = {}
    for lib in ("httpx", "httpx2", "openai", "anthropic"):
        mod = sys.modules.get(lib)
        if mod is not None:
            versions[lib] = str(getattr(mod, "__version__", "?"))
    import importlib.metadata as md

    # the parent names what to look up: importing agenticreq.ast here would need pyyaml in the agent's environment
    for dist in cfg.get("probe_dists") or []:
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    for mod_name in (
        cfg.get("probe_modules") or []
    ):  # frameworks the run actually loaded from site-packages (not a same-named local package)
        loaded = sys.modules.get(mod_name)
        where = str(getattr(loaded, "__file__", "") or "")
        if loaded is not None and ("site-packages" in where or "dist-packages" in where):
            versions[f"module:{mod_name}"] = str(getattr(loaded, "__version__", "loaded"))
    out = {
        "run_id": cfg["run_id"],
        "calls": [c.model_dump(mode="json") for c in session.calls],
        "egress_blocked": session.blocked + guard.blocked + procs.blocked,
        "llm_misses": guard.llm_misses,
        "notes": guard.notes + [f"capture: {m}" for m in session.errors],
        "libs_hooked": session.libs_hooked,
        "gates_fired": gates.fired,
        "patched": patched,
        "patched_tools": [name for name, m in mocks.tools.items() if m.patch and m.patch in patched],
        "error": error,
        "failure": failure,
        "rejections": session.rejections,
        "versions": versions,
    }
    Path(cfg["out"]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
