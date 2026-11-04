"""Safety gate and best-effort guards for a mock run (`agenticreq mock-run`).

**These guards are not a security boundary.** A mock run executes the agent's own code. The guards catch the
mistakes an honest agent makes (a stray HTTP call, a subprocess, a file written outside a scratch area). They patch
Python-level APIs, so code that goes around the interpreter (`_socket.socket.connect`, ctypes / native extensions such
as libcurl, reading `~/.ssh`, writing through a symlink) is NOT stopped. Run a mock run on projects you trust; for
a project from someone else, use a separate machine.

What it does guarantee (no bypass flag exists for 1-3, by design):
1. A mock run refuses to start without a mocks file (mocks.yml).
2. The child process never receives real credentials: the environment is rebuilt from an
   allowlist, provider keys are set to obvious fakes, and any credential name found in a
   local .env file is pinned to a fake so load_dotenv() cannot override it.
3. A mocks file that tries to pass a live-looking credential under `env:` is refused.
4. Inside the child, the common ways to leave the run are blocked on a best-effort basis: stdlib / HTTP-client
   network connects, starting other programs, and file writes outside a scratch area (plus the folders listed under
   `allow_write:` in mocks.yml). This is a seat belt, not a wall.
"""

from __future__ import annotations

import os
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any


FAKE_PREFIX = "agenticreq-fake-"

PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
)

_CRED_NAME = re.compile(
    r"(API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|DATABASE_URL|DSN|_AUTH)", re.IGNORECASE
)
LIVE_VALUE = re.compile(
    r"^(sk-(proj|ant|live|svcacct)-|sk_live_|rk_live_|AKIA[0-9A-Z]{12}|ASIA[0-9A-Z]{12}|ghp_|gho_|github_pat_|"
    r"xox[abpr]-|AIza[0-9A-Za-z_-]{20}|glpat-|gsk_[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|xai-[A-Za-z0-9]{20}|"
    r"pplx-[A-Za-z0-9]{20}|r8_[A-Za-z0-9]{20}|csk-[A-Za-z0-9]{20}|fw_[A-Za-z0-9]{20})"
)
#: env vars passed through to the child (everything else is dropped)
_PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "SYSTEMROOT",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    # Windows: Path.home() / expanduser need these, os.getlogin etc. need the rest
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "COMSPEC",
    "PATHEXT",
    "WINDIR",
    "USERNAME",
    "ALLUSERSPROFILE",
    "PROGRAMFILES",
    "NUMBER_OF_PROCESSORS",
)


class MockRunRefused(Exception):
    """The mock run must not start. Deliberately not overridable."""


#: printed when `agenticreq mock-run` finds no mocks file
NO_MOCKS = "mock-run needs a mocks file: run `agenticreq mocks` to create mocks.yml"


def is_credential_name(name: str) -> bool:
    return bool(_CRED_NAME.search(name))


def dotenv_credential_names(root: Path) -> list[str]:
    names: list[str] = []
    for p in [root / ".env", *root.glob("*.env"), root.parent / ".env"]:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and is_credential_name(m.group(1)):
                names.append(m.group(1))
    return sorted(set(names))


def _provider_env() -> dict[str, Any]:
    from agenticreq.data import load_yaml  # parent process only

    return load_yaml("providers.yml")  # type: ignore[no-any-return]


def build_child_env(
    parent: dict[str, str], agent_root: Path, extra: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Return (env, scrubbed_names). The child env contains no real credential. ``extra``: settings the user
    passes on purpose (mocks.yml ``env:``), checked like everything else for live-looking credentials."""
    env = {k: parent[k] for k in _PASSTHROUGH if k in parent}
    scrubbed = sorted(k for k in parent if is_credential_name(k))
    for k in PROVIDER_KEYS:
        env[k] = FAKE_PREFIX + k.lower()
    for k in dotenv_credential_names(agent_root):
        env[k] = FAKE_PREFIX + k.lower()
    table = _provider_env()
    for k in table.get("credential_env") or []:
        env[k] = FAKE_PREFIX + k.lower()
    for k, v in (table.get("env_defaults") or {}).items():
        env.setdefault(k, str(v))
    live = sorted(k for k, v in (extra or {}).items() if LIVE_VALUE.match(v))
    if live:
        raise MockRunRefused(f"mocks.yml env {live} look like live credentials; a mock run never starts with them")
    env.update(extra or {})
    env["AGENTICREQ_MOCK_RUN"] = "1"
    # point SDKs at a dead local address: anything the hook misses fails instead of leaving the box
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", f"http://127.0.0.1:{DEAD_LLM_PORT}/v1")
    env["ANTHROPIC_BASE_URL"] = env.get("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{DEAD_LLM_PORT}")
    for k, v in env.items():
        if is_credential_name(k) and LIVE_VALUE.match(v):
            raise MockRunRefused(f"refusing: {k} would reach the agent with a live-looking credential")
    return env, scrubbed


def norm_path(path: str | Path) -> str:
    """One spelling of a path, for comparing: / and \\ alike, `..` and symlinks resolved, and on Windows the case
    folded. A file reached through ../ or a symlink is judged where it really is."""
    text = str(path)
    if os.sep == "/":
        text = text.replace("\\", "/")
    return os.path.normcase(os.path.realpath(os.path.abspath(text)))


def under(path: str | Path, roots: list[str]) -> bool:
    """``path`` is one of ``roots`` or inside one (an allowed folder includes everything under it)."""
    p = norm_path(path)
    return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def gate(mock_tools: str | Path | None) -> Any:
    """The mock-run safety gate: no mocks file, no run. Raises MockRunRefused; there is no flag to skip it."""
    if not mock_tools or not Path(mock_tools).is_file():
        raise MockRunRefused(NO_MOCKS)
    import yaml  # parent process only: the agent's environment does not need pyyaml

    from agenticreq.capture.responder import MockConfig

    try:
        return MockConfig.load(mock_tools)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise MockRunRefused(f"{mock_tools}: {str(e).splitlines()[0]}") from None


# ---------------------------------------------------------------- in-child socket guard
#: SDK base URLs in the child point here; a connect attempt means the HTTP hook missed an LLM call
DEAD_LLM_PORT = 9

LLM_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "openai.azure.com",
    "bedrock-runtime",
    "api.mistral.ai",
    "api.groq.com",
)


def _is_ip(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    return True


class SocketGuard:
    """Blocks the outbound INET connects that go through Python's `socket` module (best-effort, not a security
    boundary: `_socket.socket.connect` and native code are not covered); records LLM-host misses.

    Works the same on Linux, macOS and Windows, for sync and async agents:

    * ``socket.connect`` / ``connect_ex`` — every blocking and selector-loop connect (Linux, macOS,
      Windows SelectorEventLoop, ``socket.create_connection``, requests, urllib, httpx).
    * Windows ``IocpProactor.connect`` — the default Windows event loop connects through ``ConnectEx``,
      which never calls ``socket.connect``; without this an async agent bypassed the guard.
    * ``socket.socketpair`` is exempt: an event loop's internal wake-up pipe. On Linux/macOS it is an
      AF_UNIX pair, but on Windows it is a loopback TCP connection — blocking it made every
      ``asyncio.run()`` agent fail before its first LLM call.
    * uvloop (common on Linux servers) does its own connect + DNS in C, out of reach of any Python
      guard, so inside a mock run it is replaced by the standard asyncio loop.
    """

    def __init__(self, allow_hosts: list[str], llm_hosts: tuple[str, ...] = LLM_HOSTS) -> None:
        self.allow_hosts = allow_hosts
        self.llm_hosts = llm_hosts
        self.blocked: list[str] = []
        self.llm_misses: list[str] = []
        self.notes: list[str] = []
        self._restore: list[tuple[Any, str, Any]] = []  # (owner, attribute, original)
        self._internal = threading.local()

    # -- the one decision every connect path goes through
    def check(self, family: int, address: Any) -> None:
        if getattr(self._internal, "active", False) or family not in (socket.AF_INET, socket.AF_INET6):
            return
        host, port = str(address[0]), address[1]
        if port == DEAD_LLM_PORT:
            self.llm_misses.append(f"{host}:{port}")
        if not self._allowed(host):
            self.blocked.append(f"connect:{host}:{port}")
            raise ConnectionRefusedError(f"agenticreq mock run: connect to {host} blocked")

    def _patch(self, owner: Any, attr: str, new: Any) -> None:
        self._restore.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, new)

    def install(self) -> None:
        guard = self
        orig_connect, orig_connect_ex = socket.socket.connect, socket.socket.connect_ex
        orig_gai, orig_pair = socket.getaddrinfo, socket.socketpair

        def getaddrinfo(host: Any, *a: Any, **k: Any) -> Any:
            h = host.decode() if isinstance(host, bytes) else str(host)
            if any(x in h for x in guard.llm_hosts):
                guard.llm_misses.append(h)
            if h not in ("localhost", "None") and not _is_ip(h) and not guard._allowed(h):
                # (an IP literal needs no DNS; connect() decides on it)
                guard.blocked.append(f"dns:{h}")
                raise socket.gaierror(f"agenticreq mock run: DNS for {h!r} blocked")
            return orig_gai(host, *a, **k)

        def connect(sock: socket.socket, address: Any) -> Any:
            guard.check(sock.family, address)
            return orig_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> Any:
            guard.check(sock.family, address)
            return orig_connect_ex(sock, address)

        def socketpair(*a: Any, **k: Any) -> Any:
            guard._internal.active = True  # the loop's own wake-up pipe, never egress
            try:
                return orig_pair(*a, **k)
            finally:
                guard._internal.active = False

        self._patch(socket, "getaddrinfo", getaddrinfo)
        self._patch(socket.socket, "connect", connect)
        self._patch(socket.socket, "connect_ex", connect_ex)
        self._patch(socket, "socketpair", socketpair)
        if sys.platform == "win32":
            self._guard_proactor()
        self._disable_uvloop()

    def _guard_proactor(self) -> None:
        if sys.platform != "win32":  # also tells the type checker the block below is Windows-only
            return
        from asyncio import windows_events

        guard = self
        orig = windows_events.IocpProactor.connect

        def proactor_connect(proactor: Any, conn: socket.socket, address: Any) -> Any:
            guard.check(conn.family, address)
            return orig(proactor, conn, address)

        self._patch(windows_events.IocpProactor, "connect", proactor_connect)

    def _disable_uvloop(self) -> None:
        import asyncio
        import importlib.util

        guard = self
        orig_set_policy = getattr(asyncio, "set_event_loop_policy", None)  # policies are removed in Python 3.16

        def set_event_loop_policy(policy: Any) -> None:
            if policy is not None and type(policy).__module__.startswith("uvloop"):
                guard.notes.append("uvloop policy replaced by asyncio's default (mock run)")
                return
            if orig_set_policy is not None:
                orig_set_policy(policy)

        if orig_set_policy is not None:
            self._patch(asyncio, "set_event_loop_policy", set_event_loop_policy)
        if importlib.util.find_spec("uvloop") is None:
            return
        uvloop = importlib.import_module("uvloop")  # optional dependency of the agent, not of agenticreq

        def install() -> None:
            guard.notes.append("uvloop.install() ignored (mock run)")

        def run(main: Any, **_kw: Any) -> Any:  # uvloop.run(): the loop_factory way, with no policy at all
            guard.notes.append("uvloop.run() ran on asyncio's default loop (mock run)")
            return asyncio.run(main)

        if hasattr(uvloop, "install"):
            self._patch(uvloop, "install", install)
        if hasattr(uvloop, "run"):
            self._patch(uvloop, "run", run)
        self._patch(uvloop, "new_event_loop", asyncio.new_event_loop)
        if hasattr(uvloop, "run"):
            self._patch(uvloop, "run", asyncio.run)

    def _allowed(self, host: str) -> bool:
        return any(host == h or host.endswith("." + h) for h in self.allow_hosts)

    def uninstall(self) -> None:
        for owner, attr, original in reversed(self._restore):
            setattr(owner, attr, original)
        self._restore.clear()


# ---------------------------------------------------------------- in-child process / file guard
_EXEC_EVENTS = frozenset(
    {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.startfile"}
)
#: audit events whose first argument is a path that would be created, changed or removed
_PATH_EVENTS = frozenset(
    {
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.mkdir",
        "os.truncate",
        "os.symlink",
        "os.link",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)
_DEVICES = frozenset({"nul", "conout$", "conin$", "con"})
#: refused writes to one folder before the guard stops answering with PermissionError: on Windows tempfile treats a
#: PermissionError as "that name is taken" and tries again, up to 10,000 times
RETRY_CAP = 25


class WriteLoopStopped(OSError):
    """The agent keeps writing to a blocked folder. Not a PermissionError, so no retry loop swallows it."""


class ProcessGuard:
    """Blocks what the network guard cannot see: starting other programs and changing files outside a scratch area.

    Built on Python audit hooks (PEP 578), so it behaves the same on Windows, macOS and Linux:

    * starting a program (subprocess, os.system, exec/spawn/fork) is refused;
    * opening a file for writing, or creating / deleting / renaming one, is refused unless it is under the
      temp folder, the usual cache / config folders, or a folder listed under ``allow_write:`` in mocks.yml.

    Honest limit: this stops Python-level actions. Native code that bypasses the interpreter (a C extension
    calling the OS directly) is out of its reach: run a mock run on projects you trust.
    """

    def __init__(
        self, write_roots: list[str] | None = None, *, cache_dirs: bool = True, project: str | None = None
    ) -> None:
        import tempfile

        home = os.path.expanduser("~")
        # temp, the places libraries keep caches / config (never the whole home folder, which holds the project),
        # and whatever mocks.yml lists under allow_write:
        caches = [os.path.join(home, d) for d in (".cache", ".local", ".config", "AppData")] if cache_dirs else []
        self.general = [self._norm(r) for r in [tempfile.gettempdir(), *caches] if r]
        self.listed = [self._norm(r) for r in (write_roots or []) if r]
        self.roots = [*self.general, *self.listed]
        #: the project folder: a project kept under the temp folder is still only writable where allow_write: says
        self.project = self._norm(project) if project else None
        self.blocked: list[str] = []
        self.active = False
        #: blocked file -> the scratch file that took its writes, so the run goes on and can read back what it wrote
        self.diverted: dict[str, str] = {}
        self._denied: dict[str, int] = {}  # folder -> refused writes
        self._restore: list[tuple[Any, str, Any]] = []

    @staticmethod
    def _norm(path: str) -> str:
        return norm_path(path)

    def _under_roots(self, path: str) -> bool:
        p = self._norm(path)
        base = os.path.basename(p)
        if base in _DEVICES or p == self._norm(os.devnull) or (os.name == "posix" and p.startswith("/dev/")):
            return True
        if self.project is not None and under(p, [self.project]):
            return under(p, self.listed)
        return under(p, self.roots)

    def _leads_to_allowed(self, path: str) -> bool:
        """``path`` is a folder above an allowed one (only creating it is allowed: it holds nothing yet)."""
        p = self._norm(path).rstrip(os.sep)
        return any(r.startswith(p + os.sep) for r in self.roots)

    @staticmethod
    def _writes(mode: Any, flags: Any) -> bool:
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return True
        write_bits = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        return isinstance(flags, int) and bool(flags & write_bits)

    def _note(self, what: str) -> None:
        if what not in self.blocked:
            self.blocked.append(what)

    def _deny(self, what: str, event: str) -> None:
        folder = os.path.dirname(what.removeprefix("write:"))
        self._denied[folder] = self._denied.get(folder, 0) + 1
        if self._denied[folder] > RETRY_CAP:
            raise WriteLoopStopped(
                f"agenticreq mock run: {RETRY_CAP} writes to {folder or 'the project folder'} were blocked and the "
                f"agent keeps trying (last: {what.removeprefix('write:')}). To allow it, list it under `allow_write:` "
                "in mocks.yml."
            )
        self._note(what)
        raise PermissionError(
            f"agenticreq mock run: {event} blocked ({what}). To allow a folder for writing, list it "
            "under `allow_write:` in mocks.yml."
        )

    def divert_writes(self, scratch: str) -> None:
        """Files opened for writing outside the allowed paths go to a scratch file instead, and the path is listed
        as blocked: the agent's real target is untouched and the run goes on. Whatever goes around ``open()``
        (``os.open``, native code) still meets the audit hook."""
        import builtins
        import io
        import pathlib

        guard, real_open, real_mkdir = self, builtins.open, os.mkdir

        def diverted(path: str, mode: str) -> str:
            key = guard._norm(path)
            if key not in guard.diverted:
                os.makedirs(scratch, exist_ok=True)
                guard.diverted[key] = os.path.join(scratch, f"{len(guard.diverted)}-{os.path.basename(key) or 'file'}")
                if "a" in mode or "+" in mode:
                    real_open(guard.diverted[key], "ab").close()
            return guard.diverted[key]

        def checked_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
            if guard.active and isinstance(file, (str, bytes, os.PathLike)):
                path = os.fsdecode(file)
                if guard._writes(mode, None) and not guard._under_roots(path):
                    guard._note(f"write:{path}")
                    return real_open(diverted(path, mode), mode.replace("x", "w"), *a, **k)
                if guard._norm(path) in guard.diverted:
                    return real_open(guard.diverted[guard._norm(path)], mode, *a, **k)
            return real_open(file, mode, *a, **k)

        def checked_mkdir(path: Any, mode: int = 0o777, *a: Any, **k: Any) -> None:
            if guard.active and isinstance(path, (str, bytes, os.PathLike)):
                p = os.fsdecode(path)
                if not guard._under_roots(p) and not guard._leads_to_allowed(p):
                    guard._note(f"write:{p}")
                    return None  # the files written under it are diverted too
            return real_mkdir(path, mode, *a, **k)

        patches: list[tuple[Any, str, Any]] = [
            (builtins, "open", checked_open),
            (io, "open", checked_open),
            (os, "mkdir", checked_mkdir),
        ]
        accessor = getattr(pathlib, "_NormalAccessor", None)  # Python 3.10 keeps its own reference to both
        if accessor is not None:
            patches += [
                (accessor, "open", staticmethod(checked_open)),
                (accessor, "mkdir", staticmethod(checked_mkdir)),
            ]
        for owner, attr, new in patches:
            self._restore.append((owner, attr, getattr(owner, attr)))
            setattr(owner, attr, new)

    def _hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event in _EXEC_EVENTS:
            first = args[0] if args else ""
            self._deny(f"exec:{os.fsdecode(first) if isinstance(first, (str, bytes, os.PathLike)) else event}", event)
        elif event == "open" and len(args) >= 3 and isinstance(args[0], (str, bytes, os.PathLike)):
            if self._writes(args[1], args[2]):
                path = os.fsdecode(args[0])
                if not self._under_roots(path):
                    self._deny(f"write:{path}", event)
        elif (
            event == "os.mkdir"
            and args
            and isinstance(args[0], (str, bytes, os.PathLike))
            and self._leads_to_allowed(os.fsdecode(args[0]))
        ):
            return  # a parent folder on the way to an allowed one: mkdir(parents=True) for data/mock_run makes data
        elif event in _PATH_EVENTS:
            for a in args[:2]:  # source and destination of rename / move / link
                if isinstance(a, (str, bytes, os.PathLike)) and not self._under_roots(os.fsdecode(a)):
                    self._deny(f"write:{os.fsdecode(a)}", event)

    def install(self) -> None:
        if not getattr(ProcessGuard, "_hooked", False):  # an audit hook can never be removed: install it once
            sys.addaudithook(ProcessGuard._dispatch)
            ProcessGuard._hooked = True  # type: ignore[attr-defined]
        ProcessGuard._current = self  # type: ignore[attr-defined]
        self.active = True

    @staticmethod
    def _dispatch(event: str, args: tuple[Any, ...]) -> None:
        cur = getattr(ProcessGuard, "_current", None)
        if cur is not None:
            cur._hook(event, args)

    def uninstall(self) -> None:
        self.active = False
        for owner, attr, original in reversed(self._restore):
            setattr(owner, attr, original)
        self._restore.clear()
        if getattr(ProcessGuard, "_current", None) is self:
            ProcessGuard._current = None  # type: ignore[attr-defined]


def environ_snapshot() -> dict[str, str]:
    return dict(os.environ)
