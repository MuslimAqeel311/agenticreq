"""The discover socket guard on every OS and loop type: event loops start, egress stays blocked.

203.0.113.0/24 (TEST-NET-3) is never routable, so a connect that slipped past the guard would hang or
fail with a *different* error — the assertions check the guard's own message.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from typing import Any

import pytest

from agenticreq.capture.guard import DEAD_LLM_PORT, SocketGuard

BLOCKED = ("203.0.113.7", 80)


@pytest.fixture
def guard() -> Any:
    g = SocketGuard(allow_hosts=["api.allowed.test"])
    g.install()
    try:
        yield g
    finally:
        g.uninstall()


def test_event_loop_starts_under_the_guard(guard: SocketGuard) -> None:
    # Windows: the loop's self-pipe is a loopback TCP socketpair; it must not be blocked
    async def main() -> int:
        await asyncio.sleep(0)
        return 42

    assert asyncio.run(main()) == 42
    a, b = socket.socketpair()
    a.close()
    b.close()
    assert not guard.blocked


def test_sync_connect_blocked(guard: SocketGuard) -> None:
    with pytest.raises(ConnectionRefusedError, match="agenticreq mock run"):
        socket.create_connection(BLOCKED, timeout=1)
    s = socket.socket()
    try:
        with pytest.raises(ConnectionRefusedError, match="agenticreq mock run"):
            s.connect_ex(BLOCKED)
    finally:
        s.close()
    assert guard.blocked.count(f"connect:{BLOCKED[0]}:{BLOCKED[1]}") == 2


def test_async_connect_blocked_on_the_default_loop(guard: SocketGuard) -> None:
    # Windows' default ProactorEventLoop connects via ConnectEx, not socket.connect
    async def main() -> None:
        await asyncio.wait_for(asyncio.open_connection(*BLOCKED), timeout=5)

    with pytest.raises(ConnectionRefusedError, match="agenticreq mock run"):
        asyncio.run(main())
    assert f"connect:{BLOCKED[0]}:{BLOCKED[1]}" in guard.blocked


@pytest.mark.skipif(sys.platform != "win32", reason="the selector loop is already the default elsewhere")
def test_async_connect_blocked_on_windows_selector_loop(guard: SocketGuard) -> None:
    async def main() -> None:
        await asyncio.wait_for(asyncio.open_connection(*BLOCKED), timeout=5)

    loop = asyncio.SelectorEventLoop()
    try:
        with pytest.raises(ConnectionRefusedError, match="agenticreq mock run"):
            loop.run_until_complete(main())
    finally:
        loop.close()


def test_dns_blocked_and_llm_miss_recorded(guard: SocketGuard) -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.org", 443)
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", DEAD_LLM_PORT), timeout=1)
    assert "dns:example.org" in guard.blocked
    assert f"127.0.0.1:{DEAD_LLM_PORT}" in guard.llm_misses


def test_allowlist_is_honoured(guard: SocketGuard) -> None:
    guard.check(socket.AF_INET, ("api.allowed.test", 443))
    guard.check(socket.AF_INET, ("eu.api.allowed.test", 443))
    with pytest.raises(ConnectionRefusedError):
        guard.check(socket.AF_INET, ("allowed.test.evil.com", 443))


def test_uvloop_policy_is_replaced(guard: SocketGuard) -> None:
    fake = type("EventLoopPolicy", (asyncio.DefaultEventLoopPolicy,), {"__module__": "uvloop"})()
    asyncio.set_event_loop_policy(fake)
    assert not type(asyncio.get_event_loop_policy()).__module__.startswith("uvloop")
    assert any("uvloop" in n for n in guard.notes)


def test_uninstall_restores_everything() -> None:
    before = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
        socket.socketpair,
        asyncio.set_event_loop_policy,
    )
    g = SocketGuard([])
    g.install()
    g.uninstall()
    after = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
        socket.socketpair,
        asyncio.set_event_loop_policy,
    )
    assert before == after
