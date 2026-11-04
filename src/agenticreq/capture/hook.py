"""S2 capture: hook ``Client.send`` / ``AsyncClient.send`` on BOTH httpx and httpx2.

Spike results this design follows:
* S1 (SDK method patching) saw 71% of calls — it misses beta.messages.create,
  responses.create and with_raw_response. Not used.
* httpx alone saw 54%: openai>=3 / anthropic>=1 moved to httpx2. Both are hooked.
* Patching the class method (not instances) means clients created before the hook
  was installed are still covered.

The user's code is never modified — only in-memory class attributes, and only in
the guarded mock-run subprocess.
"""

from __future__ import annotations

import importlib
import io
import json
import time
from types import ModuleType
from typing import Any, Callable

from agenticreq.capture.responder import CannedResponder
from agenticreq.capture.wire import detect_provider, parse_request, parse_response_tool_calls, provider_table
from agenticreq.models.observed import CapturedCall

HOOKED_LIBS = ("httpx", "httpx2")
#: other HTTP clients agents reach a model with; each is hooked when it is installed
OTHER_LIBS = ("requests", "aiohttp")


class EgressBlocked(RuntimeError):
    pass


def trace_header(headers: Any) -> str | None:
    """The trace id a model request carries: the id part of a W3C ``traceparent``, or a plain id header."""
    if not headers:
        return None
    for key in provider_table().get("trace_headers") or ():
        value = headers.get(key)
        if value:
            parts = str(value).split("-")
            return parts[1] if key == "traceparent" and len(parts) >= 3 else str(value)
    return None


class _AioContent:
    """The part of aiohttp's StreamReader that SDKs read a reply through."""

    def __init__(self, data: bytes) -> None:
        self._lines = data.splitlines(keepends=True)
        self._data = data

    def __aiter__(self) -> Any:
        return self._iter()

    async def _iter(self) -> Any:
        for line in self._lines:
            yield line

    async def read(self, _n: int = -1) -> bytes:
        data, self._data, self._lines = self._data, b"", []
        return data

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def iter_any(self) -> Any:
        return self._iter()

    def iter_chunked(self, _n: int) -> Any:
        return self._iter()


class _AioResponse:
    """A canned reply with the surface of ``aiohttp.ClientResponse``."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes, url: str) -> None:
        self.status, self.headers, self._body, self.url = status, headers, body, url
        self.reason = "OK" if status == 200 else "Bad Request"
        self.ok = status < 400
        self.content = _AioContent(body)

    async def read(self) -> bytes:
        return self._body

    async def text(self, encoding: str = "utf-8") -> str:
        return self._body.decode(encoding)

    async def json(self, **_kw: Any) -> Any:
        return json.loads(self._body)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"agenticreq mock run: HTTP {self.status}")

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def __aenter__(self) -> _AioResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class CaptureSession:
    def __init__(
        self,
        responder: CannedResponder | None,
        run_id: str = "run-0",
        egress_allow: list[str] | None = None,
        passthrough: bool = False,
    ) -> None:
        self.responder = responder
        self.run_id = run_id
        self.egress_allow = egress_allow or []
        self.passthrough = passthrough  # record only, forward real requests (non-discover use)
        self.calls: list[CapturedCall] = []
        self.blocked: list[str] = []
        self.libs_hooked: list[str] = []
        #: problems inside the hook itself (a request body or response that could not be read); reported, not dropped
        self.errors: list[str] = []
        #: requests the mock model could not answer (a request shape it does not support): the first one is
        #: usually the real cause of a failed run, whatever the SDK raised after it
        self.rejections: list[str] = []
        self._originals: list[tuple[type, str, Callable[..., Any]]] = []

    # ------------------------------------------------------------ core
    def _body(self, request: Any) -> dict[str, Any] | None:
        try:
            raw = request.content
        except Exception:
            try:
                raw = request.read()
            except Exception as e:
                self.errors.append(f"request body could not be read ({type(e).__name__}); that call was not captured")
                return None
        if not raw:
            return None
        try:
            v = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return None
        return v if isinstance(v, dict) else None

    def handle(self, lib: ModuleType, libname: str, request: Any) -> Any | None:
        """Return a lib.Response to short-circuit, or None to forward."""
        url = str(request.url)
        method = str(request.method).upper()
        body = self._body(request) if method == "POST" else None
        provider = detect_provider(url, body) if body is not None else None
        if provider is None:
            host = request.url.host or ""
            if self.passthrough or any(host == h or host.endswith("." + h) for h in self.egress_allow):
                return None
            self.blocked.append(f"{method} {url}")
            raise lib.ConnectError(f"agenticreq mock run: egress to {host!r} blocked", request=request)

        answer = self.answer(url, body or {}, libname, trace_header(getattr(request, "headers", None)))
        if answer is None:
            return None
        status, headers, content = answer
        return lib.Response(status, headers=headers, content=content, request=request)

    def answer(
        self, url: str, body: dict[str, Any], libname: str, trace_id: str | None = None
    ) -> tuple[int, dict[str, str], bytes] | None:
        """Record one model request and build the canned reply, the same for every HTTP client."""
        t0 = time.perf_counter()
        call = parse_request(len(self.calls), url, body, run_id=self.run_id, client_lib=libname)
        call.timestamp = time.time()
        call.trace_id = trace_id
        call.raw_request = body
        call.overhead_ms = (time.perf_counter() - t0) * 1000.0
        self.calls.append(call)

        if self.passthrough or self.responder is None:
            return None
        try:
            tcs = self.responder.plan(call, body)
            call.tool_calls = tcs
            call.synthetic = True
            stream = bool(body.get("stream")) or "streamGenerateContent" in url
            content, ctype = self.responder.render(call, tcs, stream=stream)
        except Exception as e:  # noqa: BLE001 - answer 400 with the reason, so the run fails where it is clear
            why = f"the mock model could not answer the {call.provider} request ({type(e).__name__}: {e})"
            self.rejections.append(why)
            return (
                400,
                {"content-type": "application/json", "x-agenticreq": "rejected"},
                json.dumps({"error": {"type": "invalid_request_error", "message": why}}).encode(),
            )
        return 200, {"content-type": ctype, "x-agenticreq": "canned"}, content

    def _llm_body(self, url: str, method: str, raw: Any) -> dict[str, Any] | None:
        """The JSON body of a model request, else None (the socket guard decides what happens to it)."""
        if method.upper() != "POST" or raw is None:
            return None
        try:
            body = raw if isinstance(raw, dict) else json.loads(raw)
        except (ValueError, UnicodeDecodeError, TypeError):
            return None
        return body if isinstance(body, dict) and detect_provider(url, body) else None

    def record_response(self, call: CapturedCall, response_json: dict[str, Any]) -> None:
        call.tool_calls = parse_response_tool_calls(call.provider, response_json)

    # ------------------------------------------------------------ install / uninstall
    def install(self) -> list[str]:
        for libname in HOOKED_LIBS:
            try:
                lib = importlib.import_module(libname)
            except ImportError:
                continue
            self._patch(lib, libname)
            self.libs_hooked.append(libname)
        for libname in OTHER_LIBS:
            try:
                lib = importlib.import_module(libname)
            except ImportError:
                continue
            (self._patch_requests if libname == "requests" else self._patch_aiohttp)(lib)
            self.libs_hooked.append(libname)
        return self.libs_hooked

    def _patch_requests(self, lib: ModuleType) -> None:
        session = self
        orig: Callable[..., Any] = lib.Session.send

        def send(self_: Any, prepared: Any, **kwargs: Any) -> Any:
            body = session._llm_body(str(prepared.url), str(prepared.method), prepared.body)
            headers = trace_header(prepared.headers)
            answer = session.answer(str(prepared.url), body, "requests", headers) if body is not None else None
            if answer is None:
                return orig(self_, prepared, **kwargs)
            resp = lib.models.Response()
            resp.status_code, resp._content, resp.url, resp.request = answer[0], answer[2], prepared.url, prepared
            resp.headers.update(answer[1])
            resp.raw = io.BytesIO(answer[2])
            return resp

        self._originals.append((lib.Session, "send", orig))
        lib.Session.send = send

    def _patch_aiohttp(self, lib: ModuleType) -> None:
        session = self
        orig: Callable[..., Any] = lib.ClientSession._request

        async def _request(self_: Any, method: str, str_or_url: Any, **kwargs: Any) -> Any:
            url = str(str_or_url)
            body = session._llm_body(url, method, kwargs.get("json") or kwargs.get("data"))
            answer = (
                session.answer(url, body, "aiohttp", trace_header(kwargs.get("headers"))) if body is not None else None
            )
            if answer is None:
                return await orig(self_, method, str_or_url, **kwargs)
            return _AioResponse(answer[0], answer[1], answer[2], url)

        self._originals.append((lib.ClientSession, "_request", orig))
        lib.ClientSession._request = _request

    def _patch(self, lib: ModuleType, libname: str) -> None:
        session = self
        sync_orig: Callable[..., Any] = lib.Client.send
        async_orig: Callable[..., Any] = lib.AsyncClient.send

        def send(self_: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            resp = session.handle(lib, libname, request)
            if resp is not None:
                return resp
            out = sync_orig(self_, request, *args, **kwargs)
            session._maybe_record_passthrough(out)
            return out

        async def asend(self_: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            resp = session.handle(lib, libname, request)
            if resp is not None:
                return resp
            out = await async_orig(self_, request, *args, **kwargs)
            session._maybe_record_passthrough(out)
            return out

        send.__agenticreq_hook__ = True  # type: ignore[attr-defined]
        asend.__agenticreq_hook__ = True  # type: ignore[attr-defined]
        self._originals += [(lib.Client, "send", sync_orig), (lib.AsyncClient, "send", async_orig)]
        lib.Client.send = send
        lib.AsyncClient.send = asend

    def _maybe_record_passthrough(self, response: Any) -> None:
        if not self.passthrough or not self.calls:
            return
        try:
            ctype = response.headers.get("content-type", "")
            if "json" in ctype and not getattr(response, "is_stream_consumed", False):
                response.read()
                self.record_response(self.calls[-1], response.json())
        except Exception as e:
            self.errors.append(f"response could not be recorded ({type(e).__name__}); its tool calls are missing")

    def uninstall(self) -> None:
        for cls, attr, orig in reversed(self._originals):
            setattr(cls, attr, orig)
        self._originals.clear()

    def __enter__(self) -> CaptureSession:
        self.install()
        return self

    def __exit__(self, *exc: object) -> None:
        self.uninstall()
