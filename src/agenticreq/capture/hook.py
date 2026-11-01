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
import json
import time
from types import ModuleType
from typing import Any, Callable

from agenticreq.capture.responder import CannedResponder
from agenticreq.capture.wire import detect_provider, parse_request, parse_response_tool_calls
from agenticreq.models.observed import CapturedCall

HOOKED_LIBS = ("httpx", "httpx2")


class EgressBlocked(RuntimeError):
    pass


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

        t0 = time.perf_counter()
        call = parse_request(len(self.calls), url, body or {}, run_id=self.run_id, client_lib=libname)
        call.timestamp = time.time()
        call.raw_request = body
        call.overhead_ms = (time.perf_counter() - t0) * 1000.0
        self.calls.append(call)

        if self.passthrough or self.responder is None:
            return None
        try:
            tcs = self.responder.plan(call, body or {})
            call.tool_calls = tcs
            call.synthetic = True
            content, ctype = self.responder.render(call, tcs, stream=bool((body or {}).get("stream")))
        except Exception as e:  # noqa: BLE001 - answer 400 with the reason, so the run fails where it is clear
            why = f"the mock model could not answer the {call.provider} request ({type(e).__name__}: {e})"
            self.rejections.append(why)
            return lib.Response(
                400,
                headers={"content-type": "application/json", "x-agenticreq": "rejected"},
                content=json.dumps({"error": {"type": "invalid_request_error", "message": why}}).encode(),
                request=request,
            )
        return lib.Response(
            200, headers={"content-type": ctype, "x-agenticreq": "canned"}, content=content, request=request
        )

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
        return self.libs_hooked

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
