"""Record human-approval gates while discover runs the agent.

Each gate that fires is attached to the last captured LLM call (``CapturedCall.gates_after``), so a rule can
tell whether an approval happened between the model asking for a tool and that tool's result reaching the
model. The probes are a table: a framework's approval API is one more row, never new logic.
"""

from __future__ import annotations

import builtins
import functools
import importlib
import importlib.util
import inspect
import sys
from typing import Any, Callable

from agenticreq.capture.hook import CaptureSession

#: (module, attribute, gate kind, when). when="call": the call itself is the approval request; when="true": the
#: function decides whether approval is needed and only a True result is one. A module that is not installed, or a
#: version without the attribute, is skipped. CrewAI human_input and AutoGen UserProxyAgent ask through input(),
#: probed separately. Never probe something a framework builds for every tool call (OpenAI Agents' ToolApprovalItem
#: is also its lookup key for ungated tools): a gate recorded for an ungated tool would be a false PASS.
PROBES: tuple[tuple[str, str, str, str], ...] = (
    ("langgraph.types", "interrupt", "interrupt", "call"),  # LangGraph human-in-the-loop
    ("agents.util._approvals", "evaluate_needs_approval_setting", "needs_approval", "true"),  # OpenAI Agents SDK
)


class GateProbe:
    def __init__(self, session: CaptureSession, answer: str = "yes") -> None:
        self.session = session
        self.answer = answer  # what a human typing at an input() prompt answers
        self.fired: list[str] = []  # every gate, also those before the first LLM call
        self._undo: list[tuple[Any, str, Any]] = []

    def record(self, kind: str) -> None:
        self.fired.append(kind)
        if self.session.calls:
            self.session.calls[-1].gates_after.append(kind)

    def install(self) -> list[str]:
        installed = []
        orig_input = builtins.input
        probe = self

        def fake_input(prompt: object = "") -> str:  # a console approval: the run answers it, never blocks
            probe.record("human_input")
            return probe.answer

        self._undo.append((builtins, "input", orig_input))
        builtins.input = fake_input
        installed.append("builtins.input")
        for mod_name, attr, kind, when in PROBES:
            if self._wrap(mod_name, attr, kind, when):
                installed.append(f"{mod_name}.{attr}")
        return installed

    def _wrap(self, mod_name: str, attr: str, kind: str, when: str = "call") -> bool:
        try:
            if importlib.util.find_spec(mod_name.split(".")[0]) is None:
                return False
            owner: Any = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - a framework that fails to import is the run's problem, not the probe's
            return False
        target = getattr(owner, attr, None)
        if not callable(target) or isinstance(target, type):
            return False
        wrapper = self._recording(target, kind, when)
        # a framework that did `from x import f` holds its own reference: rebind it wherever it is bound
        for mod in [owner, *[m for m in list(sys.modules.values()) if m is not None and m is not owner]]:
            try:
                names = [n for n, v in vars(mod).items() if v is target]
            except TypeError:
                continue
            for n in names:
                self._undo.append((mod, n, target))
                setattr(mod, n, wrapper)
        return True

    def _recording(self, fn: Callable[..., Any], kind: str, when: str) -> Callable[..., Any]:
        probe = self

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*a: Any, **k: Any) -> Any:
                if when == "call":
                    probe.record(kind)
                out = await fn(*a, **k)
                if when == "true" and out:
                    probe.record(kind)
                return out

            return awrapper

        @functools.wraps(fn)
        def wrapper(*a: Any, **k: Any) -> Any:
            if when == "call":
                probe.record(kind)
            out = fn(*a, **k)
            if when == "true" and out:
                probe.record(kind)
            return out

        return wrapper

    def uninstall(self) -> None:
        for owner, name, orig in reversed(self._undo):
            setattr(owner, name, orig)
        self._undo.clear()
