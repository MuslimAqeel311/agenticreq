"""Optional per-framework recognizers (~50 lines each). Used only for graph reachability.

Plugin contract (versioned so community plugins don't break silently):

    class MyRecognizer:
        api_version = 1                 # must equal RECOGNIZER_API_VERSION's major
        framework = "myframework"       # matches ObservedAgent.framework / frameworks
        def recognize(self, obs: ObservedAgent, root: Path) -> None: ...

Register via the entry-point group ``agenticreq.recognizers``.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

from agenticreq.models.observed import ExtractionError, ObservedAgent

RECOGNIZER_API_VERSION = 1


@runtime_checkable
class Recognizer(Protocol):
    api_version: int
    framework: str

    def recognize(self, obs: ObservedAgent, root: Path) -> None: ...


def _builtin() -> list[Recognizer]:
    from agenticreq.recognizers.langgraph import LangGraphRecognizer

    return [LangGraphRecognizer()]


def load_recognizers() -> list[Recognizer]:
    found: dict[str, Recognizer] = {r.framework: r for r in _builtin()}
    for ep in importlib.metadata.entry_points(group="agenticreq.recognizers"):
        try:
            obj = ep.load()()
        except Exception as e:  # noqa: BLE001  (a broken plug-in must not stop the run, but must not vanish either)
            warnings.warn(
                f"agenticreq: framework plug-in {ep.name!r} could not be loaded and was skipped "
                f"({type(e).__name__}: {e}); its framework will show as not recognised",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if isinstance(obj, Recognizer) and obj.api_version == RECOGNIZER_API_VERSION:
            found.setdefault(obj.framework, obj)
    return list(found.values())


def run_recognizers(obs: ObservedAgent, root: Path) -> None:
    for r in load_recognizers():
        if r.framework in obs.frameworks:
            try:
                r.recognize(obs, root)
            except Exception as e:  # a broken plugin must not break the check
                obs.extraction_errors.append(ExtractionError(file=str(root), message=f"recognizer {r.framework}: {e}"))
