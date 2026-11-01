"""S2 boundary capture (httpx + httpx2 client hook), canned responder, mock-run guards."""

from agenticreq.capture.discover import DiscoverError, DiscoverResult, discover, load_captures, save_captures
from agenticreq.capture.guard import MockRunRefused
from agenticreq.capture.hook import CaptureSession

__all__ = [
    "CaptureSession",
    "DiscoverError",
    "DiscoverResult",
    "MockRunRefused",
    "discover",
    "load_captures",
    "save_captures",
]
