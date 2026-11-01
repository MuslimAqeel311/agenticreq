"""mock-run never fails silently: every failed or empty run says, in one sentence, what was captured, where the run
stopped (file:line), why, and the next step, and exits 1. A clean run exits 0 with no error text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture.discover import DiscoverResult, explain
from agenticreq.cli import app

runner = CliRunner()
NEXT = "then run: agenticreq mock-run"
#: one model call through httpx (the hook captures it), an OpenAI chat request
CALL = (
    "import httpx\n"
    "def _ask():\n"
    "    httpx.post('https://api.openai.com/v1/chat/completions', headers={'Authorization': 'Bearer x'},\n"
    "               json={'model': 'gpt-4o-2024-08-06', 'messages': [{'role': 'user', 'content': 'hi'}]})\n"
)


def _run(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, body: str, entry: str = "agent.py:main", *extra: str
) -> tuple[int, str]:
    (tmp / "agent.py").write_text(body, encoding="utf-8")
    (tmp / "mocks.yml").write_text(f"entry: {entry}\ntools: {{}}\n", encoding="utf-8")
    monkeypatch.chdir(tmp)
    r = runner.invoke(app, ["mock-run", "--quiet", *extra], terminal_width=250)
    return r.exit_code, " ".join(r.output.split())


def test_entry_function_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, "def main():\n    pass\n", "agent.py:nope")
    assert code == 1 and "The entry point agent.py:nope was not found" in out and "Fix entry: in mocks.yml" in out


def test_entry_module_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, "", "no_such_module:run")
    assert code == 1 and "The entry point no_such_module:run was not found" in out


def test_a_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, "import not_a_real_package_xyz\ndef main():\n    pass\n")
    assert code == 1 and "stopped at agent.py:1 with ModuleNotFoundError" in out
    assert "a dependency is missing" in out and "Install not_a_real_package_xyz there" in out and NEXT in out


def test_a_crash_before_the_model_is_called(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, "def main():\n    cfg = {}\n    return cfg['model']\n")
    assert code == 1
    assert (
        "mock-run captured 0 model calls. The run stopped at agent.py:3 with KeyError: 'model'. Fix the error, "
        "then run: agenticreq mock-run" in out
    )


def test_the_sdk_rejects_the_request_before_sending_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = tmp_path / "lib" / "site-packages" / "anthropic"  # an installed SDK that checks the request first
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text(
        "def create(temperature):\n    if temperature > 1:\n"
        "        raise ValueError(f'temperature {temperature} is not supported')\n",
        encoding="utf-8",
    )
    body = (
        "import sys\nsys.path.insert(0, r'" + str(tmp_path / "lib" / "site-packages") + "')\nimport anthropic\n"
        "def main():\n    return anthropic.create(temperature=2.5)\n"
    )
    code, out = _run(tmp_path, monkeypatch, body)
    assert code == 1
    assert (
        "mock-run captured 0 model calls. The run stopped at agent.py:5 with ValueError: temperature 2.5 is not "
        "supported: the anthropic SDK rejected the request before sending it. Fix the value or the model, then "
        "run: agenticreq mock-run"
    ) in out


def test_the_mock_model_rejects_a_request() -> None:
    import httpx

    from agenticreq.capture.hook import CaptureSession

    class Broken:
        def plan(self, call: object, body: object) -> list[object]:
            raise KeyError("unknown model 'gpt-99'")

        def render(self, *a: object, **k: object) -> tuple[bytes, str]:
            return b"", "application/json"

    s = CaptureSession(Broken(), run_id="r")  # type: ignore[arg-type]
    req = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-99", "messages": [{"role": "user", "content": "hi"}]},
    )
    resp = s.handle(httpx, "httpx", req)
    assert resp is not None and resp.status_code == 400
    assert s.rejections and "could not answer" in s.rejections[0] and "gpt-99" in s.rejections[0]
    res = DiscoverResult(entry="agent.py:main", rejections=s.rejections)
    text = explain("run-0", {"type": "BadRequestError", "message": "400", "where": "agent.py:9"}, "", 1, res, None)
    assert "because the mock model could not answer" in text and text.endswith(NEXT)


def test_an_http_client_that_is_not_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "import urllib.request\ndef main():\n    try:\n"
        "        urllib.request.urlopen('https://api.openai.com/v1/chat/completions', data=b'{}', timeout=3)\n"
        "    except Exception:\n        pass\n"
    )
    code, out = _run(tmp_path, monkeypatch, body)
    assert code == 1 and "through an HTTP client the mock run cannot capture" in out and NEXT in out


def test_a_timeout_says_the_limit_and_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        CALL + "def main():\n    _ask()\n    import time\n    time.sleep(30)\n",
        "agent.py:main",
        "--timeout",
        "8",
    )
    assert code == 1 and "before the 8s limit stopped run-0" in out and "last step: model call 1" in out
    assert "Raise --timeout" in out


def test_no_model_call_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, "CACHE = {'q': 'cached answer'}\ndef main():\n    return CACHE['q']\n")
    assert code == 1 and "returned without calling a model" in out and "Check entry: and args:" in out


def test_a_retry_that_hid_the_first_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "def main():\n    for attempt in range(3):\n        try:\n"
        "            raise ConnectionError('refused by proxy')\n        except ConnectionError:\n"
        "            if attempt == 2:\n                raise RuntimeError('gave up after 3 attempts')\n"
    )
    code, out = _run(tmp_path, monkeypatch, body)
    assert code == 1 and "with RuntimeError: gave up after 3 attempts" in out
    assert "The first error was ConnectionError: refused by proxy" in out


def test_a_partial_run_keeps_what_was_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, CALL + "def main():\n    _ask()\n    raise RuntimeError('tool failed')\n")
    assert (
        code == 1
        and "mock-run captured 1 model call, then stopped. The run stopped at agent.py:7 with RuntimeError: tool failed"
        in out
    )
    data = json.loads((tmp_path / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))
    assert len(data["calls"]) == 1  # the call before the crash is kept


def test_a_clean_run_exits_0_with_no_error_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(tmp_path, monkeypatch, CALL + "def main():\n    _ask()\n    return 'done'\n")
    assert code == 0, out
    assert "captured 1/1 LLM calls" in out
    for word in ("failed", "stopped", "Traceback", "error"):
        assert word not in out.lower() or word == "error" and "error" not in out.lower(), (word, out)
