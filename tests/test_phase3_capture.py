"""Phase 3 acceptance: S2 capture on the fixture agents — 100% coverage, zero-touch, gate refuses."""

import importlib.util
from pathlib import Path

import pytest

from agenticreq.capture import MockRunRefused, discover
from agenticreq.capture.hook import CaptureSession
from agenticreq.capture.responder import CannedResponder
from agenticreq.capture.guard import build_child_env

AGENTS = Path(__file__).parent / "fixtures" / "agents"
MISC = Path(__file__).parent / "fixtures" / "agents_misc"

REQUIRES = {
    "langgraph_po": "langgraph",
    "oai_agents_po": "agents",
    "smol_tools": "smolagents",
    "smol_code": "smolagents",
    "anthropic_beta": "anthropic",
    "httpx_legacy": "httpx",
    "httpx2_async_stream": "openai",
    "pydantic_ai_po": "pydantic_ai",
}


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.parametrize("name", sorted(REQUIRES))
def test_full_coverage_and_zero_touch(name: str) -> None:
    if not _have(REQUIRES[name]):
        pytest.skip(f"{REQUIRES[name]} not installed")
    d = AGENTS / name
    r = discover(str(d / "agent.py"), mock_tools=str(d / "mocks.yml"), timeout=120)
    cap, tot = r.coverage
    assert cap > 0 and cap == tot, (name, r.coverage, r.run_errors)
    assert r.zero_touch, r.changed_files
    assert not r.llm_misses


def test_both_client_libraries_hooked() -> None:
    pytest.importorskip("anthropic")  # the anthropic_beta fixture drives httpx2 through the SDK
    libs = set()
    for name in ("httpx_legacy", "anthropic_beta"):
        d = AGENTS / name
        r = discover(str(d / "agent.py"), mock_tools=str(d / "mocks.yml"))
        libs |= {c.client_lib for c in r.calls}
    assert libs == {"httpx", "httpx2"}


def test_refuses_without_mocks() -> None:
    """No mocks file, no run: the gate has no bypass (the sandbox.yml route is gone)."""
    from agenticreq.capture.guard import NO_MOCKS

    with pytest.raises(MockRunRefused, match="mock-run needs a mocks file"):
        discover(str(AGENTS / "httpx_legacy" / "agent.py"), mock_tools=None)
    with pytest.raises(MockRunRefused) as e:
        discover(str(AGENTS / "httpx_legacy" / "agent.py"), mock_tools="does-not-exist.yml")
    assert str(e.value) == NO_MOCKS


def test_mocks_with_live_credential_refused(tmp_path: Path) -> None:
    m = tmp_path / "mocks.yml"
    m.write_text("tools: {}\nenv:\n  OPENAI_API_KEY: sk-proj-abcdefghijklmnop\n")
    with pytest.raises(MockRunRefused, match="live"):
        discover(str(AGENTS / "httpx_legacy" / "agent.py"), mock_tools=str(m))


def test_child_env_never_has_real_credentials(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("STRIPE_SECRET_KEY=sk_live_123\nDB_PASSWORD=hunter2\n")
    parent = {"PATH": "/bin", "OPENAI_API_KEY": "sk-proj-REAL", "AWS_SECRET_ACCESS_KEY": "x", "GITHUB_TOKEN": "ghp_x"}
    env, scrubbed = build_child_env(parent, tmp_path)
    assert env["OPENAI_API_KEY"].startswith("agenticreq-fake-")
    assert "AWS_SECRET_ACCESS_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["STRIPE_SECRET_KEY"].startswith("agenticreq-fake-")  # pinned so load_dotenv can't override
    assert env["DB_PASSWORD"].startswith("agenticreq-fake-")
    assert set(scrubbed) >= {"OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"}


def test_no_bypass_flag_exists() -> None:
    import inspect

    params = set(inspect.signature(discover).parameters)
    assert not {p for p in params if any(w in p for w in ("force", "unsafe", "bypass", "skip", "allow_prod"))}


def test_non_httpx_client_counted_as_miss() -> None:
    d = MISC / "urllib_bypass"
    r = discover(str(d / "agent.py"), mock_tools=str(d / "mocks.yml"))
    cap, tot = r.coverage
    assert cap == 0 and tot == 1  # honest: not 100%


def test_in_process_hook_overhead() -> None:
    import statistics

    import httpx

    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
    }
    with CaptureSession(CannedResponder()) as s:
        with httpx.Client() as c:
            first = c.post("https://api.openai.com/v1/chat/completions", json=body).json()
            for _ in range(49):
                c.post("https://api.openai.com/v1/chat/completions", json=body)
        assert first["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "t"
    median = statistics.median(x.overhead_ms for x in s.calls)
    print(f"capture overhead median {median:.3f} ms over {len(s.calls)} calls")
    assert median < 2.0  # spike: 0.2–0.9 ms per call
    assert not hasattr(httpx.Client.send, "__agenticreq_hook__")  # uninstalled cleanly
