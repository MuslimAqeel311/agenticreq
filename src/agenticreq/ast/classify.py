"""Name/shape heuristics. Everything here is `heuristic` confidence, never `proven`."""

from __future__ import annotations

import re
from typing import Any

from agenticreq.models.observed import ToolSchema


def _vocab() -> dict[str, Any]:
    """The action vocabulary (data/action_verbs.yml): extend it for your domain, no code change needed."""
    from agenticreq.data import load_yaml

    return load_yaml("action_verbs.yml") or {}


_VOCAB = _vocab()
_WRITE_VERBS = tuple(str(w) for w in _VOCAB.get("write") or ())
#: writes that are irreversible or leave the system: money, deletion, messages to people, bookings,
#: permissions, deploys. Zero-config audits ask for an approval gate only on these (AS-212).
_HIGH_IMPACT_VERBS = {str(w) for w in _VOCAB.get("high_impact") or ()}
BOUND_PARAMS = _BOUND_PARAMS = {
    "limit",
    "max_results",
    "top_k",
    "k",
    "n",
    "page_size",
    "max_items",
    "count",
    "max_rows",
    "num_results",
    "max_count",
    "per_page",
    "first",
    "last",
}
# `export_*` is not here: exporting moves data elsewhere (a write), it does not return it
_UNBOUNDED_PREFIX = ("search", "list", "get_all", "query", "fetch_all", "find", "scan", "read_all", "dump", "select")
_UNTRUSTED_HINT = re.compile(
    r"(email|inbox|mail|web|browse|url|scrape|crawl|http|rss|slack|ticket|comment|review|document|retriev|"
    r"read_file|upload|chat_history|message|search_internet|search_web|webpage|pdf)",
    re.IGNORECASE,
)
_IDEMPOTENCY = {
    "idempotency_key",
    "idempotencykey",
    "request_id",
    "dedupe_key",
    "dedup_key",
    "client_token",
    "client_reference_id",
    "operation_id",
}

HANDOFF_PREFIXES = ("transfer_to_", "handoff_to_")


def canonical(name: str) -> str:
    """Normalise names so `Delegate work to coworker` == `delegate_work_to_coworker` (names.norm: the one
    normalization every name comparison uses)."""
    from agenticreq.names import norm

    return norm(name)


def near_miss(name: str, candidates: set[str] | list[str]) -> str | None:
    """A candidate that is probably the same tool spelled differently: `pay_invoice` vs `payInvoice`,
    `pay_invoices`, `invoice_pay`, `pay_invioce`. None when the name exists exactly or nothing is close."""
    import difflib

    c = canonical(name)
    squash = c.replace("_", "")
    tokens = set(c.split("_"))
    best: tuple[float, str] | None = None
    for cand in candidates:
        k = canonical(cand)
        if k == c:
            return None  # it exists: not a near miss
        ks = k.replace("_", "")
        if ks == squash or set(k.split("_")) == tokens or ks.rstrip("s") == squash.rstrip("s"):
            score = 1.0
        else:
            score = difflib.SequenceMatcher(None, squash, ks).ratio() if min(len(squash), len(ks)) >= 6 else 0.0
        if score >= 0.88 and (best is None or score > best[0]):
            best = (score, cand)
    return best[1] if best else None


def is_handoff_tool(name: str) -> bool:
    c = canonical(name)
    # OpenAI Agents / AutoGen Swarm transfer_to_<x>, ADK transfer_to_agent(agent_name=), LlamaIndex
    # AgentWorkflow handoff(to_agent=), CrewAI delegation tools
    return c.startswith(HANDOFF_PREFIXES) or c in ("delegate_work_to_coworker", "ask_question_to_coworker", "handoff")


_READ_VERBS = {str(w) for w in _VOCAB.get("read") or ()}
#: read words that are also nouns (review, audit, plan, test ...): a verb at the start of a name (`review_pull_request`),
#: just an object later on (`post_review_comment` posts; it does not review)
_READ_NOUNISH = {str(w) for w in _VOCAB.get("read_noun_capable") or ()}


def action_verb(name: str) -> str:
    """The first word of a tool name that is a known verb: `reddit_create_post` -> create,
    `gmail_get_message` -> get. Service prefixes are skipped; with no known verb, the first word."""
    words = canonical(name).split("_")
    verbs = [
        i
        for i, w in enumerate(words)
        if w in _WRITE_VERBS or (w in _READ_VERBS and not (0 < i < len(words) - 1 and w in _READ_NOUNISH))
    ]
    if not verbs:
        return words[0]
    if "and" in words:  # `search_and_replace`, `read_and_delete`: the write after "and" is what it does
        tail = [w for w in words[words.index("and") + 1 :] if w in _WRITE_VERBS]
        if tail:
            return tail[0]
    if verbs[:2] == [0, 1] and (
        len(words) >= 3 or words[1] in _READ_NOUNISH
    ):  # `close_create_lead`, `post_get_comments`: the first word is the service
        # ... unless it is a read verb: `read_discharge_summary`, `get_refill_status` read a noun that looks like a verb
        return words[0] if words[0] in _READ_VERBS and words[0] not in _WRITE_VERBS else words[1]
    return words[verbs[0]]


def verb_class(name: str) -> str:
    """'write' | 'read' | 'unknown': what the tool name's action verb says. 'unknown' = the verb is in neither list,
    so the name alone cannot tell (callers must say UNKNOWN, never assume read)."""
    v = action_verb(canonical(name))
    if v in _WRITE_VERBS:
        return "write"
    return "read" if v in _READ_VERBS else "unknown"


#: a tool that says, in its name or description, that it only reads (run_sql_readonly, "Run a read-only query")
_READ_ONLY_CLAIM = re.compile(
    r"(?<!not )(?<!non[- ])(?<!not a )read[- _]?only", re.IGNORECASE
)  # only the explicit words: "no side effect" often
#                                                                  # describes a mocked WRITE ("WRITE action, mocked, no side effect")


def claims_read_only(t: ToolSchema) -> bool:
    return bool(_READ_ONLY_CLAIM.search(t.name.replace("_", " ")) or _READ_ONLY_CLAIM.search(t.description or ""))


def is_write_tool(t: ToolSchema) -> bool:
    if t.kind == "agent-as-tool":
        return False  # it has no body: what the delegate's own tools do is judged on the delegate
    return _is_write_tool(t)


def _is_write_tool(t: ToolSchema) -> bool:
    c = canonical(t.name)
    if is_handoff_tool(c):
        return False
    if "mcp:readOnlyHint=true" in t.features:  # the server declares it: stronger than any name
        return False
    if "mcp:destructiveHint=true" in t.features or "mcp:readOnlyHint=false" in t.features:
        return True
    # what the body does (ast/effects.py) outranks the name: a write in the body is a write, whatever is declared
    if t.effect == "write":
        return True
    if t.declared_effect is not None:  # tool_effects: in spec.yml, for a body that only reads or cannot be followed
        return t.declared_effect == "write"
    if t.effect == "read":
        return False
    return name_says_write(t)  # no body, or one agenticreq cannot follow: the name is the only (weak) hint


def name_says_write(t: ToolSchema) -> bool:
    """The weak hint: what the tool's name (or its own read-only claim) says."""
    if claims_read_only(t):
        return False
    return action_verb(canonical(t.name)) in _WRITE_VERBS


def effect_unsure(t: ToolSchema) -> bool:
    """A write decision rests on the name alone because the body calls something agenticreq cannot follow: a result
    built on it waits for a mock run, never a gap."""
    return t.effect == "unclear" and t.declared_effect is None


def is_mcp_server_tool(t: ToolSchema) -> bool:
    """Defined by an MCP server in the scanned code: provided to clients, not called by an agent here."""
    return "mcp_server" in t.features


def is_high_impact_write(t: ToolSchema) -> bool:
    """Irreversible or external by name, or by what the body does (remote write, file write, exec)."""
    if not is_write_tool(t):
        return False
    if "mcp:destructiveHint=true" in t.features:
        return True
    if "mcp:destructiveHint=false" in t.features:
        return False
    return (
        action_verb(t.name) in _HIGH_IMPACT_VERBS
        or bool({"fs_write", "exec"} & set(t.features))
        or ("http" in t.features and "non_idempotent" in t.features)
    )


def has_idempotency(t: ToolSchema) -> bool:
    return (
        any(canonical(p.name) in _IDEMPOTENCY for p in t.params)
        or "idempotency" in t.features
        or "mcp:idempotentHint=true" in t.features
    )


_FREE_TEXT_PARAMS = {
    "query",
    "q",
    "text",
    "search",
    "keyword",
    "keywords",
    "term",
    "terms",
    "question",
    "prompt",
    "search_query",
    "pattern",
}


def is_unbounded(t: ToolSchema) -> bool:
    if "unbounded_read" in t.features and "bounded_output" not in t.features:
        # the body reads without a cap (SELECT with no LIMIT, .all()), whatever the name says, and a
        # `limit` parameter the body never applies bounds nothing
        return True
    c = canonical(t.name)
    verb = action_verb(c)
    c = c[c.index(verb) :] if verb in c else c  # `reddit_search_posts` is judged as `search_posts`
    if not c.startswith(_UNBOUNDED_PREFIX) or "bounded_output" in t.features:
        return False  # the body caps the result itself: TavilySearch(max_results=...), results[:10]
    if c.startswith(("search", "find", "query", "select")) and not t.params and t.source == "ast":
        return False  # a search with nothing to search for is an action: find_my_phone() rings the phone
    if (
        t.source == "ast"
        and not t.func_name
        and t.kind in ("mcp-server tool", "dict-schema")
        and not t.body_bounds
        and "bounded_output" not in t.features
    ):
        return False  # schema only, its handler is elsewhere: the name alone is not evidence of unbounded output
    for p in t.params:
        if canonical(p.name) in _BOUND_PARAMS:
            return False
        if any(k in p.constraints for k in ("maxItems", "maximum", "maxLength")):
            return False
    # a required argument that narrows the question (an id, a number, an airport, a date) scopes the result; only a
    # lone free-text query leaves it open. Without body evidence the name alone is not enough to call it unbounded
    if any(p.required and canonical(p.name) not in _FREE_TEXT_PARAMS for p in t.params):
        return False
    return True


#: verbs that themselves mean "go and read outside content" (browse_forum, scrape_page): matched on the whole name
_UNTRUSTED_VERB_HINT = re.compile(r"(browse|scrape|crawl|retriev|http)", re.IGNORECASE)
#: more outside-content objects, matched as whole words of the object part
_UNTRUSTED_EXTRA = re.compile(
    r"(^|_)(mentions?|tweets?|forum|threads?|feeds?|posts?|replies|reply)(_|$)", re.IGNORECASE
)


def looks_untrusted_source(t: ToolSchema) -> bool:
    """The tool reads outside content: judged on WHAT it reads (the words after its verb), not on the verb itself, so
    `read_reviews` is a source and `review_booking` (an action verb, an internal object) is not."""
    if is_write_tool(t):
        return False
    words = canonical(t.name).split("_")
    verb = action_verb(t.name)
    tail = "_".join(words[words.index(verb) + 1 :]) if verb in words else ""
    name = canonical(t.name)
    return bool(
        _UNTRUSTED_HINT.search(tail or name)
        or _UNTRUSTED_EXTRA.search(tail or name)
        or _UNTRUSTED_VERB_HINT.search(name)
    )


# ---------------------------------------------------------------- models
_MODEL_NAME = re.compile(
    r"^(?:[\w.-]+/)?(gpt-|o1|o3|o4|chatgpt|claude|gemini|mistral|mixtral|llama|command|grok|deepseek|qwen|"
    r"anthropic\.|us\.anthropic|openai/|anthropic/|bedrock/|vertex_ai/|azure/)",
    re.IGNORECASE,
)
_PINNED = [
    re.compile(r"^(models/)?gemini-\d+(\.\d+)?-(?!.*latest)[a-z0-9.-]+$"),  # Google stable ids are fixed versions
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),  # gpt-4o-2024-08-06
    re.compile(r"-20\d{6}(-v\d+:\d+)?$"),  # claude-sonnet-4-5-20250929, bedrock ...-v1:0
    re.compile(r"@20\d{6}$"),  # vertex claude-3-5-sonnet@20240620
    re.compile(r"-\d{3,4}$"),  # gpt-4-0613, gemini-1.5-pro-002
    re.compile(r"-v\d+:\d+$"),
]


def strip_provider(s: str) -> str:
    """'anthropic:claude-sonnet-4-5' / 'openai:gpt-4o' (PydanticAI style) -> bare model id."""
    head, sep, tail = s.partition(":")
    return tail if sep and head.replace("-", "").isalpha() and not head.startswith(("claude", "gpt")) else s


def looks_like_model(s: str) -> bool:
    s = strip_provider(s.strip())
    return bool(_MODEL_NAME.match(s)) and " " not in s and len(s) < 120


def is_pinned(model: str) -> bool:
    m = strip_provider(model.strip())
    if m.endswith("-latest") or m.endswith(":latest"):
        return False
    return any(p.search(m) for p in _PINNED)


def model_key(name: str) -> str:
    """Identity of a model across spellings: 'openai:gpt-4o', 'openai/gpt-4o' and 'gpt-4o' are one model."""
    m = strip_provider(name).strip().lower()
    while "/" in m and m.split("/", 1)[0] in _PROVIDERS:  # litellm / crewai: "anthropic/claude-..."
        m = m.split("/", 1)[1]
    return m


_PROVIDERS = {
    "openai",
    "anthropic",
    "azure",
    "bedrock",
    "gemini",
    "google",
    "vertex_ai",
    "groq",
    "mistral",
    "ollama",
    "openrouter",
    "together_ai",
    "deepseek",
    "xai",
    "cohere",
    "fireworks_ai",
}


_TEST_FILE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


def is_test_file(path: str) -> bool:
    """Test / fixture code configures models and tools too, but it is not what ships."""
    return bool(_TEST_FILE.search(path.replace("\\", "/")))


# ---------------------------------------------------------------- providers (data/provider_pinning.yml)
def _providers() -> dict[str, dict[str, Any]]:
    from agenticreq.data import load_yaml

    raw = (load_yaml("provider_pinning.yml") or {}).get("providers") or {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def _strs(p: dict[str, Any], key: str) -> list[str]:
    return [str(x) for x in p.get(key) or []]


def provider_from(
    *, model: str | None = None, cls: str | None = None, host: str | None = None, hint: str | None = None
) -> str | None:
    """The provider a model call goes to, from whatever evidence is available; None when unknown."""
    for name, p in _providers().items():
        if cls and cls in _strs(p, "classes"):
            return name
        if host and any(host == h or host.endswith("." + h) for h in _strs(p, "hosts")):
            return name
        if model and any(model.lower().startswith(x) for x in _strs(p, "prefixes")):
            return name
        if hint and any(x.lower() in hint.lower() for x in _strs(p, "env_hints")):
            return name
    return None


def offers_snapshots(provider: str | None) -> bool:
    """False only for providers documented as having no dated snapshots; unknown -> True (normal AS-210)."""
    if provider is None:
        return True
    return _providers().get(provider, {}).get("snapshots", True) is not False
