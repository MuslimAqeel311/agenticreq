"""`agenticreq init` — mock run + AST → spec.yml scaffold with TODO markers.

Business intent (prohibited, limits, ...) is NEVER auto-filled; `check` fails on every TODO (AS-101).
"""

from __future__ import annotations

import json
import re

from agenticreq.ast.classify import (
    canonical,
    is_high_impact_write,
    is_unbounded,
    is_write_tool,
    looks_untrusted_source,
)
from agenticreq.data import InjectedTools
from agenticreq.models.observed import ObservedAgent
from agenticreq.models.spec import TIER2_FIELDS

#: the 12 Level-1 fields (owner + technical_owner count as one). Only a human can write these.
_TIER1_TODO = [
    ("agent_id", None, "required"),
    ("risk_level", None, "required — low | medium | high | critical (selects every Level-2 default)"),
    ("goal", None, "required — {statement, done_criteria: [...]}"),
    ("out_of_scope", None, "required — [what this agent must not try to do]"),
    ("role_boundary", None, "{allowed_tools: [...]}"),
    ("prohibited", None, "required — [{tool, reason}] NEVER auto-filled (business intent)"),
    ("approval_required", None, "required — [{tool, approver, timeout, on_timeout: safe_stop}]"),
    ("limits", None, "required — [{tool, arg, limit: {max: ...}}] NEVER auto-filled"),
    ("escalation", None, "required — [{metric, op, threshold, to}]"),
    ("untrusted_inputs", None, "required — [{source: <tool or channel>}]"),
    ("owner", None, "required — business owner"),
    ("technical_owner", None, "engineering owner who acts on findings"),
    ("data_rules", None, "{retention: {...}, never_store: [...]} — write {} if the agent stores nothing"),
]


def _flow(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


#: next to each TODO that takes names: the spec holds the requirement, _derived what the code has
NAMES_HINT = "the names found in the code are listed in _derived; write the ones that SHOULD exist"
#: `init --from-code` prints this
FROM_CODE_WARNING = "names copied from the code: a spec copied from the code cannot find mismatches; review every line."


def code_agents(obs: ObservedAgent) -> list[str]:
    """Every agent the code declares, exactly as the code spells it (a team / group chat is not an agent)."""
    return list(dict.fromkeys(a.name for a in obs.declared_agents if a.kind != "group_chat"))


def derived(obs: ObservedAgent) -> dict[str, object]:
    from agenticreq.graph.builder import declared_edges

    injected = InjectedTools()
    declared = sorted(obs.declared_tools)
    has_obs = obs.has_runtime or bool(obs.mcp_tools)
    offered = sorted(obs.offered_tools()) if has_obs else []
    observed_user = [t for t in offered if canonical(t) in {canonical(d) for d in declared}]
    return {
        "framework": f"{obs.framework or 'not recognised'}"
        + (f" {obs.framework_version}" if obs.framework_version else ""),
        "execution_model": obs.execution_model,
        "declared_agents": code_agents(obs),
        "declared_tools": declared,
        # a team's coordinator -> member edges are the team itself, not a delegation (AS-301 skips them too)
        "delegation_in_code": list(
            dict.fromkeys(
                f"{a} -> {b}"
                for a, b, _ in declared_edges(obs)
                if a not in {g.name for g in obs.declared_agents if g.kind == "group_chat"}
            )
        ),
        "observed_tools": observed_user if has_obs else None,
        "never_reached": [t for t in declared if canonical(t) not in {canonical(o) for o in offered}]
        if has_obs
        else None,
        "injected_by_framework": [
            t for t in offered if injected.framework_of(t) and canonical(t) not in {canonical(d) for d in declared}
        ]
        + [
            f"agent {n.id}: a call the framework made itself (its prompt has no text from this project)"
            for n in obs.nodes
            if n.origin == "framework"
        ],
        "write_tools": [t.name for t in obs.declared_tools.values() if is_write_tool(t)],
        "unbounded_tools": [t.name for t in obs.declared_tools.values() if is_unbounded(t)],
        "models": [(m.name, m.pinned) for m in obs.models],
        "delegation_observed": [f"{e.src} -> {e.dst} via {e.via_tool}" for e in obs.edges],
        "system_prompt_hashes": [n.system_prompt_sha256[:6] for n in obs.nodes],
    }


_NUMERIC = {"integer", "number"}
_SENSITIVE = re.compile(
    r"(card|cvv|ssn|social_security|passport|password|passwd|secret|iban|account_number|"
    r"routing|dob|date_of_birth|national_id|tax_id)",
    re.IGNORECASE,
)


def suggested_risk(obs: ObservedAgent) -> str:
    """The risk level the tools suggest (a suggestion: risk_level itself stays TODO, a human decision)."""
    tools = list(obs.declared_tools.values())
    writes = [t for t in tools if is_write_tool(t)]
    if any(is_high_impact_write(t) for t in writes) or any(
        "exec" in t.features or "self_modify" in t.features for t in tools
    ):
        return "high"
    return "medium" if writes else "low"


#: the numeric spec fields: init lists their names only. Every number is yours to write (or not_needed with a reason).
NUMERIC_FIELDS: tuple[str, ...] = (
    "bounds.max_steps",
    "bounds.max_cost_usd",
    "bounds.max_duration_s",
    "bounds.max_retries",
    "bounds.max_tool_calls",
    "progress_guard.max_no_progress_steps",
    "output_size_limits.max_chars",
    "confidence_threshold",
)


def numeric_field_notes(multi_agent: bool = False) -> list[str]:
    """Comment lines naming the numbers the user writes. No value is written: agenticreq never proposes a number."""
    names = [*NUMERIC_FIELDS, *(["max_delegation_depth"] if multi_agent else [])]
    return [
        "# Numbers: write your own value for each one you need, or not_needed with a reason.",
        "#   " + ", ".join(names),
    ]


def suggest(obs: ObservedAgent) -> dict[str, list[str]]:
    """Level-1 hints read from the code, written as comments under each TODO.

    They never fill a value: what is forbidden, what needs whose approval, and where the money limit is
    are business decisions (the scaffold stays TODO and `check` fails until a human writes them). The
    code only shows which tools exist and what they can do.
    """
    tools = list(obs.declared_tools.values())
    writes = [t for t in tools if is_write_tool(t)]
    high = [t for t in writes if is_high_impact_write(t)]
    dangerous = [t for t in tools if "exec" in t.features or "self_modify" in t.features]
    out: dict[str, list[str]] = {}

    level = suggested_risk(obs)
    why = (
        ", ".join(t.name for t in (high + dangerous)[:4]) or ", ".join(t.name for t in writes[:4]) or "read-only tools"
    )
    out["risk_level"] = [f"{level}  ({why})"]
    names = [a.name for a in obs.declared_agents if a.kind != "group_chat"]
    if names:
        out["goal"] = [f"what {', '.join(names[:3])} must achieve; make every done_criteria item checkable"]
    cands = dangerous + [t for t in high if t not in dangerous and _destructive(t)]
    if cands:
        out["prohibited"] = [f"{{tool: {t.name}, reason: TODO}}  # {_what(t)}" for t in cands[:6]]
    gate = [t for t in high if t not in cands]
    if gate:
        out["approval_required"] = [
            f"{{tool: {t.name}, approver: TODO, timeout: TODO, on_timeout: safe_stop}}" for t in gate[:6]
        ]
    lims = [(t, p) for t in writes for p in t.params if p.type in _NUMERIC and not p.name.endswith("_id")]
    if lims:
        out["limits"] = [f"{{tool: {t.name}, arg: {p.name}, limit: {{max: TODO}}}}" for t, p in lims[:6]]
        t, p = lims[0]
        out["escalation"] = [f"{{metric: {p.name}, op: '>', threshold: TODO, to: TODO}}  # from {t.name}.{p.name}"]
    sources = [t for t in tools if looks_untrusted_source(t) or "url_from_arg" in t.features]
    if sources:
        out["untrusted_inputs"] = ["[" + ", ".join(f"{{source: {t.name}}}" for t in sources[:6]) + "]"]
    sensitive = sorted({p.name for t in tools for p in t.params if _SENSITIVE.search(p.name)})
    if sensitive:
        out["data_rules"] = ["{never_store: " + _flow(sensitive) + "}"]
    return out


def _topology(obs: ObservedAgent) -> str:
    """A guess at the multi-agent shape, from how the code wires the agents."""
    from agenticreq.graph.builder import declared_edges

    kinds = {a.kind for a in obs.declared_agents}
    if "group_chat" in kinds:
        team = next(a for a in obs.declared_agents if a.kind == "group_chat")
        return (
            "swarm" if team.var and any(a.handoffs for a in obs.declared_agents if a.kind != "group_chat") else "peer"
        )
    edges = [(s, d) for s, d, _ in declared_edges(obs)]
    if not edges:
        return "single" if len(obs.declared_agents) <= 1 else "pipeline"
    if any((d, s) in edges for s, d in edges):
        return "swarm"
    out_deg: dict[str, int] = {}
    for s, _ in edges:
        out_deg[s] = out_deg.get(s, 0) + 1
    depth_two = any(d in out_deg for _, d in edges)
    return "hierarchical" if depth_two else "supervisor"


def _destructive(t: object) -> bool:
    from agenticreq.ast.classify import action_verb, canonical

    name = getattr(t, "name", "")
    return action_verb(canonical(name)) in (
        "delete",
        "drop",
        "truncate",
        "pay",
        "transfer",
        "wire",
        "charge",
        "refund",
        "terminate",
        "revoke",
        "grant",
        "destroy",
        "wipe",
        "kill",
    )


def _what(t: object) -> str:
    feats = getattr(t, "features", [])
    if "exec" in feats:
        return "runs code / shell"
    if "self_modify" in feats:
        return "changes the agent itself"
    return "irreversible / moves money"


def _y(v: object) -> str:
    """A YAML-safe scalar / flow list: a tool named `a: b` or `#x` must not break the file."""
    return json.dumps(v, ensure_ascii=False)


def unresolved(obs: ObservedAgent) -> list[str]:
    """What extraction could not see, so `_derived` never reads as complete when it is not."""
    out = [f"tool built at {u.code_ref}: {u.reason}" for u in obs.unresolved_tools]
    out += [
        f"agent created at runtime (prompt {n.system_prompt_sha256[:6]}, tools: {', '.join(n.tools) or 'none'}): "
        "not named in the code; confirm what it is under confirmed: (runtime_agents)"
        for n in obs.nodes
        if n.origin == "unresolved"
    ]
    out += [f"agents built at {u.code_ref}: {u.reason}" for u in obs.unresolved_agents]
    out += [f"{e.file}:{e.line or '?'} {e.message}" for e in obs.extraction_errors]
    if not obs.models:
        out.append("model: not a literal in code (env / config); pinning not verified")
    if not obs.has_runtime and not obs.mcp_tools:
        out.append("observed_tools / never_reached: no mock run (run `agenticreq mock-run`, or pass --captures)")
    return out


def render_init(
    obs: ObservedAgent, agent_id: str | None = None, notes: list[str] | None = None, from_code: bool = False
) -> str:
    """The scaffold. Whatever the agent looks like, the result loads with zero shape errors: only the
    user's own decisions are TODO (a test asserts this over every fixture and over broken code).

    Agent and tool names are the user's requirement, so role_boundary and agent_registry are TODO and the names
    the code has go to _derived only; ``from_code`` copies them in instead (`init --from-code`)."""
    from agenticreq.schema import MODELINE

    d = derived(obs)
    # first line: editors that read it complete field names and flag a wrong one (agenticreq export-schema)
    L: list[str] = [MODELINE, "# AUTO-DERIVED — no need to edit; regenerated by `agenticreq init`"]
    L += [f"# {n}" for n in notes or []]
    L.append("# _derived: what the code has today; not your requirement")
    L.append("_derived:")
    fw = obs.framework
    L.append(
        f"  framework: {_y(d['framework'])}"
        + ("" if fw else "   # no recognised agent framework: only framework-agnostic checks apply")
    )
    L.append(f"  execution_model: {_y(d['execution_model'])}")
    L.append(f"  files_scanned: {len(obs.files)}")
    L.append(f"  declared_agents: {_y(d['declared_agents'])}")
    L.append(f"  declared_tools: {_y(d['declared_tools'])}")
    L.append(f"  delegation_in_code: {_y(d['delegation_in_code'])}")
    if d["observed_tools"] is None:
        L.append("  observed_tools: not seen yet       # ← run `agenticreq mock-run` or pass --captures")
        L.append("  never_reached: not seen yet")
    else:
        L.append(f"  observed_tools: {_y(d['observed_tools'])}")
        nr = d["never_reached"]
        L.append(
            f"  never_reached: {_y(nr)}" + ("              # ← not reached in the mock run, not a PASS" if nr else "")
        )
    L.append(f"  injected_by_framework: {_y(d['injected_by_framework'])}")
    L.append(f"  write_tools: {_y(d['write_tools'])}")
    L.append(f"  unbounded_tools: {_y(d['unbounded_tools'])}")
    models: list[tuple[str, bool]] = d["models"]  # type: ignore[assignment]
    uniq = list(dict.fromkeys(models))
    if not uniq:
        L.append("  model: not in code             # ← not a literal in code (env / config); pinning unchecked")
    elif len(uniq) == 1:
        name, pinned = uniq[0]
        L.append(f"  model: {_y(name)}" + ("" if pinned else "                   # ← NOT pinned"))
    else:
        L.append("  models:")
        for name, pinned in uniq:
            L.append(f"    - {_y(name)}" + ("" if pinned else "   # ← NOT pinned"))
    L.append(f"  delegation_observed: {_y(d['delegation_observed'])}")
    L.append(f"  system_prompt_hashes: {_y(d['system_prompt_hashes'])}")
    gaps = unresolved(obs)
    L.append(
        f"  unresolved: {_y(gaps)}"
        if not gaps
        else "  unresolved:   # not visible to static analysis: not read from code, never a PASS"
    )
    L += [f"    - {_y(g)}" for g in gaps]
    L.append("")
    width = max(len(k) for k, _, _ in _TIER1_TODO) + 1
    hints = suggest(obs)
    for key, _, note in _TIER1_TODO:
        if key == "role_boundary" and from_code:
            L.append(f"{key + ':':<{width}} {{allowed_tools: {_y(d['declared_tools'])}}}   # copied from the code")
            continue
        val = agent_id if key == "agent_id" and agent_id else "TODO"
        comment = f"# {note}" if val == "TODO" else ""
        shown = _y(val) if val != "TODO" else val
        L.append(f"{key + ':':<{width}} {shown:<12} {comment}".rstrip())
        if key == "role_boundary":
            L.append(f"#   {NAMES_HINT}")
        if val == "TODO":
            L += [f"#   read from the code: {h}" for h in hints.get(key, [])]
    L.append("")
    L.append(
        f"# Level 2: {len(TIER2_FIELDS)} engineering settings come from risk_level (`agenticreq params` lists them)."
    )
    L.append("#          Level 2 numbers are never filled in for you; agenticreq only checks your code against them.")
    agents = [a for a in obs.declared_agents if a.kind != "group_chat"]
    # two or more agents, or a pool whose size only the run knows, turn Level 3 on
    multi = len(agents) > 1 or bool(obs.edges) or bool(obs.unresolved_agents)
    L += numeric_field_notes(multi_agent=multi)
    L.append("# Level 3 (multi-agent): topology, agent_registry, declared_delegation, conflict_resolution;")
    L.append("#          the other 10 have defaults.")
    if multi:  # several agents: the Level 3 fields are the user's to write, as real TODOs (check reports each one)
        L.append("# More than one agent was found: fill these in.")
        L.append(
            f"topology: TODO   # pipeline | supervisor | hierarchical | swarm | peer"
            f"   (read from the code: {_topology(obs)})"
        )
    if from_code:
        reg = ", ".join(f"{{id: {_y(a.name)}, tools: {_y(list(a.tools))}}}" for a in agents)
        L.append(f"agent_registry: [{reg}]   # copied from the code")
    else:
        L.append("agent_registry: TODO   # [{id, tools: [...]}]" + ("" if multi else "; one agent: write not_needed"))
        L.append(f"#   {NAMES_HINT}")
    if multi:
        L.append(
            "declared_delegation: TODO   # [{from: ..., to: ...}]; the delegations in the code are in "
            "_derived.delegation_in_code: write the ones that SHOULD exist"
        )
        L.append("conflict_resolution: TODO   # who decides when agents disagree (or not_needed)")
    L.append("# Level 4 (derived, never written): trace_coverage, enforceability")
    return "\n".join(L) + "\n"
