"""Infer the delegation graph.

No request carries a parent/child id, so (spike, §2.6):
* a distinct system_prompt sha256 = one agent node
* a handoff / delegate tool call = an edge from the calling node to the next node seen

This depends on request ordering and breaks for parallel / async delegation, so every
inferred edge is ``heuristic``. Edges from traces that DO carry parent span ids are
``observed``. Edges from code (Agent(handoffs=[...])) are ``proven`` declarations.
"""

from __future__ import annotations

from collections import defaultdict

from agenticreq.ast.classify import canonical, is_handoff_tool
from agenticreq.models.confidence import Confidence
from agenticreq.models.observed import AgentNode, ObservedAgent, ObservedEdge


def _node_name(
    obs: ObservedAgent,
    prompt: str | None,
    taken: set[str] = frozenset(),  # type: ignore[assignment]
    tools: set[str] | None = None,
) -> str | None:
    """Which declared agent sent this request, strongest evidence first:

    1. the request offers exactly one declared agent's tools — counting the agents it calls like tools
       (managed_agents, AgentTool, as_tool) and ignoring the framework's own tools. First, because
       managers' prompts embed their sub-agents' descriptions (smolagents, ADK), so a prompt match
       alone would name the manager after its sub-agent;
    2. its declared system prompt appears in the request (the longest, most specific one wins);
    3. the agent's name appears in the prompt (weak: managers and specialists name each other),
       skipping names already given to another node.
    """
    if not prompt:
        return None
    from agenticreq.data import InjectedTools

    injected = InjectedTools()
    # the framework's own tools (final_answer, transfer_to_agent, handoff ...) say nothing about WHICH agent
    own = {canonical(t) for t in (tools or set()) if not is_handoff_tool(t) and not injected.framework_of(t)}
    if own:  # one agent may send several prompt hashes (instructions built per turn): same name, merged

        def offers(a: object) -> set[str]:
            names = {canonical(t) for t in getattr(a, "tools", [])}
            for d in getattr(a, "delegates", []):
                tgt = next((b for b in obs.declared_agents if b.var == d or b.name == d), None)
                names.add(canonical(tgt.name if tgt else d))
            return names

        by_tools = [a for a in obs.declared_agents if a.kind != "group_chat" and offers(a) == own]
        if len(by_tools) == 1:
            return by_tools[0].name
    by_prompt = [
        a
        for a in obs.declared_agents
        if a.system_prompt and a.system_prompt.strip() and a.system_prompt.strip()[:60] in prompt
    ]
    if by_prompt:
        return max(by_prompt, key=lambda a: len(a.system_prompt or "")).name
    named = [a for a in obs.declared_agents if a.name and a.name in prompt and a.name not in taken]
    if not named:
        return None
    return min(named, key=lambda a: prompt.find(a.name or "")).name  # "You are <role>" comes first


def _target_from_call(name: str, args: dict[str, object]) -> str | None:
    """The agent a handoff names: an explicit argument first (ADK transfer_to_agent(agent_name=...),
    LlamaIndex handoff(to_agent=...), CrewAI coworker=...), else the tool's own suffix (transfer_to_billing)."""
    for key in ("agent_name", "to_agent", "coworker", "agent", "to", "target"):
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    c = canonical(name)
    for p in ("transfer_to_", "handoff_to_"):
        if c.startswith(p) and c != "transfer_to_agent":
            return c[len(p) :]
    return None


def _match_node(nodes: list[AgentNode], target: str) -> AgentNode | None:
    t = canonical(target)
    for n in nodes:
        if n.name and (canonical(n.name) == t or t in canonical(n.name)):
            return n
    return None


def build_delegation_graph(obs: ObservedAgent) -> None:
    nodes: dict[str, AgentNode] = {}
    ordered = sorted(obs.captures, key=lambda c: (c.run_id, c.index))
    for c in ordered:
        h = c.system_prompt_sha256
        if not h:
            continue
        own = [t for t in c.tools if not obs.is_output_schema(t.name)]  # a schema sent as a tool is not held
        if h not in nodes:
            taken = {n.name for n in nodes.values() if n.name}
            name = _node_name(obs, c.system_prompt, taken, {t.name for t in own})
            node_id = name or h[:6]  # several prompt hashes of one declared agent share its node
            nodes[h] = AgentNode(id=node_id, system_prompt_sha256=h, name=name)
            if name is None:  # no declared agent matches: the framework's own call, or one the code does not name
                from agenticreq.graph.origin import origin

                nodes[h].origin = origin(obs, c.system_prompt, [t.name for t in own])  # type: ignore[assignment]
        n = nodes[h]
        n.calls += 1
        for t in own:
            if t.name not in n.tools:
                n.tools.append(t.name)
    merged: dict[str, AgentNode] = {}  # one node per agent, however many prompt hashes it sent
    for n in nodes.values():
        if n.id in merged:
            m = merged[n.id]
            m.calls += n.calls
            m.tools += [t for t in n.tools if t not in m.tools]
        else:
            merged[n.id] = n
    obs.nodes = list(merged.values())

    edges: dict[tuple[str, str, str], ObservedEdge] = {}
    by_run: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(ordered):
        by_run[c.run_id].append(i)
    for idxs in by_run.values():
        for pos, i in enumerate(idxs):
            c = ordered[i]
            if not c.system_prompt_sha256:
                continue
            for tc in c.tool_calls:
                if not is_handoff_tool(tc.name):
                    continue
                src = nodes[c.system_prompt_sha256]
                target = _target_from_call(tc.name, tc.arguments)
                dst = _match_node(obs.nodes, target) if target else None
                if dst is None:  # next request with a different prompt hash, in order
                    for j in idxs[pos + 1 :]:
                        h2 = ordered[j].system_prompt_sha256
                        if h2 and h2 != c.system_prompt_sha256:
                            dst = nodes[h2]
                            break
                if dst is None or dst.id == src.id:
                    continue
                key = (src.id, dst.id, tc.name)
                if key not in edges:
                    edges[key] = ObservedEdge(
                        src=src.id,
                        dst=dst.id,
                        via_tool=tc.name,
                        confidence=Confidence.heuristic(),
                        evidence=f"{tc.name}({_short(tc.arguments)}) in request #{c.index} "
                        f"[{src.system_prompt_sha256[:6]} → {dst.system_prompt_sha256[:6]}]",
                    )
    # traces with parent ids give observed edges
    spans = {s.span_id: s for s in obs.trace_spans if s.span_id}
    for s in obs.trace_spans:
        parent = spans.get(s.parent_span_id or "")
        if (
            parent
            and parent.agent_name
            and s.agent_name
            and parent.agent_name != s.agent_name
            and s.operation in ("invoke_agent", "create_agent", None)
        ):
            key = (parent.agent_name, s.agent_name, "trace:parent_span_id")
            edges.setdefault(
                key,
                ObservedEdge(
                    src=parent.agent_name,
                    dst=s.agent_name,
                    via_tool="trace:parent_span_id",
                    confidence=Confidence.observed(1, 1),
                    evidence=f"span {s.span_id} parent {parent.span_id}",
                ),
            )
    obs.edges = list(edges.values())


def declared_edges(obs: ObservedAgent) -> list[tuple[str, str, str]]:
    """Edges declared in code, for every framework:

    * ``handoffs=[...]`` (OpenAI Agents, AutoGen Swarm), ``sub_agents=[...]`` (Google ADK),
      ``can_handoff_to=[...]`` (LlamaIndex), ``a.initiate_chat(b)`` (AutoGen 0.2);
    * agents exposed as tools: ``b.as_tool(...)`` (OpenAI Agents), ``AgentTool(agent=b)`` (Google ADK);
    * team coordinators -> each participant: AutoGen group chats, LlamaIndex AgentWorkflow, a CrewAI
      hierarchical crew (manager -> agents);
    * CrewAI ``allow_delegation=True`` (any other agent of the crew).
    """

    def resolve(h: str) -> str:
        by_var = next((b.name for b in obs.declared_agents if b.var == h), None)
        if by_var:
            return by_var
        return next((b.name for b in obs.declared_agents if canonical(b.name) == canonical(h)), h) or h

    out = []
    for a in obs.declared_agents:
        via = "group_chat" if a.kind == "group_chat" else "handoffs="
        for h in a.handoffs:
            out.append((a.name, resolve(h), via))
        for d in a.delegates:  # called like a tool, answers back: managed_agents, a tool that runs an agent
            out.append((a.name, resolve(d), "as_tool"))
        for t in a.tools:
            tool = obs.declared_tools.get(t)
            target = (tool.raw or {}).get("agent") if tool is not None and tool.kind == "agent-as-tool" else None
            if isinstance(target, str) and target:
                out.append((a.name, resolve(target), "as_tool"))
        if a.allow_delegation:
            for b in obs.declared_agents:
                if b.name != a.name and b.kind == a.kind:
                    out.append((a.name, b.name, "allow_delegation=True"))
    return out


def _short(args: dict[str, object]) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    return s[:80]
