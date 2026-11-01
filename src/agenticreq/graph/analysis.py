"""Graph algorithms over the merged delegation graph (code-declared + observed edges)."""

from __future__ import annotations

from dataclasses import dataclass, field

from agenticreq.ast.classify import canonical, is_handoff_tool
from agenticreq.graph.builder import declared_edges
from agenticreq.models.confidence import Confidence
from agenticreq.models.observed import AgentNode, ObservedAgent
from agenticreq.models.spec import AgentSpec


@dataclass
class Edge:
    src: str
    dst: str
    via: str
    confidence: Confidence
    evidence: str


@dataclass
class DelegationGraph:
    nodes: dict[str, set[str]] = field(default_factory=dict)  # agent id -> tools
    edges: list[Edge] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)  # agent id -> prompt sha prefix
    #: team / workflow coordinators (AutoGen group chats, AgentWorkflow, hierarchical crews): they route
    #: work but hold no tools of their own, so "holds fewer tools than its delegate" is not escalation
    groups: set[str] = field(default_factory=set)
    #: nodes seen in code or at runtime (not only listed in the spec's agent_registry)
    backed: set[str] = field(default_factory=set)

    def succ(self, n: str) -> list[Edge]:
        return [e for e in self.edges if e.src == n]

    def roots(self) -> list[str]:
        """Entry agents: never a delegation target. When every agent is a target (a hub whose
        specialists hand back), every agent that delegates is a candidate entry."""
        targets = {e.dst for e in self.edges}
        r = [n for n in self.nodes if n not in targets and self.succ(n)] or [n for n in self.nodes if n not in targets]
        if r and any(self.succ(n) for n in r):
            return r
        return [n for n in self.nodes if self.succ(n)] or list(self.nodes)  # no edges: every agent is an entry

    def paths_from(self, start: str, max_len: int = 12) -> list[list[Edge]]:
        """All simple paths starting at `start` (as edge lists)."""
        out: list[list[Edge]] = []

        def dfs(cur: str, path: list[Edge], seen: set[str]) -> None:
            for e in self.succ(cur):
                if e.dst in seen or len(path) >= max_len:
                    continue
                p = [*path, e]
                out.append(p)
                dfs(e.dst, p, seen | {e.dst})

        dfs(start, [], {start})
        return out

    def cycles(self) -> list[list[Edge]]:
        found: list[list[Edge]] = []
        keys: set[frozenset[tuple[str, str]]] = set()
        for n in self.nodes:
            for p in self.paths_from(n):
                back = [e for e in self.succ(p[-1].dst) if e.dst == n]
                for b in back:
                    cyc = [*p, b]
                    k = frozenset((e.src, e.dst) for e in cyc)
                    if k not in keys:
                        keys.add(k)
                        found.append(cyc)
            for e in self.succ(n):
                if e.dst == n:
                    found.append([e])
        return found

    def depth(self) -> tuple[int, list[Edge]]:
        best: list[Edge] = []
        for r in self.roots():
            for p in self.paths_from(r):
                if len(p) > len(best):
                    best = p
        return len(best), best


def path_names(path: list[Edge]) -> list[str]:
    return [path[0].src, *[e.dst for e in path]] if path else []


def path_confidence(path: list[Edge]) -> Confidence:
    """A path is only as strong as its weakest edge."""
    order = ["proven", "observed", "heuristic", "insufficient_evidence", "unknown"]
    worst = max(path, key=lambda e: order.index(e.confidence.level.value))
    return worst.confidence


def resolve_registry(spec: AgentSpec | None, obs: ObservedAgent) -> dict[str, str]:  # noqa: C901
    """observed node id / code agent name -> registry id."""
    out: dict[str, str] = {}
    if not spec or not spec.agent_registry:
        return out
    prompts = {c.system_prompt_sha256: c.system_prompt or "" for c in obs.captures if c.system_prompt_sha256}
    reg = spec.agent_registry

    def pick(name: str | None, sha: str | None, prompt: str | None) -> str | None:
        """Name / id / prompt hash first; a `match` substring in the prompt only as the last resort,
        because agents' prompts name each other ("hand back to the Triage Agent")."""
        for a in reg:
            if sha and a.system_prompt_sha256 and sha.startswith(a.system_prompt_sha256):
                return a.id
        if name:
            for a in reg:
                if canonical(a.id) == canonical(name) or (a.match and canonical(a.match) == canonical(name)):
                    return a.id
        if prompt:
            hits = [a for a in reg if a.match and a.match in prompt]
            if hits:  # the earliest mention is the agent's own ("You are the X")
                return min(hits, key=lambda a: prompt.find(a.match or "")).id
        return None

    for n in obs.nodes:
        rid = pick(n.name or n.id, n.system_prompt_sha256, prompts.get(n.system_prompt_sha256))
        if rid:
            out[n.id] = rid
    for d in obs.declared_agents:
        rid = pick(d.name, None, d.system_prompt)
        if rid:
            out.setdefault(d.name, rid)
            if d.var:
                out.setdefault(d.var, rid)
    return out


def unidentified(obs: ObservedAgent, spec: AgentSpec | None, origin: str | None = None) -> list[AgentNode]:
    """Runtime agents neither the code nor the spec's agent_registry (an id or a `match`) names: framework calls
    and the project's unnamed agents (``origin`` picks one of the two)."""
    alias = resolve_registry(spec, obs)
    return [n for n in obs.nodes if not n.identified and n.id not in alias and (origin is None or n.origin == origin)]


def build(obs: ObservedAgent, spec: AgentSpec | None) -> DelegationGraph:
    alias = resolve_registry(spec, obs)
    g = DelegationGraph()

    def nid(x: str) -> str:
        return alias.get(x, x)

    hidden = {n.id for n in unidentified(obs, spec, "framework")}  # a call the framework makes itself: not an agent
    for n in obs.nodes:
        if n.id in hidden:
            continue
        g.nodes.setdefault(nid(n.id), set()).update(t for t in n.tools if not is_handoff_tool(t))
        g.hashes[nid(n.id)] = n.system_prompt_sha256[:6]
        g.backed.add(nid(n.id))
    for a in obs.declared_agents:  # every agent declared in code is a node, delegating or not
        g.nodes.setdefault(nid(a.name), set()).update(t for t in a.tools if not is_handoff_tool(t))
        g.backed.add(nid(a.name))
        if a.kind == "group_chat":
            g.groups.add(nid(a.name))
    if spec and spec.agent_registry:
        for r in spec.agent_registry:
            g.nodes.setdefault(r.id, set()).update(r.tools)
    by_key: dict[tuple[str, str], Edge] = {}
    # code-declared edges are proven; observed edges add evidence to them or stand alone as heuristic
    for src, dst, via in declared_edges(obs):
        k = (nid(src), nid(dst))
        if k[0] != k[1] and k not in by_key:
            g.nodes.setdefault(k[0], set())
            g.nodes.setdefault(k[1], set())
            by_key[k] = Edge(k[0], k[1], via, Confidence.proven(), f"declared in code: {src} {via}[{dst}]")
    for e in obs.edges:
        if e.src in hidden or e.dst in hidden:
            continue
        k = (nid(e.src), nid(e.dst))
        if k in by_key:
            by_key[k].evidence += f"; observed: {e.evidence}"
        else:
            by_key[k] = Edge(k[0], k[1], e.via_tool, e.confidence, e.evidence)
    g.edges = list(by_key.values())
    g.backed |= {e.src for e in g.edges} | {e.dst for e in g.edges}
    return g


def own_node(g: DelegationGraph, spec: AgentSpec | None) -> str | None:
    """The node the spec's top-level clauses (prohibited, role_boundary) apply to."""
    if spec and spec.agent_id and spec.agent_id in g.nodes:
        return spec.agent_id
    roots = g.roots()
    return roots[0] if len(roots) == 1 else None
