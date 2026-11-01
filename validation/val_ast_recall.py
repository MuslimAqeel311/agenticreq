"""AST-only inventory vs ground truth, run 1 vs now. Usage: python validation/val_ast_recall.py <repos dir> validation"""
import json
import sys
from collections import Counter
from pathlib import Path

from agenticreq.ast.classify import canonical
from agenticreq.engine import run
from agenticreq.observe import build_observed

repos = Path(sys.argv[1])
val = Path(sys.argv[2])
truth = json.loads((val / "results/ground_truth.json").read_text())
before = json.loads((val / "results/audit_results.json").read_text())
targets = {
    "react-agent": "react-agent", "memory-agent": "memory-agent", "data-enrichment": "data-enrichment",
    "open_deep_research": "open_deep_research", "gpt-researcher/multi_agents": "gpt-researcher/multi_agents",
    "openai-cs-agents-demo": "openai-cs-agents-demo/python-backend",
    "openai-agents/financial_research_agent": "openai-agents-python/examples/financial_research_agent",
    "crewAI-examples/stock_analysis": "crewAI-examples/crews/stock_analysis",
    "pydantic-ai/bank_support": "pydantic-ai/examples/pydantic_ai_examples/bank_support.py",
    "smolagents/open_deep_research": "smolagents/examples/open_deep_research",
}
tot_t = tot_b = tot_a = 0
rows = []
for name, rel in targets.items():
    path = repos / rel
    if not path.exists():
        rows.append((name, "MISSING"))
        continue
    t = {canonical(x) for x in truth[name]["tools"]}
    obs = build_observed(path)
    r = run(mode="audit", observed=obs)
    after = {canonical(x) for x in obs.declared_tools} | {canonical(v) for v, n in obs.tool_vars.items() if n in obs.declared_tools and v[0].isupper()}
    prev = {canonical(x) for x in before[name].get("declared", [])}
    extra = sorted(after - t)
    tot_t += len(t); tot_b += len(prev & t); tot_a += len(after & t)
    rules = Counter(f.rule_id for f in r.findings)
    rows.append((name, f"truth {len(t)} | before {len(prev & t)}/{len(t)} | after {len(after & t)}/{len(t)} "
                       f"| missing {sorted(t - after)} | extra {extra} | rules {dict(rules)}"))
for n, s in rows:
    print(n.ljust(42), s)
print(f"\nTOOL RECALL  before {tot_b}/{tot_t}   after {tot_a}/{tot_t}")
