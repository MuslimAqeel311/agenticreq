import json, os, sys, traceback
from pathlib import Path
from agenticreq.capture import discover, save_captures
S = Path(sys.argv[1]); R = S / "repos"; D = S / "drivers"; V = S / "venvs"
RUNS = {  # name: (entry, venv, cwd for hashing, PYTHONPATH)
 "react-agent": (D/"react_agent_drv.py", "react-agent", R/"react-agent", R/"react-agent/src"),
 "memory-agent": (D/"memory_agent_drv.py", "memory-agent", R/"memory-agent", R/"memory-agent/src"),
 "data-enrichment": (D/"data_enrichment_drv.py", "data-enrichment", R/"data-enrichment", R/"data-enrichment/src"),
 "open_deep_research": (D/"open_deep_research_drv.py", "open_deep_research", R/"open_deep_research", R/"open_deep_research/src"),
 "openai-cs-agents-demo": (D/"cs_agents_drv.py", "cs-agents", R/"openai-cs-agents-demo", R/"openai-cs-agents-demo/python-backend"),
 "openai-agents/financial_research_agent": (D/"financial_drv.py", "financial", R/"openai-agents-python", R/"openai-agents-python"),
 "pydantic-ai/bank_support": (R/"pydantic-ai/examples/pydantic_ai_examples/bank_support.py", "bank_support", R/"pydantic-ai", R/"pydantic-ai/examples"),
 "crewAI-examples/stock_analysis": (D/"stock_analysis_drv.py", "stock_analysis", R/"crewAI-examples", f"{R}/crewAI-examples/crews/stock_analysis/src{os.pathsep}{R}/crewAI-examples/crews/stock_analysis/src/stock_analysis"),
}
only = sys.argv[2:] or list(RUNS)
results = json.loads((S/"val_runs.json").read_text()) if (S/"val_runs.json").exists() else {}
for name in only:
    entry, venv, cwd, pp = RUNS[name]
    os.environ["PYTHONPATH"] = str(pp)
    try:
        r = discover(str(entry), mock_tools=str(D/"mocks.yml"),
                     python=str(V/venv/"bin/python"), cwd=str(cwd), timeout=180)
        save_captures(r, S/"caps"/f"{name.replace('/', '__')}.json")
        cap, tot = r.coverage
        err = "; ".join(v.strip().splitlines()[-1] for v in r.run_errors.values() if v.strip())
        results[name] = {"coverage": f"{cap}/{tot}", "zero_touch": r.zero_touch, "changed": r.changed_files[:5],
                         "libs": r.libs_hooked, "blocked": r.egress_blocked[:4], "error": err[:400],
                         "offered": sorted({t.name for c in r.calls for t in c.tools})}
    except Exception as e:
        results[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
    print(name, json.dumps(results[name])[:600], flush=True)
(S/"val_runs.json").write_text(json.dumps(results, indent=1))
