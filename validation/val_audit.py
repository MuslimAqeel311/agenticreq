import json, sys
from collections import Counter
from pathlib import Path
from agenticreq.engine import run
from agenticreq.observe import build_observed
from agenticreq.capture.discover import load_capture_bundle

targets = json.loads(Path(sys.argv[1]).read_text())
caps_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
out = {}
for name, path in targets.items():
    calls, versions = ([], {})
    cap = caps_dir / f"{name.replace('/', '__')}.json" if caps_dir else None
    if cap and cap.exists():
        calls, versions = load_capture_bundle(cap)
    try:
        obs = build_observed(path, captures=calls, runtime_versions=versions)
        r = run(mode="audit", observed=obs)
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        continue
    conf = Counter(f.confidence.level.value for f in r.findings)
    b = r.by_bucket()
    out[name] = {
        "framework": obs.framework_tag(), "frameworks": obs.frameworks, "exec": obs.execution_model,
        "declared": sorted(obs.declared_tools), "bound": sorted(obs.bound_tools),
        "observed": sorted(obs.offered_tools()) if obs.has_runtime else None,
        "proven": conf["proven"], "heuristic": conf["heuristic"], "unknown": conf["unknown"] + conf["insufficient_evidence"],
        "observed_conf": conf["observed"],
        "blockers": len(b[list(b)[0]]), "warnings": len(b[list(b)[1]]), "unknowns": len(b[list(b)[2]]),
        "passed": len(r.passed), "coverage": [r.coverage.extracted_clauses, r.coverage.total_clauses],
        "rules": Counter(f.rule_id for f in r.findings),
        "findings": [{"rule": f.rule_id, "sev": f.severity, "conf": str(f.confidence), "title": f.title,
                      "where": str(f.location()), "detail": f.detail[:220]} for f in r.findings],
        "files": len(obs.files), "errors": [e.message for e in obs.extraction_errors],
    }
print(json.dumps(out, indent=1, default=str))
