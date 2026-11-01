"""Run agenticreq on one of your own agents and get a sheet to judge every finding: sahi (real) or shor (noise).

    python validation/try_my_agent.py <agent dir or file> --name procurement \
        [--entry run.py] [--mocks mocks.yml] [--python /path/to/venv/bin/python] \
        [--spec spec.yml]

Steps (each one is skipped when its input is missing, and the summary says so):

1. ``mocks``    without --mocks, writes ``<out>/mocks.yml``: every tool found in code, write tools
                marked ``call: false`` so the canned model never asks for them. Edit it, then rerun.
2. ``mock-run`` with --entry and --mocks: runs the agent with the canned model, saves captures.
3. ``audit``    always: zero-config findings.
4. ``check``    with --spec: spec vs code.
5. writes ``<out>/judge.csv``: one row per finding, with an empty ``verdict`` column.
   Fill it with ``sahi`` / ``shor`` (or ``tp`` / ``fp``), then run ``--score <out>/judge.csv``.

Nothing is sent anywhere and no LLM is called. Your code is parsed; it only runs in the mock run,
inside its guards (network blocked at the Python level, credentials withheld).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from agenticreq.ast.classify import is_write_tool
from agenticreq.engine import run
from agenticreq.loader import load_spec
from agenticreq.observe import build_observed
from agenticreq.report.json_report import render_json
from agenticreq.report.sarif import render_sarif

TP = {"sahi", "tp", "true", "real", "yes", "y"}
FP = {"shor", "fp", "false", "noise", "no", "n"}


def write_mocks(path: Path, out: Path) -> Path:
    obs = build_observed(path)
    lines = ["# agenticreq mocks — generated. Review every WRITE tool before running agenticreq mock-run.",
             "# call: false  -> the canned model never requests the tool (nothing runs)",
             "# patch: 'module:function' + returns: {...}  -> the real function is replaced by a stub",
             "tools:"]
    for name, t in sorted(obs.declared_tools.items()):
        where = f"{t.code_ref.file}:{t.code_ref.line}" if t.code_ref else "?"
        if is_write_tool(t):
            lines.append(f"  {name!r}: {{call: false}}   # WRITE tool ({where}) — or patch it to exercise it")
        else:
            lines.append(f"  {name!r}: {{}}   # read tool ({where})")
    lines.append('final_text: "Done."')
    p = out / "mocks.yml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--name", default="agent")
    ap.add_argument("--entry")
    ap.add_argument("--mocks", type=Path)
    ap.add_argument("--python")
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--score", type=Path, help="score a filled judge.csv instead of running")
    a = ap.parse_args(argv)
    if a.score:
        return score(a.score)
    if a.path is None:
        ap.error("path is required")
    out = a.out or Path("agenticreq-try") / a.name
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    calls, versions = [], {}
    if a.entry and a.mocks:
        from agenticreq.capture import discover, save_captures

        r = discover(a.entry, mock_tools=str(a.mocks), python=a.python,
                     cwd=str(a.path if a.path.is_dir() else a.path.parent))
        cap = save_captures(r, out / "captures.json")
        from agenticreq.capture.discover import load_capture_bundle

        calls, versions = load_capture_bundle(cap)
        got, tot = r.coverage
        log.append(f"mock-run: {got}/{tot} LLM calls captured, zero-touch={r.zero_touch}"
                   + (f", run errors: {'; '.join(v.strip().splitlines()[-1] for v in r.run_errors.values() if v.strip())}"
                      if r.run_errors else ""))
    elif a.entry:
        p = write_mocks(a.path, out)
        log.append(f"mock-run: SKIPPED — wrote {p}; review it, then rerun with --mocks {p}")
    else:
        log.append("mock-run: SKIPPED (no --entry): static analysis only, wire rules will not run")

    obs = build_observed(a.path, captures=calls, runtime_versions=versions)
    log.append(f"extracted: framework={obs.framework_tag()}, execution model={obs.execution_model}, "
               f"{len(obs.declared_tools)} tools, {len(obs.declared_agents)} agents, {len(obs.files)} files")
    reports = [("audit", run(mode="audit", observed=obs))]
    if a.spec:
        reports.append(("check", run(mode="check", loaded=load_spec(a.spec), observed=obs)))
    else:
        log.append("check: SKIPPED (no --spec). Start one with: agenticreq init <path> --agent-id <id>")

    rows = []
    for mode, rep in reports:
        (out / f"{mode}.json").write_text(render_json(rep), encoding="utf-8")
        (out / f"{mode}.sarif").write_text(render_sarif(rep), encoding="utf-8")
        counts = {b.name: len(v) for b, v in rep.by_bucket().items()}
        log.append(f"{mode}: {counts} (coverage {rep.coverage.extracted_clauses}/{rep.coverage.total_clauses})")
        for bucket, fs in rep.by_bucket().items():
            for f in fs:
                rows.append({"mode": mode, "bucket": bucket.name, "rule_id": f.rule_id, "severity": f.severity,
                             "confidence": str(f.confidence), "title": f.title, "where": str(f.location() or ""),
                             "detail": f.detail, "verdict": "", "note": ""})
    with (out / "judge.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["rule_id", "verdict"])
        w.writeheader()
        w.writerows(rows)
    log.append(f"judge sheet: {out / 'judge.csv'} ({len(rows)} findings). Fill `verdict` with sahi/shor, "
               f"then: python validation/try_my_agent.py --score {out / 'judge.csv'}")
    (out / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0


def score(sheet: Path) -> int:
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    per: dict[str, Counter[str]] = defaultdict(Counter)
    total: Counter[str] = Counter()
    for r in rows:
        if r.get("bucket") in ("UNKNOWNS", "ATTESTATIONS"):
            continue  # not claims of a defect: they are judged separately
        v = (r.get("verdict") or "").strip().lower()
        k = "tp" if v in TP else "fp" if v in FP else "unjudged"
        per[r["rule_id"]][k] += 1
        total[(r.get("bucket") or "") + ":" + k] += 1
    print(f"{'rule':8} {'sahi':>5} {'shor':>5} {'unjudged':>9}")
    for rule in sorted(per):
        c = per[rule]
        print(f"{rule:8} {c['tp']:>5} {c['fp']:>5} {c['unjudged']:>9}")
    for b in ("BLOCKERS", "WARNINGS"):
        tp, fp = total[f"{b}:tp"], total[f"{b}:fp"]
        if tp + fp:
            print(f"{b}: {fp}/{tp + fp} noise ({100 * fp // (tp + fp)}%), {total[f'{b}:unjudged']} unjudged")
    print(json.dumps({"sheet": str(sheet), "judged": sum(c["tp"] + c["fp"] for c in per.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
