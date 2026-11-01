# agenticreq on every pull request

The action runs `agenticreq check` (with a spec) or `agenticreq audit` (without one), writes SARIF, and
uploads it to GitHub code scanning. Findings then show up as annotations on the PR's changed lines, and
the job fails according to `fail-on`.

Every result lands in one of four groups: **Spec vs code gaps** (the code does not do what the spec says, or a
spec field is missing or TODO), **Human sign-off** (a person confirms it under `confirmed:` in spec.yml; SIGNED
once confirmed, never counted as passed), **Check with a mock run** (a run of the agent could show it: run
`agenticreq mock-run`, or pass `--mcp-tools`) and **Passed**. In SARIF, a gap is level `error`, an open sign-off or
a mock-run check is level `note`, a signed sign-off is `none`. The job's pass/fail comes from the exit code under
`fail-on`, not from the annotation levels.

It reads the agent's code (AST) and never installs or runs the agent's own dependencies, so it needs no
API keys and no secrets.

```yaml
# .github/workflows/agenticreq.yml
name: agenticreq
on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read
  security-events: write   # upload SARIF to code scanning

jobs:
  agenticreq:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: <owner>/agenticreq@v0.1.0a1    # this repository, at a release tag
        with:
          path: agents/procurement          # where the agent code lives
          spec: agents/procurement/spec.yml # omit for a zero-config audit
          fail-on: gaps                     # gaps | all | never
```

| input | default | meaning |
|---|---|---|
| `path` | `.` | folder or file with the agent code, relative to the repo root |
| `spec` | (empty) | spec.yml; empty runs `audit` (Level 2 only; Level 1/3 need a spec) |
| `captures` | (empty) | a mock run file (`.agenticreq/captures.json` from `agenticreq mock-run`), for wire-level checks |
| `traces` | (empty) | OTel GenAI spans, for runtime checks (AS-228..231) |
| `fail-on` | `gaps` | `gaps` (fail on a spec vs code gap), `all` (also on an open sign-off or a check waiting for a mock run), or `never` (report only) |
| `python-version` | `3.12` | Python that runs agenticreq |
| `sarif-file` | `agenticreq.sarif` | where the report is written |
| `upload-sarif` | `true` | upload to code scanning (needs `security-events: write`) |

SARIF paths are written relative to the repository root, so annotations land on the right file even
when `path` is a subfolder. On private repositories, code scanning requires GitHub Advanced Security;
set `upload-sarif: false` there and read the summary in the job log instead.

Without the action, the same in any CI:

```bash
pip install agenticreq
agenticreq check agents/procurement --spec agents/procurement/spec.yml -f sarif -o agenticreq.sarif --fail-on gaps
```

Exit codes: `0` the `fail-on` condition passed, `1` it failed (`gaps`: a spec vs code gap; `all`: also an open
sign-off or a pending mock-run check), `2` the spec is invalid (nothing was checked), `3` no agent or readable code was
found (nothing was checked). `never` always exits `0`.
