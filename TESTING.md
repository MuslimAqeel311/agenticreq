# Test agenticreq on your own agents

One script runs everything and writes a sheet where you mark every finding **sahi** (real) or
**shor** (noise). Nothing leaves your machine, and no LLM is called.

## 0. Once

```bash
cd Agent_Spec
pip install -e .                     # or use .venv/bin/python as below
```

## 1. Static first (2 minutes, safe: your code is only parsed)

```bash
.venv/bin/python validation/try_my_agent.py /path/to/procurement_agent --name procurement
```

Read `agenticreq-try/procurement/summary.txt`. The `extracted:` line must list **your** tools and
agents. If a tool is missing, stop and report it: every later number depends on it.

## 2. Runtime capture (the agent runs, but with a fake model, no network, no real keys)

```bash
# first time: writes agenticreq-try/procurement/mocks.yml with every tool it found
.venv/bin/python validation/try_my_agent.py /path/to/procurement_agent --name procurement \
    --entry /path/to/procurement_agent/run.py
```

Open `mocks.yml`. Write tools start as `call: false` (never requested). To exercise one safely,
replace it with a stub: `pay_invoice: {patch: "myapp.tools:pay_invoice", returns: {status: ok}}`.
Then run again with `--mocks agenticreq-try/procurement/mocks.yml` (add `--python /path/to/your/venv/bin/python`
if the agent needs its own environment).

## 3. Spec vs code

```bash
cp validation/templates/procurement.yml procurement.spec.yml   # also: support.yml, qc.yml
# replace every TODO with your real tool names and rules (check fails while a TODO is left)
.venv/bin/agenticreq validate procurement.spec.yml               # must show 0 findings
.venv/bin/python validation/try_my_agent.py /path/to/procurement_agent --name procurement \
    --entry /path/to/procurement_agent/run.py --mocks agenticreq-try/procurement/mocks.yml \
    --spec procurement.spec.yml
```

## 4. Judge every finding, then score

Open `agenticreq-try/procurement/judge.csv`. For each row, open the file:line in `where`. Write `sahi`
when the claim is true about your code **and** matters, `shor` when it does not, plus a short note.

```bash
.venv/bin/python validation/try_my_agent.py --score agenticreq-try/procurement/judge.csv
```

Repeat for `--name support` and `--name qc`. Send back the three `judge.csv` files: the `shor` rows
with notes are what gets fixed next.

**Reference:** on 10 public repos (VALIDATION.md, Run 2) all audit blockers were real, and 26% of
warnings were noise, mostly AS-213 (name-based "unbounded output"). Expect similar, and mark them honestly.
