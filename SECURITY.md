# Security policy

## Supported versions

agenticreq is in alpha (`0.1.0a1`). Security fixes go to the latest released version only.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
(**Security → Report a vulnerability** on the repository) or email the maintainer at muslimarain321@gmail.com with:

- what you found and the version (`agenticreq version`),
- the smallest project, spec or command that shows it,
- the impact you expect.

You should get an acknowledgement within 7 days. We will agree on a fix and a disclosure date with you, and credit you
in the CHANGELOG unless you prefer not to be named.

## What is in scope

- `agenticreq audit`, `check`, `init` and the other read-only commands read your code as text (AST); they do not
  import or run it. A way to make them execute project code, write outside the project or read secrets is a
  vulnerability.
- `agenticreq mock-run` **does run your agent** with mock tools and no network. Run it only on projects you trust,
  and on a separate machine for a project from someone else. Escapes from its network and process guards are in
  scope; the fact that it runs the project's own code is not.
- The GitHub Action (`action.yml`): script injection through inputs, or leaking tokens.
- Packaging: anything that makes the published files differ from this repository.

## What is out of scope

Findings about the agents you scan (that is the tool's job), and issues in third-party frameworks.
