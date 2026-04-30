# Contributing to sarpyx

This file provides reviewer-facing community guidance for how to report issues, request
support, and propose changes.

## Support and questions

Use GitHub Issues for bug reports, feature requests, installation problems, and workflow
questions. Before opening a new issue, search existing issues and include enough detail
for the maintainers to reproduce the problem.

For command-line or workflow questions, include:

- the exact command or API call you ran;
- the `sarpyx` version;
- your operating system and Python version;
- whether the run used local installation, Docker, or SNAP;
- the traceback or relevant log output.

## Reporting bugs

Please open a GitHub issue and include:

1. expected behavior;
2. observed behavior;
3. minimal reproduction steps;
4. platform details (OS, Python, `sarpyx` version, SNAP/Docker usage);
5. a small sample log or screenshot if relevant.

For security-sensitive issues, do **not** open a public issue. Use the private contact
described in `SECURITY.md`.

## Development setup

```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
uv venv .venv
source .venv/bin/activate
uv sync --group dev --extra copernicus
uv run pytest -q
```

SNAP + Java are required for SNAP-dependent commands such as `worldsar` and `shipdet`.

Docker-based workflows are also supported:

```bash
docker compose version
make recreate
make docker-test
```

## Making changes

- Create a feature branch from `main`.
- Keep changes scoped and well-described.
- Add or update tests for behavior changes.
- Update documentation when CLI flags, APIs, or workflow expectations change.
- Preserve CLI and API compatibility where practical, or document breaking changes.

## Pull requests

Each pull request should include:

- a short motivation and summary of the change;
- a link to the related issue, if one exists;
- the checks you ran locally (for example `pytest -q`, `make docker-test`, or specific
  CLI smoke checks);
- any assumptions about sample data, environment variables, or external tools.

## Code and documentation style

- Use type annotations and docstrings for public APIs where practical.
- Prefer small, deterministic tests over large binary fixtures.
- Keep examples reproducible and explicit about external dependencies.
- When user-facing behavior changes, update the docs and release notes.
