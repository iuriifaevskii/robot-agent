---
name: python-quality
description: >-
  Run this repository's Python checks and keep tests tied to issue scenarios.
  Use when editing Python or tests, creating a GitHub issue, or when the user
  mentions tests, test scenarios, lint, format, code quality, or checks.
---

# Python quality

Runtime dependencies stay in `requirements.txt`. Test and lint tools stay in `requirements-dev.txt`.

## Checks

From the repository root, with the project venv:

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
```

Run all three after a Python change. Fix failures before finishing. Report the commands that actually ran.

`ruff format .` may rewrite files. Review the diff and keep it limited to the current task.

## Tests

- For a behavior change, the issue lists test scenarios: the situation and the expected result. Implement those scenarios. Skip scenarios for documentation and workflow-only changes.
- Do not add tests that do not protect a listed scenario.
- Test behavior through the existing functions. Do not call the OpenAI API.
- Replace `load_dotenv` in tests so a local `.env` cannot supply variables or secrets.
- Use fake values such as `test-key` and `test-model`. Never print or assert a real API key.
- Cover the missing-configuration and success paths when changing configuration handling.

## Scope

Do not add mypy, pre-commit, or coverage thresholds unless the user asks.

The GitHub workflow stays in `AGENTS.md`. Do not add a workflow skill that copies it.
