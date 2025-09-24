# Contributing Guidelines

- Use feature branches (e.g., `feature/new-module`, `fix/doc-typo`).
- Keep pull requests focused and include tests/docs for the changes you make.
- Run the formatting and test suite (`pre-commit run --all-files`, `pytest`) before pushing.

Versioning
- Bump the version in `pyproject.toml` when you cut a release.
- Run `pre-commit` to keep `CITATION.cff` in sync via `scripts/sync_citation.py`.
