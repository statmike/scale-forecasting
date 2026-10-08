## Summary

<!-- Briefly describe what this pull request changes and why. -->

## Co-update & hygiene checklist (`AGENTS.md` §1 & §3)

- [ ] No customer names, internal hostnames, personal home paths, email addresses, or credential artifacts are included in code, configs, docs, or notebook outputs.
- [ ] If models, metrics, runtimes, `RunConfig` fields, registry views, smoke configs, dependency extras, or public modules changed, all co-updated files in [`AGENTS.md`](../AGENTS.md) §3 are updated in this PR.
- [ ] If `pyproject.toml` or `uv.lock` changed, `make lock` was run and `docker/requirements.txt` is committed.

## Verification gates (`AGENTS.md` §5)

- [ ] **Gate 1 (Consistency & docs tripwires):** `.venv/bin/pytest tests/unit/test_validation_ledger.py tests/unit/test_config_coverage.py tests/unit/test_docs_integrity.py tests/unit/test_api_docs_coverage.py tests/unit/test_test_dependencies_declared.py tests/unit/test_notebook_hygiene.py tests/unit/test_packaging_extras.py tests/smokes/test_smoke_configs.py -q`
- [ ] **Gate 2 (Format, lint, mypy, lock, strict docs):** `.venv/bin/ruff format --check src/ tests/ && .venv/bin/ruff check src/ tests/ && .venv/bin/mypy src/scale_forecasting && make lock-check && .venv/bin/mkdocs build --strict`
- [ ] **Gate 3 (Offline unit & coverage gate):** `make test` (or `make ci-offline` if dependencies/imports changed)
- [ ] **Gate 4 (Live cloud smoke, if touching cloud engines/runtimes):** Verified `status = 'SUCCESS'`, zero orphaned compute resources, and recorded in `docs/validation.md`
