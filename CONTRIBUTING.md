# Contributing to `scale-forecasting`

Thank you for contributing to `scale-forecasting`. This guide covers local setup, adding new models or metrics, dependency management, and the verification gates every pull request must pass.

For the full architectural invariants and co-update rules, read [`AGENTS.md`](./AGENTS.md).

---

## 1. Development setup

`scale-forecasting` targets **Python 3.11** (`>=3.11,<3.12`) and manages dependencies with [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/statmike/scale-forecasting.git
cd scale-forecasting

# Install Python 3.11, all optional extras, and the dev group from uv.lock
make sync

# Optional: install the documentation site toolchain (MkDocs Material + mkdocstrings)
uv sync --frozen --all-extras --group docs

# Enable local pre-commit and pre-push verification hooks
make hooks
```

---

## 2. Public-repository hygiene

`scale-forecasting` is a public, generic enterprise forecasting platform:

- **Zero customer or proprietary leakage:** Never include customer names, engagement identifiers, internal hostnames, or personal home directory paths (`/Users/<name>/...`, `/home/<name>/...`) in code, comments, commit messages, configs, or persisted notebook outputs.
- **Sanitized identity in validation records:** When adding live validation entries to [`docs/validation.md`](./docs/validation.md), always use the literal placeholder `<the launching user email>` for `user_id` lineage examples.
- **No credential files:** Never commit service account keys (`*.json.key`, `service-account*.json`), `.env` files, or `.tfvars` files. Authenticate locally via Application Default Credentials (`gcloud auth application-default login`).

---

## 3. Common contribution workflows

### Adding a forecasting model (1 file)

Every Python model lives in its own module under [`src/scale_forecasting/models/`](./src/scale_forecasting/models/) and inherits from `BaseModel`. Follow the step-by-step guide in [`docs/adding_a_model.md`](./docs/adding_a_model.md) and update the canonical inventory files listed in [`AGENTS.md`](./AGENTS.md) §3.

### Adding an evaluation metric (1 file)

Every metric lives in its own module under [`src/scale_forecasting/metrics/`](./src/scale_forecasting/metrics/) and inherits from `BaseMetric`. Follow [`docs/adding_a_metric.md`](./docs/adding_a_metric.md) and update the metric inventory files listed in [`AGENTS.md`](./AGENTS.md) §3.

### Updating dependencies (`pyproject.toml` and `uv.lock`)

`pyproject.toml` is the single source of truth for dependencies, `uv.lock` pins exact versions, and [`docker/requirements.txt`](./docker/requirements.txt) is a derived export used by the container and Dataproc Serverless builds.

Whenever you edit `pyproject.toml` or pull a Dependabot `uv` update branch, regenerate the lock and container export together:

```bash
make lock
make ci-offline
```

> **Dependabot `uv` pull requests:** Dependabot updates `uv.lock` directly without running the Makefile export step. Because CI's `lock-check` job verifies that `docker/requirements.txt` matches `uv.lock`, a Dependabot `uv` PR will fail `lock-check` until a maintainer checks out the branch, runs `make lock`, and commits the updated `docker/requirements.txt`.

---

## 4. Verification gates (Definition of Done)

Every pull request must pass all offline gates before merge. CI runs these automatically on every push and pull request ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).

### Gate 1 — Fast consistency & documentation tripwires (seconds)

```bash
.venv/bin/pytest \
  tests/unit/test_validation_ledger.py \
  tests/unit/test_config_coverage.py \
  tests/unit/test_docs_integrity.py \
  tests/unit/test_api_docs_coverage.py \
  tests/unit/test_test_dependencies_declared.py \
  tests/unit/test_notebook_hygiene.py \
  tests/unit/test_packaging_extras.py \
  tests/smokes/test_smoke_configs.py -q
```

### Gate 2 — Formatting, linting, static typing, lock drift & strict docs build

```bash
.venv/bin/ruff format --check src/ tests/
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/scale_forecasting
make lock-check
.venv/bin/mkdocs build --strict
```

- **Ruff (`BLE001`):** Every deliberate `except Exception` in `src/` or `tests/` must carry an inline `# noqa: BLE001 - <reason>` comment explaining why a broad catch is required at that boundary.
- **Mypy:** Zero errors across `src/scale_forecasting` (`py.typed` is shipped in the wheel). Prefer narrowing types with `isinstance`, small typed helpers, or `Protocol` definitions rather than `# type: ignore`.

### Gate 3 — Offline unit & contract test suite (85 % coverage floor)

```bash
make test
# Or reproduce the exact isolated CI environment when changing dependencies or test imports:
make ci-offline
```

### Gate 4 — Live cloud verification (when modifying cloud engines or runtimes)

If your change modifies a cloud execution engine (`src/scale_forecasting/engines/`), runtime submitter, or smoke configuration (`configs/smokes/`), execute the affected smoke configuration on Google Cloud, verify zero orphaned compute resources remain, and record the run in [`docs/validation.md`](./docs/validation.md).
