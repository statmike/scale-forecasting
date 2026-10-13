# Contributing to `scale-forecasting`

Thank you for contributing to `scale-forecasting`. This guide covers local setup, adding new models or metrics, dependency management, and the verification gates every pull request must pass.

For the full architectural invariants and co-update rules, read [`AGENTS.md`](./AGENTS.md).

---

## 1. Development setup

`scale-forecasting` installs on **Python 3.11–3.12** (`requires-python = ">=3.11,<3.13"`; CI proves the bare install on each) and develops on **3.11.15** (`.python-version`, the interpreter the cloud stack is pinned to — see [docs/version_matrix.md](./docs/version_matrix.md)). Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

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
  tests/unit/test_agent_surfaces.py \
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

---

## Releasing

`scale-forecasting` follows Semantic Versioning (`MAJOR.MINOR.PATCH`) and publishes to PyPI and GitHub Releases from git tags via [`.github/workflows/release.yml`](./.github/workflows/release.yml) using PyPI Trusted Publishing (OIDC + PEP 740 cryptographic attestations).

### Semantic Versioning charter (`MAJOR.MINOR.PATCH`)

| Version Level | When to Increment (`X.Y.Z`) | Concrete `scale-forecasting` Examples | `run_id` & Registry Compatibility Guarantee |
| :--- | :--- | :--- | :--- |
| **`MAJOR` (`X.0.0`)**<br/>*1st Level — Breaking Architectural / Contract Changes* | Breaking changes to `RunConfig` JSON/dict schema, CLI flags, Python SDK (`Forecaster`) signatures, compute runtime/runner execution contracts, or BigQuery registry table/view schemas. | Removing or renaming a `RunConfig` field without a backward-compatible alias; changing how a runtime job is launched/routed in a way that requires new IAM/infra; non-additive BigQuery DDL changes; raising the minimum Python floor; intentional `run_id` digest changes across existing configs. | May introduce a new `run_id_inputs` architecture axis in `docs/validation.md` or require a schema migration (`ensure_tables`). |
| **`MINOR` (`1.X.0`)**<br/>*2nd Level — Backward-Compatible Capability Expansions* | Expanding the platform with new models, model capabilities, evaluation metrics, ensemble/reconciliation methods, optional `RunConfig` knobs, or new SDK/MCP/reporting tools. | Adding a new model or upgrading an existing model's capabilities (e.g., `ARIMA_PLUS_XREG` + `ML.EXPLAIN_FORECAST` on `arima_plus` in `v1.1.0`); adding a new metric to `METRIC_NAMES`; adding new optional `RunConfig` fields; adding new `Forecaster` or MCP methods. | **100% backward-compatible:** all new `RunConfig` fields default to prior behavior and are elided in `_DEFAULT_ELIDED` / `_NOT_IDENTITY` so existing configs keep their exact `run_id`. |
| **`PATCH` (`1.x.Y`)**<br/>*3rd Level — Bug Fixes, Packaging & Polish* | Backward-compatible bug fixes, numerical edge-case guards, documentation/packaging improvements, and dependency patch updates with zero new models, metrics, or config fields. | Fixing an edge case in a model/metric wrapper; packaging/README rendering improvements (`hatch-fancy-pypi-readme`); lockfile security/patch refreshes (`make lock`). | **100% drop-in:** zero config, schema, or `run_id` changes. |

### Version single-sourcing & synchronized files

`pyproject.toml` (`[project].version`) is the canonical version string read at runtime via `importlib.metadata.version("scale-forecasting")`. When preparing a new release (`X.Y.Z`), update these 6 locations in lockstep:

1. **`pyproject.toml`:** Set `version = "X.Y.Z"`.
2. **`src/scale_forecasting/__init__.py`:** Update the `except PackageNotFoundError:` fallback string (`__version__ = "X.Y.Z"`, used by zip-staged cloud workers without `.dist-info`; enforced by [`tests/unit/test_smoke.py`](./tests/unit/test_smoke.py)).
3. **`plugin.json`:** Set `"version": "X.Y.Z"` (enforced by [`tests/unit/test_agent_surfaces.py`](./tests/unit/test_agent_surfaces.py)).
4. **`skills/scale-forecasting/SKILL.md`:** Set `metadata.version: X.Y.Z` in the YAML frontmatter (enforced by [`tests/unit/test_agent_surfaces.py`](./tests/unit/test_agent_surfaces.py)).
5. **`CHANGELOG.md`:** Add a `## [X.Y.Z] - YYYY-MM-DD` section summarizing user-visible changes (enforced by [`tests/unit/test_docs_integrity.py`](./tests/unit/test_docs_integrity.py)).
6. **`CITATION.cff`:** Update `version: "X.Y.Z"` and `date-released` (enforced by [`tests/unit/test_docs_integrity.py`](./tests/unit/test_docs_integrity.py)).
7. **Lock & agent surfaces:** Run `UV_NO_CONFIG=1 make lock` (using `UV_NO_CONFIG=1` on Google Cloudtop workstations so `uv.lock` resolves against public `https://pypi.org/simple`) and `make agent-surfaces` (`docs/llms-full.txt` embeds the package version header), then verify with `make ci-offline`.

### Publishing via TestPyPI (`vX.Y.Zrc1`) and PyPI (`vX.Y.Z`)

Once the release-prep PR is merged to `main` and CI is green:

```bash
git checkout main && git pull --ff-only origin main

# 1. Recommended dry-run on TestPyPI (stamps X.Y.Zrc1 in the build runner and publishes to testpypi)
git tag -a vX.Y.Zrc1 -m "Release candidate vX.Y.Zrc1"
git push origin vX.Y.Zrc1

# 2. Official release to PyPI + GitHub Releases (pauses for statmike approval on the `pypi` GitHub environment)
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z

# 3. Verify live install from public PyPI (bypassing any workstation mirror cache)
UV_NO_CONFIG=1 uvx --default-index https://pypi.org/simple/ --refresh \
  --from scale-forecasting==X.Y.Z \
  python -c "import scale_forecasting; print(scale_forecasting.__version__)"
```

Note that root [`README.md`](./README.md) is rendered as the PyPI project description (`dynamic = ["readme"]` via `hatch-fancy-pypi-readme`): every file link in `README.md` must use an absolute `https://github.com/statmike/scale-forecasting/blob/main/...` URL (enforced by `test_docs_integrity.py`) and the ` ```mermaid ` architecture flowchart is automatically swapped to `docs/assets/social_card.png` in built `METADATA` so both PyPI and GitHub render cleanly.

---

## 6. Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License Agreement](https://cla.developers.google.com/about) (CLA). You (or your employer) retain the copyright to your contribution; this simply gives permission to use and redistribute your contributions as part of the project.

If you have not already done so, please complete your CLA at <https://cla.developers.google.com/> before submitting a pull request.

---

## 7. Community guidelines & license

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/) and [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).

By contributing to `scale-forecasting`, you agree that your contributions will be licensed under the [Apache License 2.0](./LICENSE).

This is not an officially supported Google product. It is built on Google products and is neither endorsed nor supported by Google.
