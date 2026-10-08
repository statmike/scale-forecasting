# `AGENTS.md` — Engineering Charter, Style Guide & Verification Gates

This file is the authoritative engineering charter and review checklist for AI coding agents and human contributors working in `scale-forecasting`. It is automatically loaded from the repository root at the start of every session.

---

## 1. Public-Repository Hygiene & Zero-Leakage Charter

`scale-forecasting` (`github.com/statmike/scale-forecasting`) is a **public, generic, reusable enterprise forecasting platform** for Google Cloud.

1. **Zero Customer or Proprietary Leakage:** Never include customer names, engagement identifiers, internal corporate hostnames, or third-party vendor migration notes in code, comments, commit messages, configs, or documentation.
2. **Sanitized Identity in Validation Records:** In [`docs/validation.md`](./docs/validation.md) and commit messages, never record personal or corporate user email addresses; always use the literal placeholder `<the launching user email>` when illustrating populated `user_id` lineage fields.
3. **No Secret or Credential Artifacts:** Never commit service account keys (`*.json.key`, `service-account*.json`), `.env` files, or `.tfvars` files. All Google Cloud execution authenticates via Application Default Credentials (ADC) locally and least-privilege attached service accounts in cloud runtimes.
4. **Persisted Notebook Outputs — What Is and Is Not Acceptable:** Notebook outputs are committed so GitHub and the docs site render results without execution. The demo **project ID and bucket names may appear** in those outputs: Google Cloud does not treat them as secrets (they appear in console URLs and official samples) and knowing them grants nothing without IAM; the buckets they name enforce public access prevention. What must **never** appear in a persisted output: e-mail addresses, internal corporate hostnames, absolute paths under a personal or corporate home directory (`/usr/local/google/home/...`, `/Users/<name>/...`, `/home/<name>/...`), OS usernames, access tokens, or service-account key material. [`tests/unit/test_notebook_hygiene.py`](./tests/unit/test_notebook_hygiene.py) enforces exactly this split; suppress or clear the offending cell output rather than widening the allowlist.

---

## 2. Core Architectural Invariants

```mermaid
flowchart LR
    subgraph Pure["Pure Offline Layer (Zero GCP Imports)"]
        direction TB
        P1["config.py · registry/ids.py"]
        P2["models/* (34 Models · BaseModel · explain())"]
        P3["metrics/* (21 Metrics · BaseMetric)"]
        P4["features.py · backtest.py · calibration.py"]
        P5["reconciliation.py · ensembler.py · hpo.py"]
        P6["worker.py (run_cell · run_panel_model · attributions)"]
        P7["profiling/* · resources/*"]
        P8["agent_surfaces.py · mcp.py"]
    end

    subgraph Cloud["Thin Cloud Engine & Registry Wrappers"]
        direction TB
        C1["engines/spark_engine.py (Serverless · Cluster · Connect)"]
        C2["engines/ray_engine.py (Vertex AI Ray · Ray on GKE)"]
        C3["engines/vertex_engine.py (Vertex CustomJob · GCE Single-VM · GKE Indexed Job)"]
        C4["engines/automl_engine.py (Tabular Workflow · AutoML Training Job)"]
        C5["engines/bigquery_engine.py (ARIMA_PLUS · AI.FORECAST)"]
        C6["registry/bq.py · storage.py · ops.py"]
    end

    subgraph Tripwires["Automated Pre-Commit & CI Tripwires"]
        direction TB
        T1["test_validation_ledger.py"]
        T2["test_config_coverage.py"]
        T3["test_docs_integrity.py"]
        T4["test_api_docs_coverage.py"]
        T5["test_test_dependencies_declared.py"]
        T6["test_notebook_hygiene.py"]
        T7["test_packaging_extras.py"]
        T8["test_agent_surfaces.py"]
    end

    Pure -->|"Shared execution contract"| Cloud
    Pure & Cloud -->|"Enforced by"| Tripwires
```

1. **Declarative Config-Driven Execution (`config.py`):**
   - Behavior changes come from [`RunConfig`](./src/scale_forecasting/config.py) JSON/dict declarations, never environment-specific code forks.
   - The same orchestration code ([`main.run`](./src/scale_forecasting/main.py)) executes identically across all 5 launch surfaces: Local CLI, Python SDK (`Forecaster`), Interactive Notebooks, Cloud Composer 3 DAGs, and Headless Cloud Batch containers.
2. **Deterministic Content-Addressed `run_id` (`registry/ids.py`):**
   - `run_id` is `<slug>-<12hex>` computed strictly from semantic `RunConfig` inputs (`_RUN_ID_EXCLUDED` strips operational plumbing such as `compute.project_id`, `compute.region`, `compute.machine_type`, `compute.workers`, and `compute.families.*.machine_type` / `workers` when `"auto"` or default).
   - Never alter `run_id` hashing behavior without verifying [`tests/unit/test_ids.py`](./tests/unit/test_ids.py) and [`tests/unit/test_prebreak_snapshots.py`](./tests/unit/test_prebreak_snapshots.py).
3. **One-File Plugin Contracts (`models/` and `metrics/`):**
   - Every model lives in its own module under [`src/scale_forecasting/models/`](./src/scale_forecasting/models/) inheriting from [`BaseModel`](./src/scale_forecasting/models/base_model.py), declares its upstream `package`, `package_url`, `family`, covariate support flags (`supports_future_covariates`, `supports_past_covariates`, `supports_static_covariates`), `supported_training_modes`, `gpu_usefulness`, and `supports_explainability` (auto-detected when overriding `feature_attributions()` or `explain()` for Tier 1 `fit_diagnostics["feature_attributions"]` and Tier 2 `forecast_predictions.explanations`), and wraps optional third-party imports gracefully.
   - Every metric lives in its own module under [`src/scale_forecasting/metrics/`](./src/scale_forecasting/metrics/) inheriting from [`BaseMetric`](./src/scale_forecasting/metrics/base_metric.py).
4. **Zero-Idle Per-Family Compute & Scaling:**
   - The DAG router ([`dag.py`](./src/scale_forecasting/dag.py)) dispatches **1 independent job per active model family in parallel** (`statistical`, `ml`, `deep_learning`, `automl`, `native`) so fast CPU families tear down immediately without waiting for slow GPU trainers.
   - Compute configuration uses the unified vocabulary across `compute` and `compute.families.<family>`: `runtime` (`"spark"` | `"ray"` | `"vertex"` | `"gce"` | `"gke"` | `"vertex_automl"`), `gke_mode` (`"job"` | `"ray"`), `ray_mode` (`"vertex"` | `"gke"`), `automl_mode` (`"tabular_workflow"` | `"training_job"`), `machine_type` (`"auto"` or explicit GCE shape validated against `gpu_type` and `accelerator_count` in [`resources/catalog.py`](./src/scale_forecasting/resources/catalog.py)), `workers` (`1` default on CPU; auto-expands `1 -> len(models)` for multi-model `deep_learning` on `vertex`/`gke`; must be `1` on single-VM `gce`), `use_gpu` / `hardware`, `gpu_type` (`"T4"`, `"L4"`, `"A100"`, `"A100_80GB"`), and `accelerator_count`.
   - Multi-VM/multi-pod local sharding (`workers > 1` on `vertex` and `gke` Indexed Jobs) pushes each worker's contiguous `[start_id, end_id]` boundary directly into BigQuery Storage Read API `row_restriction` ([`build_worker_series_range`](./src/scale_forecasting/engines/vertex_engine.py)) and schedules chunks longest-first (`order_chunks_lpt`).
   - GCE Single-VM (`runtime = "gce"`) enforces triple-redundant zero-orphan cleanup (`scheduling.maxRunDuration` + `instanceTerminationAction="DELETE"`, guest COS `trap cleanup EXIT` REST API self-delete + `shutdown -h now`, and launcher `try ... finally` deletion).

---

## 3. Canonical Inventories & Mandatory Co-Update Matrix

Whenever you add or modify a model, metric, compute runtime, `RunConfig` field, BigQuery table/view, smoke configuration, or dependency extra, **you must update the corresponding documentation and tripwire files in the same work unit before declaring the task complete**:

| System Surface | Canonical Source of Truth | Current Count / Set | Files That MUST Be Updated Together |
| :--- | :--- | :--- | :--- |
| **Forecasting Models** | [`models/__init__.py`](./src/scale_forecasting/models/__init__.py) (`list_models()`) | **34 models** (`28` Python + `4` Vertex AI AutoML + `2` BQ SQL: `18` `statistical`, `5` `ml`, `5` `deep_learning`, `4` `automl`, `2` `native`) | [`README.md`](./README.md), [`docs/overview.md`](./docs/overview.md), [`docs/models_reference.md`](./docs/models_reference.md), [`docs/adding_a_model.md`](./docs/adding_a_model.md), [`src/scale_forecasting/README.md`](./src/scale_forecasting/README.md), [`src/scale_forecasting/models/README.md`](./src/scale_forecasting/models/README.md), [`notebooks/README.md`](./notebooks/README.md), [`tests/README.md`](./tests/README.md), [`docs/workshop.md`](./docs/workshop.md) |
| **Evaluation Metrics** | [`metrics/__init__.py`](./src/scale_forecasting/metrics/__init__.py) (`METRIC_NAMES`) | **21 metrics** (`16` point + `5` interval) | [`README.md`](./README.md), [`docs/overview.md`](./docs/overview.md), [`docs/metrics_reference.md`](./docs/metrics_reference.md), [`docs/adding_a_metric.md`](./docs/adding_a_metric.md), [`src/scale_forecasting/README.md`](./src/scale_forecasting/README.md), [`src/scale_forecasting/metrics/README.md`](./src/scale_forecasting/metrics/README.md), [`src/scale_forecasting/engines/README.md`](./src/scale_forecasting/engines/README.md), [`src/scale_forecasting/registry/README.md`](./src/scale_forecasting/registry/README.md), [`tests/README.md`](./tests/README.md) |
| **Compute Runtimes & Hardware** | [`config.py`](./src/scale_forecasting/config.py) & [`resources/catalog.py`](./src/scale_forecasting/resources/catalog.py) | **7 runtimes** (`spark`, `ray`, `vertex`, `gce`, `gke`, `vertex_automl`, `bigquery`) · **4 GPU types** (`T4`, `L4`, `A100`, `A100_80GB`) | [`README.md`](./README.md), [`docs/overview.md`](./docs/overview.md), [`docs/runtimes_reference.md`](./docs/runtimes_reference.md), [`docs/configuration_reference.md`](./docs/configuration_reference.md), [`docs/architecture.md`](./docs/architecture.md), [`docs/quota_and_scale.md`](./docs/quota_and_scale.md), [`src/scale_forecasting/engines/README.md`](./src/scale_forecasting/engines/README.md), [`src/scale_forecasting/resources/README.md`](./src/scale_forecasting/resources/README.md), [`src/scale_forecasting/probes/README.md`](./src/scale_forecasting/probes/README.md), [`tests/unit/test_config_coverage.py`](./tests/unit/test_config_coverage.py) |
| **Registry Tables & Views** | [`registry/ddl.py`](./src/scale_forecasting/registry/ddl.py) & [`registry/views.py`](./src/scale_forecasting/registry/views.py) | **4 source tables** · **5 registry tables** · **5 analytical SQL views** (`v_model_leaderboard`, `v_model_leaderboard_comparable`, `v_backtest_coverage`, `v_run_summary`, `v_run_jobs`) | [`README.md`](./README.md), [`docs/overview.md`](./docs/overview.md), [`docs/output_schemas.md`](./docs/output_schemas.md), [`docs/writing_results.md`](./docs/writing_results.md), [`docs/reading_source_data.md`](./docs/reading_source_data.md), [`src/scale_forecasting/registry/README.md`](./src/scale_forecasting/registry/README.md), [`docs/workshop.md`](./docs/workshop.md) |
| **Smoke & Demo Configs** | [`configs/smokes/*.json`](./configs/smokes/) & [`configs/*.json`](./configs/) | **42 smoke configs** (`01`–`42`) · **20 root demo configs** | [`configs/smokes/README.md`](./configs/smokes/README.md), [`configs/README.md`](./configs/README.md), [`docs/smoke_testing.md`](./docs/smoke_testing.md), [`docs/validation.md`](./docs/validation.md), [`tests/README.md`](./tests/README.md), [`tests/smokes/test_smoke_configs.py`](./tests/smokes/test_smoke_configs.py) |
| **Dependency Extras** | [`pyproject.toml`](./pyproject.toml) (`[project.optional-dependencies]`) | **12 extras** (`gcp`, `notebook`, `spark`, `ray`, `submit`, `models-stats`, `models-trees`, `models-prophet`, `models-dl`, `models-automl`, `models`, `all`); core `dependencies` are the pure offline layer and `[gcp]` is the single home of every Google client | [`README.md`](./README.md), [`docs/overview.md`](./docs/overview.md), [`docs/getting_started.md`](./docs/getting_started.md), [`docs/runtime_dependencies.md`](./docs/runtime_dependencies.md), [`docs/models_reference.md`](./docs/models_reference.md), [`docs/running_and_reviewing.md`](./docs/running_and_reviewing.md), [`docs/notebook_runtimes.md`](./docs/notebook_runtimes.md), [`docs/troubleshooting.md`](./docs/troubleshooting.md), [`docker/Dockerfile`](./docker/Dockerfile), [`Makefile`](./Makefile) (`EXPORT_ARGS`), [`.github/workflows/ci.yml`](./.github/workflows/ci.yml), the notebooks' bootstrap `EXTRAS` lists, [`errors.py`](./src/scale_forecasting/errors.py) (`EXTRA_MODULES`), then `make lock` (enforced by [`tests/unit/test_packaging_extras.py`](./tests/unit/test_packaging_extras.py) and [`tests/unit/test_core_install.py`](./tests/unit/test_core_install.py)) |
| **AI Agent Surfaces & MCP** | [`agent_surfaces.py`](./src/scale_forecasting/agent_surfaces.py) & [`mcp.py`](./src/scale_forecasting/mcp.py) | **6 generated agent files** (`docs/schemas/run_config.schema.json`, `docs/llms.txt`, `docs/llms-full.txt`, `skills/scale-forecasting/references/*.md`) · **1 portable skill** (`skills/scale-forecasting/SKILL.md`) · **1 built-in MCP server** (`7` resources, `9` tools) | Run `make agent-surfaces` (`python -m scale_forecasting.agent_surfaces --write`) after changing `RunConfig`, models, metrics, runtimes, views, or docs nav; enforced by [`tests/unit/test_agent_surfaces.py`](./tests/unit/test_agent_surfaces.py) and [`tests/unit/test_mcp_server.py`](./tests/unit/test_mcp_server.py) |
| **Public Python Modules** | [`src/scale_forecasting/**/*.py`](./src/scale_forecasting/) | All non-private modules | [`docs/api/*.md`](./docs/api/index.md), [`mkdocs.yml`](./mkdocs.yml) (enforced by [`tests/unit/test_api_docs_coverage.py`](./tests/unit/test_api_docs_coverage.py)) |

---

## 4. Writing Style, Markdown Tables & Mermaid Rendering Rules

All documentation across `README.md`, `docs/*.md`, and directory `README.md` files must meet publication-grade readability and rendering standards:

### 4.1. Prose & Code Examples
1. **High-Signal, Direct Technical Prose:** Lead with what a component does, why it exists, and how to use it. Avoid filler or vague claims.
2. **Directory `README.md` Architecture Maps:** Every directory (`src/scale_forecasting/**`, `configs/**`, `notebooks/`, `tests/`, `docker/`, `terraform/**`) maintains a `README.md` containing a visual Mermaid flow diagram and a table mapping every file in that directory to its responsibility.
3. **Executable JSON & CLI Snippets:**
   - Every full `RunConfig` JSON example in `README.md` and `docs/*.md` must pass `RunConfig.model_validate()` without validation errors or dropped-strategy warnings (e.g., if `ensemble.strategies` includes learned stackers `nnls`, `ridge`, or `xgb`, the snippet must also include `"backtest": {"enabled": true, ...}`).
   - Every `python -m scale_forecasting.<module>` CLI command and `configs/<name>.json` path in documentation must reference a real module and file on disk.
4. **Symlink-Aware Links in `notebooks/README.md`:**
   - Because `docs/notebooks` is a symlink to `../notebooks` for MkDocs rendering, links from `notebooks/README.md` to pages in `docs/` must use `../<page>.md` (e.g., `../notebook_runtimes.md`) so `mkdocs build --strict` resolves them cleanly.

### 4.2. Mermaid Diagram Standards
1. **Supported Diagram Types Only:** Use `flowchart TD`, `flowchart LR`, `sequenceDiagram`, `stateDiagram-v2`, `classDiagram`, or `erDiagram`.
2. **Mandatory Quoting of Special Characters:** Always wrap node and `subgraph` labels in double quotes (`id["Label (Extra Info)"]` or `subgraph ID["Subgraph Title (Detail)"]`) whenever the label contains parentheses `()`, brackets `[]`, braces `{}`, colons `:`, semicolons `;`, ampersands `&`, mathematical symbols, or `<br/>` line breaks.
3. **Balanced Subgraphs:** Every `subgraph` declaration must have a matching `end` keyword on its own line.
4. **No Raw HTML Tags Beyond `<br/>`:** Use `<br/>` for line breaks inside quoted Mermaid node labels; avoid arbitrary HTML tags or unquoted markdown formatting inside nodes.

### 4.3. Markdown Table & LaTeX Math Standards
1. **Strict Column Count Parity:** Every row in a Markdown table (header, alignment separator `| :--- |`, and all body rows) must have the exact same number of unescaped pipe delimiters (`|`).
2. **Escaped Pipes Inside Cells:** Any literal pipe inside a table cell (such as union types `"spark" \| "ray"` or absolute value bars) must be escaped as `\|`.
3. **LaTeX / Dollar-Sign Hygiene:** Inline math uses `$...$` and display math uses `$$...$$`. Always escape literal currency dollar signs as `\$` (or wrap shell variables like `$PROJECT_ID` and prices in backticks) so two `$` characters in the same paragraph never corrupt prose into math mode.

---

## 5. Automated Review & Verification Protocol ("Definition of Done")

Never declare a feature, bugfix, documentation update, or phase complete until **all applicable verification gates below have been executed and shown passing**. During review rounds, agents should run these commands directly before signing off.

### Gate 1: Fast Consistency & Documentation Tripwires (seconds, runs in `.githooks/pre-commit`)
Run this after *any* code, config, or documentation edit:
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
- **`test_validation_ledger.py`:** Verifies every smoke config (`01`–`42`), root demo config (`20`), and notebook (`11`) has a valid row in `docs/validation.md` whose architecture axes match current code.
- **`test_config_coverage.py`:** Verifies all reachable `Literal` and `bool` values on `RunConfig` are proven live or exercised offline.
- **`test_docs_integrity.py`:** Audits all 83+ `.md` files for valid `RunConfig` JSON examples, valid relative links and config paths, valid `python -m` module references, balanced Markdown table columns, valid Mermaid syntax, absence of deprecated parameter names, backticked exception-class names that resolve to real classes in `errors.py`, and dynamic model/metric/view/smoke count parity.
- **`test_api_docs_coverage.py`:** Verifies every public Python module has a corresponding `docs/api/*.md` page and `mkdocs.yml` nav entry.
- **`test_test_dependencies_declared.py`:** Verifies every third-party module imported anywhere under `tests/` (including lazy, function-level imports) belongs to a distribution that CI's `uv sync --frozen --all-extras` installs, computed from `uv.lock`. A package that is only in your `.venv` because of `make docs` or an ad-hoc `uv pip install` is **not** declared; add it to `[dependency-groups].dev` (or an extra) in `pyproject.toml` and run `make lock`.
- **`test_notebook_hygiene.py`:** Scans every notebook's sources **and persisted outputs** (stream text, text/JSON display data, tracebacks) for the identifiers §1 rule 4 forbids — e-mail addresses, personal or corporate home paths, internal hostnames and short links, credential material. The failure message names the notebook, cell, and pattern, never the matched text. Fix the cell (silence the warning at its source, clear or re-run the output); do not widen the allowlists.
- **`test_packaging_extras.py`:** Holds `pyproject.toml` to the §3 extras layout (pure core, `[gcp]` as the single home of every Google client, composed `spark`/`ray`/`models-automl`/`models`/`all`, each floor declared once) and verifies every surface that names an extra — Makefile, `ci.yml`, Dockerfile, `requirements.txt` header, Markdown, notebook bootstraps, `errors.EXTRA_MODULES`, the model→extra table — names one that exists. Adding or renaming an extra means updating all of them in the same change.
- **`test_agent_surfaces.py`:** Enforces zero drift between live Python reflection (`RunConfig`, model/metric/runtime/view catalogs, `mkdocs.yml`) and the 6 generated agent surface files (`docs/schemas/run_config.schema.json`, `docs/llms.txt`, `docs/llms-full.txt`, and `skills/scale-forecasting/references/*.md`), validates all 62 shipped configs against the JSON Schema, and checks `skills/scale-forecasting/SKILL.md` and `gemini-extension.json`.

### Gate 2: Formatting, Linting, Type-Checking, Lock Drift & Strict MkDocs Site Build
```bash
.venv/bin/ruff format --check src/ tests/
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/scale_forecasting
make lock-check
.venv/bin/mkdocs build --strict
```
- The first four are also run by `.githooks/pre-push` (enabled with `make hooks`), with exactly the flags CI uses, so a push can never be the first place they fail.
- **ruff runs `E`, `F`, `I`, `B`, `UP`, `SIM`, `C4`, `PIE`, `PERF`, `RUF`, and `BLE`** (`[tool.ruff.lint]` in `pyproject.toml`, each family and the one ignored rule explained there). `BLE` means every `except Exception` carries `# noqa: BLE001 - <why this catch is deliberate>`; a broad catch that re-raises or logs with `exc_info` needs no marker. Fix a report rather than adding a `noqa`; when a `noqa` is the right answer, it carries its reason on the same line.
- **mypy is a zero-error gate on `src/` only.** The package ships `py.typed`, so every public signature is a promise to downstream type-checkers. Fix a report by narrowing (a small typed helper, an `isinstance` branch, a `Protocol` for a duck-typed estimator) rather than by `cast`, `Any`, or `# type: ignore`; tests stay dynamically typed on purpose.
- After editing `pyproject.toml` or any test import, run `make ci-offline`: it syncs a second environment (`.venv-ci`) with the CI `offline` job's exact command and runs the full offline gate there. Your working `.venv` is a superset of CI's and cannot reproduce a missing-package failure.

### Gate 3: Offline Unit & Contract Test Suite
```bash
.venv/bin/pytest tests/unit/ -q
```
- CI's `offline` job, `make test`, and `make ci-offline` run the same selection with `--cov` (the Makefile's `OFFLINE_PYTEST`; `tests/unit/test_packaging_extras.py` holds the three identical). Line coverage of `src/scale_forecasting` must stay at or above `fail_under` in `[tool.coverage.report]` — **85 %** at v1.0.0, set just under the 85.77 % measured when the floor was introduced. The floor is a ratchet: raise it when coverage rises; never lower it, never add `omit`, never move it onto a command line. New cloud-launch code that only live smokes can exercise is the one accepted reason coverage moves down, and it is paid for by raising coverage elsewhere, not by lowering the number.

### Gate 4: Live Cloud & BigQuery Output Verification (When Touching Runtimes, Engines, or Smokes)
When validating a runtime or smoke configuration on Google Cloud:
1. Confirm the run reaches `status = 'SUCCESS'` in `run_registry` with `n_failed = 0`.
2. Query BigQuery directly (`run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof`, and `v_model_leaderboard`) to verify that every expected `(ts_id, model_type)` cell produced non-null forecasts, finite evaluation metrics, and populated `$.sizing` / `$.sizing_executed` telemetry.
3. Verify zero orphaned cloud compute resources remain after run completion (`gcloud compute instances list`, `gcloud dataproc batches list`, `gcloud ai custom-jobs list`, `gcloud container clusters list`).
4. Record the live `run_id`, date, and architecture axes in [`docs/validation.md`](./docs/validation.md) and re-run Gate 1.
