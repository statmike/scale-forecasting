# Test Suite (`tests/`)

`scale-forecasting` uses a three-tier test architecture designed so that **every core invariant, model contract, SQL query, sizing formula, and documentation claim is verified offline in minutes** (`make test`), while live cloud integration and end-to-end multi-runtime smoke runs have dedicated, reproducible harnesses.

```mermaid
flowchart TB
    subgraph offline["Tier 1: Offline Gate (tests/unit/ + tests/smokes/test_*.py)"]
        direction LR
        unit["Pure Unit & Seam Tests<br/>~4,800+ tests · Zero cloud/Spark/Ray required<br/>(make test)"]
        contracts["Catalogue Contract Suites<br/>test_models_contract.py (all 37 models)<br/>test_worker_model_matrix.py (full run_cell)<br/>test_metrics_contract.py (all 21 metrics)"]
        snaps["Pinned Golden Snapshots (snapshots/)<br/>SQL DDL & views · BQML SQL<br/>run_id digests · numeric golden panel"]
        trips["Consistency Tripwires (pre-commit hook)<br/>test_validation_ledger.py · test_config_coverage.py<br/>test_docs_integrity.py · test_api_docs_coverage.py<br/>test_test_dependencies_declared.py · test_notebook_hygiene.py<br/>test_packaging_extras.py · test_agent_surfaces.py · test_smoke_configs.py"]
        guards["Architectural Guards<br/>test_code_delivery.py · test_launch_point_lean.py<br/>test_source_conventions.py"]
        push["Deterministic CI Half (pre-push hook)<br/>ruff format --check · ruff check · mypy<br/>uv lock --check · requirements export drift"]
    end

    subgraph integ["Tier 2: Live Integration Tests (tests/integration/)"]
        it["@gcp · @spark · @ray Pytest Suites<br/>Registry round-trips · Engine smokes<br/>Headless Colab notebook acceptance"]
    end

    subgraph smokes["Tier 3: Live System Validation Smokes (tests/smokes/)"]
        sh["smoke_harness.py & airflow_smoke.py<br/>Executes configs/smokes/01..42 on live GCP,<br/>verifies BigQuery rows, metrics & teardown"]
    end

    offline --> integ --> smokes
```

---

## Running the Tests

### 1. Fast Consistency Tripwires (pre-commit hook, seconds)

Two tracked hooks in [`.githooks/`](../.githooks/) run automatically once enabled via `make hooks` (one-time per clone):

- **`pre-commit`** runs the AGENTS.md Gate 1 tripwires — the validation ledger, config coverage, docs integrity, API-docs coverage, test-dependency declaration, notebook hygiene, packaging extras, agent-surface sync, and smoke-config validity. They compare prose, declarations, and persisted notebook outputs against code and policy, so an edit made after a green `make test` can turn them red with no code change.
- **`pre-push`** runs the deterministic half of CI at the moment a commit leaves the machine — `ruff format --check`, `ruff check`, `mypy src/scale_forecasting` (zero errors; the package ships `py.typed`), `uv lock --check`, and a `docker/requirements.txt` export-drift check (same flags as `make lock`, read from the Makefile's `EXPORT_ARGS`).

```bash
make hooks  # one-time setup per clone
uv run pytest tests/unit/test_validation_ledger.py tests/unit/test_config_coverage.py tests/unit/test_docs_integrity.py tests/unit/test_api_docs_coverage.py tests/unit/test_test_dependencies_declared.py tests/unit/test_notebook_hygiene.py tests/unit/test_packaging_extras.py tests/unit/test_agent_surfaces.py tests/smokes/test_smoke_configs.py -q
```

To reproduce the CI `offline` job's environment exactly (a second `.venv-ci` synced with `uv sync --frozen --all-extras` and nothing else — no `docs` group, no ad-hoc installs), run `make ci-offline`. It catches the one failure shape your working venv cannot: a test that imports a package only your machine has.

### 2. Full Offline Gate (`make test`)

Runs `ruff format --check`, `ruff check` (pyflakes, pycodestyle, isort, bugbear, pyupgrade, plus `SIM`, `C4`, `PIE`, `PERF`, `RUF`, and `BLE` — every `except Exception` must carry `# noqa: BLE001` and a reason), `mypy` over `src/` (`make typecheck` on its own), and all offline unit, contract, and snapshot tests (deselecting `@gcp`, `@spark`, and `@ray` markers) **under the coverage floor** — the CI `offline` job, step for step. The pytest step runs with `--cov`; line coverage of `src/scale_forecasting` must stay at or above `fail_under` in `pyproject.toml`'s `[tool.coverage.report]` (85 %, set just under the 85.77 % measured when it was introduced). The floor is a ratchet: raise it when coverage rises, never lower it. The uncovered tail is the cloud-launch code the live smoke ledger proves instead.

```bash
make test
# Or directly with pytest (the exact command is the Makefile's OFFLINE_PYTEST):
uv run pytest -m "not gcp and not spark and not ray" -q --cov --cov-report=term-missing
```

### 3. Live Cloud Integration & Smoke Suites

Requires an active GCP deployment and `SF_*` environment variables exported:

```bash
# Run live BigQuery/GCP integration tests
uv run pytest -m gcp tests/integration/

# Run headless Colab Enterprise acceptance across all 11 notebooks
uv run pytest -m gcp tests/integration/test_notebook_acceptance.py

# Run a numbered end-to-end smoke config from configs/smokes/
uv run python -m tests.smokes.smoke_harness --only 01_serverless_cpu
```

---

## Directory Structure

| Directory / File | Purpose |
| :--- | :--- |
| [`conftest.py`](./conftest.py) | Shared pytest fixtures, marker registrations (`gcp`, `spark`, `ray`, `airflow`), and autouse guards that prevent offline unit tests from reading ambient `SF_*` environment variables or making accidental cloud calls. |
| **[`unit/`](./unit/)** | 80+ test modules covering every module in `src/scale_forecasting/`. Includes contract suites (`test_models_contract.py`, `test_metrics_contract.py`, `test_worker_model_matrix.py`), Forecaster Workbench, Causal Guardrails, Cross-Temporal Coherence, Inventory Policy & Lifecycle Clearance/Contract Optimization suites (`test_phase_g1_workbench.py`, `test_phase_g2_hygiene_and_quantiles.py`, `test_phase_g3_diagnostics_and_scenarios.py`, `test_phase_g4_lifecycle_and_report.py`, `test_inventory_policy.py`, `test_monotonicity.py`, `test_holiday_window.py`, `test_elasticity.py`, `test_scenarios.py`, `test_track_j_e2e.py`, `test_temporal_reconciliation.py`, `test_simplex.py`, `test_goal_seek.py`, `test_track_k_e2e.py`, `test_clearance_optimizer.py`), architectural guards (`test_code_delivery.py`, `test_launch_point_lean.py`, `test_source_conventions.py`, `test_core_install.py`), MCP protocol tests (`test_mcp_server.py`), and doc/config/dependency/hygiene/agent-surface tripwires (`test_validation_ledger.py`, `test_config_coverage.py`, `test_docs_integrity.py`, `test_api_docs_coverage.py`, `test_test_dependencies_declared.py`, `test_notebook_hygiene.py`, `test_packaging_extras.py`, `test_agent_surfaces.py`). |
| **[`unit/snapshots/`](./unit/snapshots/README.md)** | Golden snapshot files pinning generated BigQuery DDL (`ddl_deployment.sql`, `ddl_drop.sql`), analyst views (`views.sql`), BQML query shapes (`bigquery_native.sql`), deterministic `run_id` digests (`run_ids.json`), and numerical forecast outputs (`golden_panel.json`). |
| **[`integration/`](./integration/)** | Live integration tests against BigQuery, Dataproc Spark Connect/Serverless, Vertex AI Ray, `Registry` operations (`test_registry_ops_live.py`), and headless notebook execution (`test_notebook_acceptance.py`). |
| **[`smokes/`](./smokes/)** | Live end-to-end smoke driver (`smoke_harness.py`), Cloud Composer / Airflow DAG smoke driver (`airflow_smoke.py`), and offline tests for the harness and smoke configs (`test_harness.py`, `test_airflow_smoke.py`, `test_smoke_configs.py`). |

---

## Key Architectural Tripwires

Several unit test modules enforce structural guarantees across the repository:
- **[`test_validation_ledger.py`](./unit/test_validation_ledger.py):** Verifies that every shipped config (`configs/*.json` and `configs/smokes/*.json`) and every notebook (`notebooks/*.ipynb`) is recorded in [`docs/validation.md`](../docs/validation.md), that `CURRENT` rows match current architecture axes, and that `run_id` citations in [`docs/quota_and_scale.md`](../docs/quota_and_scale.md) resolve to valid ledger entries.
- **[`test_config_coverage.py`](./unit/test_config_coverage.py):** Joins every enumerable value reachable from `RunConfig` against shipped configs and `CURRENT` ledger rows, ensuring 100% of config options are either proven live or backed by a verified unit test span.
- **[`test_docs_integrity.py`](./unit/test_docs_integrity.py):** Audits every `.md` file in the repository for Markdown table column parity, Mermaid diagram syntax and label quoting, valid `RunConfig` JSON examples, valid relative links and `configs/*.json` paths, valid `python -m` CLI module references, absence of deprecated parameter names, backticked exception-class names that resolve to real classes in [`errors.py`](../src/scale_forecasting/errors.py) (or Python builtins), and dynamic model/metric/view/smoke count alignment.
- **[`test_prebreak_snapshots.py`](./unit/test_prebreak_snapshots.py):** Locks the deterministic `<slug>-<12hex>` `run_id` digests of all shipped configurations and the numerical outputs of the golden panel so refactoring never alters run identity or model math silently.
- **[`test_code_delivery.py`](./unit/test_code_delivery.py):** Ensures `docker/Dockerfile` and `docker/cloudbuild.yaml` never bake `src/scale_forecasting` into the container image.
- **[`test_launch_point_lean.py`](./unit/test_launch_point_lean.py):** Ensures importing model classes and planning DAGs never eagerly imports heavy compute libraries (`torch`, `statsmodels`, `xgboost`, `lightgbm`, `prophet`, `neuralprophet`), keeping thin submission environments (such as Cloud Composer workers) lean.
- **[`test_source_conventions.py`](./unit/test_source_conventions.py):** Static checks ruff does not make: every relative import (including the lazy, function-level ones on the submit path) names a module that exists; no module name shadows a stdlib module (a Composer plugins delivery puts inner directories on `sys.path`); the four deliberately duplicated terminal-status sets agree; and cross-module imports of underscore-prefixed names can only **ratchet down** — the test carries an exact allowlist of today's debt, fails on any new entry (promote the helper instead) and on any stale entry (so the list is always a true picture).
- **[`test_api_docs_coverage.py`](./unit/test_api_docs_coverage.py):** Ensures every module re-exported by `scale_forecasting.__init__` and every subpackage is documented in `docs/api/`, every `docs/api/*.md` page is wired into `mkdocs.yml`, and every folder `README.md` in the repository spine includes a Mermaid diagram.
- **[`test_test_dependencies_declared.py`](./unit/test_test_dependencies_declared.py):** Computes, from `uv.lock`, the closure of what the CI `offline` job installs (`uv sync --frozen --all-extras`: core dependencies, every extra, and the default dependency groups) and fails if any file under `tests/` imports a third-party module — at module level or lazily inside a function — whose distribution is outside that closure. Imports guarded by `try/except ImportError` or `pytest.importorskip` are exempt. It exists because a package present in a developer venv (via `make docs` or an ad-hoc install) but absent from CI's sync left `main` red for two merges.
- **[`test_notebook_hygiene.py`](./unit/test_notebook_hygiene.py):** Scans every notebook's markdown and code sources **and its persisted outputs** (stream text, `text/*` and JSON display data, tracebacks) for the identifiers [AGENTS.md §1 rule 4](../AGENTS.md) forbids in a public repository: e-mail addresses (service-account and RFC 2606 example domains excepted), personal or corporate home paths, internal hostnames and short links, and credential material. The demo project ID, bucket, and dataset names are deliberately allowed. Failures name the notebook, cell, and pattern — never the matched text, because CI logs are public. It exists because a library warning's first token is the absolute path of the file that raised it, which on a workstation is a home directory and a username; nothing else reads notebook outputs.
- **[`test_packaging_extras.py`](./unit/test_packaging_extras.py):** Reads `pyproject.toml` with `tomllib` and holds the extras layout to its documented shape: the core `dependencies` carry no Google client, plotting, Spark, Ray, torch, or model library; `[gcp]` is the single home of every Google client and no other extra redeclares one; `spark`, `ray`, and `models-automl` compose `[gcp]` rather than copying it; `models` is composed from exactly the `models-*` family extras; every version floor is declared once; `all` is the closure of everything. It then checks the surfaces that *name* extras — `Makefile EXPORT_ARGS`, `ci.yml`, `docker/Dockerfile`, the `docker/requirements.txt` header, every `scale-forecasting[...]` and `--extra` in the Markdown, every notebook bootstrap `EXTRAS` list, [`errors.EXTRA_MODULES`](../src/scale_forecasting/errors.py), and the model→extra table in `docs/models_reference.md` against each model's `optional_extra` — so a documented extra always exists and an existing extra is always documented. It exists because `[models-automl]` was documented for a release before it was declared.
- **[`test_core_install.py`](./unit/test_core_install.py):** Proves the pure-core promise in a subprocess that installs a meta-path finder refusing every optional top-level package (Google clients, pyspark, ray, matplotlib, torch, the model libraries): all 37 models and 21 metrics list, the playground forecasts, `main.run(..., dry_run=True)`, `Forecaster.dag()`, and `--dry-run` all succeed with nothing optional importable; and every cloud-facing entry point — `main.run`, `Forecaster.run()`, `--probe`, the submitters, `registry.ops`, plotting — raises `MissingExtraError` naming the right extra instead of a `ModuleNotFoundError`. The CI `core-install` job runs the same test in a genuinely bare virtualenv (`uv pip install .` plus pytest) so the finder is a fast local proxy, not the only evidence.
- **[`test_agent_surfaces.py`](./unit/test_agent_surfaces.py), [`test_mcp_server.py`](./unit/test_mcp_server.py) & [`test_bq_agent.py`](./unit/test_bq_agent.py):** Enforces zero drift between live Python reflection and all 6 generated agent surface files (`docs/schemas/run_config.schema.json`, `docs/llms.txt`, `docs/llms-full.txt`, and `skills/scale-forecasting/references/*.md`), validates all 62 shipped JSON configs against the JSON Schema, verifies `"$schema"` acceptance and `run_id` invariance, tests the built-in `stdio` JSON-RPC 2.0 MCP server across all 7 resources, 14 tools (`ask_forecast_registry`), and `--allow-launch` safety locks, and verifies self-documenting registry metadata parity (`metadata.py`), Knowledge Catalog Business Glossary caps (`<= 8` links/table), long-scan-ID collision safety, strict scope isolation, Golden SQL parameterization, and stateful `DataAgent` chat parsing.
- **[`test_clearance_optimizer.py`](./unit/test_clearance_optimizer.py):** Verifies Track Q.1 (`v1.9.0`) lifecycle markdown clearance, perishable expiry, warehouse space reclamation (`optimize_clearance`, `optimize_clearance_df`, `plot_clearance_schedule`), and multi-SKU vendor contract volume / Take-or-Pay MILP optimization (`optimize_contract_volume`, `optimize_contract_volume_df`) across exact analytical fixtures, cross-SKU cannibalization, hierarchical & `THieF` / `cross_temporal_mint` / `@transform_ilr` reconciliation (`hierarchy_rollup_df` additivity and aggregate-node filtering), 3-tier covariates with Frisch–Waugh–Lovell partialling & Double-ML `promo_elasticity_df`, all 5 model families / ensembles / OOF per-SKU champion selection / Optuna HPO / `@<tag>` modifiers, calibrated Online ACI / Conformal / Copula `CVaR` downside shortfall, and all 7 cloud runtime configurations.
- **[`test_auto_forecaster.py`](./unit/test_auto_forecaster.py):** Verifies the 20-panel Synthetic DGP Benchmark Bank + 9-dataset Public Open-Data Benchmark Suite (`benchmarks.py`), the 7-runtime parametric wall-clock & cost predictor (`profiling/duration.py`), the Glass-Box Auto-Forecaster (`Forecaster.build_config()` and `Forecaster.auto()` across `"first_light"`, `"wide_field"`, `"deep_sky"`, and `"deep_field"`), the 5 `n_series` scale regimes (`micro` to `hyperscale`), the 4-anchor baseline ladder (`baseline_value_df()`), hierarchical reconciliation value (`hierarchy_value_df()`), composable compound `@<tag>` variants, the Mode A AlphaEvolve policy (`auto_policy.py`), and the human-first release notes generator (`render_release_notes`).
- **[`test_evolve.py`](./unit/test_evolve.py):** Verifies the opt-in AlphaEvolve runtime harness (`scale_forecasting.evolve` and `Forecaster.evolve()`): AST security sandbox (`validate_candidate_ast`), `+1e6` future-perturbation causality detector (`verify_no_future_leakage`), all 4 `# EVOLVE-BLOCK` targets (`"features"`, `"ensemble"`, `"model"`, `"policy"`), the 3-Gate Anti-Overfitting Ratchet (including Gate 3 rejection on held-out Lockdown Fold degradation), `probe_alphaevolve_access()` pre-flight `ConfigError` guidance, `backend="vertex_gemini"` backup mutation loop, `EvolveResult.emit_airflow_dag()`, and `python -m scale_forecasting.evolve --dry-run`.
- **[`test_monotonicity.py`](./unit/test_monotonicity.py), [`test_holiday_window.py`](./unit/test_holiday_window.py), [`test_elasticity.py`](./unit/test_elasticity.py), [`test_scenarios.py`](./unit/test_scenarios.py) & [`test_track_j_e2e.py`](./unit/test_track_j_e2e.py):** Verifies Track J (`v1.5.0`) causal & structural econometric guardrails and full-chain What-If simulation across 4 test tiers: `features.monotonic_constraints` & `@monotone` (`lightgbm`, `xgboost`, `catboost`, `regression_lags` / `ridge_lags`), Double/Debiased ML `promo_elasticity_df` (`baseline_yhat`, `incremental_promo_lift`, `pull_forward_dip`, `cannibalization_effect`, `net_causal_elasticity`), multi-day `features.holiday_window` (`pre_holiday_days`, `post_holiday_days`), and 5-stage `simulate_scenarios(..., unit_cost=..., reconcile=True, include_inventory_impact=True)`.
- **[`test_temporal_reconciliation.py`](./unit/test_temporal_reconciliation.py), [`test_simplex.py`](./unit/test_simplex.py), [`test_goal_seek.py`](./unit/test_goal_seek.py) & [`test_track_k_e2e.py`](./unit/test_track_k_e2e.py):** Verifies Track K (`v1.6.0`) 3-mode spatial, temporal (`THieF`), and joint cross-temporal (`cross_temporal_mint` with `spatial_temporal_weight="auto"` and `sample_coherent_paths`) reconciliation, Aitchison compositional market-share simplex closure (`@transform_ilr`, `@transform_clr`, `simplex_sum`), and top-down Goal-Seek constrained QP target allocation (`allocate_target` with `locked_nodes`, capacity bounds, and `smoothness_weight`).
- **[`test_similarity.py`](./unit/test_similarity.py) & [`test_track_p2_cross_industry.py`](./unit/test_track_p2_cross_industry.py):** Verifies Track P (`v1.8.0` / `v1.8.1`) 7-lens & multi-modal time-series similarity (`hybrid`, `physics_attractor`, `spectral_phase`, `elastic_dtw`, `minirocket`, `structural_signature`, `foundation_latent`), stockout SKU substitution (`recommend_substitutes_df`), emerging trend detection (`detect_emerging_trends_df`), positive complementary halo ($\gamma_{ij} > 0$) & geometric ad-stock decay ($\lambda$), piecewise weather degree-day splines (`degree_day_cols`), Arps decline curves (`@arps_decline` & `EUR`), and multi-generation product supersession (`supersession_map`).
- **[`test_pathological_torture_suite.py`](./unit/test_pathological_torture_suite.py) & [`test_sql_injection_safety.py`](./unit/test_sql_injection_safety.py):** Verifies Track F.2 (`v1.8.2`) 12-case pathological input resilience (all-zero, $T=1$/$T=2$, interior `NaN`/`±inf`, zero-variance constant, $[10^{-9}, 10^{12}]$ dynamic range, negative $y$, duplicate/unsorted/gapped timestamps, collinear covariates, boundary simplex zeros, infeasible Goal-Seek bounds, and $O(N \cdot K)$ 100k-series canopy similarity scaling in $< 30\text{s}$ and $< 250\text{ MB}$) and BigQuery SQL identifier / `row_restriction` injection safety.
- **[`test_cross_version_resilience.py`](./unit/test_cross_version_resilience.py):** Verifies backward compatibility across `v1.0.0`, `v1.1.0`, and `v1.2.x` registry payloads — including `Forecaster.from_run_id()`, cross-version model borrowing (`match_reusable_models` and `render_borrow_model_sql`), `review_run()` / tidy frame builders on legacy rows lacking newer columns, and `diagnostics.py` (`compute_fva`, `compute_inventory_policy`, `compute_anomalies`, `compute_change_points`).
