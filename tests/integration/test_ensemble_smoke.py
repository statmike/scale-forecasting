"""Live ensemble-orchestration smoke (BUILD B5, ``@gcp``).

Runs :func:`scale_forecasting.main.run` end-to-end against live GCP with **ensembles enabled**, and
asserts the B5 contract the offline tests can't reach: after both engines join, the ensembler
blends the base forecasts (a true beyond-data forecast into ``forecast_predictions``), scores each
consensus on the **backtest OOF window** (post-C2 the predictions have no actuals to join — the
ensemble earns its metric on exactly the window the base models are scored on), and lands
``ensemble_*`` rows on the **same ``v_model_leaderboard``** as the base Spark + BigQuery models —
under one shared ``run_id``, with a non-NULL ``mean_wape`` (i.e. the OOF-consensus scoring actually
produced a metric, not an empty row).

This is the missing leaderboard link (``forecast_metadata WHERE fold_id IS NULL`` for each
``ensemble_*`` pseudo-model). Once those rows exist, the ensembles surface on the leaderboard with
**no view change** — that automatic surfacing is what this test proves.

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). Because ensembles run
inside :func:`main.run`, this launches a real Dataproc Serverless batch (the Spark model) *and* the
in-BigQuery engine, so it carries the same spend + latency as the orchestration smoke. Run::

    SF_PROJECT_ID=statmike-scale-forecasting \\
    SF_CONNECTION=statmike-scale-forecasting.us-central1.sf-iceberg \\
    SF_WAREHOUSE_URI=gs://statmike-scale-forecasting-warehouse/warehouse \\
    SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_ensemble_smoke.py

**Self-contained data.** Like the B3 native + Arc B orchestration smokes, this seeds its own tiny
univariate scratch ``source_series`` table and tears it down after — "the test owns its data".
``run_name`` varies per invocation so the deterministic ``run_id`` is unique (append-only cell
tables can't be DELETE-d while buffered).

The ensemble uses **calculated** strategies only (mean / median / inverse_error): post-C2 both the
base Spark and BigQuery models forecast the same true future into ``forecast_predictions`` and
score the same ``backtest_oof`` folds, so the calculated blends have overlapping base rows in both
spaces. Scoring happens in OOF space (where ``y_true`` lives). (Learned strategies are covered by
the offline unit tests.)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp

_SPARK_MODEL = "theta"
_NATIVE_MODELS = ["arima_plus", "timesfm"]
_BASE_MODELS = [_SPARK_MODEL, *_NATIVE_MODELS]
_STRATEGIES = ["mean", "median", "inverse_error"]
_ENSEMBLE_MODELS = [f"ensemble_{s}" for s in _STRATEGIES]
_SERIES_LIMIT = 10
_HORIZON = 28
_HISTORY = 730  # ~2 years daily → training window > 1 year, so ARIMA_PLUS holidays apply
_SCRATCH_TABLE = "b5_ensemble_smoke_source"


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


@pytest.fixture(scope="module")
def scratch_source(settings: Settings) -> Iterator[str]:
    """Seed a tiny univariate scratch ``source_series`` table, yield its name, drop it after.

    The same generator the production seed uses; loaded as a plain BQ table both the Spark connector
    and the BigQuery engine read. Truncate-on-write so a rerun is clean.
    """
    from google.cloud import bigquery

    from scale_forecasting.data_gen.generator import GenConfig, generate_panel
    from scale_forecasting.data_gen.seed_spark import _to_source_rows

    client = bigquery.Client(project=settings.project_id)
    table_ref = settings.table_ref(_SCRATCH_TABLE)

    gen = GenConfig(history=_HISTORY, freq="D", start="2021-01-01", holidays=("US",))
    panel = generate_panel(_SERIES_LIMIT, gen, seed=7)
    rows = _to_source_rows(panel, ("US",))
    rows = rows.astype({"y": "float64", "is_holiday": "bool"})

    schema = [
        bigquery.SchemaField("ts_id", "STRING"),
        bigquery.SchemaField("ds", "DATE"),
        bigquery.SchemaField("y", "FLOAT"),
        bigquery.SchemaField("archetype", "STRING"),
        bigquery.SchemaField("is_holiday", "BOOL"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(rows, table_ref, job_config=job_config).result()
    try:
        yield _SCRATCH_TABLE
    finally:
        client.delete_table(table_ref, not_found_ok=True)


def _cfg(source_table: str) -> RunConfig:
    # A per-invocation run_name → a unique deterministic run_id (see module docstring).
    return RunConfig(
        run_name=f"b5 ensemble smoke {int(time.time())}",
        python_runtime="spark",
        spark_method="explode",
        data={"source_table": source_table, "horizon": _HORIZON, "series_limit": _SERIES_LIMIT},
        models=_BASE_MODELS,
        features={"holidays": ["US"]},
        # Backtest ON so the Spark path emits an OOF metric panel comparable to the natives (the
        # inverse_error strategy also weights off OOF error). A small fold count keeps it cheap.
        backtest={"enabled": True, "n_folds": 2, "horizon": _HORIZON, "step": _HORIZON},
        ensemble={"enabled": True, "strategies": _STRATEGIES},
    )


def _rows(client: Any, sql: str, run_id: str) -> list[Any]:
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )
    return list(client.query(sql, job_config=job_config).result())


def _poll_rows(client: Any, sql: str, run_id: str, predicate: Any) -> list[Any]:
    """Write API rows are async-visible; poll briefly until ``predicate(rows)`` holds."""
    rows: list[Any] = []
    for _ in range(12):
        rows = _rows(client, sql, run_id)
        if predicate(rows):
            return rows
        time.sleep(3)
    return rows


def test_ensemble_orchestration_smoke(settings: Settings, scratch_source: str) -> None:
    from google.cloud import bigquery

    from scale_forecasting import main

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(scratch_source)
    run_id = make_run_id(cfg)
    d = settings.dataset_ref

    # One call: Spark ∥ BigQuery under one run_id, then B5 ensembles fire after the join.
    returned = main.run(cfg)
    assert returned == run_id

    # Header closed COMPLETED — the ensembles are part of the run's success contract, so a scoring
    # failure would have flipped this FAILED (and re-raised).
    header = next(
        iter(
            _rows(
                client,
                f"SELECT status, runtime_seconds FROM `{d}.run_registry` WHERE run_id=@run_id",
                run_id,
            )
        )
    )
    assert header.status == "COMPLETED"
    assert header.runtime_seconds > 0

    # The payoff: base Spark + base BigQuery + every ensemble_* on one leaderboard, one run_id.
    expected = set(_BASE_MODELS) | set(_ENSEMBLE_MODELS)
    board = _poll_rows(
        client,
        f"SELECT model_type, compute_engine, n_cells, mean_wape "
        f"FROM `{d}.v_model_leaderboard` WHERE run_id=@run_id",
        run_id,
        lambda rows: {r.model_type for r in rows} >= expected,
    )
    by_model = {r.model_type: r for r in board}
    assert set(by_model) >= expected, f"missing: {expected - set(by_model)}"

    # Base models keep their engine tags; ensembles are tagged 'ensemble' and scored (non-NULL).
    assert by_model[_SPARK_MODEL].compute_engine == "spark"
    for m in _NATIVE_MODELS:
        assert by_model[m].compute_engine == "bigquery", m
    for m in _ENSEMBLE_MODELS:
        row = by_model[m]
        assert row.compute_engine == "ensemble", m
        assert row.mean_wape is not None, f"{m} scored no metric — actuals join produced nothing"
        assert row.n_cells > 0, m

    # C4 — two ensemble configs coexist under one run_id, distinctly keyed by ensemble_id. Re-run
    # the ensemble stage over the *same* base predictions with a different strategy set (the
    # standalone path, no base recompute). It must land under a *new* ensemble_id beside the first,
    # never overwriting it (append-only) and never colliding on model_type.
    from scale_forecasting.ensemble_run import _override_ensemble, run_ensembles
    from scale_forecasting.registry.ids import make_ensemble_id

    first_id = make_ensemble_id(cfg.ensemble)
    cfg2 = _override_ensemble(cfg, ["mean"])  # a subset → a different ensemble_id
    second_id = make_ensemble_id(cfg2.ensemble)
    assert second_id != first_id
    run_ensembles(cfg2, run_id, settings=settings)

    keyed = _poll_rows(
        client,
        f"SELECT ensemble_id FROM `{d}.v_model_leaderboard` "
        f"WHERE run_id=@run_id AND model_type='ensemble_mean'",
        run_id,
        lambda rows: {r.ensemble_id for r in rows} >= {first_id, second_id},
    )
    landed = {r.ensemble_id for r in keyed}
    assert landed >= {first_id, second_id}, (
        f"both ensemble configs should coexist; saw {landed}, want {{{first_id}, {second_id}}}"
    )
