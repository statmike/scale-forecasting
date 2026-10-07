"""The read surface over a run — everything selected back out of the registry.

What the review layer, the smokes and the notebooks read: the run summary, the model leaderboard,
prediction counts, per-cell timing, the stored config, live progress, and the metric aggregates.
Read-only — nothing here writes. The one shared coercion is `parse_ts`, which both the review and
probe age arithmetic go through.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .params import header_param
from .rows import METRIC_COLUMNS
from .tables import resolve_settings

if TYPE_CHECKING:
    from ..settings import Settings


def parse_ts(value: Any) -> datetime | None:
    """Coerce a registry timestamp to a timezone-aware UTC ``datetime``, or ``None`` if it isn't.

    A registry row's timestamp arrives as a ``datetime`` from the BigQuery client but as an ISO
    string from a JSON-shaped reader dict (and from every offline test), so both readers that do
    age arithmetic — `review._assemble_progress`'s quiet-time and
    `probes.reconcile._is_stale`'s escalation grace — need the same coercion. Pure and defensive:
    anything unparseable comes back ``None`` rather than raising, so a malformed timestamp costs
    a *signal*, never a monitor.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def read_run_summary(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_run_summary`` row for ``run_id`` (status + scaling/efficiency), or ``None``.

    The view already keeps one row per run (latest header wins), so this is a single-row lookup —
    the run-level answer to "how did this run go, and how efficiently?". Raises `RegistryError` on
    failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = f"SELECT * FROM `{resolved.registry_table_ref('v_run_summary')}` WHERE run_id=@run_id"
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_run_summary failed for run {run_id}: {exc}") from exc
    return dict(rows[0]) if rows else None


def read_leaderboard(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_model_leaderboard`` rows for ``run_id`` — one per (model, ensemble).

    Ordered by the mean decision-metric error (WAPE) ascending so the best model is first; a model
    with no scored metric (NULL) sorts last. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.registry_table_ref('v_model_leaderboard')}` "
        "WHERE run_id=@run_id ORDER BY mean_wape ASC NULLS LAST"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_leaderboard failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_prediction_counts(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, int]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return ``{model_type: forecast row count}`` for ``run_id`` from ``forecast_predictions``.

    The direct proof that fits produced forecasts (not just metadata): a model whose cells all
    failed writes metadata rows but **zero** predictions, so it still shows on the leaderboard yet
    has a count of 0 here. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "SELECT model_type, COUNT(*) AS n "
        f"FROM `{resolved.registry_table_ref('forecast_predictions')}` "
        "WHERE run_id=@run_id GROUP BY model_type"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_prediction_counts failed for run {run_id}: {exc}") from exc
    return {str(r["model_type"]): int(r["n"]) for r in rows}


def read_cell_groups(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """The per-cell outcome of ``run_id``, grouped to the shape the repair classifier reads.

    `retry_policy.classify_cell` looks at exactly four per-cell facts — did the cell write metadata,
    did it write predictions, what is its ``cell_status``, what is its ``error_class`` — so cells
    that agree on all four are guaranteed the same verdict. This groups by that tuple and returns
    one row per distinct combination with an ``n_cells`` weight and an ``example_ts_id``: a
    hundred-thousand-cell run comes back as a few dozen rows with nothing the classifier can use
    thrown away.

    Ensemble rows are excluded on both sides — an ensemble is calculated from cells rather than
    being one, and it is not resubmittable at this grain. Metadata is deduped latest-write-wins per
    cell, the same rule the serving views apply, so a re-run's second attempt is what gets
    classified. Note the shape only covers cells that wrote *something*: a cell whose job died
    before reaching it has no row here at all and must be derived as expected-minus-observed
    against the run's snapshot-pinned source. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    meta_ref = resolved.registry_table_ref("forecast_metadata")
    pred_ref = resolved.registry_table_ref("forecast_predictions")
    sql = (
        "WITH meta AS ("
        "SELECT ts_id, model_type, cell_status, error_class "
        f"FROM `{meta_ref}` "
        "WHERE run_id=@run_id AND fold_id IS NULL AND ensemble_id IS NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1"
        "), preds AS ("
        "SELECT DISTINCT ts_id, model_type "
        f"FROM `{pred_ref}` "
        "WHERE run_id=@run_id AND ensemble_id IS NULL"
        ") "
        "SELECT m.model_type AS model_type, m.cell_status AS cell_status, "
        "m.error_class AS error_class, p.ts_id IS NOT NULL AS has_predictions, "
        "COUNT(*) AS n_cells, MIN(m.ts_id) AS example_ts_id "
        "FROM meta m LEFT JOIN preds p ON p.ts_id=m.ts_id AND p.model_type=m.model_type "
        "GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_cell_groups failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_cell_timing(
    run_id: str, *, limit: int = 5000, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-cell wall-clock brackets for ``run_id`` — the fine grain under the per-job trace.

    Reads ``forecast_metadata`` directly (no view — this is a fine-grained, bounded read, not a
    curated roll-up): one row per full-fit cell (``fold_id IS NULL``) that recorded a wall-clock
    bracket (``cell_started_at IS NOT NULL`` — older rows predate the trace columns, so skipped),
    carrying ``ts_id`` / ``model_type`` / ``compute_engine`` / ``worker_id`` and the
    ``cell_started_at`` / ``cell_ended_at`` stamps. Writes are append-only + at-least-once, so it
    collapses to one row per cell (latest write wins) with the same grain as ``v_model_leaderboard``
    before returning. Ordered by ``cell_started_at`` and capped at ``limit`` rows (a 100k-cell run
    would swamp a trace plot; the cap keeps the read bounded — the per-job trace stays the
    whole-run view). Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "SELECT ts_id, model_type, compute_engine, worker_id, cell_started_at, cell_ended_at "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND fold_id IS NULL AND cell_started_at IS NOT NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "ORDER BY cell_started_at LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_cell_timing failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_run_config(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the validated ``raw_config`` a run landed under, as a dict — or ``None`` if never run.

    Reads the latest ``run_registry`` header's ``raw_config`` JSON column for ``run_id`` (the config
    *is* the experiment record). The review layer rebuilds a `config.RunConfig` from this to recover
    the run's plan — models per family, decision metric, ensemble strategies — so a monitor keyed on
    a bare ``run_id`` knows the *expected* work, not just what has landed. ``TO_JSON_STRING`` +
    ``json.loads`` sidesteps the client's native-JSON decoding so the shape is a plain dict. Raises
    `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        f"SELECT TO_JSON_STRING(raw_config) AS raw_config "
        f"FROM `{resolved.registry_table_ref('run_registry')}` "
        "WHERE run_id=@run_id ORDER BY created_at DESC LIMIT 1"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_run_config failed for run {run_id}: {exc}") from exc
    if not rows or rows[0]["raw_config"] is None:
        return None
    return json.loads(rows[0]["raw_config"])


def read_progress(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-model landed-cell counts + mean fit time for ``run_id`` — the live-progress read.

    One row per ``(model_type, ensemble_id)`` with ``n_cells_done`` (deduped full-fit cells that
    have landed) and ``mean_fit_seconds``. Deliberately light (COUNT + AVG, no quantiles) so a
    monitor can poll it cheaply while a run is in flight, at any scale. Progress is coarse-grained:
    cells land when a job's ``write_cells`` runs (often at job end), so counts step up per job
    rather than ticking per series — pair with the per-job status from `read_run_jobs` for the live
    picture. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "WITH deduped AS ("
        "  SELECT * FROM `" + resolved.registry_table_ref("forecast_metadata") + "`"
        "  WHERE run_id=@run_id AND fold_id IS NULL"
        "  QUALIFY ROW_NUMBER() OVER ("
        "    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id"
        "    ORDER BY created_at DESC) = 1"
        ") "
        "SELECT model_type, ensemble_id, "
        "COUNT(*) AS n_cells_done, AVG(fit_seconds) AS mean_fit_seconds "
        "FROM deduped GROUP BY model_type, ensemble_id"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_progress failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_metric_aggregates(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the full metric panel aggregated across series, per model — the finished-run stats.

    One row per ``(model_type, ensemble_id)``: ``n_series`` (deduped full-fit cells),
    ``mean_fit_seconds``, and for **every** metric in `METRIC_COLUMNS` its cross-series ``mean_``
    plus ``p10_``/``p50_``/``p90_`` (via ``APPROX_QUANTILES``). The aggregation runs server-side so
    it holds at 100k+ series without pulling per-series rows to the client — the review layer reads
    distribution shape from here, falling back to `read_cell_metrics` only for a bounded drill-down.
    The projection is generated from `METRIC_COLUMNS`, so it tracks the one metric source of truth.
    Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    metric_aggs = ", ".join(
        f"AVG({m}) AS mean_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(1)] AS p10_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(5)] AS p50_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(9)] AS p90_{m}"
        for m in METRIC_COLUMNS
    )
    sql = (
        "WITH deduped AS ("
        "  SELECT * FROM `" + resolved.registry_table_ref("forecast_metadata") + "`"
        "  WHERE run_id=@run_id AND fold_id IS NULL"
        "  QUALIFY ROW_NUMBER() OVER ("
        "    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id"
        "    ORDER BY created_at DESC) = 1"
        ") "
        "SELECT model_type, ensemble_id, ANY_VALUE(compute_engine) AS compute_engine, "
        "COUNT(*) AS n_series, AVG(fit_seconds) AS mean_fit_seconds, " + metric_aggs + " "
        "FROM deduped GROUP BY model_type, ensemble_id"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_metric_aggregates failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_cell_metrics(
    run_id: str, *, limit: int = 20000, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-series metric rows for ``run_id`` — the bounded drill-down under the aggregates.

    One deduped full-fit row per ``(ts_id, model_type, ensemble_id)`` carrying ``compute_engine`` /
    ``fit_seconds`` and the whole `METRIC_COLUMNS` panel — the raw material for per-series
    distribution plots and outlier hunts. Capped at ``limit`` rows (a 100k×N run would swamp the
    client and any plot); use `read_metric_aggregates` for the whole-run distribution shape and this
    for a bounded sample or a small run. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    metric_cols = ", ".join(METRIC_COLUMNS)
    sql = (
        "SELECT ts_id, model_type, ensemble_id, compute_engine, fit_seconds, " + metric_cols + " "
        "FROM `" + resolved.registry_table_ref("forecast_metadata") + "` "
        "WHERE run_id=@run_id AND fold_id IS NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_cell_metrics failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_arm_comparison(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Per-model rollup of the point-forecast arm choice and what it was worth.

    One row per ``model_type``: how many of its series shipped the raw arm, how many had that arm
    chosen for them rather than configured, how its band was calibrated, how many series the
    corrected arm beat the raw one on, and the mean/median relative margin over this run's
    ``decision_metric``. ``point_forecast_margin`` is signed the same way for every row — positive
    means the correction helped — so the counts and the average are comparable across models, and
    across series within a model that selected differently under ``point_forecast="auto"``.

    Aggregated server-side for the same reason as `read_metric_aggregates`: at 100k series this is
    the difference between a summary and a download. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "WITH deduped AS ("
        "  SELECT * FROM `" + resolved.registry_table_ref("forecast_metadata") + "`"
        "  WHERE run_id=@run_id AND fold_id IS NULL AND ensemble_id IS NULL"
        "  QUALIFY ROW_NUMBER() OVER ("
        "    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id"
        "    ORDER BY created_at DESC) = 1"
        ") "
        "SELECT model_type, "
        "ANY_VALUE(compute_engine) AS compute_engine, "
        # Not ANY_VALUE for the arm: under `point_forecast="auto"` the cells of one model can
        # legitimately disagree, and picking one at random to stand for all of them would report
        # the split as unanimous. The count is the honest summary, and it degrades correctly —
        # a fleetwide arm gives 0 or n_series, which reads as the unanimity it is.
        "COUNTIF(point_forecast_source = 'raw') AS n_raw_arm, "
        "COUNTIF(STARTS_WITH(point_forecast_decision, 'auto')) AS n_auto_decided, "
        "ANY_VALUE(interval_calibration) AS interval_calibration, "
        "COUNT(*) AS n_series, "
        "COUNTIF(point_forecast_margin > 0) AS n_corrected_wins, "
        "COUNTIF(point_forecast_margin IS NOT NULL) AS n_compared, "
        "AVG(point_forecast_margin) AS mean_margin, "
        "APPROX_QUANTILES(point_forecast_margin, 10)[OFFSET(5)] AS median_margin "
        "FROM deduped GROUP BY model_type ORDER BY model_type"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_arm_comparison failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_coverage_by_step(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Achieved interval coverage per model per horizon step, over the whole out-of-fold panel.

    The receipt for the calibrated band, and the one number a fleet average hides: a band can sit
    at nominal on average while step 1 is over-covered and step 28 badly under-covered, which is
    precisely what a horizon-flat band does. ``mean_width`` comes along because coverage alone is
    gameable — an infinitely wide band covers everything.

    Rows with a NULL ``horizon_step`` are excluded rather than pooled — folding them into a
    ``NULL`` bucket would put an unordered bar in the middle of a chart that is otherwise ordered
    by distance. Both engines project the step now, so on a current run the exclusion should catch
    nothing; it stays because runs written before they did are still in the table.
    Raises `RegistryError`.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "SELECT model_type, horizon_step, "
        "COUNT(*) AS n, "
        "AVG(CAST(y_true BETWEEN yhat_lower AND yhat_upper AS INT64)) AS coverage, "
        "AVG(yhat_upper - yhat_lower) AS mean_width "
        "FROM `" + resolved.registry_table_ref("backtest_oof") + "` "
        "WHERE run_id=@run_id AND horizon_step IS NOT NULL "
        "AND yhat_lower IS NOT NULL AND yhat_upper IS NOT NULL "
        "GROUP BY model_type, horizon_step ORDER BY model_type, horizon_step"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_coverage_by_step failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_backtest_coverage(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_backtest_coverage`` rows for ``run_id`` — the panel behind each model's score.

    One row per ``(model_type, ensemble_id, backtest_status, n_folds_achieved, backtest_refit)``, so
    a model with a ragged panel returns several. Ordered so a reader walking the rows sees each
    model's healthiest cohort first. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.registry_table_ref('v_backtest_coverage')}` "
        "WHERE run_id=@run_id "
        "ORDER BY model_type, ensemble_id NULLS FIRST, n_folds_achieved DESC NULLS LAST"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_backtest_coverage failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_comparable_leaderboard(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_model_leaderboard_comparable`` rows — one per (model, ensemble) on ``run_id``.

    The holdout-fold, pooled-WAPE ranking (see `registry.views`), ordered best-first with an
    unscorable model last. Read ``n_series`` alongside the score: rows that disagree on it are not
    yet comparable, whatever the ranking says. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.registry_table_ref('v_model_leaderboard_comparable')}` "
        "WHERE run_id=@run_id ORDER BY pooled_wape ASC NULLS LAST"
    )
    params = [header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_comparable_leaderboard failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_predictions(
    run_id: str,
    *,
    ts_ids: list[str] | None = None,
    models: list[str] | None = None,
    limit: int = 10000,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Return deduped ``forecast_predictions`` rows for ``run_id``, optionally filtered by series
    and model.

    Deduplicates with the same ``QUALIFY ROW_NUMBER()`` rule as the serving views (latest write
    wins per ``(run_id, ts_id, model_type, ensemble_id, forecast_date)``) and orders by
    ``ts_id, model_type, forecast_date``. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    clauses = ["run_id=@run_id"]
    params: list[Any] = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    if ts_ids:
        clauses.append("ts_id IN UNNEST(@ts_ids)")
        params.append(bigquery.ArrayQueryParameter("ts_ids", "STRING", list(ts_ids)))
    if models:
        clauses.append("model_type IN UNNEST(@models)")
        params.append(bigquery.ArrayQueryParameter("models", "STRING", list(models)))
    where_sql = " AND ".join(clauses)
    sql = (
        "SELECT ts_id, model_type, ensemble_id, compute_engine, forecast_date, "
        "yhat, yhat_lower, yhat_upper, TO_JSON_STRING(explanations) AS explanations "
        f"FROM `{resolved.registry_table_ref('forecast_predictions')}` "
        f"WHERE {where_sql} "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, ensemble_id, forecast_date "
        "ORDER BY created_at DESC NULLS LAST) = 1 "
        "ORDER BY ts_id, model_type, forecast_date LIMIT @limit"
    )
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_predictions failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_oof(
    run_id: str,
    *,
    ts_ids: list[str] | None = None,
    models: list[str] | None = None,
    limit: int = 10000,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Return deduped ``backtest_oof`` rows for ``run_id``, optionally filtered by series and model.

    Deduplicates per ``(run_id, ts_id, model_type, ensemble_id, fold_id, forecast_date)`` and
    orders by ``ts_id, model_type, fold_id, forecast_date``. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    clauses = ["run_id=@run_id"]
    params: list[Any] = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    if ts_ids:
        clauses.append("ts_id IN UNNEST(@ts_ids)")
        params.append(bigquery.ArrayQueryParameter("ts_ids", "STRING", list(ts_ids)))
    if models:
        clauses.append("model_type IN UNNEST(@models)")
        params.append(bigquery.ArrayQueryParameter("models", "STRING", list(models)))
    where_sql = " AND ".join(clauses)
    sql = (
        "SELECT ts_id, model_type, ensemble_id, fold_id, cutoff_date, forecast_date, "
        "horizon_step, y_true, yhat, yhat_lower, yhat_upper "
        f"FROM `{resolved.registry_table_ref('backtest_oof')}` "
        f"WHERE {where_sql} "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, ensemble_id, fold_id, forecast_date "
        "ORDER BY created_at DESC NULLS LAST) = 1 "
        "ORDER BY ts_id, model_type, fold_id, forecast_date LIMIT @limit"
    )
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_oof failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_series_history(
    run_id: str,
    *,
    ts_ids: list[str] | None = None,
    history_tail: int = 120,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Read the last ``history_tail`` historical observations per series for ``run_id``'s source
    panel.

    Uses the run's stored `RunConfig` (`read_run_config`) to locate the source table and column
    mappings (and builds hierarchical aggregate series when ``hierarchy.enabled=True``). Returns
    ``[]`` if the run has no stored config. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..config import RunConfig
    from ..engines.bigquery_sql import build_history_query
    from ..errors import RegistryError

    raw = read_run_config(run_id, settings=settings)
    if not raw:
        return []
    cfg = RunConfig.model_validate(raw)
    resolved = resolve_settings(settings)
    base_sql = build_history_query(cfg, resolved.registry_dataset_ref).rstrip().rstrip(";")
    if cfg.hierarchy.enabled:
        client = bigquery.Client(project=resolved.project_id)
        try:
            df = client.query(base_sql).result().to_dataframe()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RegistryError(f"read_series_history failed for run {run_id}: {exc}") from exc
        if df.empty:
            return []
        hier_cols = {col for level in cfg.hierarchy.levels for col in level}
        if hier_cols and hier_cols.issubset(df.columns):
            from ..reconciliation import build_hierarchy

            hist_cfg = cfg.model_copy(
                update={
                    "data": cfg.data.model_copy(
                        update={"ts_id_col": "ts_id", "date_col": "ds", "target_col": "y"}
                    )
                }
            )
            df, _ = build_hierarchy(df, hist_cfg)
        if ts_ids:
            df = df[df["ts_id"].astype(str).isin(set(ts_ids))]
        if history_tail > 0:
            df = df.sort_values(["ts_id", "ds"]).groupby("ts_id", as_index=False).tail(history_tail)
        return [
            {"ts_id": str(rec["ts_id"]), "ds": rec["ds"], "y": float(rec["y"])}
            for rec in df[["ts_id", "ds", "y"]].to_dict(orient="records")
        ]

    params: list[Any] = [bigquery.ScalarQueryParameter("tail", "INT64", max(1, history_tail))]
    ts_clause = ""
    if ts_ids:
        ts_clause = " WHERE ts_id IN UNNEST(@ts_ids)"
        params.append(bigquery.ArrayQueryParameter("ts_ids", "STRING", list(ts_ids)))
    sql = (
        f"WITH hist AS ({base_sql}) "
        f"SELECT ts_id, ds, y FROM hist{ts_clause} "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY ts_id ORDER BY ds DESC) <= @tail "
        "ORDER BY ts_id, ds"
    )
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_series_history failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_best_params(
    run_id: str,
    *,
    limit: int = 5000,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Return deduped full-fit ``forecast_metadata`` rows carrying ``best_params`` for ``run_id``.

    Captures both Optuna hyperparameter tuning winners (on base models) and learned ensemble
    stacking weights (``nnls`` / ``ridge`` / ``xgb`` on ``ensemble_*`` rows). Raises
    `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    sql = (
        "SELECT ts_id, model_type, compute_engine, ensemble_id, "
        "TO_JSON_STRING(best_params) AS best_params, fit_seconds, wape, mae, rmse, mase "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND fold_id IS NULL AND best_params IS NOT NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC NULLS LAST) = 1 "
        "ORDER BY model_type, ts_id LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_best_params failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_recent_runs(
    *,
    limit: int = 25,
    status: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Return the most recent ``v_run_summary`` rows in this registry, newest ``created_at`` first.

    Optionally filters by ``status`` (e.g. ``"COMPLETED"``). Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    where_clause = " WHERE status=@status" if status is not None else ""
    sql = (
        "SELECT run_id, created_at, status, python_runtime, n_series, n_models, "
        "backtest_on, runtime_seconds, total_wall_s, overhead_seconds, overhead_fraction, "
        "dcu_milli_seconds "
        f"FROM `{resolved.registry_table_ref('v_run_summary')}`"
        f"{where_clause} "
        "ORDER BY created_at DESC NULLS LAST LIMIT @limit"
    )
    params: list[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", max(1, limit))]
    if status is not None:
        params.append(bigquery.ScalarQueryParameter("status", "STRING", status))
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_recent_runs failed: {exc}") from exc
    return [dict(r) for r in rows]


def read_series_covariates(
    run_id: str,
    *,
    ts_id: str,
    history_tail: int = 120,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Read the last ``history_tail`` rows of configured dynamic covariates for ``ts_id`` on
    ``run_id`` (plus future covariates from ``synthetic_future_covariates`` when available).

    Returns ``[]`` if the run has no stored config or no configured dynamic covariates.
    """
    from google.cloud import bigquery

    from ..config import RunConfig
    from ..engines.bigquery_names import source_ref
    from ..errors import RegistryError

    raw = read_run_config(run_id, settings=settings)
    if not raw:
        return []
    cfg = RunConfig.model_validate(raw)
    cov_cols = cfg.features.dynamic_covariates
    if not cov_cols or cfg.hierarchy.enabled:
        return []

    resolved = resolve_settings(settings)
    source = source_ref(cfg, resolved.registry_dataset_ref)
    idc, datec = cfg.data.ts_id_col, cfg.data.date_col
    col_sql = ", ".join([f"{idc} AS ts_id", f"{datec} AS ds", *cov_cols])
    sql = (
        f"SELECT {col_sql} FROM `{source}` "
        f"WHERE {idc}=@ts_id "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {idc} ORDER BY {datec} DESC) <= @tail "
        f"ORDER BY {datec}"
    )
    params: list[Any] = [
        bigquery.ScalarQueryParameter("ts_id", "STRING", ts_id),
        bigquery.ScalarQueryParameter("tail", "INT64", max(1, history_tail)),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = [
            dict(r)
            for r in client.query(
                sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
            ).result()
        ]
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_series_covariates failed for run {run_id}: {exc}") from exc

    # When known_future_covariates are configured on the shipped synthetic panel, also fetch
    # future horizon rows from synthetic_future_covariates if present.
    future_cols = cfg.features.known_future_covariates
    if future_cols and cfg.data.source_table.endswith("synthetic_daily_series"):
        fut_table = f"{resolved.registry_dataset_ref}.synthetic_future_covariates"
        fut_col_sql = ", ".join(["ts_id", "ds", *future_cols])
        fut_sql = (
            f"SELECT {fut_col_sql} FROM `{fut_table}` WHERE ts_id=@ts_id ORDER BY ds LIMIT @horizon"
        )
        fut_params = [
            bigquery.ScalarQueryParameter("ts_id", "STRING", ts_id),
            bigquery.ScalarQueryParameter("horizon", "INT64", cfg.data.horizon),
        ]
        try:
            fut_rows = [
                dict(r)
                for r in client.query(
                    fut_sql, job_config=bigquery.QueryJobConfig(query_parameters=fut_params)
                ).result()
            ]
            rows.extend(fut_rows)
        except Exception:  # noqa: BLE001, S110 - optional future table
            pass
    return rows


def read_feature_attributions(
    run_id: str,
    *,
    ts_ids: list[str] | None = None,
    models: list[str] | None = None,
    level: str = "global",
    limit: int = 5000,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O
    """Read Tier 1 (``level="global"``) or Tier 2 (``level="local"``) feature attributions.

    - ``level="global"`` queries deduped full-fit ``forecast_metadata`` rows carrying
      ``fit_diagnostics`` (where ``$.feature_attributions`` is stored).
    - ``level="local"`` queries deduped ``forecast_predictions`` rows carrying non-null
      ``explanations`` JSON per forecast horizon step.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = resolve_settings(settings)
    clauses = ["run_id=@run_id", "ensemble_id IS NULL"]
    params: list[Any] = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    if ts_ids:
        clauses.append("ts_id IN UNNEST(@ts_ids)")
        params.append(bigquery.ArrayQueryParameter("ts_ids", "STRING", list(ts_ids)))
    if models:
        clauses.append("model_type IN UNNEST(@models)")
        params.append(bigquery.ArrayQueryParameter("models", "STRING", list(models)))

    if level == "local":
        clauses.append("explanations IS NOT NULL")
        where_sql = " AND ".join(clauses)
        sql = (
            "SELECT ts_id, model_type, compute_engine, forecast_date, yhat, "
            "TO_JSON_STRING(explanations) AS explanations "
            f"FROM `{resolved.registry_table_ref('forecast_predictions')}` "
            f"WHERE {where_sql} "
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY run_id, ts_id, model_type, ensemble_id, forecast_date "
            "ORDER BY created_at DESC NULLS LAST) = 1 "
            "ORDER BY ts_id, model_type, forecast_date LIMIT @limit"
        )
    else:
        clauses.append("fold_id IS NULL")
        clauses.append("fit_diagnostics IS NOT NULL")
        where_sql = " AND ".join(clauses)
        sql = (
            "SELECT ts_id, model_type, compute_engine, "
            "TO_JSON_STRING(fit_diagnostics) AS fit_diagnostics "
            f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
            f"WHERE {where_sql} "
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
            "ORDER BY created_at DESC NULLS LAST) = 1 "
            "ORDER BY ts_id, model_type LIMIT @limit"
        )

    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_feature_attributions failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]
