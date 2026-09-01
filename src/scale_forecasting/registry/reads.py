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

from .params import _header_param
from .rows import METRIC_COLUMNS
from .tables import _resolve_settings

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

    resolved = _resolve_settings(settings)
    sql = f"SELECT * FROM `{resolved.registry_table_ref('v_run_summary')}` WHERE run_id=@run_id"
    params = [_header_param("run_id", run_id)]
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

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.registry_table_ref('v_model_leaderboard')}` "
        "WHERE run_id=@run_id ORDER BY mean_wape ASC NULLS LAST"
    )
    params = [_header_param("run_id", run_id)]
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

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT model_type, COUNT(*) AS n "
        f"FROM `{resolved.registry_table_ref('forecast_predictions')}` "
        "WHERE run_id=@run_id GROUP BY model_type"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_prediction_counts failed for run {run_id}: {exc}") from exc
    return {str(r["model_type"]): int(r["n"]) for r in rows}


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

    resolved = _resolve_settings(settings)
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

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT TO_JSON_STRING(raw_config) AS raw_config "
        f"FROM `{resolved.registry_table_ref('run_registry')}` "
        "WHERE run_id=@run_id ORDER BY created_at DESC LIMIT 1"
    )
    params = [_header_param("run_id", run_id)]
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

    resolved = _resolve_settings(settings)
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
    params = [_header_param("run_id", run_id)]
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

    resolved = _resolve_settings(settings)
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
    params = [_header_param("run_id", run_id)]
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

    resolved = _resolve_settings(settings)
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
