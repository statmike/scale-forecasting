"""Compute-harvest reads — the measured evidence a cost profile is fitted from.

What ``compute.profile.source`` resolves against: pull the full-fit cells of a prior run
(`read_compute_harvest`), or find the most recent run that has any (`discover_harvest_run`). Kept
apart from the ordinary read surface (`registry.reads`) because the filters and the caps are the
ones a cost model needs, not the ones an analyst does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import get_logger
from .reads import read_run_config
from .tables import _resolve_settings

if TYPE_CHECKING:
    from ..settings import Settings

_log = get_logger(__name__)


# The most cells one harvest read will pull back. A profiling run is meant to be *small* — §3.10's
# "an ordinary small run of the same config" — so the cap is not the expected case; it is what keeps
# pointing `source` at a finished 100k run from materialising half a million dicts on a submit host
# that was deliberately given no model stack and not much memory. When it trips, the read is
# truncated and says so, and the resulting profile's signature honestly reports the smaller panel.
_MAX_HARVEST_CELLS = 50_000

# How far back `discover_harvest_run` looks. Two jobs at once: it prunes the `created_at` partition
# scan (the whole table is otherwise in play), and it puts a floor under staleness — evidence older
# than a quarter is evidence about a fleet, a package set and a data volume that have all moved.
_HARVEST_LOOKBACK_DAYS = 90

# The full-fit filter every harvest read shares. Fold rows are bracketed by the full-fit row that
# already covers them and ensemble rows are arithmetic rather than fits, so both would corrupt a
# cost model; `cpu_seconds IS NOT NULL` is what distinguishes a measured cell from one written
# before the columns existed or under `measure="off"`.
_HARVEST_WHERE = (
    "fold_id IS NULL AND ensemble_id IS NULL AND cpu_seconds IS NOT NULL"
)


def read_compute_harvest(
    run_id: str, *, limit: int = _MAX_HARVEST_CELLS, settings: Settings | None = None
) -> tuple[list[dict[str, Any]], str | None] | None:  # pragma: no cover - GCP I/O
    """Return ``(harvest rows, the table the run read)`` for ``run_id``, or ``None`` if unmeasured.

    The read half of "a profile is a query over a `run_id`". Pulls the per-cell measurements
    `worker.run_cell` recorded — `profiling.cost.harvest_profile` aggregates them into the same
    `ComputeProfile` the in-run pre-pass builds, so there is one aggregation and this is only a
    reader. ``None`` (rather than an empty list) means the run has no measurements at all, which is
    what lets the caller fall through to the next source instead of sizing off nothing.

    ``source_table`` comes back alongside because ``forecast_metadata`` does not record one — the
    run header's ``raw_config`` does — and it is the axis `profiling.signature.compare_signatures`
    cares most about: the same fits on a different table are not evidence about this run.

    Writes are append-only and at-least-once, so the rows are deduped to one per cell (latest write
    wins) before returning, the same grain ``v_model_leaderboard`` serves. Ordering is by a
    fingerprint of ``ts_id`` rather than by time or by cost: when ``limit`` truncates, a
    deterministic pseudo-random slice of the panel is a usable sample, while the first N by arrival
    would be whichever bucket happened to finish first. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT ts_id, model_type, fold_id, ensemble_id, fit_seconds, cpu_seconds, "
        "process_rss_bytes, peak_gpu_bytes, intraop_threads, n_obs, created_at "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        f"WHERE run_id=@run_id AND {_HARVEST_WHERE} "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "ORDER BY FARM_FINGERPRINT(ts_id) LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
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
        raise RegistryError(f"read_compute_harvest failed for run {run_id}: {exc}") from exc
    if not rows:
        return None
    if len(rows) == limit:
        _log.warning(
            "harvest for %s truncated at %d cells; the profile is sized from a sample of the "
            "panel, so its memory bound may under-state the true worst case",
            run_id,
            limit,
        )
    config = read_run_config(run_id, settings=settings) or {}
    return rows, (config.get("data") or {}).get("source_table")


def discover_harvest_run(
    *,
    source_table: str | None,
    freq: str | None,
    lookback_days: int = _HARVEST_LOOKBACK_DAYS,
    settings: Settings | None = None,
) -> str | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """The newest completed run carrying a harvest of ``source_table`` at ``freq``, or ``None``.

    What ``compute.profile.source = "auto"`` resolves against. Only the two *identity* axes are
    filtered here — same table, same frequency — because those are the ones where a mismatch means
    "not evidence about this run at all". The scale axes (series count, history length) are checked
    after the rows are loaded, by `profiling.signature.compare_signatures`, where a mismatch is a
    warning rather than a disqualification: a 1k-series run is imperfect but usable evidence for a
    100k one,
    and preferring nothing over it would leave the common case unsized.

    Restricted to ``COMPLETED`` runs. A run that died partway measured only the cells that finished,
    which on a bucketed engine is a biased slice, and biased evidence sizes a fleet badly in a
    direction nobody can see. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    # run_registry is append-only (the header is written, then updated on finalize), so it is
    # deduped to the latest row per run before the join — otherwise a finalized run fans out.
    sql = (
        "WITH headers AS ("
        "  SELECT run_id, raw_config, status FROM `{registry}` "
        "  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1"
        ") "
        "SELECT m.run_id AS run_id, MAX(m.created_at) AS measured_at "
        "FROM `{metadata}` AS m JOIN headers AS h USING (run_id) "
        "WHERE m.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback DAY) "
        "  AND {harvest_where} "
        "  AND h.status = 'COMPLETED' "
        "  AND (@source_table IS NULL OR "
        "       JSON_VALUE(h.raw_config, '$.data.source_table') = @source_table) "
        "  AND (@freq IS NULL OR JSON_VALUE(h.raw_config, '$.data.freq') = @freq) "
        "GROUP BY m.run_id ORDER BY measured_at DESC LIMIT 1"
    ).format(
        registry=resolved.registry_table_ref("run_registry"),
        metadata=resolved.registry_table_ref("forecast_metadata"),
        harvest_where=_HARVEST_WHERE.replace("fold_id", "m.fold_id")
        .replace("ensemble_id", "m.ensemble_id")
        .replace("cpu_seconds", "m.cpu_seconds"),
    )
    params = [
        bigquery.ScalarQueryParameter("lookback", "INT64", lookback_days),
        bigquery.ScalarQueryParameter("source_table", "STRING", source_table),
        bigquery.ScalarQueryParameter("freq", "STRING", freq),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"discover_harvest_run failed: {exc}") from exc
    return str(rows[0]["run_id"]) if rows else None
