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
_HARVEST_WHERE = "fold_id IS NULL AND ensemble_id IS NULL AND cpu_seconds IS NOT NULL"


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


# How many candidate runs one discovery will rank. One row per run, not per cell, so this is a
# sanity bound rather than a real limit — a 90-day window would have to hold this many *distinct
# measured runs* to trip it. When it does, the oldest candidates are the ones dropped.
_MAX_HARVEST_CANDIDATES = 500


def rank_harvest_candidates(
    candidates: list[dict[str, Any]],
    *,
    target_series: int | None,
    target_runtime: str | None = None,
) -> str | None:
    """Pick the best harvest to size ``target_series`` from, or ``None`` if there are none (pure).

    Each candidate is ``{"run_id", "measured_at", "n_series", "engines"}``. Three keys, in strict
    order: **runtime comparability, then closeness of scale, then recency**.

    *Runtime* comes first because a mismatch there is not an inaccuracy, it is a category error.
    ``process_rss_bytes`` is the absolute footprint of the process that ran the cell: on Ray that is
    one task, on Spark it is an executor running many cells at once. Reading a Spark number as a Ray
    per-task bound therefore over-states it by roughly the executor's density, and the result is not
    a fleet that is merely too large — it is a per-task ``memory`` request no node can satisfy,
    which Ray answers by queueing the task forever. Live 2026-09-03: a 100k Ray run sat at zero
    cells for an hour holding a 21 GiB per-task request harvested from a Spark run. A run whose
    cells are *all* the target runtime ranks above a mixed run (whose harvest read would blend both
    kinds of number), which ranks above one with none of it.

    *Scale* comes second, and ahead of recency — the opposite of what "the newest evidence"
    suggests, and deliberately so. A campaign writes small harvests often and large ones rarely, so
    ordering by recency alone reliably selects the *least* representative run in the window:
    observed live three times, once handing a 100k-series plan a profile measured on six series.

    Distance is measured in **log space**, because sizing error scales multiplicatively. Against a
    100k target a 1k harvest is off by two orders of magnitude and a 6-series one by four, and the
    log makes that a 2-vs-4 gap rather than a 99,000-vs-99,994 one where every small run looks
    equally bad. Distance is symmetric: an over-large harvest is no more trusted than an equally
    over-small one, since both are extrapolations.

    Each axis degrades to *no opinion* when its input is missing — no ``target_runtime``, or no
    ``target_series`` (a config with no ``series_limit``, so the scale is not known until the source
    is read) — and with both missing this is pure recency, the original behaviour.
    """
    if not candidates:
        return None
    import math

    def runtime_tier(cand: dict[str, Any]) -> int:
        if not target_runtime:
            return 0
        engines = {e for e in (cand.get("engines") or []) if e}
        if not engines:
            return 1  # an older harvest that recorded no engine — unknown, not disqualified
        if engines == {target_runtime}:
            return 0
        return 1 if target_runtime in engines else 2

    def distance(cand: dict[str, Any]) -> float:
        n = cand.get("n_series") or 0
        if not target_series or n <= 0:
            return 0.0
        # Rounded, so that two candidates whose log-distance differs in the fifteenth decimal are a
        # genuine tie and recency decides. A 0.01 band in log space is a ~1% difference in ratio —
        # far below the resolution at which one harvest is better evidence than another.
        return round(abs(math.log(n) - math.log(target_series)), 2)

    # Sort keys are (runtime tier asc, distance asc, measured_at desc). measured_at is a datetime,
    # so negating it is not available; sorting repeatedly with a stable sort, least-significant key
    # first, gives the same order and needs no key arithmetic.
    ordered = sorted(candidates, key=lambda c: c["measured_at"], reverse=True)
    ordered.sort(key=distance)
    ordered.sort(key=runtime_tier)
    return str(ordered[0]["run_id"])


def discover_harvest_run(
    *,
    source_table: str | None,
    freq: str | None,
    target_series: int | None = None,
    target_runtime: str | None = None,
    lookback_days: int = _HARVEST_LOOKBACK_DAYS,
    settings: Settings | None = None,
) -> str | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """The completed run whose harvest of ``source_table`` at ``freq`` best fits, or ``None``.

    What ``compute.profile.source = "auto"`` resolves against. Only the two *identity* axes are
    filtered here — same table, same frequency — because those are the ones where a mismatch means
    "not evidence about this run at all". The scale axes (series count, history length) are checked
    after the rows are loaded, by `profiling.signature.compare_signatures`, where a mismatch is a
    warning rather than a disqualification: a 1k-series run is imperfect but usable evidence for a
    100k one, and preferring nothing over it would leave the common case unsized.

    But *which* imperfect run gets picked is a choice, and it is made here rather than in SQL: this
    query returns every candidate in the window with its measured series count and the set of
    engines its cells ran on, and `rank_harvest_candidates` — pure, and therefore tested offline —
    chooses. See it for why runtime beats scale and scale beats recency.

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
        "SELECT m.run_id AS run_id, MAX(m.created_at) AS measured_at, "
        "       COUNT(DISTINCT m.ts_id) AS n_series, "
        "       ARRAY_AGG(DISTINCT m.compute_engine IGNORE NULLS) AS engines "
        "FROM `{metadata}` AS m JOIN headers AS h USING (run_id) "
        "WHERE m.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback DAY) "
        "  AND {harvest_where} "
        "  AND h.status = 'COMPLETED' "
        "  AND (@source_table IS NULL OR "
        "       JSON_VALUE(h.raw_config, '$.data.source_table') = @source_table) "
        "  AND (@freq IS NULL OR JSON_VALUE(h.raw_config, '$.data.freq') = @freq) "
        "GROUP BY m.run_id ORDER BY measured_at DESC LIMIT @candidates"
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
        bigquery.ScalarQueryParameter("candidates", "INT64", _MAX_HARVEST_CANDIDATES),
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
        raise RegistryError(f"discover_harvest_run failed: {exc}") from exc
    return rank_harvest_candidates(rows, target_series=target_series, target_runtime=target_runtime)
