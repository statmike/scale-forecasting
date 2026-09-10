"""What terminal status did this run earn? — the job tier's, and the header's roll-up of them.

Two questions, one module, because the second is only as honest as the first. `audit_cells` decides
what one family job's ``run_jobs`` row should say; `combined_run_status` folds a run's job statuses
into the header's. Both `main.run` and the Airflow `finalize_run` task call the second one — a
single definition rather than a local copy and a mirror, which is how the two tiers came to disagree
in the first place.

**The job tier.** A job row used to go terminal on one fact: the launch call returned without
raising. That is a
statement about the *submission*, not about the run. Everything the engine knows — how many cells
it fitted, how many raised — is computed on the remote driver by
`engines.spark_io.aggregate_status` and then thrown away, because a submitter hands back a probe
handle rather than an outcome. The 2026-09-10 negative arms are what made the gap concrete: a
Dataproc cluster job whose six cells all failed, and a Ray job that wrote nothing at all, both
closed ``COMPLETED`` on both tiers of the registry.

So this asks the question the same way `device_audit` asks its own: **one aggregate over
``forecast_metadata``, read from the driver after the job has finished, identical for every
runtime**. Serverless, a Dataproc cluster and Ray all write the same rows, so none of the three
needs a channel of its own and no engine has to learn to report anything.

Scoped by ``created_at`` rather than by attempt, because ``forecast_metadata`` has no attempt
column and it is append-only: a second attempt of a family sees the first attempt's rows sitting
under the same ``run_id``. See `read_cell_counts` for why a clock margin is safe here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from .settings import Settings

_log = get_logger(__name__)

COMPLETED = "COMPLETED"
PARTIAL = "PARTIAL"
FAILED = "FAILED"

# How far before the launch call to start counting cells. The worker that stamps ``created_at`` is
# not the process that reads it, and their clocks are only as aligned as the fleet's NTP; a minute
# is orders of magnitude more than any skew observed and orders of magnitude less than the gap
# between two attempts of one family, which cannot be shorter than a cluster create.
_CLOCK_MARGIN_S = 60


def job_status(*, cells: int, errors: int) -> str:
    """Fold a family job's cell tallies into its terminal status (pure).

    The same three-way rule `engines.spark_io.aggregate_status` applies on the remote driver, kept
    deliberately identical: every cell ok is ``COMPLETED``, no cell ok is ``FAILED``, a mix is
    ``PARTIAL`` — a run with surviving forecasts is not a failed run, and the cells that landed are
    real and usable.

    **Zero cells is ``FAILED``, not ``COMPLETED``.** A job that wrote no metadata at all did not
    finish quietly; it did not run. That is the Ray case from 2026-09-10, where the workers crashed
    before any cell could record itself and the only evidence was four ``WorkerCrashedError`` chunks
    in the driver's telemetry.
    """
    if cells <= 0 or errors >= cells:
        return FAILED
    return COMPLETED if errors <= 0 else PARTIAL


def read_cell_counts(
    run_id: str,
    models: list[str],
    *,
    since: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """The one aggregate a status is decided from: ``{cells, errors}``.

    Scoped to this family's ``models`` for the reason `device_audit.read_device_use` is — a run's
    other families wrote into the same table under the same ``run_id`` — and to full-fit rows only,
    since a fold row is bracketed by the full-fit row that covers it and an ensemble row is
    arithmetic rather than a fit.

    ``since`` is the launch time, less `_CLOCK_MARGIN_S`. Without it a re-run counts its own cells
    *and the previous attempt's*, which is how a clean second attempt after a total failure would
    be judged ``PARTIAL`` on the strength of rows it did not write.

    Never raises: a read that fails returns ``{}`` and the caller keeps the status it already had.
    Getting the status wrong in the pessimistic direction — failing a good job because BigQuery
    was briefly unreachable — is worse than the gap this closes.
    """
    from google.cloud import bigquery

    from .registry.tables import _resolve_settings

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT COUNT(*) AS cells, COUNTIF(cell_status='error') AS errors "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND model_type IN UNNEST(@models) "
        "AND fold_id IS NULL AND ensemble_id IS NULL "
        "AND (@since IS NULL OR created_at >= @since)"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ArrayQueryParameter("models", "STRING", models),
        bigquery.ScalarQueryParameter("since", "TIMESTAMP", since),
    ]
    try:
        client = bigquery.Client(project=resolved.project_id)
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - a status audit must never sink the job it audits
        _log.warning("cell-count query failed for run %s: %r", run_id, exc)
        return {}
    return dict(rows[0]) if rows else {}


def cells_written(
    run_id: str,
    *,
    since: datetime | None = None,
    settings: Settings | None = None,
) -> int | None:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """How many ``forecast_metadata`` rows this run has written so far — the liveness signal.

    Deliberately unlike `read_cell_counts`, which is a verdict on a finished family. This is asked
    *while a job is still running* and only has to answer "is anything at all happening", so it
    counts every row the run has: any family, any model, folds and ensemble rows included. Filtering
    to full fits the way the audit does would report zero for a run that is hours into backtesting
    and perfectly healthy, which is precisely the false alarm a watchdog must not raise.

    ``None`` when the read fails — no evidence, therefore no verdict. A watchdog that treated an
    unreachable BigQuery as "the run is dead" would cancel healthy runs during an outage.
    """
    from google.cloud import bigquery

    from .registry.tables import _resolve_settings

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT COUNT(*) AS cells "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND (@since IS NULL OR created_at >= @since)"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("since", "TIMESTAMP", since),
    ]
    try:
        client = bigquery.Client(project=resolved.project_id)
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - a liveness probe must never sink the job it watches
        _log.warning("liveness cell count failed for run %s: %r", run_id, exc)
        return None
    return int(rows[0]["cells"]) if rows else 0


def audit_cells(
    run_id: str,
    family: str,
    models: list[str],
    *,
    since: datetime | None = None,
    settings: Settings | None = None,
    read: Any = None,
) -> tuple[str | None, dict[str, Any]]:
    """Read the tallies and judge them: ``(status, blob)``, or ``(None, {})`` when unreadable.

    ``None`` means "no opinion" and the caller leaves the row's status alone — an unreadable
    aggregate is not evidence of failure. The blob is filed at ``job_telemetry.$.cells`` beside
    ``device_use``, so the numbers behind the status are on the row rather than only in the log.

    ``read`` injects the aggregate for offline tests, exactly as `device_audit.audit_device_use`
    does with its own.
    """
    agg = (read or read_cell_counts)(run_id, models, since=since, settings=settings)
    if not agg:
        return None, {}
    cells = int(agg.get("cells") or 0)
    errors = int(agg.get("errors") or 0)
    status = job_status(cells=cells, errors=errors)
    if status != COMPLETED:
        _log.warning(
            "family '%s' finished %s: %d of %d cell(s) failed", family, status, errors, cells
        )
    return status, {"cells": cells, "errors": errors, "status": status}


def launch_window_start() -> datetime:
    """The ``since`` bound for a job about to be launched — now, less the clock margin."""
    from datetime import UTC, datetime, timedelta

    return datetime.now(UTC) - timedelta(seconds=_CLOCK_MARGIN_S)


def combined_run_status(job_statuses: dict[str, str | None], *, ensemble_enabled: bool) -> str:
    """Roll the per-family job statuses into the one status the run header carries (pure).

    Over the base families (every key but ``"ensemble"``): all ``COMPLETED`` → ``COMPLETED``; all
    non-``COMPLETED`` → ``FAILED``; a mix → ``PARTIAL`` (surviving families' forecasts stay usable).
    A missing or still-``RUNNING`` family counts as failed — from here that is indistinguishable
    from a family whose task died before it could finalize. An ensemble that did not complete
    downgrades an otherwise-``COMPLETED`` run to ``FAILED`` (the requested output is incomplete); it
    never masks a family ``PARTIAL``/``FAILED``.

    A repair's row (``statistical_repair``, from `registry.ids.REPAIR_JOB_FAMILIES`) counts as an
    ordinary job here rather than folding onto the family it repairs, which is the same reading
    `registry.ops.roll_up_job_statuses` gives. A repair only exists because cells were missing, so a
    repair that failed leaves the run genuinely incomplete and the header should say ``PARTIAL``;
    folding it in the other direction — letting a forty-cell repair report its
    hundred-thousand-cell family ``COMPLETED`` — is exactly the lie the separate row exists to
    prevent.

    Both tiers reach this with the *statuses the job rows hold*, but by different routes, and the
    difference is worth naming. `airflow_tasks.finalize_run` runs in its own process after every
    family task, so it re-reads ``run_jobs``. `main.run` holds the launch calls' return values and
    never re-reads, because a row it just finalized cannot have changed and a second read is a
    second way to be wrong.
    """
    base = {family: status for family, status in job_statuses.items() if family != "ensemble"}
    n_jobs = len(base)
    n_failed = sum(1 for status in base.values() if status != COMPLETED)
    if n_jobs == 0 or n_failed == 0:
        engine_status = COMPLETED
    elif n_failed == n_jobs:
        engine_status = FAILED
    else:
        engine_status = PARTIAL

    if ensemble_enabled and engine_status == COMPLETED:
        if job_statuses.get("ensemble") != COMPLETED:
            return FAILED
    return engine_status
