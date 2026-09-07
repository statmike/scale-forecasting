"""Did the accelerator this job paid for actually do anything? — the GPU contract's verdict.

Layer 3 makes a cell state its device instead of guessing; this is the layer that checks, after the
fact, whether the statement came true. It exists because the failure it detects is silent by
construction: a job attaches a card, every cell fits on the CPU, every forecast is correct, every
metric is fine, and the only symptom is the bill.

Three verdicts and a ``None``, decided per **family job** on the driver once its cells are written:

* ``MISSING_DEVICE`` — the family asked for a device and no cell reports running on one. After
  Layer 3 this should be unreachable, which is exactly why it is kept: it is the regression
  detector for the whole contract.
* ``ENGAGED_IDLE`` — cells ran on the device and it was barely touched.
* ``ENGAGED_UTILISED`` — cells ran on the device and used a real share of it.
* ``None`` — the family never asked for a device, so there is nothing to judge.

**Expect ``ENGAGED_IDLE`` on every GPU run shipped today, and read it as a cost finding rather than
a fault.** Across 31,356 NeuralProphet fits on live T4s, peak device memory was 50–78 KB against a
17 GB card — 0.00045% of it. That is why this warns and never fails a job: the run is correct, it is
just paying for hardware it does not need at the shipped hyperparameters.

**Only MAX and counts, never a percentile.** ``peak_gpu_bytes`` is a per-process high-water that is
never reset between cells, so it is already monotone within a worker; a median across cells would be
the median of a running maximum, which is a number about arrival order rather than about the model.

The aggregate is one query over ``forecast_metadata``, so this is service-agnostic by construction —
Serverless, a Dataproc cluster and Ray all write the same rows, and the native BigQuery family is
covered for free by returning ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import get_logger

if TYPE_CHECKING:
    from .settings import Settings

_log = get_logger(__name__)

MISSING_DEVICE = "MISSING_DEVICE"
ENGAGED_IDLE = "ENGAGED_IDLE"
ENGAGED_UTILISED = "ENGAGED_UTILISED"

# The share of a card's memory a fit has to touch before the device counts as earning its cost,
# as a whole percent. Three orders of magnitude above the 0.00045% Phase 0 measured, so it cannot
# false-positive on the shape we already know about; low enough that anything doing genuine tensor
# work clears it. Integer percent rather than a float fraction so the threshold is an exact byte
# count and a peak sitting precisely on it lands the same way every time.
_ENGAGED_FLOOR_PERCENT = 1


def device_verdict(
    *,
    hardware: str | None,
    gpu_type: str | None,
    cells_on_device: int,
    max_peak_gpu_bytes: int | None,
) -> str | None:
    """Judge one family job's device use from its aggregate (pure; see the module docstring).

    ``hardware`` is what the family resolved to, so a CPU family and the native family both return
    ``None`` — there is no claim to check. ``cells_on_device`` counts cells whose model reported
    ``device_used="cuda"``: a **recorded** fact, never inferred from ``peak_gpu_bytes``, whose
    ``None`` is overloaded across four unrelated causes.

    A missing or zero ``max_peak_gpu_bytes`` on a family whose cells *did* run on the device reads
    as idle rather than missing. The device was there and the model was on it; that the allocator
    high-water is unreadable is a gap in the measurement, not evidence the card was absent.
    """
    if hardware != "gpu":
        return None
    if cells_on_device <= 0:
        return MISSING_DEVICE
    from .engines.ray_io import device_memory_bytes  # lazy: keep the lean launch path lean

    floor = device_memory_bytes(gpu_type) * _ENGAGED_FLOOR_PERCENT // 100
    return ENGAGED_UTILISED if (max_peak_gpu_bytes or 0) >= floor else ENGAGED_IDLE


def format_verdict(family: str, blob: dict[str, Any]) -> str:
    """The one human-readable line a verdict is worth, for the submit log."""
    verdict = blob.get("verdict")
    if verdict == MISSING_DEVICE:
        return (
            f"family '{family}' was provisioned onto GPU hardware but no cell reports having run "
            f"on a device ({blob.get('cells', 0)} cells). The run is correct and the accelerator "
            f"was billed for nothing."
        )
    if verdict == ENGAGED_IDLE:
        peak = blob.get("max_peak_gpu_bytes") or 0
        return (
            f"family '{family}' used its device but barely touched it (peak {peak:,} bytes on a "
            f"{blob.get('gpu_type')}). Correct, and paying for hardware it does not need at these "
            f"hyperparameters — see `gpu_useful` for the two remedies."
        )
    return f"family '{family}': {verdict}"


_LABELS = {
    MISSING_DEVICE: "no gpu",
    ENGAGED_IDLE: "gpu idle",
    ENGAGED_UTILISED: "gpu used",
}


def verdict_label(verdict: str | None) -> str | None:
    """Two words for a chart label — ``None`` for a family with no verdict, or an unknown one.

    The vocabulary lives here rather than at the display site so the words a reader sees and the
    words a query filters on cannot drift apart. An unknown string is dropped rather than printed
    raw: a bar-end label is not where someone should first meet a verdict nobody defined.
    """
    return _LABELS.get(verdict or "")


def read_device_use(
    run_id: str,
    models: list[str],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """The one aggregate a verdict is decided from: ``{cells, cells_on_device, …}``.

    Scoped to this family's ``models`` because a run's other families wrote into the same table
    under the same ``run_id``, and a statistical family's CPU cells would drown the deep-learning
    family's evidence. Full-fit rows only — fold rows are bracketed by the full-fit row that
    already covers them, and ensemble rows are arithmetic rather than fits.

    Never raises. This is an audit of a job that has already succeeded; failing the job because the
    audit query failed would trade a correct run for a missing note.
    """
    from google.cloud import bigquery

    from .registry.tables import _resolve_settings

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT COUNT(*) AS cells, "
        "COUNTIF(device_used='cuda') AS cells_on_device, "
        "COUNTIF(peak_gpu_bytes IS NULL) AS cells_no_device, "
        "MAX(peak_gpu_bytes) AS max_peak_gpu_bytes, "
        "ANY_VALUE(device_name) AS device_name "
        f"FROM `{resolved.registry_table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND model_type IN UNNEST(@models) "
        "AND fold_id IS NULL AND ensemble_id IS NULL"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ArrayQueryParameter("models", "STRING", models),
    ]
    try:
        client = bigquery.Client(project=resolved.project_id)
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - an audit must never sink the job it audits
        _log.warning("device audit query failed for run %s: %r", run_id, exc)
        return {}
    return dict(rows[0]) if rows else {}


def audit_device_use(
    run_id: str,
    family: str,
    models: list[str],
    *,
    hardware: str | None,
    gpu_type: str | None,
    settings: Settings | None = None,
    read: Any = None,
) -> dict[str, Any] | None:  # pragma: no cover - the read half is GCP I/O
    """Read the aggregate, judge it, and return the blob to file — or ``None`` when there is none.

    The blob lands on ``run_jobs.job_telemetry.$.device_use``, which is already JSON and already
    sits beside the ``hardware`` column the verdict is about — no ``.<family>`` nesting, because
    ``run_jobs`` is keyed by ``(run_id, family, attempt)`` and the row is already this family's. No
    new column, no wire change, and it costs a CPU family exactly one comparison because such a
    family short-circuits before the query runs.

    ``read`` injects the aggregate for offline tests, the same way `profiling.source` injects its
    measurement function.
    """
    if hardware != "gpu":
        return None
    agg = (read or read_device_use)(run_id, models, settings=settings)
    if not agg:
        return None
    blob = {
        "verdict": device_verdict(
            hardware=hardware,
            gpu_type=gpu_type,
            cells_on_device=int(agg.get("cells_on_device") or 0),
            max_peak_gpu_bytes=agg.get("max_peak_gpu_bytes"),
        ),
        "gpu_type": gpu_type,
        "cells": int(agg.get("cells") or 0),
        "cells_on_device": int(agg.get("cells_on_device") or 0),
        "cells_no_device": int(agg.get("cells_no_device") or 0),
        "max_peak_gpu_bytes": agg.get("max_peak_gpu_bytes"),
        "device_name": agg.get("device_name"),
    }
    if blob["verdict"] in (MISSING_DEVICE, ENGAGED_IDLE):
        _log.warning("%s", format_verdict(family, blob))
    return blob
