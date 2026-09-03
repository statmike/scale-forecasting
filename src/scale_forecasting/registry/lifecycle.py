"""The two context managers that bracket a run and a job.

`run_header` opens a run's ``run_registry`` row at RUNNING and finalizes it exactly once on the way
out; `run_job` does the same for one ``run_jobs`` row. Both stamp wall-clock and a terminal status
whether the body returns or raises — which is what keeps a crashed run from leaving a row stuck at
RUNNING forever. Both also decline to write a *failure* over a row that another process already
cancelled; see `_sticky_guard`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .rows import assemble_job_row

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ..config import RunConfig
    from ..settings import Settings


# A cancellation is sticky against the failure it caused.
#
# `probes.cancel` runs in a *different process* from the one that launched the run. When it stops a
# job, the launcher's own poll loop sees a job that went STOPPED, which from inside that process is
# indistinguishable from a job that died — so seconds later it finalizes FAILED (or PARTIAL, for a
# run whose other families finished) straight over the CANCELLED the cancel had just written. The
# run then reads as broken rather than deliberately stopped, on the header, in `doctor`, and in
# every status filter. Observed live 2026-09-02, seventeen seconds apart.
#
# Guarding the write rather than reading first keeps it to one statement, which matters when the
# writer being raced is in another process. Green statuses are not guarded: if the work genuinely
# finished, nothing about the run was lost.
_STICKY_STATUSES: tuple[str, ...] = ("CANCELLED",)
_NON_GREEN_STATUSES = frozenset({"FAILED", "PARTIAL"})


def _sticky_guard(status: str) -> tuple[str, ...]:
    """Statuses that a write of ``status`` must not overwrite — empty when it may overwrite (pure).

    Only a non-green finalize is held back. Splitting on the status being *written* is what keeps
    the rule to the case that was observed, instead of quietly making `CANCELLED` unwritable-over
    in general.
    """
    return _STICKY_STATUSES if status in _NON_GREEN_STATUSES else ()


class HeaderFinalizer:
    """Mutable finalize state a `run_header` body fills in before a clean exit.

    A run's terminal ``status`` (default ``COMPLETED``) plus any extra header columns to stamp on
    success (``n_series``, ``n_models``, ``bq_models``, …). Left untouched, the block finalizes a
    plain ``COMPLETED`` with only the wall-clock ``runtime_seconds`` `run_header` measures.
    """

    def __init__(self) -> None:
        self.status: str = "COMPLETED"
        self.extra: dict[str, Any] = {}

    def finalize(self, *, status: str | None = None, **fields: Any) -> None:
        """Set the terminal ``status`` (if given) and merge extra columns for the success write."""
        if status is not None:
            self.status = status
        self.extra.update(fields)


@contextmanager
def run_header(
    cfg: RunConfig,
    run_id: str,
    *,
    settings: Settings | None = None,
    manage: bool = True,
) -> Iterator[HeaderFinalizer]:
    """Own a run's ``run_registry`` header for the duration of a block (the one lifecycle seam).

    In **owner mode** (``manage=True``): on entry ``ensure_tables`` + ``write_header`` (RUNNING);
    on a clean exit ``update_header`` with the finalizer's ``status`` (default COMPLETED), the
    measured wall-clock ``runtime_seconds``, and any extra columns the body set via
    `HeaderFinalizer.finalize`; on an exception ``update_header(status=FAILED, runtime_seconds=…)``
    then re-raise, so a crashed run records a terminal status instead of stranding at RUNNING.
    Either way a non-green finalize carries the sticky-cancellation guard (`_sticky_guard`).

    In **contributor mode** (``manage=False``): touches no header at all — `main.run` owns the
    single shared row — so this only yields the finalizer for uniform call shape. The body may
    still populate it; nothing is written.
    """
    # Imported per call, not at module load: these are the GCP writers, and binding them late is
    # what lets a caller's test substitute `header.write_header` / `tables.ensure_tables` the way it
    # could when every one of these lived in a single module.
    from .header import update_header, write_header
    from .tables import ensure_tables

    fin = HeaderFinalizer()
    if manage:
        ensure_tables(cfg, settings=settings)
        write_header(cfg, run_id, settings=settings)
    started = time.perf_counter()
    try:
        yield fin
    except BaseException:
        if manage:
            update_header(
                run_id,
                settings=settings,
                status="FAILED",
                runtime_seconds=time.perf_counter() - started,
                unless_status_in=_STICKY_STATUSES,
            )
        raise
    if manage:
        update_header(
            run_id,
            settings=settings,
            status=fin.status,
            runtime_seconds=time.perf_counter() - started,
            unless_status_in=_sticky_guard(fin.status),
            **fin.extra,
        )


class JobFinalizer:
    """Mutable finalize state a `run_job` body fills in before a clean exit.

    A job's terminal ``status`` (default ``COMPLETED``), any extra ``run_jobs`` columns to stamp on
    success — notably ``system_job_id``, once the platform assigns/accepts it — and a
    ``job_telemetry`` patch. Left untouched, the block finalizes a plain ``COMPLETED`` with only the
    wall-clock ``runtime_seconds`` `run_job` measures.
    """

    def __init__(self) -> None:
        self.status: str = "COMPLETED"
        self.extra: dict[str, Any] = {}
        self.telemetry: dict[str, Any] = {}

    def finalize(
        self,
        *,
        status: str | None = None,
        telemetry: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """Set the terminal ``status`` (if given), merge extra columns, accrete a telemetry patch.

        ``telemetry`` is a ``{dotted.path: value}`` patch merged into the row's ``job_telemetry``
        (`registry.jobs.update_job`'s ``merge_telemetry``) rather than replacing the column — which
        is what a body wants nearly always, because the row it is finalizing may already carry a
        capacity ledger somebody else wrote while it was waiting to start.
        """
        if status is not None:
            self.status = status
        self.extra.update(fields)
        if telemetry:
            self.telemetry.update(telemetry)


@contextmanager
def run_job(
    run_id: str,
    family: str,
    attempt: int,
    *,
    runtime: str | None = None,
    spark_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    system_job_id: str | None = None,
    probe_handle: dict[str, Any] | None = None,
    settings: Settings | None = None,
    manage: bool = True,
) -> Iterator[JobFinalizer]:
    """Own one family's ``run_jobs`` row for the duration of a block (the per-job lifecycle seam).

    The `run_header` analog for the per-job tier: on entry write the job row (RUNNING) with its
    deterministic id (`assemble_job_row` → `registry.ids.make_job_key`) and resolved compute; on a
    clean exit ``update_job`` with the finalizer's ``status`` (default COMPLETED), the measured
    wall-clock ``runtime_seconds``, any extra columns set via `JobFinalizer.finalize` (e.g. the
    platform ``system_job_id``) and its ``job_telemetry`` patch **merged** into whatever the row
    already carries; on an exception ``update_job(status=FAILED,
    runtime_seconds=…)`` then re-raise, so a crashed job records a terminal status instead of
    stranding at RUNNING — with the same sticky-cancellation guard `run_header` applies, since a
    cancelled family is exactly the job this block is most likely to see raise. The run header is
    owned separately by `run_header`; a job row sits *under*
    it. ``manage=False`` yields the finalizer without touching ``run_jobs`` (uniform call shape for
    a caller that records the job elsewhere). Assumes the tables exist (the header owner ran
    `ensure_tables`), so it does not re-create them.

    The failure write carries ``fin.extra`` too. A body that told the finalizer something and
    *then* raised knew that thing at the moment it failed, and discarding it is how a row ends up
    saying FAILED with nothing about why: `job_launch` uses exactly this to attach
    ``failure_reason`` and the capacity ledger to a job that ran out of regions to try. The
    finalizer's ``status`` is deliberately not honoured here — a raising body is FAILED regardless
    of what it hoped to write — and `JobFinalizer.finalize` keeps ``status`` out of ``extra``, so
    the two can never collide.
    """
    from .ids import make_job_key
    from .jobs import update_job, write_job  # late-bound writers, as in `run_header`

    fin = JobFinalizer()
    job_id = make_job_key(run_id, family, attempt)
    if manage:
        from datetime import UTC, datetime

        row = assemble_job_row(
            run_id,
            family,
            attempt,
            datetime.now(UTC),
            runtime=runtime,
            spark_mode=spark_mode,
            hardware=hardware,
            gpu_type=gpu_type,
            system_job_id=system_job_id,
            probe_handle=probe_handle,
        )
        write_job(row, settings=settings)
    started = time.perf_counter()
    try:
        yield fin
    except BaseException:
        if manage:
            update_job(
                job_id,
                settings=settings,
                status="FAILED",
                runtime_seconds=time.perf_counter() - started,
                ended_at=datetime.now(UTC),
                unless_status_in=_STICKY_STATUSES,
                merge_telemetry=fin.telemetry or None,
                **fin.extra,
            )
        raise
    if manage:
        update_job(
            job_id,
            settings=settings,
            status=fin.status,
            runtime_seconds=time.perf_counter() - started,
            ended_at=datetime.now(UTC),
            unless_status_in=_sticky_guard(fin.status),
            merge_telemetry=fin.telemetry or None,
            **fin.extra,
        )
