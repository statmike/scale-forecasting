"""Blocking on a long Spark job without abandoning a healthy one or babysitting a dead one.

Both submitters — Serverless (`submit.submit_batch`) and cluster
(`cluster_submit.submit_cluster_job`) — hand a long-running operation to the same problem: block
until it is terminal, and decide what to do when it is not. The naive answer,
``operation.result(timeout=...)``, gets one of two failures wrong whichever number you pick. Too
short and a healthy run that simply takes a while is abandoned; too long and a run that will never
produce anything bills the whole ceiling before anyone notices.

The loop here answers them separately, because they are separate questions.

**How long to wait** is a patience setting, and it belongs high. Neither runtime's spend is bounded
by the client watching it — a batch has its own ttl, a cluster has an idle bound and a max age — so
a short client wait buys no safety at all. It only converts long runs into lost ones, which is
exactly what it did on 2026-09-13: a GPU cluster job two hours into a three-hour fit, killed by its
own launcher's two-hour ceiling, taking 1,941 landed cells and the cluster with it.

**Whether the run is alive** is a different question entirely, and the clock cannot answer it. A run
that has written even one forecast row is alive, however slow; a run past its grace period with zero
rows is stuck, however recently it started. So the watch is on *output*, not on elapsed time, and it
switches off permanently the moment the first row appears — after that, wedging is the ttl's problem
and not this loop's.

Nothing here is specific to a runtime. The caller supplies the operation, a label for the logs, and
its own two numbers.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .errors import EngineError, get_logger

_log = get_logger(__name__)

# How often the wait wakes up to check on the job. Not a timeout in any sense the caller cares
# about — the loop's own deadline is what bounds the wait — it is only how coarse the watchdog's
# view of a stall is allowed to be. Five minutes keeps the BigQuery liveness read to at most one a
# poll on a run that may last a day.
_WATCHDOG_INTERVAL_SECONDS = 300.0


def is_stalled(*, elapsed_s: float, grace_s: int, cells: int | None) -> bool:
    """Has this job produced nothing for long enough to call it stuck rather than slow? (pure)

    Three ways to answer no, and each of them is a false alarm this deliberately refuses to raise.
    ``grace_s <= 0`` is the operator switching the watchdog off. Inside the grace period nothing is
    concluded, because a run that has not started is not a run that has failed. And ``cells is
    None`` — the count could not be read — is *no evidence*, not evidence of death; a watchdog that
    cancelled healthy runs during a BigQuery outage would be worse than the failure it guards.

    So the only yes is: past the grace period, the count was read, and it is zero.
    """
    if grace_s <= 0 or elapsed_s < grace_s:
        return False
    return cells == 0


def wait_for_job(
    operation: Any,
    *,
    run_id: str,
    label: str,
    wait_timeout: float,
    grace_s: int,
    since: Any,
) -> Any:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """Block until the job is terminal, cancelling it if it stops producing anything.

    Behaviourally identical to the bare ``operation.result(timeout=wait_timeout)`` it replaces for
    every run that writes even one cell: the same terminal object comes back, and the same
    client-side ``TimeoutError`` is raised — the original one, re-raised at the same deadline —
    for a run that outlives the wait. The loop only exists so there is somewhere to look up from.

    ``label`` names the job the way its own runtime does (``batch sf-…``, ``cluster job …``) so the
    log line and the cancellation message read as that runtime's operator would expect.
    """
    import time
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    from .job_outcome import cells_written

    started = time.monotonic()
    deadline = started + wait_timeout
    watching = grace_s > 0
    while True:
        try:
            return operation.result(timeout=_WATCHDOG_INTERVAL_SECONDS)
        except FuturesTimeoutError:
            if time.monotonic() >= deadline:
                raise
        if not watching:
            continue
        elapsed = time.monotonic() - started
        if elapsed < grace_s:
            continue
        cells = cells_written(run_id, since=since)
        if cells:
            _log.info("%s has written %d cell(s); watchdog stands down", label, cells)
            watching = False
            continue
        if is_stalled(elapsed_s=elapsed, grace_s=grace_s, cells=cells):
            _log.error("%s has written no cells in %.0f min; cancelling", label, elapsed / 60)
            with contextlib.suppress(Exception):  # a cancel that fails must not mask the diagnosis
                operation.cancel()
            raise EngineError(
                f"{label} wrote no forecast rows in {elapsed / 60:.0f} minutes and was "
                f"cancelled (stall watchdog; raise SF_STALL_GRACE_S, or set it to 0, if this run "
                f"legitimately takes that long to produce its first cell)"
            )
