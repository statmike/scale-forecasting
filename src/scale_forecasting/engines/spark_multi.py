"""Spark method B — one serverless batch per model family (DESIGN §2.1, §6).

``multi`` is orchestrated **submit-side, not on-cluster**: it fans a run out into one independent
Dataproc batch per model family (statistical / ml / deep_learning / native), each an ``explode`` run
over that family's models — separate autoscaling and failure domains, one ``run_id``/header apiece.

The decisive reason it can't run on the cluster: family-splitting needs ``google-cloud-dataproc`` to
*submit* the child batches, and that client lives in the ``[spark]`` extra which is **excluded from
the runtime container** (the container is deps-only; code arrives via ``python_file_uris``). So the
orchestration lives in :func:`scale_forecasting.submit.submit_multi`, where the ``[spark]`` extra
and ADC credentials both exist. The launcher (:mod:`scale_forecasting.spark_entry`) maps only
``explode``/``naive`` to on-cluster engines, so ``multi`` never dispatches here in normal use.

This module's ``run`` is therefore a **guard**, not an implementation: if a ``multi`` config somehow
reaches an on-cluster engine, fail loudly and immediately with a pointer to the submit helper rather
than silently running a single un-split batch. Public surface: ``run(cfg) -> None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import EngineError

if TYPE_CHECKING:
    from ..config import RunConfig


def run(cfg: RunConfig) -> None:
    """Guard: ``multi`` is orchestrated by the submit helper, never run on-cluster (see module doc).

    Always raises :class:`~scale_forecasting.errors.EngineError`. The ``multi`` fan-out is performed
    by :func:`scale_forecasting.submit.submit_multi`, which loops model families and submits one
    child ``explode`` batch each; there is nothing for an on-cluster engine to do.
    """
    raise EngineError(
        "spark_method='multi' is orchestrated by the submit helper, not run on-cluster: "
        "use `scale_forecasting.submit.submit_multi(cfg)` (or `python -m scale_forecasting.submit "
        "--engine multi ...`), which fans out one child 'explode' batch per model family. "
        "google-cloud-dataproc is in the [spark] extra and absent from the runtime container, so "
        "family-splitting cannot happen here."
    )
