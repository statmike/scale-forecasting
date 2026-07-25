"""The cell runner — the G1 unit of work that runs identically local / Spark / Ray.

``run_cell`` (BUILD step 2.6) fits, optionally backtests, and predicts ONE
``(ts_id, model)`` cell and returns a :class:`CellResult`. Engines differ only in how
they *call* it and *collect* its results — that symmetry is what makes "same code
everywhere" (G1) real.

:class:`CellResult` is defined here because it is the worker's output type; the registry
writers (``registry/bq.py``) consume it. It carries plain data (frames + scalars), no
behavior, so it is the clean seam between compute and lineage (CONTRACTS §3.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from .config import RunConfig


@dataclass(frozen=True)
class CellResult:
    """The result of one ``(ts_id, model)`` cell (CONTRACTS §3.2).

    A failing cell sets ``status="error"`` with ``error`` populated and empty
    ``predictions`` — it never raises out of ``run_cell`` (CONTRACTS §3.3), so one bad
    cell can't sink a 100k-series batch.
    """

    run_id: str
    ts_id: str
    model_type: str
    compute_engine: str  # "spark" | "ray" | "bigquery"
    model_hash: str
    status: str  # "ok" | "error"
    error: str | None
    predictions: pd.DataFrame  # canonical §2.1
    oof: pd.DataFrame | None  # canonical §2.2, or None if backtest off
    metrics: dict[str, float]  # §2.3 (full-fit metrics)
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    artifact_local_path: str | None = None


def run_cell(
    series: pd.DataFrame, model_name: str, cfg: RunConfig
) -> CellResult:  # pragma: no cover
    raise NotImplementedError("worker.run_cell — BUILD step 2.6")
