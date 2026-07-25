"""The cell runner — the G1 unit of work that runs identically local / Spark / Ray.

Owned by BUILD step 2.6. Public surface: ``run_cell(series, model_name, cfg) -> CellResult``
(CONTRACTS §3.1-§3.3).
"""

from __future__ import annotations


def run_cell(series: object, model_name: str, cfg: object) -> object:  # pragma: no cover - stub
    raise NotImplementedError("worker.run_cell — BUILD step 2.6")
