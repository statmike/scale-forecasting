"""Split a run's model list by runtime.

A run's ``cfg.models`` mixes Python-runtime models (executed by the Spark/Ray engines via
``worker.run_cell``) and BigQuery-native models (executed as SQL by ``engines.bigquery_engine``).
The two runtimes run *in parallel* — adding BQ models costs wall-clock ``max(python, bq)``, not the
sum. This module owns the run-level split that decides which engine gets which models,
by reading each model's ``runtime`` class attribute from the factory.

Public surface: ``split_by_runtime(cfg) -> (python_models, bq_models)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import get_model

if TYPE_CHECKING:
    from .config import RunConfig


def split_by_runtime(cfg: RunConfig) -> tuple[list[str], list[str]]:
    """Partition ``cfg.models`` into ``(python_models, bq_models)`` by declared runtime.

    A model routes to BigQuery iff its class sets ``runtime == "bigquery"`` (the native models
    ``arima_plus`` / ``timesfm``); everything else routes to the Python
    runtime selected by ``cfg.python_runtime`` (spark xor ray). This is per-model, not per-run:
    one config can list both kinds and they execute concurrently. Input order is
    preserved within each list so downstream logs/SQL are deterministic.

    Unknown model names raise ``ModelError`` (via `get_model`) — the
    same validation the engines rely on, surfaced once up front.
    """
    python_models: list[str] = []
    bq_models: list[str] = []
    for name in cfg.models:
        if get_model(name).runtime == "bigquery":
            bq_models.append(name)
        else:
            python_models.append(name)
    return python_models, bq_models
