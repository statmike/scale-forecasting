"""The easy path: a thin, well-typed ``Forecaster`` facade over `main.run`.

This is the "simple SDK on top" — one class you point at a config and call. It adds **no**
forecasting logic; every method delegates to the same code the CLI and Composer run
(`scale_forecasting.main.run`, the run registry, the config layer), so a run driven from the
SDK is byte-for-byte the run driven from ``python -m scale_forecasting.main``. Users who need
to drive Spark or Ray themselves skip this class entirely and call the direct surface
(`run_group`, ``make_group_runner``,
``make_chunk_runner``, ``run_cell``) — both paths reuse the identical model machinery. See
``docs/using_the_sdk.md``.

Import cost: this module is cheap to import. The heavy model modules load only when a method that
runs models is called (``main`` is imported lazily inside ``run``/``dry_run``), which preserves the
near-instant ``import scale_forecasting`` contract enforced by ``test_sdk.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Fanout, RunConfig, estimate_fanout, load_config
from .registry.ids import make_run_id
from .registry.views import VIEW_NAMES
from .router import split_by_runtime

if TYPE_CHECKING:
    from pathlib import Path

    from .settings import Settings

__all__ = ["Forecaster", "DryRunResult", "RunResult", "ModelResult"]

# Registry statuses that mean a run has stopped changing — what `Forecaster.wait` blocks for.
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "PARTIAL"})


@dataclass(frozen=True)
class DryRunResult:
    """What a run *would* do, computed offline — no GCP calls, no compute launched.

    ``run_id`` is the deterministic config hash (identical to what an actual run lands under);
    ``fanout`` is the estimated cell count; ``python_models``/``bq_models`` are the runtime split.
    """

    run_id: str
    fanout: Fanout
    python_models: list[str]
    bq_models: list[str]


@dataclass(frozen=True)
class RunResult:
    """A pointer to where a run's results live — the run_id plus how to query them.

    ``dataset_ref`` is ``project.dataset`` (``None`` when the GCP identity can't be resolved, e.g.
    an offline `Forecaster.review`); ``views`` are the registry view names to query under it
    (e.g. ``v_run_summary``, ``v_model_leaderboard``). Filter any of them by ``run_id``.
    """

    run_id: str
    dataset_ref: str | None
    views: tuple[str, ...]


@dataclass(frozen=True)
class ModelResult:
    """One model's outcome on a run, read from ``v_model_leaderboard``.

    ``ensemble_id`` is ``None`` for a base model and the ensemble-config digest for an ensemble
    pseudo-model. ``mean_wape`` / ``mean_mae`` are the mean decision metrics where a backtest scored
    them (``None`` otherwise); ``no_artifact_rate`` near ``1.0`` flags a model that failed most
    cells. Rows come back best-first (lowest ``mean_wape``).
    """

    model_type: str
    ensemble_id: str | None
    compute_engine: str | None
    n_cells: int
    no_artifact_rate: float | None
    median_fit_seconds: float | None
    mean_wape: float | None
    mean_mae: float | None


class Forecaster:
    """The easy path: point it at a config, then `dry_run`, `run`, `status`, `wait`, or `results`.

    A run driven here is identical to the CLI/Composer run — this class only wraps
    `scale_forecasting.main.run`. Construct from an in-memory `RunConfig`, or use
    `from_file` / `from_dict`. An optional ``settings`` injects the GCP infra identity;
    ``None`` resolves it from the ``SF_*`` environment at run time (the default deployments use).

    The lifecycle closes the loop from one object: `dry_run` (offline plan), `run` (execute),
    `status`/`wait` (track a submission), and `results` (read the per-model leaderboard) — all keyed
    by the config's deterministic ``run_id``, so `status`/`results` work as a reattach path even in
    a fresh process.
    """

    def __init__(self, config: RunConfig, *, settings: Settings | None = None) -> None:
        self._config = config
        self._settings = settings

    @classmethod
    def from_file(cls, path: str | Path, *, settings: Settings | None = None) -> Forecaster:
        """Build from a JSON config file (delegates to
        `load_config`, which raises
        `ConfigError` on a bad file)."""
        return cls(load_config(path), settings=settings)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, settings: Settings | None = None) -> Forecaster:
        """Build from an already-parsed config dict (validates via `RunConfig`, raising
        `ConfigError` on a schema violation)."""
        try:
            cfg = RunConfig.model_validate(data)
        except Exception as exc:  # pydantic ValidationError → the package's ConfigError
            from .errors import ConfigError

            raise ConfigError(f"invalid config: {exc}") from exc
        return cls(cfg, settings=settings)

    @property
    def config(self) -> RunConfig:
        """The validated run configuration this forecaster will execute."""
        return self._config

    @property
    def run_id(self) -> str:
        """The deterministic run_id for this config (pure hash; no GCP call)."""
        return make_run_id(self._config)

    def dry_run(self) -> DryRunResult:
        """Validate the config and report the planned fan-out + runtime split — touches no GCP.

        Delegates the run_id to `main.run` (``dry_run=True``) so it is the single source of
        truth, then adds the estimated fanout and the python/BigQuery model split.
        """
        from . import main

        run_id = main.run(self._config, dry_run=True)
        python_models, bq_models = split_by_runtime(self._config)
        return DryRunResult(
            run_id=run_id,
            fanout=estimate_fanout(self._config),
            python_models=python_models,
            bq_models=bq_models,
        )

    def run(self, *, spark: object | None = None) -> RunResult:
        """Execute the run (Spark/Ray ∥ BigQuery under one run_id) and return where to query it.

        Delegates to `main.run`, threading this forecaster's ``settings`` (so an injected
        identity is honored). ``spark`` optionally injects a `SparkSession` /
        ``DataprocSparkSession`` for the in-process Spark path (notebook / Connect demo). Returns a
        `RunResult` pointing at the registry views under the resolved dataset.
        """
        from . import main

        run_id = main.run(self._config, spark=spark, settings=self._settings)
        return RunResult(run_id=run_id, dataset_ref=self._resolved_dataset_ref(), views=VIEW_NAMES)

    def review(self) -> RunResult:
        """Return a pointer to this config's results **without** running anything (offline).

        The run_id is the deterministic config hash, so this resolves the same location a completed
        run landed under. ``dataset_ref`` is ``None`` when the GCP identity can't be resolved (no
        ``SF_*`` env and no injected ``settings``) — the run_id + view names are still returned so a
        caller can construct the query once infra is known.
        """
        return RunResult(
            run_id=self.run_id, dataset_ref=self._resolved_dataset_ref(), views=VIEW_NAMES
        )

    def status(self, run_id: str | None = None) -> str | None:
        """The current registry status of a run (``RUNNING``/``COMPLETED``/``FAILED``/``PARTIAL``).

        Reads the run's header (`registry.bq.header_status`); returns ``None`` when this config has
        never run. ``run_id`` defaults to this config's deterministic id, so ``forecaster.status()``
        answers "did my config's run finish?" — the reattach path for a ``wait=False`` submit.
        """
        from .registry import bq

        return bq.header_status(run_id or self.run_id, settings=self._settings)

    def wait(
        self, run_id: str | None = None, *, timeout: float = 3600.0, poll_seconds: float = 15.0
    ) -> str:
        """Block until a run reaches a terminal status and return it; raise on timeout/not-found.

        Polls `status` every ``poll_seconds`` until the status is terminal
        (``COMPLETED``/``FAILED``/``PARTIAL``). Raises `TimeoutError` if ``timeout`` elapses first,
        or `ConfigError` if the run has no header yet (never submitted). Each poll re-reads the
        header through a fresh client, so a long wait re-authenticates rather than reusing a stale
        token.
        """
        from .errors import ConfigError

        rid = run_id or self.run_id
        deadline = time.monotonic() + timeout
        while True:
            status = self.status(rid)
            if status is None:
                raise ConfigError(f"no run found for run_id {rid}: nothing to wait on")
            if status in _TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run {rid} still {status} after {timeout:.0f}s")
            time.sleep(poll_seconds)

    def results(self, run_id: str | None = None) -> list[ModelResult]:
        """The per-model leaderboard for a run — one `ModelResult` each, best (lowest WAPE) first.

        Reads ``v_model_leaderboard`` (`registry.bq.read_leaderboard`) for ``run_id`` (default: this
        config's id). Returns ``[]`` when the run has produced no scored rows yet. This is the
        first-class "which model won?" read that `RunResult` only pointed at before.
        """
        from .registry import bq

        rows = bq.read_leaderboard(run_id or self.run_id, settings=self._settings)
        return [
            ModelResult(
                model_type=r["model_type"],
                ensemble_id=r.get("ensemble_id"),
                compute_engine=r.get("compute_engine"),
                n_cells=r["n_cells"],
                no_artifact_rate=r.get("no_artifact_rate"),
                median_fit_seconds=r.get("median_fit_seconds"),
                mean_wape=r.get("mean_wape"),
                mean_mae=r.get("mean_mae"),
            )
            for r in rows
        ]

    def _resolved_dataset_ref(self) -> str | None:
        """``project.dataset`` from the injected/resolved `Settings`, or ``None`` if
        unresolvable (missing ``SF_*`` env) — keeps `review` graceful offline."""
        settings = self._settings
        if settings is None:
            from .errors import ConfigError
            from .settings import Settings

            try:
                settings = Settings.resolve()
            except ConfigError:
                return None
        return settings.dataset_ref
