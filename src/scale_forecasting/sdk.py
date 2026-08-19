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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Fanout, RunConfig, estimate_fanout, load_config
from .registry.ids import make_run_id
from .registry.views import VIEW_NAMES
from .router import split_by_runtime

if TYPE_CHECKING:
    from pathlib import Path

    from .settings import Settings

__all__ = ["Forecaster", "DryRunResult", "RunResult"]


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


class Forecaster:
    """The easy path: point it at a config, then `dry_run`, `run`, or `review`.

    A run driven here is identical to the CLI/Composer run — this class only wraps
    `scale_forecasting.main.run`. Construct from an in-memory `RunConfig`, or use
    `from_file` / `from_dict`. An optional ``settings`` injects the GCP infra identity;
    ``None`` resolves it from the ``SF_*`` environment at run time (the default deployments use).
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
