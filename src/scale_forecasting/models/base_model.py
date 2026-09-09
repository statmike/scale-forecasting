"""The model interface every model file implements.

This is the linchpin of the one-model-one-file rule and the factory. Every Python model
is a subclass of `BaseModel` living in its own file that ends with ``register(...)``;
the factory (``models/__init__.py``) builds ``{name: class}`` at import. Adding a model is
a new file plus one register call — no edits anywhere else.

Models never read global config: everything they need at fit/predict time arrives through
`ModelContext`. Models that don't emit their own prediction intervals
(``supports_native_intervals = False``) call `BaseModel.residual_intervals` so every
model still returns the canonical frame with ordered bounds.

Public surface: ``BaseModel``, ``ModelContext``, ``register`` (plus the ``_REGISTRY`` the
factory reads, and the ``Runtime``/``Family`` type aliases).
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import optuna

Runtime = Literal["python", "bigquery"]
Family = Literal["statistical", "ml", "deep_learning", "native"]

# Canonical prediction-frame columns, in order.
#
# `yhat_raw` is the model's own point forecast, always preserved. `yhat` may differ from it: for a
# model that builds its band from residual quantiles, `yhat` is the 0.5 quantile, which is
# `prediction + median(residual)` — a real bias correction that measurement says earns its keep,
# but one that must never be the *only* number stored. Keeping both is what lets `calibration.py`
# score the two arms against each other on held-out folds instead of asking anyone to take the
# choice on faith.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "ds",
    "yhat",
    "yhat_raw",
    "yhat_lower",
    "yhat_upper",
    "quantiles",
)

# Default quantile set for predict() and the residual helper.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ModelContext:
    """Per-run context handed to every model so it never reads global config."""

    freq: str
    # The largest horizon `predict` will be asked for in this run — `cfg.max_horizon`, which is
    # `max(data.horizon, backtest.gap + backtest.horizon)`. **Not the forward horizon.** One context
    # is shared by the backtest folds and the final fit, so a field that meant only the forward
    # horizon was wrong on every fold of any run whose backtest horizon differs. The embargo is in
    # there because a fold forecasts *across* it before reaching the scored window. Models should
    # predict the `horizon` argument they are *handed*; this is here for sizing decisions made at
    # construction, before that argument exists.
    horizon: int
    seed: int = 0
    holidays: pd.DataFrame | None = None
    transform: str = "none"
    # Fitted Box-Cox λ for the cell (from features.fit_transform_lambda), or None for the
    # stateless transforms. Set once per cell and shared by the backtest folds + final fit, so
    # every invert_transform in predict() uses the same λ — never refit at predict.
    transform_lambda: float | None = None
    # Which device this cell's model should fit on. "auto" lets the library choose and can never
    # fail, which is why it is the default and why every run before this field existed is
    # reproduced byte-for-byte by it — but it also means "I asked for a GPU and silently got none"
    # was structurally unobservable, and that `hardware: "cpu"` could not push a model OFF a card
    # a mixed-hardware cluster happens to expose. "gpu"/"cpu" say it outright, and a model asked
    # for a device that is not there raises instead of quietly running on the CPU it was billed to
    # avoid. Set by `worker._model_context` from the job's provisioned hardware (see `hardware`),
    # never from config intent — the config is environment-blind and would break a laptop run.
    # A context field, not a config field: `ModelContext` is not in `cfg.model_dump`, so no run_id
    # moves.
    device: Literal["auto", "cpu", "gpu"] = "auto"


# The factory registry: name → concrete model class. Populated by register() at import.
_REGISTRY: dict[str, type[BaseModel]] = {}


def register(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Register a model class under its ``name``. Returns the class so it
    doubles as a decorator. Raises on a missing or duplicate name.
    """
    name = getattr(model_cls, "name", None)
    if not name:
        raise ModelError(f"{model_cls.__name__} must set a class-level 'name' before register()")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not model_cls:
        raise ModelError(
            f"duplicate model name '{name}': {existing.__name__} vs {model_cls.__name__}"
        )
    _REGISTRY[name] = model_cls
    return model_cls


class BaseModel(ABC):
    """Base class for every forecasting model."""

    # --- class-level registration metadata (read by the factory) ---
    name: ClassVar[str]
    runtime: ClassVar[Runtime]
    family: ClassVar[Family]
    supports_exog: ClassVar[bool] = False
    supports_native_intervals: ClassVar[bool] = False
    # Can this model put a device to work *at all* — is there a tensor library under it? A static
    # property of the model, not of a run. False for everything that fits on CPU by construction
    # (statsmodels, the tree models, the naives). Distinct from `gpu_useful`, which asks the much
    # harder question of whether a device would earn its cost at the hyperparameters actually
    # authored. Capable-but-not-useful is the normal state, and it is what Phase 0 measured.
    gpu_capable: ClassVar[bool] = False
    # --- frozen-backtest capabilities (see `recondition` / `advance_origin`) ---
    #
    # Both default to False so an out-of-tree model is never *assumed* capable of something it has
    # not implemented. A model that declines is not broken: under a frozen scheme it refits per fold
    # and the cell records `backtest_refit="unsupported"`, which is a fact a reader can filter on
    # rather than a silent substitution of a weaker estimand.
    supports_recondition: ClassVar[bool] = False
    supports_extrapolate: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any], ctx: ModelContext) -> None:
        self.params = dict(params)
        self.ctx = ctx
        # Residuals stashed by fit() for models that lean on the residual-interval helper.
        self._residuals: np.ndarray | None = None
        # How far past the fit's last observation this model now forecasts from — see
        # `advance_origin`. Zero for every ordinary fit-then-predict cell, which is what makes the
        # offset arithmetic in each model's predict() a no-op on the default path.
        self._origin_offset: int = 0
        # Exog covering the skipped span, for the models that forecast by recursion and so have to
        # be rolled through it. None everywhere else, and None whenever the offset is 0.
        self._gap_exog: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        """Fit on one series. ``y`` is indexed by ds (datetime64); ``X`` is aligned exog or None."""

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Return the canonical prediction frame in original units."""

    # --- frozen backtesting: moving the forecast origin without refitting ------------

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        """Take in observations that arrived *after* the current origin, without re-estimating.

        This is the seam behind ``backtest.scheme="expanding_frozen"``: the model is fit once on the
        oldest fold's window and then walked forward, so each fold's score answers "what does
        refitting less often cost me?" rather than "how good is this model freshly trained?".
        Parameters stay exactly as fitted; only what the model is *conditioned on* moves.

        ``y_new`` is the **new observations only** — not the whole window. That is
        ``statsmodels``' own ``append`` contract, and it is the contract worth matching because
        getting it wrong there is silent: appending a span twice corrupts the state filter and
        raises nothing. Models that would rather hold the whole history (the naives, ``croston``,
        the lag models) accumulate it themselves in one line, and those are exactly the models where
        the operation is cheap and hard to get wrong.

        Default: refuse. A model that has no way to absorb an observation without re-estimating
        should say so rather than approximate, and `supports_recondition` is how the driver knows
        in advance.
        """
        raise ModelError(f"{type(self).name} cannot re-condition; it must be refit")

    def advance_origin(self, periods: int, X_gap: pd.DataFrame | None = None) -> None:
        """Move the forecast origin ``periods`` steps forward *blind* — no new actuals.

        The seam behind ``backtest.scheme="expanding_stale"``: the model's already-fitted curve is
        evaluated at later dates and never told what actually happened in between, so its score
        measures **staleness, not skill**. It cannot notice a level shift. That is a real question
        ("how fast does a fitted model decay if nobody touches it?") and a different one from
        `recondition`'s, which is why the two never share a leaderboard slice.

        ``periods`` is **absolute** — counted from the last observation the fit saw, not from
        wherever the origin currently is. So the driver sets it per fold instead of accumulating,
        and setting it twice is idempotent. ``0`` restores the as-fitted origin.

        ``X_gap`` is exog over the skipped span, for the models whose forecast is a recursion that
        has to be rolled through it. Handing it over is not leakage: exog is known-in-advance by
        construction here — `features.build_future_features` fabricates it for the forward forecast
        too. Only the *target* is withheld.

        Unlike `recondition`, this one has a real default: it records the offset and the gap exog,
        and each model honours them in ``predict`` (the base helper `_forecast_index` does the date
        half). There is nothing model-specific left to override, so a model opts in by setting
        `supports_extrapolate` rather than by reimplementing the same four lines sixteen times.
        """
        if not self.supports_extrapolate:
            raise ModelError(f"{type(self).name} cannot advance its forecast origin")
        if periods < 0:
            raise ModelError(f"advance_origin: periods must be >= 0, got {periods}")
        self._origin_offset = int(periods)
        self._gap_exog = X_gap

    def get_params(self) -> dict[str, Any]:
        """Resolved params actually used (post-HPO). Logged to ``forecast_metadata.best_params``."""
        return dict(self.params)

    def serialize(self) -> bytes | None:
        """Serialize the fitted model for artifact persistence.

        Returns the bytes to store as the cell's ``model_artifact`` — the registry writer
        uploads them under ``<warehouse>/artifacts/<run_id>/`` and stamps the GCS ObjectRef
        onto the ``forecast_metadata`` row (lineage). Default: pickle the instance.
        Override to use a model-native format (e.g. ``Booster.save_model``) or return
        ``None`` to opt out for models cheap enough to refit. Called only when the run sets
        ``persist_models``; a ``None`` return (or a raised error, which the worker catches)
        simply means no artifact for that cell — persistence never sinks a forecast.
        """
        import pickle

        return pickle.dumps(self)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        """HPO search space (optional; used only when ``hpo.enabled``). Default: no search."""
        return {}

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Refuse an authored ``model_params`` block this model cannot honour. Default: accept all.

        Called at plan time, before anything is provisioned, so a parameter combination that would
        produce a horizon of NaNs costs nothing instead of a fleet-hour. Override when a model has
        a constraint that is only knowable from its own library's behaviour — the point of the seam
        is that such knowledge stays in the model's file rather than leaking into the config layer.

        ``max_horizon`` is the largest horizon this run will ask ``predict`` for: the forward
        horizon and the backtest horizon, whichever is greater. Raise `ConfigError`, not
        `ModelError` — nothing has been fitted, the config is what is wrong.
        """
        return  # accept: a model with no library-level constraint has nothing to refuse

    @classmethod
    def gpu_useful(cls, params: Mapping[str, Any]) -> bool:
        """Would a device do meaningful work at these **authored** hyperparameters? Default: no.

        `gpu_capable` says a device *can* be used; this says it would be *worth paying for*. The
        two came apart when they were measured: across 31,356 NeuralProphet fits on live T4s, peak
        device memory was 50–78 KB against a card holding 17 GB and ``cpu_seconds / fit_seconds``
        sat at 0.93–0.996. The model was on the device the whole time and the device was doing
        essentially nothing, because at the shipped defaults the network is a few hundred trend and
        Fourier parameters. Every GPU run in the ledger was, in substance, a CPU run.

        **Authored hyperparameters only** — no `ModelContext`, no config. Building a plan-time
        context would duplicate `worker._model_context` (which does real work), and no model's
        device use depends on the frequency or the holiday frame. Widen the signature later if a
        model genuinely needs the horizon; that is an internal change, not a config change.

        **This never feeds routing or provisioning.** Consuming it at
        `engines.ray_io.split_gpu_cpu_models` would empty the GPU pool on every config shipped
        today while the submitter still bought the cards — manufacturing the exact
        provisioned-but-unrouted incoherence `dag.check_hardware_coherence` exists to refuse. It
        feeds one thing: a warning (`dag.gpu_usefulness_report`).
        """
        return False

    def device_used(self) -> str | None:
        """After ``fit``, where did the parameters actually end up — ``"cuda"``, ``"cpu"``, None?

        The evidence half of the GPU contract, and it has to come from the fitted object rather
        than from what the run asked for. ``ModelContext.device`` records the *request*; a request
        is not a receipt, and the whole reason this contract exists is that for twenty-one jobs
        the two silently disagreed.

        ``None`` is the default and the honest answer for every model with no tensor library under
        it: a statsmodels fit does not run "on the CPU" in any sense worth recording, it simply has
        no device concept. Do not return ``"cpu"`` to mean "not applicable" — a reader auditing a
        GPU family needs to tell "this model looked and found no device" from "this model cannot
        answer". Override only where the library can be asked; see `gpu_capable`.

        Called once per cell after the final fit, and it must never raise — a probe that sank a
        good forecast would be worse than no probe.
        """
        return None

    # --- shared helpers ---------------------------------------------------------

    def residual_intervals(
        self, yhat: np.ndarray, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    ) -> dict[float, np.ndarray]:
        """Empirical residual-quantile prediction intervals.

        For models without native intervals: fit() records residuals via
        `_set_residuals`, and this adds their empirical quantiles to the point
        forecast so bounds are naturally ordered (lower ≤ yhat ≤ upper) for any monotone
        quantile set. Falls back to a point-mass band (bounds == yhat) if no residuals
        were recorded.
        """
        yhat = np.asarray(yhat, dtype=float)
        if self._residuals is None or self._residuals.size == 0:
            return {q: yhat.copy() for q in quantiles}
        return {q: yhat + float(np.quantile(self._residuals, q)) for q in quantiles}

    def _set_residuals(self, residuals: np.ndarray | pd.Series) -> None:
        """Record in-sample residuals (actual − fitted) for `residual_intervals`."""
        arr = np.asarray(residuals, dtype=float)
        self._residuals = arr[~np.isnan(arr)]

    def _future_index(self, last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
        """The ``horizon`` future dates after ``last_date`` at the context frequency."""
        return pd.date_range(start=last_date, periods=horizon + 1, freq=self.ctx.freq)[1:].as_unit(
            "ns"
        )

    def _forecast_index(self, horizon: int) -> pd.DatetimeIndex:
        """The ``horizon`` dates this prediction covers, honouring any advanced origin.

        At the default offset of 0 this is exactly ``_future_index(self._last_date, horizon)`` —
        which is what every model's ``predict`` used to call directly, and why swapping the call is
        safe across the whole suite. After `advance_origin(k)` it is the same date grid walked ``k``
        further out, so the fold's validation dates line up without the model needing to know what a
        fold is.
        """
        offset = self._origin_offset
        return self._future_index(self._last_date, offset + horizon)[offset:]

    def _forecast_steps(self, horizon: int) -> int:
        """How many steps a step-indexed forecaster must produce to cover `_forecast_index`.

        The companion to `_forecast_index` for the models whose library counts *steps from the fit*
        rather than taking dates: ask for this many, then keep the last ``horizon``. Reads as
        ``horizon`` at the default offset.
        """
        return self._origin_offset + horizon

    def _forecast_exog(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        """Exog covering the whole step span: the skipped gap first, then the requested window.

        Returns ``X`` untouched at the default offset, so the ordinary path allocates nothing. When
        the origin has been advanced past a span the caller supplied exog for, the two are stacked
        in date order — which is the order every consumer here reads them in.
        """
        if self._origin_offset == 0 or self._gap_exog is None:
            return X
        return self._gap_exog if X is None else pd.concat([self._gap_exog, X])

    def _assemble_frame(
        self,
        ds: pd.DatetimeIndex | pd.Series,
        quantile_map: dict[float, np.ndarray],
        raw: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Build the canonical prediction frame from a quantile map.

        ``yhat`` is the 0.5 quantile (median); bounds are the min/max quantiles so they
        stay ordered. ``quantiles`` is the full map serialized to a JSON string per row.

        ``raw`` is the model's own point forecast **in original units** — pass
        ``invert_transform(mean, t, lam)``, the same value the quantile map is built around. It
        matters only for the models whose band comes from residual quantiles, where the 0.5
        quantile is `prediction + median(residual)` rather than the prediction; for a model with a
        symmetric native band the two coincide. Optional, defaulting to the median, so an
        out-of-tree model written against the older contract still assembles — but such a model
        forfeits the arm comparison, because there is nothing to compare against.
        """
        qs = sorted(quantile_map)
        if not qs:
            raise ModelError("quantile_map is empty")
        median = quantile_map.get(0.5, quantile_map[qs[len(qs) // 2]])
        lower = quantile_map[qs[0]]
        upper = quantile_map[qs[-1]]
        raw_values = median if raw is None else np.asarray(raw, dtype=float)
        n = len(median)
        # Drop non-finite quantile values per step. json.dumps defaults to allow_nan=True, minting
        # the bare literals NaN/Infinity — invalid JSON that BigQuery's JSON-column parser rejects
        # ("syntax error while parsing value - invalid literal"), failing the whole Storage Write
        # API append and cascading to a whole-run FAILURE. A runaway series (log1p's expm1 inverse
        # overflowing to ±Inf) is a per-series pathology and must not take the fleet down; a step
        # whose values are all non-finite serializes to "{}". (The scalar median/bounds columns are
        # nulled independently at the write boundary by _as_float.)
        quantiles_json = [
            json.dumps({str(q): fv for q in qs if math.isfinite(fv := float(quantile_map[q][i]))})
            for i in range(n)
        ]
        # The contract requires datetime64[ns]; pandas 2.x may infer coarser units.
        ds_ns = pd.DatetimeIndex(ds).as_unit("ns")
        return pd.DataFrame(
            {
                "ds": ds_ns,
                "yhat": np.asarray(median, dtype=float),
                "yhat_raw": np.asarray(raw_values, dtype=float),
                "yhat_lower": np.asarray(lower, dtype=float),
                "yhat_upper": np.asarray(upper, dtype=float),
                "quantiles": pd.array(quantiles_json, dtype="string"),
            },
            columns=list(PREDICTION_COLUMNS),
        )
