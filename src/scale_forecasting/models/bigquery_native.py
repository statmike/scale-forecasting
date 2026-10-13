"""BigQuery-native models — arima_plus / timesfm.

These register through the *same* factory (``runtime="bigquery"``) so the router and registry
treat them uniformly with the Python models, but they are **executed as SQL** by
``engines/bigquery_engine.py`` — never by this Python code. That is by design, not a gap: the
classes exist so fan-out, routing, and the registry see them as first-class models, while the
actual ``CREATE MODEL`` / ``ML.FORECAST`` / ``AI.FORECAST`` runs in BigQuery. The in-process
``fit``/``predict`` therefore raise `BigQueryNativeExecutionError` — reaching them means a
native model was mistakenly dispatched to the Python worker path instead of the BigQuery engine.

Two models, two classes, one file — they share nothing but a runtime and an identical
"executed in BigQuery" stance, so a thin in-file base keeps each a real registered model while
avoiding copies of the same guard (the one-model-one-file rule is about *authorship locality*, and
both native models are authored here against the same BigQuery contract).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import ConfigError
from .base_model import BaseModel, register

if TYPE_CHECKING:
    from collections.abc import Mapping

_EXECUTED_IN_BIGQUERY = (
    "BigQuery-native models execute as SQL in engines/bigquery_engine.py, not in the Python "
    "worker — this method should never be called. Route native models through the BigQuery engine."
)

# Canonical BigQuery AI.FORECAST model names (`model => '...'`).
# Shorthands ("2.0", "2.5", "3.0") normalize to the canonical BigQuery literal.
TIMESFM_VERSIONS: dict[str, str] = {
    "2.0": "TimesFM 2.0",
    "2.5": "TimesFM 2.5",
    "3.0": "TimesFM 3.0",
    "timesfm 2.0": "TimesFM 2.0",
    "timesfm 2.5": "TimesFM 2.5",
    "timesfm 3.0": "TimesFM 3.0",
}

_TIMESFM_2X_CONTEXT_WINDOWS: frozenset[int] = frozenset(
    {64, 128, 256, 512, 1024, 2048, 4096, 8192, 15360}
)
_TIMESFM_30_CONTEXT_WINDOWS: frozenset[int] = frozenset(n * 32 for n in range(2, 65))
_TIMESFM_ALLOWED_KEYS: frozenset[str] = frozenset(
    {"model", "version", "context_window", "training_mode"}
)

_ARIMA_PLUS_FORBIDDEN_BOUND_KEYS: frozenset[str] = frozenset(
    {"forecast_limit_lower_bound", "forecast_limit_upper_bound"}
)
_ARIMA_PLUS_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "auto_arima_max_order",
        "auto_arima_min_order",
        "clean_spikes_and_dips",
        "adjust_step_changes",
        "decompose_time_series",
        "holiday_region",
        "seasonalities",
        "time_series_length_fraction",
        "min_time_series_length",
        "max_time_series_length",
        "trend_smoothing_window_size",
        "training_mode",
    }
)
_ARIMA_PLUS_SEASONALITIES: frozenset[str] = frozenset(
    {"NO_SEASONALITY", "AUTO", "DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY"}
)


def resolve_arima_plus_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize ``model_params.arima_plus`` into BigQuery ``OPTIONS(...)`` entries.

    Returns an ordered dict of authored ``CREATE MODEL`` options (empty when un-authored so default
    SQL generation stays byte-identical). Explicitly rejects ``forecast_limit_lower_bound`` and
    ``forecast_limit_upper_bound`` (pointing to ``output.floor`` / ``output.cap``) as well as
    ``auto_arima=False`` / ``non_seasonal_order`` (unsupported on multi-series
    ``time_series_id_col`` models in BigQuery ML).
    """
    bound_keys = sorted(set(params) & _ARIMA_PLUS_FORBIDDEN_BOUND_KEYS)
    if bound_keys:
        raise ConfigError(
            f"model_params.arima_plus specifies {bound_keys}, which BigQuery ML applies via an "
            "internal transformation that requires both bounds together, forces "
            "decompose_time_series=TRUE, and frequently degrades multi-series fits at scale. "
            "Configure physical forecast bounds via 'output.floor' and 'output.cap' instead."
        )
    if "non_seasonal_order" in params or params.get("auto_arima") is False:
        raise ConfigError(
            "model_params.arima_plus specifies manual ARIMA order ('auto_arima=False' or "
            "'non_seasonal_order'), which BigQuery ML does not support on multi-series models "
            "using 'time_series_id_col'. Use 'auto_arima_min_order' and 'auto_arima_max_order' "
            "(1..5) instead."
        )
    if "auto_arima" in params and params["auto_arima"] is not True:
        raise ConfigError(
            f"model_params.arima_plus.auto_arima={params['auto_arima']!r} must be True "
            "(BigQuery ML multi-series ARIMA_PLUS requires auto_arima=TRUE)."
        )
    unknown = sorted(set(params) - _ARIMA_PLUS_ALLOWED_KEYS - {"auto_arima"})
    if unknown:
        raise ConfigError(
            f"model_params.arima_plus contains unsupported keys {unknown}; "
            f"supported keys: {sorted(_ARIMA_PLUS_ALLOWED_KEYS - {'training_mode'})}."
        )

    out: dict[str, Any] = {}
    max_order = params.get("auto_arima_max_order")
    if max_order is not None:
        if (
            isinstance(max_order, bool)
            or not isinstance(max_order, int)
            or not (1 <= max_order <= 5)
        ):
            raise ConfigError(
                f"model_params.arima_plus.auto_arima_max_order={max_order!r} must be an integer "
                "between 1 and 5."
            )
        out["auto_arima_max_order"] = int(max_order)

    min_order = params.get("auto_arima_min_order")
    if min_order is not None:
        if (
            isinstance(min_order, bool)
            or not isinstance(min_order, int)
            or not (0 <= min_order <= 5)
        ):
            raise ConfigError(
                f"model_params.arima_plus.auto_arima_min_order={min_order!r} must be an integer "
                "between 0 and 5."
            )
        out["auto_arima_min_order"] = int(min_order)

    if (
        "auto_arima_min_order" in out
        and "auto_arima_max_order" in out
        and out["auto_arima_min_order"] > out["auto_arima_max_order"]
    ):
        raise ConfigError(
            f"model_params.arima_plus.auto_arima_min_order ({out['auto_arima_min_order']}) "
            f"cannot exceed auto_arima_max_order ({out['auto_arima_max_order']})."
        )

    for bool_key in ("clean_spikes_and_dips", "adjust_step_changes", "decompose_time_series"):
        val = params.get(bool_key)
        if val is not None:
            if not isinstance(val, bool):
                raise ConfigError(f"model_params.arima_plus.{bool_key}={val!r} must be a boolean.")
            out[bool_key] = val

    hol_reg = params.get("holiday_region")
    if hol_reg is not None:
        if isinstance(hol_reg, str):
            if not hol_reg.strip():
                raise ConfigError("model_params.arima_plus.holiday_region must be non-empty.")
            out["holiday_region"] = hol_reg.strip()
        elif (
            isinstance(hol_reg, list)
            and hol_reg
            and all(isinstance(x, str) and x.strip() for x in hol_reg)
        ):
            out["holiday_region"] = [x.strip() for x in hol_reg]
        else:
            raise ConfigError(
                f"model_params.arima_plus.holiday_region={hol_reg!r} must be a non-empty string "
                "or list of strings (e.g. 'US' or ['US', 'GB'])."
            )

    seas = params.get("seasonalities")
    if seas is not None:
        if not isinstance(seas, list) or not seas or not all(isinstance(x, str) for x in seas):
            raise ConfigError(
                f"model_params.arima_plus.seasonalities={seas!r} must be a non-empty list of "
                f"strings from {sorted(_ARIMA_PLUS_SEASONALITIES)}."
            )
        norm_seas = [x.strip().upper() for x in seas]
        bad_seas = [x for x in norm_seas if x not in _ARIMA_PLUS_SEASONALITIES]
        if bad_seas:
            raise ConfigError(
                f"model_params.arima_plus.seasonalities contains invalid entry {bad_seas}; "
                f"supported values: {sorted(_ARIMA_PLUS_SEASONALITIES)}."
            )
        out["seasonalities"] = norm_seas

    ts_frac = params.get("time_series_length_fraction")
    if ts_frac is not None:
        if (
            isinstance(ts_frac, bool)
            or not isinstance(ts_frac, (int, float))
            or not (0.0 < float(ts_frac) <= 1.0)
        ):
            raise ConfigError(
                f"model_params.arima_plus.time_series_length_fraction={ts_frac!r} must be a "
                "float in (0.0, 1.0]."
            )
        out["time_series_length_fraction"] = float(ts_frac)

    for int_key in (
        "min_time_series_length",
        "max_time_series_length",
        "trend_smoothing_window_size",
    ):
        ival = params.get(int_key)
        if ival is not None:
            if isinstance(ival, bool) or not isinstance(ival, int) or ival <= 0:
                raise ConfigError(
                    f"model_params.arima_plus.{int_key}={ival!r} must be a positive integer."
                )
            out[int_key] = int(ival)

    if (
        "min_time_series_length" in out
        and "max_time_series_length" in out
        and out["min_time_series_length"] > out["max_time_series_length"]
    ):
        raise ConfigError(
            f"model_params.arima_plus.min_time_series_length ({out['min_time_series_length']}) "
            f"cannot exceed max_time_series_length ({out['max_time_series_length']})."
        )

    return out


def resolve_timesfm_params(params: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """Normalize and validate ``model_params.timesfm`` into ``(canonical_model, context_window)``.

    Accepts either ``version`` or ``model`` (e.g. ``"3.0"`` or ``"TimesFM 3.0"``) and optional
    ``context_window``. Returns ``(None, None)`` when un-authored so default SQL generation omits
    optional arguments and leaves BigQuery's own default (``TimesFM 2.5``) in place.
    """
    unknown = sorted(set(params) - _TIMESFM_ALLOWED_KEYS)
    if unknown:
        raise ConfigError(
            f"model_params.timesfm contains unsupported keys {unknown}; "
            "supported keys: ['model', 'version', 'context_window']."
        )
    raw_model = params.get("model")
    raw_version = params.get("version")
    if raw_model is not None and raw_version is not None:
        norm_m = TIMESFM_VERSIONS.get(str(raw_model).strip().lower())
        norm_v = TIMESFM_VERSIONS.get(str(raw_version).strip().lower())
        if norm_m != norm_v or norm_m is None:
            raise ConfigError(
                f"model_params.timesfm specifies conflicting model={raw_model!r} and "
                f"version={raw_version!r}; specify only one."
            )
    raw = raw_model if raw_model is not None else raw_version
    canonical_model: str | None = None
    if raw is not None:
        canonical_model = TIMESFM_VERSIONS.get(str(raw).strip().lower())
        if canonical_model is None:
            raise ConfigError(
                f"model_params.timesfm version {raw!r} is invalid; supported versions: "
                "['TimesFM 2.0', 'TimesFM 2.5', 'TimesFM 3.0'] (or '2.0', '2.5', '3.0')."
            )

    raw_ctx = params.get("context_window")
    context_window: int | None = None
    if raw_ctx is not None:
        if isinstance(raw_ctx, bool) or not isinstance(raw_ctx, int):
            raise ConfigError(
                f"model_params.timesfm.context_window={raw_ctx!r} must be an integer."
            )
        context_window = int(raw_ctx)
        effective_model = canonical_model or "TimesFM 2.5"
        if effective_model == "TimesFM 3.0":
            if context_window not in _TIMESFM_30_CONTEXT_WINDOWS:
                raise ConfigError(
                    f"model_params.timesfm.context_window={context_window} is invalid for "
                    "TimesFM 3.0; must be a multiple of 32 between 64 and 2048."
                )
        elif context_window not in _TIMESFM_2X_CONTEXT_WINDOWS:
            raise ConfigError(
                f"model_params.timesfm.context_window={context_window} is invalid for "
                f"{effective_model}; supported values: {sorted(_TIMESFM_2X_CONTEXT_WINDOWS)}."
            )

    return canonical_model, context_window


class BigQueryNativeExecutionError(NotImplementedError):
    """Raised if a BigQuery-native model's in-process fit/predict is called.

    Native models run as SQL in `engines.bigquery_engine`; hitting this means one was
    dispatched to the Python worker path by mistake. Subclasses `NotImplementedError` so
    existing ``except NotImplementedError`` handlers and tests keep working.
    """


class _BigQueryNativeModel(BaseModel):
    """Shared stance for native models: registered here, executed as SQL in BigQuery."""

    runtime = "bigquery"
    family = "native"
    supports_native_intervals = True  # ML.FORECAST returns prediction-interval bounds
    package = "bigquery-ml"
    # Executed by BigQuery, submitted by the client: without [gcp] there is no way to run them, so
    # a bare install lists them as unavailable rather than failing at submit.
    optional_import = "google.cloud.bigquery"
    optional_extra = "gcp"
    package_url = "https://cloud.google.com/bigquery/docs/bqml-introduction"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        raise BigQueryNativeExecutionError(_EXECUTED_IN_BIGQUERY)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        raise BigQueryNativeExecutionError(_EXECUTED_IN_BIGQUERY)


class ArimaPlus(_BigQueryNativeModel):
    """BigQuery ML ``ARIMA_PLUS`` / ``ARIMA_PLUS_XREG`` with ``ML.EXPLAIN_FORECAST``."""

    name = "arima_plus"
    supports_exog = True
    supports_future_covariates = True
    supports_past_covariates = False
    supports_static_covariates = False
    supports_explainability = True

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Validate ``model_params.arima_plus`` options and reject unsupported bound knobs."""
        super().validate_params(params, max_horizon=max_horizon)
        resolve_arima_plus_params(params)


class TimesFm(_BigQueryNativeModel):
    """BigQuery ``AI.FORECAST`` with TimesFM (2.0 / 2.5 / 3.0; univariate + covariates)."""

    name = "timesfm"
    supports_exog = True
    supports_future_covariates = True
    supports_past_covariates = True
    supports_static_covariates = False

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Validate ``model_params.timesfm`` (version 2.0/2.5/3.0, context_window, horizon)."""
        super().validate_params(params, max_horizon=max_horizon)
        canonical_model, _ = resolve_timesfm_params(params)
        effective_model = canonical_model or "TimesFM 2.5"
        if effective_model == "TimesFM 3.0" and max_horizon > 1024:
            raise ConfigError(
                f"model_params.timesfm selects 'TimesFM 3.0', which supports a maximum horizon of "
                f"1024 in BigQuery AI.FORECAST, but this run requires max_horizon={max_horizon}."
            )
        if max_horizon > 10000:
            raise ConfigError(
                f"BigQuery AI.FORECAST ({effective_model}) supports a maximum horizon of 10000, "
                f"but this run requires max_horizon={max_horizon}."
            )


register(ArimaPlus)
register(TimesFm)
