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
    """BigQuery ML ``ARIMA_PLUS`` (univariate; custom holidays for parity with features.py)."""

    name = "arima_plus"
    supports_exog = False


class TimesFm(_BigQueryNativeModel):
    """BigQuery ``AI.FORECAST`` with TimesFM (2.0 / 2.5 / 3.0; pretrained, no training step)."""

    name = "timesfm"
    supports_exog = False

    @classmethod
    def validate_params(cls, params: dict[str, Any], *, max_horizon: int) -> None:
        """Validate ``model_params.timesfm`` (version 2.0/2.5/3.0, context_window, horizon)."""
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
