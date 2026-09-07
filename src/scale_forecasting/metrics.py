"""Forecast metric panel — pure.

One entry point, ``compute_metrics``, returns the full panel every run so users never
re-run to get a different metric; the decision metric is then a pure config choice.
All values are floats, with NaN where a metric is undefined (e.g. MAPE
with zeros, MASE/RMSSE without training history, coverage without intervals) rather than
raising — a metric that can't be computed for one cell must not sink the batch.

Public surface: ``compute_metrics``, ``METRIC_NAMES``, ``METRIC_DIRECTION``, ``loss_of``.

Definitions (n = horizon, e = yhat - y_true):
- mae   = mean(|e|)
- rmse  = sqrt(mean(e²));  mse = mean(e²)
- mape  = mean(|e| / |y_true|)          (NaN if any y_true == 0)
- smape = mean(2|e| / (|y_true| + |yhat|))
- wape  = sum(|e|) / sum(|y_true|)      (NaN if sum(|y_true|) == 0)
- mase  = mae / mae_naive,   naive = one-step (m=1) on y_train
- rmsse = rmse / rmse_naive, naive = one-step (m=1) on y_train
- bias  = mean(e)   (mean error / ME)
- coverage = fraction of y_true within [lower, upper]  (needs intervals)
- pinball  = mean quantile loss across the interval bounds (needs intervals)
- mase_seasonal = mae / mae_naive, naive = *seasonal* (m=`seasonal_period`) on y_train
- maape = mean(arctan(|e| / |y_true|))  — defined at y_true == 0, unlike MAPE
- interval_score  = mean Winkler score of [lower, upper] at α = 0.2  (needs intervals)
- interval_width  = mean(upper - lower)  (needs intervals)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# The interval bounds follow the canonical convention: lower at the 0.1
# quantile, upper at the 0.9 quantile. Pinball loss is averaged over both.
_LOWER_Q = 0.1
_UPPER_Q = 0.9

# The nominal miss rate of the [0.1, 0.9] interval — the α the Winkler interval score
# penalises with. Derived from the bounds above so the two can never drift apart.
_INTERVAL_ALPHA = 2.0 * _LOWER_Q

# Panel order — kept identical to config.DecisionMetric / the DDL (single source of truth).
# New metrics append at the *tail*: the DDL and the Storage Write API spec are generated
# from this order, and only a tail append leaves the existing columns where they are.
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "rmse",
    "mse",
    "mape",
    "smape",
    "wape",
    "mase",
    "rmsse",
    "bias",
    "coverage",
    "pinball",
    "mase_seasonal",
    "maape",
    "interval_score",
    "interval_width",
)

# Which way is better, for every metric in the panel. Three answers, not two:
#
#   "lower"  — an error. Smaller is better, zero is perfect. Most of the panel.
#   "higher" — a fraction of successes. Larger is better, one is perfect. Only `coverage`.
#   "zero"   — a signed quantity where either sign is a fault. Only `bias`.
#
# This table exists because three separate places used to answer the question by assuming "lower",
# and were therefore wrong for two of the fifteen metrics — `inverse_error` gave a model with 50%
# interval coverage nearly twice the weight of one with 95%, `prune_threshold` dropped the accurate
# model and kept the broken one, and a model whose bias happened to be negative got weight zero for
# being *good*. A config can name any of the fifteen as its `decision_metric`, so a lower-is-better
# assumption is not a safe default; it is a silent inversion.
METRIC_DIRECTION: dict[str, str] = {
    "mae": "lower",
    "rmse": "lower",
    "mse": "lower",
    "mape": "lower",
    "smape": "lower",
    "wape": "lower",
    "mase": "lower",
    "rmsse": "lower",
    "bias": "zero",
    "coverage": "higher",
    "pinball": "lower",
    "mase_seasonal": "lower",
    "maape": "lower",
    "interval_score": "lower",
    "interval_width": "lower",
}


def loss_of(metric: str, value: float) -> float:
    """``value`` restated as a **loss**: non-negative, zero is perfect, smaller is better (pure).

    One direction map, one conversion, for every caller that has to rank models — the optimiser,
    the inverse-error weighting and the pruner. Each of those wants the same thing, and each used
    to hard-code its own idea of what "worse" means.

    The conversions:

    * ``lower`` → the value itself.
    * ``higher`` → ``1 - value``. Coverage is the only such metric and it is a fraction, so the
      shortfall from perfect coverage is the natural loss. Returning ``-value`` would rank
      identically but be *negative*, and a negative loss cannot be inverted into a weight.
    * ``zero`` → ``abs(value)``. A bias of −0.1 is better than one of +0.4, and both are worse
      than 0.
    * A metric this table does not know, or a NaN, → ``inf``: unrankable, so it can never win.

    Two caveats worth stating rather than burying. ``coverage`` is treated as higher-is-better
    although it is really best *at nominal* — 98% coverage from a wildly over-wide interval is not
    better than 80% from a calibrated one. Judging it against nominal needs the interval's α,
    which is not in the panel, and `interval_score` is the metric that already penalises width
    honestly. And ``interval_width`` is scored lower-is-better, which in isolation rewards an
    interval of zero width; it is a diagnostic to read beside `coverage`, not a `decision_metric`
    to optimise alone.
    """
    if value != value:  # NaN — a metric that could not be computed ranks last, never first
        return float("inf")
    direction = METRIC_DIRECTION.get(metric)
    if direction == "higher":
        return 1.0 - value
    if direction == "zero":
        return abs(value)
    if direction == "lower":
        return value
    return float("inf")  # unknown metric: no opinion is safer than a wrong one


def compute_metrics(
    y_true: Sequence[float] | np.ndarray,
    yhat: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray | None = None,
    lower: Sequence[float] | np.ndarray | None = None,
    upper: Sequence[float] | np.ndarray | None = None,
    seasonal_period: int | None = None,
) -> dict[str, float]:
    """Compute the full metric panel for one forecast window.

    Args:
        y_true: actuals over the evaluation window.
        yhat: point forecasts, aligned to ``y_true``.
        y_train: training-history actuals; required for scale-free MASE/RMSSE (else NaN).
        lower: lower prediction bound; with ``upper`` enables coverage/pinball (else NaN).
        upper: upper prediction bound.
        seasonal_period: steps in one seasonal cycle, from `seasonality.seasonal_period`;
            required for ``mase_seasonal`` (else NaN). Callers pass the run frequency's
            period rather than a default, because guessing it here would silently score
            an hourly run against a weekly naive.

    Returns:
        ``{name: float}`` for every name in `METRIC_NAMES`. Undefined metrics are NaN.

    Raises:
        ValueError: if ``y_true`` and ``yhat`` have different lengths or are empty.
    """
    yt = np.asarray(y_true, dtype=float)
    yh = np.asarray(yhat, dtype=float)
    if yt.shape != yh.shape:
        raise ValueError(f"y_true and yhat shape mismatch: {yt.shape} vs {yh.shape}")
    if yt.size == 0:
        raise ValueError("y_true is empty")

    err = yh - yt
    abs_err = np.abs(err)

    out: dict[str, float] = {}
    out["mae"] = float(np.mean(abs_err))
    out["mse"] = float(np.mean(err**2))
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["bias"] = float(np.mean(err))
    out["mape"] = _mape(yt, abs_err)
    out["smape"] = _smape(yt, yh, abs_err)
    out["wape"] = _wape(yt, abs_err)
    out["mase"] = _scaled(out["mae"], y_train, kind="mae")
    out["rmsse"] = _scaled(out["rmse"], y_train, kind="rmse")
    out["coverage"] = _coverage(yt, lower, upper)
    out["pinball"] = _pinball(yt, lower, upper)
    out["mase_seasonal"] = _scaled_seasonal(out["mae"], y_train, seasonal_period)
    out["maape"] = _maape(yt, abs_err)
    out["interval_score"] = _interval_score(yt, lower, upper)
    out["interval_width"] = _interval_width(lower, upper)
    return out


# --- individual metrics (each NaN-safe) ----------------------------------------


def _mape(yt: np.ndarray, abs_err: np.ndarray) -> float:
    if np.any(yt == 0):
        return float("nan")  # undefined near zeros
    return float(np.mean(abs_err / np.abs(yt)))


def _smape(yt: np.ndarray, yh: np.ndarray, abs_err: np.ndarray) -> float:
    denom = np.abs(yt) + np.abs(yh)
    # Where both actual and forecast are 0 the term is 0/0 → define as 0 (perfect match).
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2.0 * abs_err / denom)
    return float(np.mean(terms))


def _wape(yt: np.ndarray, abs_err: np.ndarray) -> float:
    denom = float(np.sum(np.abs(yt)))
    if denom == 0:
        return float("nan")
    return float(np.sum(abs_err) / denom)


def _scaled(numerator: float, y_train: object, *, kind: str) -> float:
    """MASE/RMSSE: scale error by the in-sample one-step (m=1) naive error."""
    if y_train is None:
        return float("nan")
    tr = np.asarray(y_train, dtype=float)
    if tr.size < 2:
        return float("nan")
    naive_err = np.abs(np.diff(tr))  # |y_t - y_{t-1}|
    if kind == "mae":
        scale = float(np.mean(naive_err))
    else:  # rmse
        scale = float(np.sqrt(np.mean(naive_err**2)))
    if scale == 0:
        return float("nan")  # flat training history → undefined scaling
    return numerator / scale


def _scaled_seasonal(mae: float, y_train: object, period: int | None) -> float:
    """MASE against a *seasonal* naive (y_t vs y_{t-m}) instead of the one-step naive.

    The m=1 `_scaled` denominator is the last observation, which on a strongly seasonal
    series is an easy baseline to beat — every model scores well and MASE stops
    discriminating. The seasonal naive is the honest baseline there.

    NaN when the period is unknown, when the history is shorter than one full cycle plus
    one step, or when the seasonal naive is exactly flat (nothing to scale by).
    """
    if y_train is None or period is None or period < 1:
        return float("nan")
    tr = np.asarray(y_train, dtype=float)
    if tr.size <= period:
        return float("nan")
    scale = float(np.mean(np.abs(tr[period:] - tr[:-period])))
    if scale == 0:
        return float("nan")
    return mae / scale


def _maape(yt: np.ndarray, abs_err: np.ndarray) -> float:
    """Mean arctangent absolute percentage error — MAPE that survives zeros.

    MAPE is NaN for the whole window if a single actual is 0. MAAPE takes the arctangent
    of the ratio, so a zero actual contributes π/2 (the bounded worst case) instead of
    poisoning everything: intermittent-demand series get a percentage-flavoured score
    they can actually be ranked by. Range is [0, π/2].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(abs_err / np.abs(yt))
    # 0/0 is a perfect match at a zero actual, not an infinite error.
    ratio = np.where((yt == 0) & (abs_err == 0), 0.0, ratio)
    return float(np.mean(np.arctan(ratio)))


def _interval_score(yt: np.ndarray, lower: object, upper: object) -> float:
    """Mean Winkler interval score at α = `_INTERVAL_ALPHA` — sharpness *and* calibration.

    Coverage says whether the actuals fell inside; width says how wide the band was.
    Either alone is trivially gamed (an infinite band covers everything, a zero-width one
    is maximally sharp). The Winkler score is the width plus a 2/α penalty for each miss,
    proportional to how far outside it landed, so it is the single number that ranks
    interval quality. Lower is better.
    """
    if lower is None or upper is None:
        return float("nan")
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    penalty = 2.0 / _INTERVAL_ALPHA
    score = (up - lo) + penalty * np.maximum(lo - yt, 0.0) + penalty * np.maximum(yt - up, 0.0)
    return float(np.mean(score))


def _interval_width(lower: object, upper: object) -> float:
    """Mean width of the prediction interval — the sharpness half of `_interval_score`.

    Reported on its own because it is the one number a planner reads directly: it is in
    the units of the series, so "how uncertain is this forecast" needs no calibration
    lesson to interpret.
    """
    if lower is None or upper is None:
        return float("nan")
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    return float(np.mean(up - lo))


def _coverage(yt: np.ndarray, lower: object, upper: object) -> float:
    if lower is None or upper is None:
        return float("nan")
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    inside = (yt >= lo) & (yt <= up)
    return float(np.mean(inside))


def _pinball(yt: np.ndarray, lower: object, upper: object) -> float:
    """Average pinball (quantile) loss across the lower (0.1) and upper (0.9) bounds."""
    if lower is None or upper is None:
        return float("nan")
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    return float(np.mean([_pinball_q(yt, lo, _LOWER_Q), _pinball_q(yt, up, _UPPER_Q)]))


def _pinball_q(yt: np.ndarray, q_forecast: np.ndarray, q: float) -> float:
    diff = yt - q_forecast
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))
