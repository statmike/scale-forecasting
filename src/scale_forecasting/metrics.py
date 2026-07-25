"""Forecast metric panel — pure (CONTRACTS §2.3, DESIGN §5.1).

One entry point, ``compute_metrics``, returns the full panel every run so users never
re-run to get a different metric; the decision metric is then a pure config choice
(DESIGN §5.1). All values are floats, with NaN where a metric is undefined (e.g. MAPE
with zeros, MASE/RMSSE without training history, coverage without intervals) rather than
raising — a metric that can't be computed for one cell must not sink the batch.

Public surface: ``compute_metrics``.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# The interval bounds follow the canonical convention (CONTRACTS §2.1): lower at the 0.1
# quantile, upper at the 0.9 quantile. Pinball loss is averaged over both.
_LOWER_Q = 0.1
_UPPER_Q = 0.9

# Panel order — kept identical to config.DecisionMetric / the DDL (single source of truth).
METRIC_NAMES: tuple[str, ...] = (
    "mae", "rmse", "mse", "mape", "smape", "wape",
    "mase", "rmsse", "bias", "coverage", "pinball",
)


def compute_metrics(
    y_true: Sequence[float] | np.ndarray,
    yhat: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray | None = None,
    lower: Sequence[float] | np.ndarray | None = None,
    upper: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full metric panel for one forecast window (CONTRACTS §2.3).

    Args:
        y_true: actuals over the evaluation window.
        yhat: point forecasts, aligned to ``y_true``.
        y_train: training-history actuals; required for scale-free MASE/RMSSE (else NaN).
        lower: lower prediction bound; with ``upper`` enables coverage/pinball (else NaN).
        upper: upper prediction bound.

    Returns:
        ``{name: float}`` for every name in :data:`METRIC_NAMES`. Undefined metrics are NaN.

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
