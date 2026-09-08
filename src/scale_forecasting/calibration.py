"""Out-of-fold calibration of the point forecast and the prediction interval — pure.

Every model has to emit one number per future date, so something decides what that number
*means*. Until this module existed, the decision was an accident: ten of sixteen models built
their band with `BaseModel.residual_intervals`, `_assemble_frame` took the 0.5 quantile of that
band as `yhat`, and the shipped point forecast was silently `prediction + median(in-sample
residual)`. Nobody chose it, nothing named it, and the other six models got no such correction —
so the leaderboard ranked corrected models against uncorrected ones and called it a comparison.

Measurement said the correction was worth keeping (it moved fleet WAPE by 5.7%, because the median
minimises absolute error and that is what the default metric rewards), so it is kept — but as a
stated, recorded, reversible choice rather than a side effect. `yhat_raw` is always preserved,
`yhat_adjusted` sits beside it, and `point_forecast_source` says which one `yhat` carries.

Two things this module fixes beyond making the choice visible:

**The residuals were in-sample.** `_set_residuals` records actual−fitted on the *training* data.
A model that nearly interpolates its training set (xgboost, lightgbm) has tiny training residuals,
so its correction is ~0 and — much worse — its interval is far too narrow. Measured fleet coverage
was **0.601 against a nominal 0.8**. The backtest's out-of-fold frame is a genuine held-out
calibration set that every backtest-enabled run already pays for, so we use it.

Measured after the change, on the same fixture (24 series × 500 obs × horizon 28 × 3 folds, the ten
models with no native interval), scored leave-one-fold-out so nothing is graded on the residuals it
was fitted from: **0.790 against nominal 0.8**, up from 0.601. The mean is the smaller half of the
result. Per-model coverage ranged 0.053 (xgboost) to 0.837 (naive_drift) before — a spread of 0.78,
which is what "the number in this column means something different per model" looks like — and
0.755 to 0.821 after. xgboost went 0.053 → 0.807, lightgbm 0.211 → 0.821. A leaderboard can compare
interval quality across models now; before, it could not.

**The band was horizon-flat.** One scalar covered step 1 and step 28, making it too wide early and
too narrow late. In-sample residuals *cannot* fix this — they are all one-step-ahead by
construction — which is why per-step calibration is available only from OOF data, and why
`Calibration.source` distinguishes the two.

**Where the correction lands relative to `transform`.** This module runs on the frame `predict`
already returned, which is in original units — so an OOF correction is a plain additive shift in
original space, whatever the transform was. The old in-sample correction was not: it added the
residual quantile in *transformed* space and inverted afterwards, which under `log1p`/`boxcox`
became multiplicative on the way out (accidentally, the textbook back-transform bias correction).
Neither is lost. With no backtest the model's own transformed-space band still ships untouched;
with one, the shift is estimated directly against the loss the metrics actually measure, in the
space they measure it in. Calibration is per cell, so a single series' scale is constant across the
residuals being pooled and an additive shift is well-posed even when the error is multiplicative.

**This is not conformal prediction**, and the distinction is worth stating rather than letting the
resemblance imply a guarantee that is not on offer. Conformal's coverage guarantee assumes an
exchangeable calibration set; rolling-origin backtest residuals are not exchangeable, which is why
the literature carries separate adaptive variants for time series. What this is: out-of-fold
empirical quantile calibration. Much better than in-sample, and honest about what it claims.

Public surface: ``Calibration``, ``StepCalibration``, ``calibrate_from_oof``,
``apply_calibration``, ``compare_arms``, ``coverage_by_step``, ``POINT_FORECAST_ARMS``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

# Which functional `yhat` carries. "raw" is the model's own output untouched; "median" and "mean"
# add the corresponding residual statistic. Config defaults this from `decision_metric` rather than
# to a constant, because the choice is not free: the median minimises absolute error and the mean
# minimises squared error, so `rmse` with a median point forecast is an incoherent pair.
POINT_FORECAST_ARMS: tuple[str, ...] = ("raw", "median", "mean")

# Calibration provenance, most to least informative. Recorded per cell so a reader can tell a
# genuinely held-out band from the in-sample fallback without inferring it from the config.
CALIBRATION_SOURCES: tuple[str, ...] = ("oof-per-step", "oof-flat", "in-sample", "native")

# The prediction frame after calibration: `models.base_model.PREDICTION_COLUMNS` plus the one
# column only this module can produce. A model cannot emit `yhat_adjusted` — the correction is
# estimated from folds the model itself never sees — so putting it in the model contract would mean
# sixteen models each filling a column none of them owns. The two constants stay separate and
# `tests/unit/test_calibration.py` pins the relationship, which is the part that could drift.
# Spelled out literally rather than derived, because importing `base_model` here would reach
# `models/__init__` and pull the whole model stack onto the submit path.
CALIBRATED_COLUMNS: tuple[str, ...] = (
    "ds",
    "yhat",
    "yhat_raw",
    "yhat_adjusted",
    "yhat_lower",
    "yhat_upper",
    "quantiles",
)

# Minimum residuals behind one horizon step before we believe a per-step quantile.
#
# This is the constraint that shapes the whole module. Within one series, horizon step h has
# exactly `n_folds` residuals — three, at the common default. Three points cannot locate a 10th
# percentile, so a naive per-step estimator would be almost pure noise, and it would be noise that
# *looks* like signal because the band would visibly vary with h.
#
# The fix that stays inside a single cell is a widening neighbourhood: step 5 borrows from steps
# 4-6, then 3-7, until it has enough samples. That captures the real growth from step 1 to step 28
# without pretending to resolve step 5 from step 6. At three folds and a 28-step horizon the window
# settles at roughly ±2 steps — fifteen residuals, a real per-step estimate over a smoothed
# neighbourhood. It degrades continuously from there: only when the horizon itself is short enough
# that ±2 steps *is* the whole horizon does every step land on the same sample, and then the result
# is the pooled band we already had, labelled `oof-flat` rather than dressed up as per-step.
#
# Twelve is where the window stops widening, not where the estimate becomes good. Small-sample
# empirical quantiles under-cover badly on their own — twelve residuals achieve 0.68 against a
# nominal 0.8 — so the window is only half the answer; `_empirical_quantile` is the other half, and
# it is what makes a twelve-residual sample produce a band worth shipping rather than a narrow one.
#
# The alternative — pooling the *shape* across series while keeping the *scale* per series — is
# strictly better and genuinely available at our scale, but it needs a cross-series reduce stage
# that does not exist in the worker. Left as follow-on work rather than faked here.
_MIN_RESIDUALS_PER_STEP = 12

# Minimum residuals overall before any OOF calibration is attempted. Below this the in-sample
# fallback is not obviously worse and the OOF estimate is certainly unstable.
_MIN_RESIDUALS_TOTAL = 8


@dataclass(frozen=True)
class StepCalibration:
    """Residual statistics for one horizon step (or pooled across all of them)."""

    quantiles: dict[float, float]
    mean: float
    n: int


@dataclass(frozen=True)
class Calibration:
    """What the out-of-fold residuals say about this cell's error, by horizon step.

    `by_step` is empty when no step cleared `_MIN_RESIDUALS_PER_STEP` on its own; `pooled` is
    always populated and is the fallback for any step missing from `by_step`. `source` records
    which of the two actually did the work, so the registry carries the distinction rather than
    leaving a reader to infer it from the fold count.
    """

    by_step: dict[int, StepCalibration]
    pooled: StepCalibration
    source: str

    def for_step(self, step: int) -> StepCalibration:
        """Per-step statistics if they were estimable, else the pooled fallback."""
        return self.by_step.get(step, self.pooled)


def _empirical_quantile(sorted_residuals: np.ndarray, q: float) -> float:
    """A residual quantile that does not systematically under-cover on a small sample.

    `np.quantile` interpolates *between* order statistics, which makes it a fine point estimate and
    a poor tail bound: at n=14 its 10th/90th pair sits at index 1.3 and 11.7, leaving two of the
    fourteen points outside on each side. Achieved coverage is then 0.71 against a nominal 0.8, and
    the gap does not close until n is in the hundreds — measured at 0.645 (n=8), 0.700 (n=14), 0.748
    (n=30), 0.787 (n=120). That is exactly the sample size a three-fold backtest produces, so left
    alone it would hand back a band that is too narrow at the fold counts people actually run.

    The fix is to take an order statistic outward of the interpolated point rather than through it:
    the ``ceil((n+1)q)``-th for an upper quantile, the ``floor((n+1)q)``-th for a lower one. The
    median is left on the plain interpolated value — this widens tails, it does not move the centre,
    and the centre is what the point-forecast arm reads. Simulated coverage becomes 0.78–0.87 across
    n=8…120 instead of 0.65–0.79: slightly conservative on the smallest samples, which is the right
    direction for an interval. A band a little too wide is honest; one too narrow is a false claim.

    This is the finite-sample rank correction, not conformal prediction — see the module docstring.
    Borrowing the arithmetic does not import the guarantee, which needs exchangeability we do not
    have.
    """
    n = sorted_residuals.size
    if q == 0.5:
        return float(np.quantile(sorted_residuals, q))
    rank = math.ceil((n + 1) * q) if q > 0.5 else math.floor((n + 1) * q)
    return float(sorted_residuals[min(max(rank - 1, 0), n - 1)])


def _summarize(residuals: np.ndarray, quantiles: Sequence[float]) -> StepCalibration:
    """Quantiles + mean of one residual sample. Assumes non-empty, NaN-free input."""
    ordered = np.sort(residuals)
    return StepCalibration(
        quantiles={float(q): _empirical_quantile(ordered, float(q)) for q in quantiles},
        mean=float(np.mean(residuals)),
        n=int(residuals.size),
    )


def _window_residuals(
    by_step: dict[int, np.ndarray], step: int, steps: Sequence[int]
) -> np.ndarray:
    """Residuals for `step`, widening symmetrically into neighbours until there are enough.

    Returns an empty array if even the full horizon cannot reach the threshold — the caller then
    falls back to pooled, which is the same sample but honestly labelled.
    """
    radius, span = 0, max(steps) - min(steps)
    while radius <= span:
        picked = [by_step[s] for s in steps if abs(s - step) <= radius and s in by_step]
        stacked = np.concatenate(picked) if picked else np.empty(0)
        if stacked.size >= _MIN_RESIDUALS_PER_STEP:
            return stacked
        radius += 1
    return np.empty(0)


def calibrate_from_oof(
    oof: pd.DataFrame,
    quantiles: Sequence[float],
    *,
    residual_column: str = "yhat_raw",
) -> Calibration | None:
    """Learn residual quantiles from a cell's out-of-fold frame, by horizon step.

    Residuals are ``y_true − yhat_raw`` — measured against the model's *own* output, never against
    an already-corrected number, or the correction would be estimated from data it had already
    been applied to. Returns None when there is not enough held-out data to beat the in-sample
    fallback; the caller keeps the model's own band and records `source="in-sample"`.
    """
    if oof is None or oof.empty:
        return None
    needed = {"y_true", residual_column}
    if not needed.issubset(oof.columns):
        return None

    resid = pd.to_numeric(oof["y_true"], errors="coerce") - pd.to_numeric(
        oof[residual_column], errors="coerce"
    )
    steps_raw = (
        pd.to_numeric(oof["horizon_step"], errors="coerce")
        if "horizon_step" in oof.columns
        else pd.Series(np.nan, index=oof.index)
    )
    ok = resid.notna()
    if int(ok.sum()) < _MIN_RESIDUALS_TOTAL:
        return None

    pooled = _summarize(resid[ok].to_numpy(dtype=float), quantiles)

    # A step column that is absent or all-NaN means the caller has residuals but no idea which
    # horizon distance produced them — pooled is then the only defensible answer, not a weaker
    # version of a per-step one.
    have_steps = ok & steps_raw.notna()
    if not bool(have_steps.any()):
        return Calibration(by_step={}, pooled=pooled, source="oof-flat")

    grouped: dict[int, np.ndarray] = {
        int(step): group.to_numpy(dtype=float)
        for step, group in resid[have_steps].groupby(steps_raw[have_steps].astype(int))
    }
    steps = sorted(grouped)
    by_step: dict[int, StepCalibration] = {}
    for step in steps:
        sample = _window_residuals(grouped, step, steps)
        if sample.size:
            by_step[step] = _summarize(sample, quantiles)

    # Every step landing on the identical widened sample means the window opened all the way and
    # "per-step" would be the pooled band wearing a better label. Say pooled.
    distinct = {tuple(sorted(s.quantiles.items())) + (s.mean,) for s in by_step.values()}
    if not by_step or len(distinct) == 1:
        return Calibration(by_step={}, pooled=pooled, source="oof-flat")
    return Calibration(by_step=by_step, pooled=pooled, source="oof-per-step")


def apply_calibration(
    pred: pd.DataFrame, cal: Calibration | None, arm: str
) -> tuple[pd.DataFrame, str]:
    """Rewrite a prediction frame's band and point forecast from `cal`, and select the arm.

    `pred` must carry `yhat_raw`; the frame is not mutated. Returns the new frame and the
    calibration source actually used, which is `"in-sample"` when `cal` is None — the model's own
    band is then left exactly as it was rather than being recomputed from nothing.

    Row *i* is horizon step *i+1*: the prediction frame is horizon-ordered by construction
    (`_future_index` builds it that way and every model returns it unchanged), which is the same
    convention `backtest.py` writes into the OOF `horizon_step` column.
    """
    if arm not in POINT_FORECAST_ARMS:
        raise ValueError(
            f"unknown point_forecast arm {arm!r}; expected one of {POINT_FORECAST_ARMS}"
        )

    out = pred.copy()
    raw = pd.to_numeric(out["yhat_raw"], errors="coerce").to_numpy(dtype=float)

    if cal is None:
        # No held-out data. The band the model produced stands; the only correction available is
        # the in-sample one already baked into `yhat`, which is what `yhat_adjusted` records.
        out["yhat_adjusted"] = out["yhat"].to_numpy(dtype=float)
        out["yhat"] = raw if arm == "raw" else out["yhat_adjusted"].to_numpy(dtype=float)
        return out[list(CALIBRATED_COLUMNS)], "in-sample"

    qs = sorted(cal.pooled.quantiles)
    lo_q, hi_q = qs[0], qs[-1]
    lower, upper, adjusted, qjson = [], [], [], []
    for i in range(len(out)):
        step = cal.for_step(i + 1)
        shift = step.mean if arm == "mean" else step.quantiles.get(0.5, 0.0)
        lower.append(raw[i] + step.quantiles[lo_q])
        upper.append(raw[i] + step.quantiles[hi_q])
        adjusted.append(raw[i] + shift)
        # The full map is rewritten, not just the two bounds it brackets. Leaving `quantiles`
        # holding the model's own band beside a recalibrated `yhat_lower`/`yhat_upper` would ship
        # a row that disagrees with itself, and a reader pulling q=0.9 out of the JSON would get a
        # different number than the one in the column named for it. Non-finite values are dropped
        # per step for the same reason `_assemble_frame` drops them: `json.dumps` would otherwise
        # mint bare `NaN`, which BigQuery's JSON parser rejects for the whole append.
        qjson.append(
            json.dumps({str(q): v for q in qs if math.isfinite(v := raw[i] + step.quantiles[q])})
        )

    out["yhat_lower"] = np.asarray(lower, dtype=float)
    out["yhat_upper"] = np.asarray(upper, dtype=float)
    out["yhat_adjusted"] = np.asarray(adjusted, dtype=float)
    out["yhat"] = raw if arm == "raw" else out["yhat_adjusted"].to_numpy(dtype=float)
    out["quantiles"] = pd.array(qjson, dtype="string")
    return out[list(CALIBRATED_COLUMNS)], cal.source


def coverage_by_step(oof: pd.DataFrame) -> pd.DataFrame:
    """Achieved coverage per horizon step — the receipt for the calibrated band.

    A single fleet-average coverage figure can sit at nominal while step 1 is over-covered and
    step 28 badly under-covered, which is exactly the failure a horizon-flat band produces. Empty
    frame if the OOF rows lack bounds or a step column.
    """
    needed = {"y_true", "yhat_lower", "yhat_upper", "horizon_step"}
    if oof is None or oof.empty or not needed.issubset(oof.columns):
        return pd.DataFrame(columns=["horizon_step", "coverage", "mean_width", "n"])

    df = oof[list(needed)].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return pd.DataFrame(columns=["horizon_step", "coverage", "mean_width", "n"])

    inside = (df["y_true"] >= df["yhat_lower"]) & (df["y_true"] <= df["yhat_upper"])
    width = df["yhat_upper"] - df["yhat_lower"]
    out = (
        pd.DataFrame(
            {"horizon_step": df["horizon_step"].astype(int), "inside": inside, "width": width}
        )
        .groupby("horizon_step")
        .agg(coverage=("inside", "mean"), mean_width=("width", "mean"), n=("inside", "size"))
        .reset_index()
    )
    return out.sort_values("horizon_step", ignore_index=True)


def _leave_one_fold_out(oof: pd.DataFrame, quantiles: Sequence[float], arm: str) -> np.ndarray:
    """The adjusted arm, refit per fold on the *other* folds. NaN where it is not estimable.

    Scoring the shipped correction against the folds it was estimated from would grade it on its
    own training data and flatter it — badly, at three folds, where one fold is a third of the
    sample. Holding out each fold in turn is the cheap honest version: the same estimator, never
    shown the rows it is graded on.
    """
    out = np.full(len(oof), np.nan)
    folds = pd.to_numeric(oof["fold_id"], errors="coerce")
    raw = pd.to_numeric(oof["yhat_raw"], errors="coerce").to_numpy(dtype=float)
    steps = (
        pd.to_numeric(oof["horizon_step"], errors="coerce")
        if "horizon_step" in oof.columns
        else pd.Series(np.nan, index=oof.index)
    )
    for fold in sorted(set(folds.dropna())):
        held_in = folds != fold
        cal = calibrate_from_oof(oof[held_in], quantiles)
        if cal is None:
            continue
        for pos in np.flatnonzero((folds == fold).to_numpy()):
            step = steps.iloc[pos]
            stats = cal.for_step(int(step)) if pd.notna(step) else cal.pooled
            shift = stats.mean if arm == "mean" else stats.quantiles.get(0.5, 0.0)
            out[pos] = raw[pos] + shift
    return out


def compare_arms(oof: pd.DataFrame, metric: str, arm: str = "median") -> dict[str, Any]:
    """Score `raw` against the corrected arm on held-out folds — the diagnostic, not a by-product.

    Two corrections could be graded here and only one is the one we ship. The `yhat_adjusted` column
    each fold wrote is that fold's *in-sample* residual shift, which is the thing item 2.5
    replaced; the shipped correction is estimated from the out-of-fold residuals of the whole cell.
    So the adjusted arm is refit leave-one-fold-out (`_leave_one_fold_out`) and graded on the folds
    it never saw. With one fold, or too few residuals to calibrate on the remainder, there is
    nothing to hold out — the stored in-sample arm is scored instead and `basis` says so, because a
    margin computed two different ways under one name is worse than a margin that names its method.

    Returns NaN losses (and no winner) when the frame lacks an arm, which is the state for the
    native engine until it emits both.
    """
    # Lazy, both of them: importing `models.base_model` reaches `models/__init__`, which imports
    # every model module and therefore the whole statsmodels/lightgbm/prophet stack. This module
    # is reachable from the submit path, where that stack is not installed.
    from .metrics import compute_metrics, loss_of
    from .models.base_model import DEFAULT_QUANTILES

    blank = {"loss_raw": float("nan"), "loss_adjusted": float("nan"), "margin": float("nan")}
    needed = {"y_true", "yhat_raw", "yhat_adjusted"}
    if oof is None or oof.empty or not needed.issubset(oof.columns):
        return {**blank, "basis": float("nan")}

    df = oof.reset_index(drop=True)
    adjusted = np.full(len(df), np.nan)
    basis = "in-sample-arm"
    if "fold_id" in df.columns and pd.to_numeric(df["fold_id"], errors="coerce").nunique() > 1:
        adjusted = _leave_one_fold_out(df, DEFAULT_QUANTILES, arm)
        if np.isfinite(adjusted).any():
            basis = "leave-one-fold-out"
    if basis == "in-sample-arm":
        adjusted = pd.to_numeric(df["yhat_adjusted"], errors="coerce").to_numpy(dtype=float)

    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    raw = pd.to_numeric(df["yhat_raw"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(raw) & np.isfinite(adjusted)
    if not keep.any():
        return {**blank, "basis": basis}

    losses = {
        name: loss_of(metric, compute_metrics(y[keep], arr[keep]).get(metric, float("nan")))
        for name, arr in (("raw", raw), ("adjusted", adjusted))
    }
    raw_loss, adj_loss = losses["raw"], losses["adjusted"]
    # Relative improvement of the adjusted arm over raw. Positive means the correction helped.
    # Always measured in that direction, whichever arm the run selected, so a fleet-wide average
    # over cells that chose differently is still a single comparable number.
    margin = float("nan")
    if np.isfinite(raw_loss) and np.isfinite(adj_loss) and raw_loss > 0:
        margin = (raw_loss - adj_loss) / raw_loss
    return {"loss_raw": raw_loss, "loss_adjusted": adj_loss, "margin": margin, "basis": basis}
