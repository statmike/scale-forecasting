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
``apply_calibration``, ``apply_output_constraints``, ``has_output_constraints``,
``compare_arms``, ``coverage_by_step``, ``select_arm``, ``resolve_arm``,
``POINT_FORECAST_ARMS``.
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

    from .config import OutputConfig

# Which functional `yhat` carries. "raw" is the model's own output untouched; "median" and "mean"
# add the corresponding residual statistic. Config defaults this from `decision_metric` rather than
# to a constant, because the choice is not free: the median minimises absolute error and the mean
# minimises squared error, so `rmse` with a median point forecast is an incoherent pair.
POINT_FORECAST_ARMS: tuple[str, ...] = ("raw", "median", "mean")

# Calibration provenance, most to least informative. Recorded per cell so a reader can tell a
# genuinely held-out band from the in-sample fallback without inferring it from the config.
CALIBRATION_SOURCES: tuple[str, ...] = ("oof-per-step", "oof-flat", "in-sample", "native")

# How a cell's arm got chosen — the companion to `point_forecast_source`, which says only *what* was
# chosen. Under a fleetwide setting every cell reads "configured" and the column is dull, which is
# correct; under `auto` it is the record of how much evidence each cell actually had.
ARM_DECISIONS: tuple[str, ...] = (
    "configured",  # the config named a concrete arm; nothing was decided here
    "auto-raw",  # enough held-out folds, and they favoured the raw arm
    "auto-corrected",  # enough held-out folds, and they favoured the correction
    "auto-few-folds",  # fewer than `_MIN_FOLDS_FOR_AUTO` usable folds; fleetwide default applied
    "auto-no-backtest",  # the cell's backtest degraded to nothing; fleetwide default applied
    "engine-native",  # BigQuery ML: one arm exists, so there was never a choice to make
)

# How many held-out folds `auto` needs before it will act on their verdict.
#
# Three, and the number is measured rather than chosen for caution. The selection rule below is a
# per-fold sign test, and at two folds it degenerates: the inner comparison has a single fold to
# grade on, so it grades the correction on the residuals it was fitted from, and the verdict is
# optimistic exactly where it is least informed. Three is the smallest count at which every inner
# comparison is genuinely held out.
#
# The temptation is to set this higher — a sign test on three folds is a weak instrument. Measured
# fleet MAE on the ten no-native-interval models over 24 series, scored nested leave-one-fold-out
# so no rule is graded on the folds it selected from:
#
#             always raw   always corrected   auto   oracle
#   3 folds      20.09          19.87        18.75    17.41
#   5 folds      20.00          17.85        17.38    16.00
#   8 folds      19.57          17.96        17.43    16.36
#
# Selection beats both fleetwide arms at every fold count, and its *largest* relative win is at
# three — the fold count most runs actually use. A minimum of five would have discarded it.
_MIN_FOLDS_FOR_AUTO = 3

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


def _calibrate_from_arrays(
    resid: np.ndarray,
    steps_raw: np.ndarray | None,
    quantiles: Sequence[float],
) -> Calibration | None:
    """Fast numpy-array core of `calibrate_from_oof` (avoids per-cell pandas Series overhead)."""
    ok = np.isfinite(resid)
    if int(np.count_nonzero(ok)) < _MIN_RESIDUALS_TOTAL:
        return None

    pooled = _summarize(resid[ok], quantiles)

    if steps_raw is None:
        return Calibration(by_step={}, pooled=pooled, source="oof-flat")

    have_steps = ok & np.isfinite(steps_raw)
    if not bool(np.any(have_steps)):
        return Calibration(by_step={}, pooled=pooled, source="oof-flat")

    valid_resid = resid[have_steps]
    valid_steps = steps_raw[have_steps].astype(int)
    unique_steps = np.unique(valid_steps)
    grouped: dict[int, np.ndarray] = {int(s): valid_resid[valid_steps == s] for s in unique_steps}
    steps = [int(s) for s in unique_steps]
    by_step: dict[int, StepCalibration] = {}
    for step in steps:
        sample = _window_residuals(grouped, step, steps)
        if sample.size:
            by_step[step] = _summarize(sample, quantiles)

    distinct = {(*sorted(s.quantiles.items()), s.mean) for s in by_step.values()}
    if not by_step or len(distinct) == 1:
        return Calibration(by_step={}, pooled=pooled, source="oof-flat")
    return Calibration(by_step=by_step, pooled=pooled, source="oof-per-step")


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

    resid = (
        pd.to_numeric(oof["y_true"], errors="coerce")
        - pd.to_numeric(oof[residual_column], errors="coerce")
    ).to_numpy(dtype=float)
    steps_raw = (
        pd.to_numeric(oof["horizon_step"], errors="coerce").to_numpy(dtype=float)
        if "horizon_step" in oof.columns
        else None
    )
    return _calibrate_from_arrays(resid, steps_raw, quantiles)


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
        raw_qmap = {q: v for q in qs if math.isfinite(v := raw[i] + step.quantiles[q])}
        mono_qmap = rearrange_monotone_quantiles(raw_qmap)
        lower.append(mono_qmap.get(lo_q, raw[i] + step.quantiles[lo_q]))
        upper.append(mono_qmap.get(hi_q, raw[i] + step.quantiles[hi_q]))
        adjusted.append(raw[i] + shift)
        # The full map is rewritten, not just the two bounds it brackets. Leaving `quantiles`
        # holding the model's own band beside a recalibrated `yhat_lower`/`yhat_upper` would ship
        # a row that disagrees with itself, and a reader pulling q=0.9 out of the JSON would get a
        # different number than the one in the column named for it. Non-finite values are dropped
        # per step for the same reason `_assemble_frame` drops them: `json.dumps` would otherwise
        # mint bare `NaN`, which BigQuery's JSON parser rejects for the whole append.
        qjson.append(json.dumps({str(q): v for q, v in mono_qmap.items()}))

    out["yhat_lower"] = np.asarray(lower, dtype=float)
    out["yhat_upper"] = np.asarray(upper, dtype=float)
    out["yhat_adjusted"] = np.asarray(adjusted, dtype=float)
    out["yhat"] = raw if arm == "raw" else out["yhat_adjusted"].to_numpy(dtype=float)
    out["quantiles"] = pd.array(qjson, dtype="string")
    return out[list(CALIBRATED_COLUMNS)], cal.source


def rearrange_monotone_quantiles(q_map: dict[float, float]) -> dict[float, float]:
    """Chernozhukov, Fernández-Val & Galichon (2010) monotone quantile rearrangement.

    Sorts the finite quantile levels ``q_1 < q_2 < ... < q_k`` and arranges their values into
    non-decreasing order ``v_(1) <= v_(2) <= ... <= v_(k)``, eliminating quantile crossing by
    construction while weakly improving $L_p$ estimation accuracy.
    """
    finite_pairs = sorted((float(q), float(v)) for q, v in q_map.items() if math.isfinite(float(v)))
    if not finite_pairs:
        return {}
    qs = [p[0] for p in finite_pairs]
    sorted_vals = np.sort(np.array([p[1] for p in finite_pairs], dtype=float))
    return {q: float(v) for q, v in zip(qs, sorted_vals, strict=True)}


def format_target_quantile_arm(q_star: float) -> str:
    """Canonical ``point_forecast_source`` label for a Newsvendor or SLA quantile target."""
    if round(q_star, 2) == round(q_star, 6):
        return f"quantile:{q_star:.2f}"
    formatted = f"{q_star:.4f}".rstrip("0").rstrip(".")
    return f"quantile:{formatted}"


def apply_quantile_and_cost_rules(
    pred: pd.DataFrame,
    output_cfg: OutputConfig | None,
) -> tuple[pd.DataFrame, str | None]:
    """Apply multi-percentile fan expansion, Chernozhukov monotone rearrangement, and Newsvendor /
    SLA quantile shifting (`underage_cost` / `overage_cost` / `target_quantile`) to ``pred``.

    Returns ``(updated_df, point_source_override)`` where ``point_source_override`` is
    ``"quantile:<q*>"`` when a Newsvendor or SLA target quantile is active, else ``None``.
    """
    if (
        pred is None
        or pred.empty
        or output_cfg is None
        or (output_cfg.quantiles is None and output_cfg.effective_target_quantile is None)
    ):
        return pred, None

    from scipy.stats import norm

    out = pred.copy()
    target_qs = output_cfg.effective_quantiles
    q_star = output_cfg.effective_target_quantile
    z_90 = float(norm.ppf(0.9))

    yh = pd.to_numeric(out["yhat"], errors="coerce").to_numpy(dtype=float)
    yl = (
        pd.to_numeric(out["yhat_lower"], errors="coerce").to_numpy(dtype=float)
        if "yhat_lower" in out.columns
        else yh.copy()
    )
    yu = (
        pd.to_numeric(out["yhat_upper"], errors="coerce").to_numpy(dtype=float)
        if "yhat_upper" in out.columns
        else yh.copy()
    )
    has_qcol = "quantiles" in out.columns
    raw_q_series = out["quantiles"].tolist() if has_qcol else [None] * len(out)

    new_yh = yh.copy()
    new_yl = yl.copy()
    new_yu = yu.copy()
    new_qjson: list[str] = []

    for i in range(len(out)):
        parsed_map: dict[float, float] = {}
        raw_q = raw_q_series[i]
        if isinstance(raw_q, str) and raw_q.strip():
            try:
                loaded = json.loads(raw_q)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if isinstance(v, (int, float)) and math.isfinite(float(v)):
                            parsed_map[round(float(k), 6)] = float(v)
            except (TypeError, ValueError):
                pass

        # Anchor center and spread for any quantile not yet in parsed_map (e.g. backtest=False).
        center = parsed_map.get(0.5, float(yh[i]) if math.isfinite(yh[i]) else 0.0)
        lo_ref = parsed_map.get(0.1, float(yl[i]) if math.isfinite(yl[i]) else center)
        hi_ref = parsed_map.get(0.9, float(yu[i]) if math.isfinite(yu[i]) else center)
        sigma = (
            max((hi_ref - lo_ref) / (2.0 * z_90), 0.0) if math.isfinite(hi_ref - lo_ref) else 0.0
        )

        for q in target_qs:
            rq = round(float(q), 6)
            if rq not in parsed_map and math.isfinite(center):
                parsed_map[rq] = float(center + float(norm.ppf(rq)) * sigma)

        mono_map = rearrange_monotone_quantiles(parsed_map)
        if output_cfg.quantiles is not None:
            keep_set = {round(float(q), 6) for q in output_cfg.quantiles}
            if q_star is not None:
                keep_set.add(q_star)
            emit_map = {q: v for q, v in mono_map.items() if q in keep_set}
        else:
            emit_map = mono_map

        if emit_map:
            emit_qs = sorted(emit_map)
            if output_cfg.quantiles is not None:
                new_yl[i] = emit_map[emit_qs[0]]
                new_yu[i] = emit_map[emit_qs[-1]]
            if q_star is not None and q_star in mono_map:
                new_yh[i] = mono_map[q_star]
            new_yl[i] = min(new_yl[i], new_yh[i])
            new_yu[i] = max(new_yu[i], new_yh[i])
        new_qjson.append(json.dumps({str(q): v for q, v in emit_map.items()}))

    out["yhat"] = new_yh
    if "yhat_lower" in out.columns:
        out["yhat_lower"] = new_yl
    if "yhat_upper" in out.columns:
        out["yhat_upper"] = new_yu
    if has_qcol or output_cfg.quantiles is not None:
        out["quantiles"] = pd.array(new_qjson, dtype="string")

    arm_override = format_target_quantile_arm(q_star) if q_star is not None else None
    return out, arm_override


def has_output_constraints(output_cfg: OutputConfig | None) -> bool:
    """Return True when ``output_cfg`` configures any physical bound or discrete rounding rule."""
    if output_cfg is None:
        return False
    return (
        output_cfg.floor is not None
        or output_cfg.cap is not None
        or output_cfg.floor_col is not None
        or output_cfg.cap_col is not None
        or output_cfg.integer_rounding != "none"
    )


def _cumulative_round(arr: np.ndarray) -> np.ndarray:
    """Error-diffusion (Bresenham) cumulative rounding across the forecast horizon.

    Rounds the running cumulative sum ``round(cumsum(arr))`` and takes first differences so a
    fractional series (for example ``0.2`` units/day over 5 days) emits discrete integer units
    (``[0, 0, 1, 0, 0]``) whose horizon total matches the unrounded cumulative demand instead of
    collapsing to all zeros under step-by-step ``round()``. Preserves ``NaN`` entries in place.
    """
    out = np.asarray(arr, dtype=float).copy()
    finite_mask = np.isfinite(out)
    if not finite_mask.any():
        return out
    finite_vals = out[finite_mask]
    cum_rounded = np.rint(np.cumsum(finite_vals))
    out[finite_mask] = np.diff(cum_rounded, prepend=0.0)
    return out


def _resolve_bound_array(
    n: int,
    scalar_bound: float | None,
    col_name: str | None,
    df: pd.DataFrame,
    bounds_df: pd.DataFrame | None,
    *,
    default: float,
    combine_fn: Any,
) -> np.ndarray:
    """Resolve a per-step lower or upper bound array of length ``n``."""
    bound = np.full(n, float(scalar_bound) if scalar_bound is not None else default, dtype=float)
    if not col_name:
        return bound

    col_vals: np.ndarray | None = None
    if col_name in df.columns:
        col_vals = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
    elif bounds_df is not None and not bounds_df.empty and col_name in bounds_df.columns:
        # Align by date column when both frames carry dates, else positional / carry-last.
        df_date_col = (
            "ds"
            if "ds" in df.columns
            else ("forecast_date" if "forecast_date" in df.columns else None)
        )
        b_date_col = (
            "ds"
            if "ds" in bounds_df.columns
            else ("forecast_date" if "forecast_date" in bounds_df.columns else None)
        )
        if df_date_col is not None and b_date_col is not None:
            b_map = (
                bounds_df[[b_date_col, col_name]]
                .assign(_dt=pd.to_datetime(bounds_df[b_date_col]))
                .drop_duplicates(subset=["_dt"], keep="last")
                .set_index("_dt")[col_name]
            )
            aligned = pd.to_numeric(
                pd.to_datetime(df[df_date_col]).map(b_map), errors="coerce"
            ).to_numpy(dtype=float)
            if np.isfinite(aligned).any():
                # Fill any unmatched future dates with the last finite observation in bounds_df.
                raw_b = pd.to_numeric(bounds_df[col_name], errors="coerce").dropna()
                fallback_val = float(raw_b.iloc[-1]) if not raw_b.empty else np.nan
                col_vals = np.where(np.isfinite(aligned), aligned, fallback_val)
        if col_vals is None:
            raw_b_arr = pd.to_numeric(bounds_df[col_name], errors="coerce").to_numpy(dtype=float)
            if len(raw_b_arr) == n:
                col_vals = raw_b_arr
            else:
                finite_b = raw_b_arr[np.isfinite(raw_b_arr)]
                if finite_b.size:
                    col_vals = np.full(n, float(finite_b[-1]), dtype=float)

    if col_vals is not None and len(col_vals) == n:
        finite = np.isfinite(col_vals)
        bound = np.where(finite, combine_fn(bound, col_vals), bound)
    return bound


def _round_array(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "round":
        return np.rint(arr)
    if mode == "ceil":
        return np.ceil(arr)
    if mode == "floor":
        return np.floor(arr)
    if mode == "cumulative_round":
        return _cumulative_round(arr)
    return arr


def apply_output_constraints(
    df: pd.DataFrame,
    output_cfg: OutputConfig | None,
    bounds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply physical bounds (``floor``, ``cap``, ``floor_col``, ``cap_col``) and discrete
    ``integer_rounding`` post-calibration (pure).

    Crucially, ``yhat_raw`` is **never** modified: `calibrate_from_oof` estimates empirical
    residuals from ``y_true - yhat_raw``, so leaving ``yhat_raw`` unclamped prevents one-sided
    truncation bias in the learned residual distribution while guaranteeing that ``yhat``,
    ``yhat_adjusted``, ``yhat_lower``, ``yhat_upper``, `` yhat_stale``, and ``quantiles`` obey
    physical constraints before metrics or registry rows are emitted.
    """
    if df is None or df.empty or not has_output_constraints(output_cfg):
        return df
    assert output_cfg is not None

    out = df.copy()
    n = len(out)
    lo = _resolve_bound_array(
        n,
        output_cfg.floor,
        output_cfg.floor_col,
        out,
        bounds_df,
        default=-np.inf,
        combine_fn=np.maximum,
    )
    hi = _resolve_bound_array(
        n,
        output_cfg.cap,
        output_cfg.cap_col,
        out,
        bounds_df,
        default=np.inf,
        combine_fn=np.minimum,
    )
    # Ensure hi >= lo if dynamic columns cross.
    hi = np.maximum(lo, hi)
    mode = output_cfg.integer_rounding

    for col in ("yhat", "yhat_adjusted", "yhat_stale"):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.any():
                clamped = np.where(finite, np.clip(vals, lo, hi), np.nan)
                rounded = _round_array(clamped, mode)
                out[col] = np.where(finite, np.clip(rounded, lo, hi), np.nan)

    for bcol, bmode in (("yhat_lower", "floor"), ("yhat_upper", "ceil")):
        if bcol in out.columns:
            vals = pd.to_numeric(out[bcol], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.any():
                clamped = np.where(finite, np.clip(vals, lo, hi), np.nan)
                eff_mode = bmode if mode == "cumulative_round" else mode
                rounded = _round_array(clamped, eff_mode)
                out[bcol] = np.where(finite, np.clip(rounded, lo, hi), np.nan)

    # Guarantee yhat_lower <= yhat <= yhat_upper after rounding.
    if {"yhat", "yhat_lower", "yhat_upper"}.issubset(out.columns):
        yh = pd.to_numeric(out["yhat"], errors="coerce").to_numpy(dtype=float)
        yl = pd.to_numeric(out["yhat_lower"], errors="coerce").to_numpy(dtype=float)
        yu = pd.to_numeric(out["yhat_upper"], errors="coerce").to_numpy(dtype=float)
        both_lo = np.isfinite(yh) & np.isfinite(yl)
        both_hi = np.isfinite(yh) & np.isfinite(yu)
        if both_lo.any():
            out["yhat_lower"] = np.where(both_lo, np.minimum(yl, yh), yl)
        if both_hi.any():
            out["yhat_upper"] = np.where(both_hi, np.maximum(yu, yh), yu)

    if "quantiles" in out.columns:
        q_mode = "round" if mode == "cumulative_round" else mode
        new_q: list[str | None] = []
        for idx, raw_q in enumerate(out["quantiles"]):
            if raw_q is None or pd.isna(raw_q) or not isinstance(raw_q, str) or not raw_q.strip():
                new_q.append(None if raw_q is None or pd.isna(raw_q) else str(raw_q))
                continue
            try:
                parsed = json.loads(raw_q)
            except (TypeError, ValueError):
                new_q.append(raw_q)
                continue
            if not isinstance(parsed, dict):
                new_q.append(raw_q)
                continue
            updated: dict[str, float] = {}
            for k, v in parsed.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    val = float(np.clip(float(v), lo[idx], hi[idx]))
                    if q_mode != "none":
                        val = float(_round_array(np.array([val], dtype=float), q_mode)[0])
                        val = float(np.clip(val, lo[idx], hi[idx]))
                    updated[str(k)] = val
            new_q.append(json.dumps(updated, sort_keys=True))
        out["quantiles"] = pd.array(new_q, dtype="string")

    return out


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

    blank = {
        "loss_raw": float("nan"),
        "loss_adjusted": float("nan"),
        "margin": float("nan"),
        "fold_win_rate": float("nan"),
        "n_folds_compared": 0,
    }
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

    def _loss(y_arr: np.ndarray, pred: np.ndarray) -> float:
        return loss_of(metric, compute_metrics(y_arr, pred).get(metric, float("nan")))

    raw_loss, adj_loss = _loss(y[keep], raw[keep]), _loss(y[keep], adjusted[keep])
    # Relative improvement of the adjusted arm over raw. Positive means the correction helped.
    # Always measured in that direction, whichever arm the run selected, so a fleet-wide average
    # over cells that chose differently is still a single comparable number.
    margin = float("nan")
    if np.isfinite(raw_loss) and np.isfinite(adj_loss) and raw_loss > 0:
        margin = (raw_loss - adj_loss) / raw_loss

    # The same comparison again, fold by fold. Two numbers out of one leave-one-fold-out pass,
    # because the pooled margin and the count of folds that agree with it are different evidence
    # and measurement says the count is the better of the two — see `select_arm`.
    wins, compared = 0, 0
    folds = pd.to_numeric(df["fold_id"], errors="coerce") if "fold_id" in df else None
    for fold in sorted(set(folds.dropna())) if folds is not None else []:
        m = (folds == fold).to_numpy() & keep
        if not m.any():
            continue
        lr, la = _loss(y[m], raw[m]), _loss(y[m], adjusted[m])
        if np.isfinite(lr) and np.isfinite(la):
            compared += 1
            wins += int(la < lr)
    return {
        "loss_raw": raw_loss,
        "loss_adjusted": adj_loss,
        "margin": margin,
        "fold_win_rate": (wins / compared) if compared else float("nan"),
        "n_folds_compared": compared,
        "basis": basis,
    }


def select_arm(oof: pd.DataFrame | None, metric: str, fallback: str) -> tuple[str, str]:
    """Choose this cell's point-forecast arm from its own held-out folds. Returns (arm, decision).

    `fallback` is the fleetwide arm the run would have used — `config.corrected_arm_for(metric)`,
    so `"median"` or `"mean"`. The question is only ever whether *this* series is better off
    without the correction, and the answer comes from the folds rather than from a global setting.

    **The rule is a per-fold sign test: the corrected arm keeps its place only if it beats raw on a
    strict majority of the cell's held-out folds.** No margin threshold, which is not the design the
    plan called for — measurement removed it. Fleet MAE over ten models × 24 series, nested
    leave-one-fold-out so no rule is graded on the folds it selected from:

    ==========================  =========  =========  =========
    rule                          3 folds    5 folds    8 folds
    ==========================  =========  =========  =========
    always raw                     20.09      20.00      19.57
    always corrected (2.5)         19.87      17.85      17.96
    margin < 0                     19.17      17.38      17.45
    margin < -0.05                 19.23      17.46      17.43
    **fold sign test**           **18.75**  **17.38**  **17.43**
    oracle (cheating)              17.41      16.00      16.36
    ==========================  =========  =========  =========

    Every selection rule beats both fleetwide arms, so per-series selection is worth doing at all.
    Among them the sign test is best or tied-best everywhere and clearly best at three folds, which
    is the fold count most runs use. A margin threshold makes things *worse* at every setting tried:
    the pooled margin is one noisy number, and a simulation on frames with no true bias put its 90th
    percentile at 0.11 — an 11% apparent improvement out of pure noise at three folds — so no
    threshold can separate a real effect from a lucky one. Counting how many folds agree is the
    robust version of the same question.

    Below `_MIN_FOLDS_FOR_AUTO` the sign test is not merely weak, it is wrong in a specific
    direction: at two folds the inner comparison has one fold to grade on, so it grades the
    correction on the residuals it was fitted from and reports the correction winning 85% of the
    time when an oracle says it wins 36%. Those cells take the fleetwide default and say so.
    """
    if fallback not in POINT_FORECAST_ARMS or fallback == "raw":
        raise ValueError(f"select_arm fallback must be a corrected arm, got {fallback!r}")
    if oof is None or oof.empty:
        return fallback, "auto-no-backtest"

    ev = compare_arms(oof, metric, fallback)
    win_rate = ev.get("fold_win_rate", float("nan"))
    enough = (
        ev.get("basis") == "leave-one-fold-out"
        and int(ev.get("n_folds_compared") or 0) >= _MIN_FOLDS_FOR_AUTO
        and isinstance(win_rate, float)
        and math.isfinite(win_rate)
    )
    if not enough:
        return fallback, "auto-few-folds"
    # A tie goes to raw. Not a coin-flip default: in the folds where the two arms genuinely cannot
    # be told apart, the corrected arm is still carrying the estimation variance of a shift it did
    # not need, so equal measured loss is not equal expected loss.
    return (fallback, "auto-corrected") if win_rate > 0.5 else ("raw", "auto-raw")


def resolve_arm(
    point_forecast: str | None, oof: pd.DataFrame | None, metric: str, fallback: str
) -> tuple[str, str]:
    """The arm a cell forecasts with, and why: ``(arm, decision)``.

    ``point_forecast`` is `config.OutputConfig.point_forecast`. A named arm is taken as written
    (``decision="configured"``); ``"auto"`` defers to `select_arm`, which reads this cell's own
    held-out folds and reports one of its four decisions. ``None`` only reaches here from a config
    built by hand — `RunConfig._normalize` resolves an unset field before a run starts — and falls
    back to ``"median"``, the model's own correction. Every engine that calibrates a point forecast
    asks this one question, so it lives here rather than in each of them.
    """
    if point_forecast == "auto":
        return select_arm(oof, metric, fallback)
    return point_forecast or "median", "configured"
