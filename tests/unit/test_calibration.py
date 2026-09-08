"""Point-forecast arms and out-of-fold interval calibration (plan item 2.5).

These tests are the exit gate for a change that moved every forecast the product produces, so they
are written to check the *claims* rather than the plumbing. Four of them matter more than the rest:

* `test_raw_arm_returns_the_models_own_number` — `raw` is genuinely uncorrected. This is the one
  that would have caught the original defect, where ten models shipped
  ``prediction + median(in-sample residual)`` and nothing said so.
* `test_band_widens_with_horizon_step` — the band is per-step. A horizon-flat band is too wide at
  step 1 and too narrow at step 28, and averages to something that looks fine.
* `test_oof_coverage_beats_the_in_sample_band` — the measured claim. The fleet number the change was
  argued from is 0.601 achieved against a nominal 0.8; a synthetic where the in-sample band
  under-covers the same way has to come out closer under calibration or the change is not earning
  its place.
* `test_arm_comparison_is_estimated_out_of_fold` — the diagnostic is honest. A margin computed from
  the folds the correction was fitted on flatters it, and that margin is what item 2.5b will
  eventually key an automatic per-series choice on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.calibration import (
    CALIBRATED_COLUMNS,
    CALIBRATION_SOURCES,
    POINT_FORECAST_ARMS,
    _summarize,
    apply_calibration,
    calibrate_from_oof,
    compare_arms,
    coverage_by_step,
    select_arm,
)
from scale_forecasting.models.base_model import DEFAULT_QUANTILES, PREDICTION_COLUMNS

_Q = DEFAULT_QUANTILES
_HORIZON = 12
_FOLDS = 6


def _pred(n: int = _HORIZON, *, raw: float = 100.0, shift: float = 5.0) -> pd.DataFrame:
    """A model's prediction frame: `yhat` already carries an in-sample shift over `yhat_raw`."""
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    raw_vals = np.full(n, raw, dtype=float)
    return pd.DataFrame(
        {
            "ds": idx,
            "yhat": raw_vals + shift,
            "yhat_raw": raw_vals,
            "yhat_lower": raw_vals + shift - 1.0,
            "yhat_upper": raw_vals + shift + 1.0,
            "quantiles": pd.array([json.dumps({"0.5": raw + shift})] * n, dtype="string"),
        },
        columns=list(PREDICTION_COLUMNS),
    )


def _oof(
    *,
    folds: int = _FOLDS,
    horizon: int = _HORIZON,
    bias: float = 8.0,
    noise_at_step: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    """An out-of-fold frame whose residuals have a known bias and a spread that grows with step.

    `y_true - yhat_raw` is `bias + noise`, with the noise standard deviation scaling as
    `noise_at_step * step` — the shape a real forecast error has and the one a horizon-flat band
    cannot represent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(folds):
        for step in range(1, horizon + 1):
            raw = 100.0 + fold
            resid = bias + (rng.normal(0.0, noise_at_step * step) if noise_at_step else 0.0)
            rows.append(
                {
                    "fold_id": fold,
                    "horizon_step": step,
                    "yhat_raw": raw,
                    "yhat_adjusted": raw + 5.0,
                    "yhat": raw + 5.0,
                    "y_true": raw + resid,
                    "yhat_lower": raw + 4.0,
                    "yhat_upper": raw + 6.0,
                }
            )
    return pd.DataFrame(rows)


# --- the contract ------------------------------------------------------------------------


def test_calibrated_columns_extend_the_model_contract_by_exactly_one() -> None:
    """`CALIBRATED_COLUMNS` is spelled out literally to keep the model stack off the submit path.

    That is a duplication, so this is the test that stops the two drifting: a column added to the
    model contract has to be threaded through calibration too, and this fails in the commit that
    forgets.
    """
    assert set(CALIBRATED_COLUMNS) - set(PREDICTION_COLUMNS) == {"yhat_adjusted"}
    assert set(PREDICTION_COLUMNS) - set(CALIBRATED_COLUMNS) == set()


def test_apply_calibration_rejects_an_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown point_forecast arm"):
        apply_calibration(_pred(), None, "mode")


@pytest.mark.parametrize("arm", POINT_FORECAST_ARMS)
def test_every_arm_returns_the_canonical_frame(arm: str) -> None:
    out, source = apply_calibration(_pred(), calibrate_from_oof(_oof(), _Q), arm)
    assert list(out.columns) == list(CALIBRATED_COLUMNS)
    assert source in CALIBRATION_SOURCES


# --- the arms ----------------------------------------------------------------------------


def test_raw_arm_returns_the_models_own_number() -> None:
    """The defect this whole item exists to fix: `raw` must be uncorrected, calibration or not."""
    pred = _pred()
    for cal in (None, calibrate_from_oof(_oof(), _Q)):
        out, _ = apply_calibration(pred, cal, "raw")
        assert np.allclose(out["yhat"], pred["yhat_raw"])


def test_median_arm_shifts_raw_by_the_out_of_fold_median_residual() -> None:
    """And `yhat_raw` plus the recorded correction reconstructs `yhat_adjusted` exactly."""
    oof = _oof(bias=8.0)
    cal = calibrate_from_oof(oof, _Q)
    assert cal is not None
    out, _ = apply_calibration(_pred(), cal, "median")
    expected = out["yhat_raw"].to_numpy() + np.array(
        [cal.for_step(i + 1).quantiles[0.5] for i in range(len(out))]
    )
    assert np.allclose(out["yhat_adjusted"], expected)
    assert np.allclose(out["yhat"], out["yhat_adjusted"])
    # A constant bias is exactly recoverable, which is what makes the assertion above a claim
    # about the estimator rather than a restatement of its own arithmetic.
    assert np.allclose(out["yhat"], out["yhat_raw"] + 8.0)


def test_mean_arm_uses_the_mean_residual_not_the_median() -> None:
    """On a skewed residual sample the two differ, and each minimises the loss it is named for."""
    oof = _oof(bias=0.0)
    # Skew: one large positive residual per horizon step drags the mean above the median at every
    # step, so the claim is about the estimator and not about which steps happened to be hit.
    outlier = oof["fold_id"] == 0
    oof.loc[outlier, "y_true"] = oof.loc[outlier, "yhat_raw"] + 100.0
    cal = calibrate_from_oof(oof, _Q)
    assert cal is not None
    median_out, _ = apply_calibration(_pred(), cal, "median")
    mean_out, _ = apply_calibration(_pred(), cal, "mean")
    assert not np.allclose(median_out["yhat"], mean_out["yhat"])
    assert (mean_out["yhat"] > median_out["yhat"]).all()


def test_no_backtest_leaves_the_models_own_band_untouched() -> None:
    """With nothing held out there is nothing to calibrate from, and the source says so."""
    pred = _pred()
    out, source = apply_calibration(pred, None, "median")
    assert source == "in-sample"
    assert np.allclose(out["yhat_lower"], pred["yhat_lower"])
    assert np.allclose(out["yhat_upper"], pred["yhat_upper"])
    assert np.allclose(out["yhat"], pred["yhat"])


# --- the band ----------------------------------------------------------------------------


def test_bounds_stay_ordered() -> None:
    cal = calibrate_from_oof(_oof(noise_at_step=0.4), _Q)
    out, _ = apply_calibration(_pred(), cal, "median")
    assert (out["yhat_lower"] <= out["yhat"] + 1e-9).all()
    assert (out["yhat"] <= out["yhat_upper"] + 1e-9).all()


def test_band_widens_with_horizon_step() -> None:
    """The defect the flat band hid: error grows with distance and a single band cannot say so."""
    cal = calibrate_from_oof(_oof(folds=12, noise_at_step=0.6), _Q)
    assert cal is not None and cal.source == "oof-per-step"
    out, _ = apply_calibration(_pred(), cal, "median")
    width = (out["yhat_upper"] - out["yhat_lower"]).to_numpy()
    # Compared end to end rather than pairwise: the widening window deliberately smooths adjacent
    # steps, so a strictly monotone sequence would be a stronger claim than the estimator makes.
    assert width[-1] > width[0] * 1.5


def test_quantile_map_is_rewritten_with_the_band() -> None:
    """A row that disagrees with itself is worse than a row with a stale band and no map."""
    cal = calibrate_from_oof(_oof(), _Q)
    out, _ = apply_calibration(_pred(), cal, "median")
    first = json.loads(out["quantiles"].iloc[0])
    assert set(first) == {str(q) for q in sorted(_Q)}
    assert first[str(min(_Q))] == pytest.approx(out["yhat_lower"].iloc[0])
    assert first[str(max(_Q))] == pytest.approx(out["yhat_upper"].iloc[0])


def test_oof_coverage_beats_the_in_sample_band() -> None:
    """The measured claim, against the recorded fleet baseline of 0.601 versus a nominal 0.8.

    The synthetic reproduces the failure mode: an in-sample band of ±1 against residuals whose real
    spread is far wider, which is what an optimistic in-sample estimate looks like. Nominal here is
    0.8 — the outer two of `DEFAULT_QUANTILES`.
    """
    # Thirty folds so each step's own residuals carry the estimate rather than a borrowed
    # neighbourhood. The small-n counterpart is
    # `test_the_band_widens_outward_on_a_small_sample_rather_than_interpolating`.
    oof = _oof(folds=30, bias=0.0, noise_at_step=0.5)
    nominal = max(_Q) - min(_Q)

    in_sample = coverage_by_step(oof)["coverage"].mean()

    cal = calibrate_from_oof(oof, _Q)
    assert cal is not None
    recalibrated = oof.copy()
    steps = recalibrated["horizon_step"].astype(int)
    lo, hi = min(_Q), max(_Q)
    recalibrated["yhat_lower"] = [
        r + cal.for_step(s).quantiles[lo] for r, s in zip(oof["yhat_raw"], steps, strict=True)
    ]
    recalibrated["yhat_upper"] = [
        r + cal.for_step(s).quantiles[hi] for r, s in zip(oof["yhat_raw"], steps, strict=True)
    ]
    achieved = coverage_by_step(recalibrated)["coverage"].mean()

    assert in_sample < 0.65, "the fixture is meant to reproduce an under-covering in-sample band"
    assert abs(achieved - nominal) < abs(in_sample - nominal)
    assert achieved == pytest.approx(nominal, abs=0.1)


def test_the_band_widens_outward_on_a_small_sample_rather_than_interpolating() -> None:
    """`np.quantile` interpolates between order statistics, which is too narrow to be a tail bound.

    Twelve residuals is what a three-fold backtest yields per step after the window settles — the
    common case, not an edge case — and there the plain 10th/90th pair leaves two points outside on
    each side. `_empirical_quantile` steps outward to an order statistic instead, so the band
    contains at least as much of its own calibration sample as it claims to.
    """
    residuals = np.sort(np.linspace(-3.0, 3.0, 12) + np.array([0.1] * 12))
    step = _summarize(residuals, _Q)
    plain_lo, plain_hi = np.quantile(residuals, 0.1), np.quantile(residuals, 0.9)

    assert step.quantiles[0.1] <= plain_lo and step.quantiles[0.9] >= plain_hi
    inside = ((residuals >= step.quantiles[0.1]) & (residuals <= step.quantiles[0.9])).mean()
    assert inside >= 0.8, "the calibration sample itself must clear nominal"
    # The centre is a location estimate, not a tail bound, and must not be pushed outward with them.
    assert step.quantiles[0.5] == pytest.approx(float(np.quantile(residuals, 0.5)))


def test_a_large_sample_converges_on_the_plain_quantile() -> None:
    """The correction is a finite-sample fix; it has to vanish once the sample is large."""
    residuals = np.sort(np.random.default_rng(0).standard_normal(4000))
    step = _summarize(residuals, _Q)
    for q in _Q:
        assert step.quantiles[q] == pytest.approx(float(np.quantile(residuals, q)), abs=0.02)


def test_coverage_by_step_is_empty_without_bounds() -> None:
    assert coverage_by_step(_oof().drop(columns=["yhat_lower"])).empty
    assert coverage_by_step(pd.DataFrame()).empty


# --- degrading honestly ------------------------------------------------------------------


def test_too_few_residuals_falls_back_rather_than_inventing_a_band() -> None:
    assert calibrate_from_oof(_oof(folds=1, horizon=4), _Q) is None
    assert calibrate_from_oof(pd.DataFrame(), _Q) is None
    assert calibrate_from_oof(_oof().drop(columns=["yhat_raw"]), _Q) is None


def test_a_short_horizon_degrades_to_pooled_and_says_so() -> None:
    """Three residuals per step cannot locate a 10th percentile, so the window widens. When the
    horizon is short enough that widening swallows all of it, every step lands on the same sample
    — which must be labelled `oof-flat`, not dressed up as per-step."""
    cal = calibrate_from_oof(_oof(folds=3, horizon=4, noise_at_step=0.4), _Q)
    assert cal is not None
    assert cal.source == "oof-flat"
    assert cal.by_step == {}
    assert cal.for_step(1) is cal.pooled


def test_three_folds_over_a_long_horizon_still_resolve_per_step() -> None:
    """The common default is not the degraded case: ±2 steps is fifteen residuals, and a band that
    grows with distance is exactly what three folds over twenty-eight steps can support."""
    cal = calibrate_from_oof(_oof(folds=3, horizon=28, noise_at_step=0.4), _Q)
    assert cal is not None
    assert cal.source == "oof-per-step"
    assert all(s.n >= 12 for s in cal.by_step.values())


def test_missing_step_column_pools_rather_than_guessing() -> None:
    cal = calibrate_from_oof(_oof().drop(columns=["horizon_step"]), _Q)
    assert cal is not None and cal.source == "oof-flat"


# --- the diagnostic ----------------------------------------------------------------------


def test_arm_comparison_is_estimated_out_of_fold() -> None:
    """The margin has to be earned on folds the correction never saw, and name its own method."""
    out = compare_arms(_oof(folds=6, bias=8.0), "wape")
    assert out["basis"] == "leave-one-fold-out"
    # A real constant bias of 8 that the raw arm ignores: correcting it must win, clearly.
    assert out["loss_adjusted"] < out["loss_raw"]
    assert out["margin"] > 0.5


def test_arm_comparison_reports_a_loss_when_the_correction_is_noise() -> None:
    """Negative margins are the point. The A/B that reversed this item's original design found
    centring lost on the fleet, and a diagnostic that cannot say so is decoration."""
    oof = _oof(folds=6, bias=0.0, noise_at_step=1.0, seed=3)
    out = compare_arms(oof, "wape")
    assert np.isfinite(out["margin"])
    assert out["margin"] < 0.2, "a correction fitted to pure noise should not look like a big win"


def test_single_fold_falls_back_to_the_stored_arm_and_names_it() -> None:
    out = compare_arms(_oof(folds=1, horizon=_HORIZON), "wape")
    assert out["basis"] == "in-sample-arm"
    assert np.isfinite(out["loss_raw"])


def test_arm_comparison_is_blank_without_both_arms() -> None:
    out = compare_arms(_oof().drop(columns=["yhat_adjusted"]), "wape")
    assert np.isnan(out["loss_raw"]) and np.isnan(out["margin"])


# --- picking the arm per series (plan item 2.5b) -----------------------------------------
#
# The rule is a per-fold sign test with a minimum fold count and no margin threshold, and every
# part of that is measured rather than chosen — `select_arm`'s docstring carries the table. These
# tests pin the three behaviours a later reader could plausibly "simplify" away: the tie goes to
# raw, two folds is not enough evidence, and the choice is a function of the data alone.


def _folds_biased(biases: list[float]) -> pd.DataFrame:
    """An out-of-fold frame where each fold carries its own constant bias.

    Selection reads *agreement between folds*, so the fixture has to be able to make the folds
    disagree with each other. A single scalar bias cannot — it makes every fold say the same thing,
    which is the easy case. Constant within a fold keeps the arithmetic exact, so these tests pin a
    stated win count rather than a seed that happened to come out the right way.
    """
    return pd.concat(
        [_oof(folds=1, bias=b).assign(fold_id=fold) for fold, b in enumerate(biases)],
        ignore_index=True,
    )


def test_auto_keeps_the_correction_when_the_folds_back_it() -> None:
    """A real, consistent bias: every fold agrees the correction helps, so it ships."""
    arm, decision = select_arm(_oof(folds=6, bias=8.0), "wape", "median")
    assert (arm, decision) == ("median", "auto-corrected")


def test_auto_drops_the_correction_when_the_folds_do_not_back_it() -> None:
    """A cell whose bias is not stable across folds: the shift fitted on some is wrong on the rest.

    This is the case a fleetwide setting cannot express. Item 2.5 measured the correction as worth
    5.7% of fleet WAPE *on average*, which says nothing about the series where it is actively
    harmful, and averaging is exactly what hides those.
    """
    # Half the folds run +8 and half −8, so the shift estimated from the others always points the
    # wrong way for the fold it is graded on. Zero of four folds back the correction.
    oof = _folds_biased([8.0, 8.0, -8.0, -8.0])
    assert compare_arms(oof, "wape")["fold_win_rate"] == 0.0
    assert select_arm(oof, "wape", "median") == ("raw", "auto-raw")


def test_the_choice_is_a_function_of_the_data_and_nothing_else() -> None:
    """Re-running an unchanged config on unchanged data must not move the forecast.

    Selection introduces a second thing that could vary between runs, on top of model fitting. It
    must not: a run that is re-submitted for an unrelated reason would otherwise come back with
    different numbers under the same run_id, which is the one thing the registry cannot survive.
    """
    oof = _oof(folds=5, bias=0.4, noise_at_step=1.0, seed=11)
    assert select_arm(oof, "wape", "median") == select_arm(oof.copy(), "wape", "median")
    # Row order is not part of the data. It changes with a shuffled read and must not change this.
    shuffled = oof.sample(frac=1.0, random_state=5).reset_index(drop=True)
    assert select_arm(shuffled, "wape", "median") == select_arm(oof, "wape", "median")


def test_a_tie_goes_to_raw_because_the_correction_is_not_free() -> None:
    """Equal measured loss is not equal expected loss, and the measurement agrees.

    Sending ties to the corrected arm instead was tried on the fixture that set the rule: it cost
    3% of fleet MAE at three folds (19.32 against 18.75) and 3.4% of fleet RMSE (29.74 against
    28.77). The corrected arm in a tie is carrying the estimation variance of a shift it did not
    need, so the tie is only a tie in the sample.
    """
    # Four folds chosen so the leave-one-out shift helps on exactly two of them: the two large
    # like-signed folds are corrected towards each other, and the two odd ones out are dragged
    # further from the truth than they started.
    oof = _folds_biased([10.0, 10.0, 1.0, -21.0])
    assert compare_arms(oof, "wape")["fold_win_rate"] == 0.5, "the fixture has to be a real tie"
    assert select_arm(oof, "wape", "median") == ("raw", "auto-raw")


def test_two_folds_is_not_evidence_and_the_row_says_which_way_it_went() -> None:
    """Below the minimum the fleetwide arm applies, and `point_forecast_decision` records that.

    Two folds is not merely weak, it is wrong in a known direction: the inner comparison has one
    fold to grade on, so it grades the correction on the residuals it was fitted from. Measured on
    the fixture, the rule then sends 15% of cells to the raw arm where an oracle sends 64%.
    """
    arm, decision = select_arm(_oof(folds=2, bias=0.0, noise_at_step=1.0, seed=3), "wape", "median")
    assert (arm, decision) == ("median", "auto-few-folds")


def test_a_cell_whose_backtest_produced_nothing_takes_the_fleetwide_arm() -> None:
    """Item 2.1 made a per-cell backtest failure survivable, so `auto` has to survive it too."""
    assert select_arm(None, "rmse", "mean") == ("mean", "auto-no-backtest")
    assert select_arm(pd.DataFrame(), "wape", "median") == ("median", "auto-no-backtest")


def test_the_fallback_must_be_a_corrected_arm() -> None:
    """A `raw` fallback would collapse "not enough evidence" and "evidence says raw" into one."""
    with pytest.raises(ValueError, match="corrected arm"):
        select_arm(_oof(), "wape", "raw")


def test_the_comparison_reports_how_many_folds_agreed_with_its_own_verdict() -> None:
    """The count is what `select_arm` reads; the pooled margin is what a human reads."""
    out = compare_arms(_oof(folds=6, bias=8.0), "wape")
    assert out["n_folds_compared"] == 6
    assert out["fold_win_rate"] == 1.0, "a constant bias should be corrected on every fold"

    noisy = compare_arms(_oof(folds=6, bias=0.0, noise_at_step=1.0, seed=3), "wape")
    assert 0.0 <= noisy["fold_win_rate"] <= 1.0
    assert noisy["n_folds_compared"] == 6


def test_a_single_fold_reports_no_agreement_to_read() -> None:
    """`n_folds_compared` gates `select_arm`, so the degenerate case must not report a majority."""
    out = compare_arms(_oof(folds=1), "wape")
    assert out["n_folds_compared"] <= 1
