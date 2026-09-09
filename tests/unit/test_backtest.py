"""Tests for backtest folds + out-of-fold predictions.

Covers fold geometry, the no-leakage invariant (train_end == val_start), expanding vs
sliding schemes, the short-series clamp (fewer folds, never an exception), and OOF frame
shape/units.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.backtest import (
    OOF_COLUMNS,
    Fold,
    achievable_folds,
    backtest_cell,
    fit_rows,
    make_folds,
    suggest_min_train,
    training_width,
    training_window,
)
from scale_forecasting.config import RunConfig
from scale_forecasting.features import invert_transform
from scale_forecasting.models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext


def _cfg(
    backtest: dict[str, Any] | None = None, features: dict[str, Any] | None = None
) -> RunConfig:
    kw: dict[str, Any] = {
        "run_name": "r",
        "data": {"source_table": "p.d.s"},
        "models": ["theta"],
    }
    if backtest is not None:
        kw["backtest"] = {"enabled": True, **backtest}
    if features is not None:
        kw["features"] = features
    return RunConfig(**kw)


def _series(n: int) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"ds": ds, "y": np.arange(1.0, n + 1.0)})


class _LastValue(BaseModel):
    """Deterministic: forecast = last training value, flat over the horizon."""

    name = "_lastval"
    runtime = "python"
    family = "statistical"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._last = float(y.iloc[-1])

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        ds = pd.date_range("2026-01-01", periods=horizon, freq="D")
        # predict returns original units: invert the transform on the way out.
        yhat = invert_transform(
            np.full(horizon, self._last), self.ctx.transform, self.ctx.transform_lambda
        )
        return self._assemble_frame(ds, {q: yhat for q in quantiles})


def _factory(transform: str = "none") -> Any:
    def make() -> BaseModel:
        return _LastValue({}, ModelContext(freq="D", horizon=3, transform=transform))

    return make


# --- fold geometry -------------------------------------------------------------


def test_folds_count_and_ids() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert len(folds) == 3
    assert [f.fold_id for f in folds] == [0, 1, 2]


def test_no_leakage_train_end_equals_val_start() -> None:
    folds = make_folds(100, _cfg({"n_folds": 4, "horizon": 7, "step": 7, "min_train": 20}))
    for f in folds:
        assert f.train_end == f.val_start  # training strictly precedes validation
        assert f.train_start < f.train_end
        assert f.val_start < f.val_end


def test_last_fold_validates_final_window() -> None:
    n = 100
    folds = make_folds(n, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert folds[-1].val_end == n
    assert folds[-1].val_start == n - 5


def test_val_windows_are_horizon_sized_and_stepped() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    for f in folds:
        assert f.val_size == 5
    # consecutive folds step by `step`
    assert folds[1].val_start - folds[0].val_start == 5


def test_expanding_scheme_grows_train_from_zero() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert all(f.train_start == 0 for f in folds)
    # train grows fold to fold
    assert folds[0].train_size < folds[1].train_size < folds[2].train_size


def test_sliding_scheme_fixed_window() -> None:
    folds = make_folds(
        100, _cfg({"scheme": "sliding", "n_folds": 3, "horizon": 5, "step": 5, "min_train": 20})
    )
    for f in folds:
        assert f.train_size == 20


def test_a_series_too_short_for_every_fold_gets_the_folds_it_can_support() -> None:
    """It used to raise, and `run_cell` turned that into an error cell — losing the forecast.

    Backtesting scores a model; it does not produce the forecast. A series with 15 observations can
    be fit and forecast perfectly well, so a shortfall in *scoring* must cost only the score.
    """
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    folds = make_folds(15, cfg)  # 15 - 5 - 10 = 0 slack: room for exactly one fold
    assert len(folds) == 1
    assert folds[0].train_size == 10  # exactly min_train, the tightest legal fold


def test_the_folds_dropped_are_the_oldest_and_the_survivors_keep_their_original_ids() -> None:
    """Both halves matter, and renumbering is the tempting mistake.

    Dropping the oldest keeps every series scored on the most recent window it can reach — the
    window a leaderboard is about. Keeping the original ids is what lets ``fold_id`` mean the same
    thing across a ragged panel: renumber the survivors 0..k and a short series' fold 0 silently
    lines up against a long series' fold 0 covering a completely different stretch of history.
    """
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    full = make_folds(100, cfg)
    clamped = make_folds(20, cfg)  # slack 5 → two folds
    assert [f.fold_id for f in clamped] == [1, 2]  # fold 0, the oldest, is the one dropped
    # The survivors are the same folds they would have been, measured from the end of the series.
    assert [f.val_end - 20 for f in clamped] == [f.val_end - 100 for f in full[1:]]


def test_a_series_that_cannot_support_one_fold_gets_no_folds_rather_than_an_exception() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    assert make_folds(14, cfg) == []  # 14 - 5 = 9 < min_train
    assert achievable_folds(14, cfg) == 0


def test_achievable_folds_saturates_at_the_requested_count() -> None:
    """A long series does not get bonus folds — ``n_folds`` is what was asked for."""
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    assert achievable_folds(10_000, cfg) == 3
    assert achievable_folds(15, cfg) == 1
    assert achievable_folds(20, cfg) == 2


def test_achievable_folds_agrees_with_the_folds_actually_built() -> None:
    """The count and the list must not drift; the cell records one and scores the other."""
    cfg = _cfg({"n_folds": 4, "horizon": 7, "step": 3, "min_train": 12})
    for n in range(0, 60):
        assert achievable_folds(n, cfg) == len(make_folds(n, cfg)), f"disagreed at n={n}"


def test_every_clamped_fold_still_honours_the_no_leakage_and_min_train_invariants() -> None:
    """Clamping must not buy folds by relaxing the geometry it was protecting."""
    cfg = _cfg({"n_folds": 4, "horizon": 7, "step": 3, "min_train": 12})
    for n in range(0, 60):
        for f in make_folds(n, cfg):
            assert f.train_end == f.val_start  # no leakage
            assert f.train_size >= 12  # min_train respected
            assert f.val_size == 7  # full-width validation window
            assert f.val_end <= n  # never reads past the series


# --- the embargo (`gap`) and the sliding width (`window`) ------------------------------------


_GEOM = {"n_folds": 3, "horizon": 5, "step": 5, "min_train": 20}


def test_the_embargo_moves_the_training_end_and_leaves_the_validation_window_alone() -> None:
    """The direction is the whole design, so it is asserted rather than described.

    A ``gap`` run and a ``gap=0`` run score the *same dates*; only how much history the model was
    allowed to see changes. Anchoring the other way round — validation pushed out past the
    embargo — would have been easier to write and would make the two runs incomparable, because
    fold 2 of one would cover a different fortnight than fold 2 of the other.
    """
    plain = make_folds(100, _cfg(_GEOM))
    embargoed = make_folds(100, _cfg({**_GEOM, "gap": 7}))
    assert [(f.val_start, f.val_end) for f in embargoed] == [
        (f.val_start, f.val_end) for f in plain
    ]
    assert [f.train_end for f in embargoed] == [f.train_end - 7 for f in plain]
    for f in embargoed:
        assert f.train_end + 7 == f.val_start


def test_the_embargo_costs_history_so_a_short_series_achieves_fewer_folds() -> None:
    """Not a bug to route around: the observations in the embargo are genuinely unusable.

    `achievable_folds` subtracts the gap for the same reason it subtracts the horizon — a series
    that cannot seat the oldest fold's training window *and* the embargo in front of it cannot run
    that fold. Reporting three folds and delivering two would be the alternative.
    """
    n = 20 + 5 + 2 * 5  # exactly three folds' worth at gap=0
    assert achievable_folds(n, _cfg(_GEOM)) == 3
    assert achievable_folds(n, _cfg({**_GEOM, "gap": 5})) == 2
    assert len(make_folds(n, _cfg({**_GEOM, "gap": 5}))) == 2


def test_a_sliding_window_is_its_own_width_and_min_train_stays_the_floor() -> None:
    """The split is the point of the field: how much the model sees, versus how much is enough.

    Before ``window`` existed, asking for a 60-observation sliding window also told the fold planner
    that 60 observations were enough to score on — one number doing two jobs, and the only way to
    say "train on the last 60 but do not bother scoring a series with less than a year" was to pick
    whichever mattered more.
    """
    cfg = _cfg({**_GEOM, "scheme": "sliding", "window": 60})
    folds = make_folds(100, cfg)
    assert [f.train_size for f in folds] == [60, 60, 60]
    # Feasibility still reads `min_train`, not `window`: the wider window did not cost a fold.
    assert len(folds) == achievable_folds(100, cfg) == 3


def test_an_unset_window_reproduces_the_min_train_geometry_exactly() -> None:
    """Why the default is ``None`` and not a copy of ``min_train``: the two stay distinguishable in
    the serialized config, and every pre-``window`` run's geometry is unchanged."""
    sliding = {**_GEOM, "scheme": "sliding"}
    assert training_width(_cfg(sliding)) == 20
    assert make_folds(100, _cfg(sliding)) == make_folds(100, _cfg({**sliding, "window": 20}))


def test_the_scale_denominator_follows_the_training_width() -> None:
    """MASE divides by the mean step of the training data, so `training_window` has to slice the
    same span `make_folds` did — one function, `training_width`, answers for both."""
    cfg = _cfg({**_GEOM, "scheme": "sliding", "window": 60})
    n = 100
    ds = pd.date_range("2026-01-01", periods=n, freq="D").to_numpy()
    y = np.arange(float(n))
    fold = make_folds(n, cfg)[-1]
    got = training_window(ds, y, pd.Timestamp(ds[fold.train_end - 1]), cfg)
    assert len(got) == 60
    assert got[0] == y[fold.train_start] and got[-1] == y[fold.train_end - 1]


def test_the_embargo_scores_the_validation_window_and_not_the_dates_it_skipped() -> None:
    """End to end through `backtest_cell`: the OOF rows land on the validation dates.

    The trap this exists for is off-by-a-gap. A model's forecast origin is its last training date,
    so with an embargo the first ``gap`` steps it produces cover dates nobody is scoring. If those
    were kept, every OOF row would be shifted ``gap`` days early and scored against the wrong
    actual — a silent, plausible-looking accuracy number.
    """
    cfg = _cfg({"n_folds": 2, "horizon": 4, "step": 4, "min_train": 10, "gap": 3})
    series = _series(40)
    oof, fold_metrics, _ = backtest_cell(series, _factory(), cfg)

    assert len(oof) == 2 * 4
    for fold in make_folds(len(series), cfg):
        rows = oof[oof["fold_id"] == fold.fold_id].reset_index(drop=True)
        assert list(rows["ds"]) == list(series["ds"].iloc[fold.val_start : fold.val_end])
        assert list(rows["y_true"]) == list(series["y"].iloc[fold.val_start : fold.val_end])
        # Numbered from the window, not from the origin — see `bigquery_sql.build_eval_query`.
        assert list(rows["horizon_step"]) == [1, 2, 3, 4]
        # `_LastValue` is flat at the last *training* value, which the embargo moved back.
        assert rows["cutoff_date"].iloc[0] == series["ds"].iloc[fold.train_end - 1]
        assert set(rows["yhat_raw"]) == {series["y"].iloc[fold.train_end - 1]}
    assert len(fold_metrics) == 2


# --- backtest_cell -------------------------------------------------------------


def test_oof_frame_shape_and_columns() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 4, "step": 4, "min_train": 10})
    oof, fold_metrics, _ = backtest_cell(_series(40), _factory(), cfg)
    assert list(oof.columns) == list(OOF_COLUMNS)
    assert len(oof) == 3 * 4  # n_folds × horizon
    assert oof["ds"].dtype == np.dtype("datetime64[ns]")
    assert sorted(oof["fold_id"].unique()) == [0, 1, 2]
    assert len(fold_metrics) == 3


def test_a_series_with_no_achievable_folds_still_returns_the_full_column_set() -> None:
    # The empty frame has to look like the populated one. A fold-less series that returned four
    # columns while its neighbours returned eight would only surface downstream, as a concat that
    # quietly widened with NaNs — or, in `assemble_oof_rows`, as rows missing keys.
    cfg = _cfg({"n_folds": 3, "horizon": 4, "step": 4, "min_train": 10})
    oof, fold_metrics, _ = backtest_cell(_series(8), _factory(), cfg)
    assert oof.empty and not fold_metrics
    assert list(oof.columns) == list(OOF_COLUMNS)


def test_each_oof_row_records_the_fold_origin_and_its_step_within_the_horizon() -> None:
    # `fold_id` is an ordinal in one series' own plan; `cutoff_date` is the date the fold forecast
    # from, which is what makes two series comparable when their histories end on different days.
    # `horizon_step` is what turns "does this model decay with horizon?" into a GROUP BY.
    cfg = _cfg({"n_folds": 2, "horizon": 4, "step": 4, "min_train": 10})
    series = _series(40)
    oof, _, _ = backtest_cell(series, _factory(), cfg)

    for fold_id, block in oof.groupby("fold_id"):
        assert list(block["horizon_step"]) == [1, 2, 3, 4]
        # One cutoff per fold, and it is the last training date — strictly before the first
        # validation date, which is the no-leakage invariant restated in date space.
        assert block["cutoff_date"].nunique() == 1
        assert block["cutoff_date"].iloc[0] < block["ds"].iloc[0], fold_id

    # The two folds step back by exactly `step` days, as the geometry says they should.
    cutoffs = sorted(oof["cutoff_date"].unique())
    assert (cutoffs[1] - cutoffs[0]) == pd.Timedelta(days=4)


def test_the_interval_metrics_are_finite_because_the_folds_now_score_the_bounds() -> None:
    # Every Python cell in every run before this scored `coverage`, `pinball`, `interval_score` and
    # `interval_width` as NaN — the model returned bounds on every fold and the fold loop dropped
    # them. Asserting FINITE, not merely present: a NaN is present too, and that is how four of
    # fifteen metric columns stayed empty through a green suite.
    cfg = _cfg({"n_folds": 2, "horizon": 4, "step": 4, "min_train": 10})
    oof, fold_metrics, _ = backtest_cell(_series(40), _factory(), cfg)

    for panel in fold_metrics:
        for metric in ("coverage", "pinball", "interval_score", "interval_width"):
            assert np.isfinite(panel[metric]), metric
        assert 0.0 <= panel["coverage"] <= 1.0
        assert panel["interval_width"] >= 0.0
    # The bounds reach the OOF frame too, ordered, so a reader can recompute the coverage.
    assert (oof["yhat_lower"] <= oof["yhat_upper"]).all()


def _random_walk(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=n, freq="D"),
            "y": 100.0 + np.cumsum(rng.normal(0.0, 1.0, n)),
        }
    )


def _real_factory(model_name: str, horizon: int) -> Any:
    from scale_forecasting.models import get_model

    model_cls = get_model(model_name)

    def make() -> BaseModel:
        return model_cls({}, ModelContext(freq="D", horizon=horizon, transform="none"))

    return make


def test_a_model_with_native_intervals_covers_near_its_nominal_rate() -> None:
    """The bounds are the 0.1 and 0.9 quantiles, so a correctly specified model should cover
    around 80% of the held-out points. The tolerance is deliberately wide: coverage over a few
    hundred held-out points is a coarse estimate, and the assertion that matters is that the
    number is *in the right neighbourhood* rather than the 0.0 a degenerate band produces or the
    NaN every Python cell reported before the folds were scored on their intervals at all.
    """
    cfg = _cfg({"n_folds": 12, "horizon": 7, "step": 7, "min_train": 60})
    _, fold_metrics, _ = backtest_cell(_random_walk(200), _real_factory("sarimax", 7), cfg)

    coverage = float(np.mean([panel["coverage"] for panel in fold_metrics]))
    assert 0.55 <= coverage <= 1.0, coverage


def test_a_residual_band_is_flat_across_the_horizon_so_late_steps_are_under_covered() -> None:
    """A model without native intervals gets the empirical quantiles of its *one-step* in-sample
    residuals added to every step of the forecast, so its band is the same width at h=1 and at
    h=14. Real uncertainty on a random walk grows with the square root of the horizon, so the late
    steps are systematically under-covered.

    This is documented rather than fixed. It is a property of `BaseModel.residual_intervals` and
    of the ten models that rely on it, and the honest response is to make it visible: that is what
    `horizon_step` on the OOF rows is for, and this test is the smallest demonstration that the
    column answers the question. Anyone comparing a residual-interval model's coverage against a
    native-interval model's should be filtering on `interval_source` and reading it by step.
    """
    horizon = 14
    cfg = _cfg({"n_folds": 20, "horizon": horizon, "step": 1, "min_train": 80})
    oof, _, _ = backtest_cell(_random_walk(300), _real_factory("naive_drift", horizon), cfg)

    covered = (oof["yhat_lower"] <= oof["y_true"]) & (oof["y_true"] <= oof["yhat_upper"])
    by_step = covered.groupby(oof["horizon_step"]).mean()

    # Flat band, growing truth: the last step of the horizon covers less than the first.
    assert by_step.loc[horizon] < by_step.loc[1], by_step.to_dict()
    # And the band really is flat — the width does not grow with the step, which is the cause.
    width = (oof["yhat_upper"] - oof["yhat_lower"]).groupby(oof["horizon_step"]).mean()
    assert width.loc[horizon] == pytest.approx(width.loc[1], rel=0.05)


def test_oof_values_match_lastvalue_model() -> None:
    # series is 1..40; last-value model on fold 0 (val at positions 32..36 for the
    # earliest window) predicts the value at the split point, flat.
    cfg = _cfg({"n_folds": 1, "horizon": 4, "step": 4, "min_train": 10})
    oof, _, _ = backtest_cell(_series(40), _factory(), cfg)
    # last training value is y at position val_start-1 = 35 → value 36.0
    assert np.allclose(oof["yhat"].to_numpy(), 36.0)
    # y_true is the actual future window: positions 36..39 → values 37..40
    assert np.allclose(oof["y_true"].to_numpy(), [37.0, 38.0, 39.0, 40.0])


def test_oof_in_original_units_under_log1p() -> None:
    # With log1p, yhat and y_true must both be back in original units.
    cfg = _cfg(
        backtest={"n_folds": 1, "horizon": 4, "step": 4, "min_train": 10},
        features={"transform": "log1p"},
    )
    oof, _, _ = backtest_cell(_series(40), _factory("log1p"), cfg)
    assert np.allclose(oof["y_true"].to_numpy(), [37.0, 38.0, 39.0, 40.0])
    # last-value model fit on log1p target, inverted → original last value 36.0
    assert np.allclose(oof["yhat"].to_numpy(), 36.0)


def test_fold_metrics_have_full_panel() -> None:
    from scale_forecasting.metrics import METRIC_NAMES

    cfg = _cfg({"n_folds": 2, "horizon": 4, "step": 4, "min_train": 10})
    _, fold_metrics, _ = backtest_cell(_series(40), _factory(), cfg)
    for m in fold_metrics:
        # The panel, plus which fold earned it. `fold_id` rides along rather than being inferred
        # from list position because a short series is exactly where position stops being the
        # fold id — and a short series is exactly where the holdout question gets interesting.
        assert set(m) == set(METRIC_NAMES) | {"fold_id"}
    assert [m["fold_id"] for m in fold_metrics] == [0, 1]


# --- the four schemes, and what each one is a measurement of ---------------------------------
#
# The geometry is identical across all four bar `sliding` (asserted in
# `test_inert_config_fields.py`), so everything below is about how the *model* is carried between
# origins — which is the only thing that changes, and the whole reason the schemes exist.


def _scheme_cfg(scheme: str) -> RunConfig:
    return _cfg({"scheme": scheme, "n_folds": 3, "horizon": 5, "step": 5, "min_train": 20})


def _shifted(n: int, at: int, jump: float) -> pd.DataFrame:
    """A flat series that steps up by ``jump`` at position ``at`` — a level shift a blind model
    cannot know about and a re-conditioned one can."""
    y = np.full(n, 100.0)
    y[at:] += jump
    return pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=n, freq="D"), "y": y})


@pytest.mark.parametrize("scheme", ["expanding", "sliding"])
def test_a_refit_scheme_says_so_and_runs_no_control_arm(scheme: str) -> None:
    oof, _, outcome = backtest_cell(
        _random_walk(200), _real_factory("naive_mean", 5), _scheme_cfg(scheme)
    )

    assert outcome.refit_mode == "per_fold"
    # Nothing to compare a fresh fit against: there is no second arm, so no gap and no column.
    assert outcome.staleness_gap is None
    assert oof["yhat_stale"].isna().all()


def test_a_frozen_scheme_fits_twice_for_the_whole_cell_not_once_per_fold() -> None:
    """The efficiency claim, stated as a count. Three folds, two fits: one for the primary arm
    carried forward on re-conditioning, one held blind for the control arm."""
    cfg = _scheme_cfg("expanding_frozen")
    inner, fits = _real_factory("naive_mean", 5), 0

    def counting() -> BaseModel:
        nonlocal fits
        fits += 1
        return inner()

    _, fold_metrics, outcome = backtest_cell(_random_walk(200), counting, cfg)
    assert len(fold_metrics) == 3
    assert fits == 2
    assert outcome.refit_mode == "recondition"


def test_the_stale_scheme_fits_once_and_never_looks_again() -> None:
    cfg = _scheme_cfg("expanding_stale")
    inner, fits = _real_factory("theta", 5), 0

    def counting() -> BaseModel:
        nonlocal fits
        fits += 1
        return inner()

    oof, fold_metrics, outcome = backtest_cell(_random_walk(200), counting, cfg)
    assert len(fold_metrics) == 3
    assert fits == 1
    assert outcome.refit_mode == "extrapolate"
    # The primary arm already *is* the blind arm, so there is nothing for a control arm to add.
    assert oof["yhat_stale"].isna().all()
    assert outcome.staleness_gap is None


def test_the_frozen_control_arm_lands_on_the_same_rows_as_the_primary_one() -> None:
    """A gap between the two is only a staleness measurement if they are scored on the same
    dates, the same actuals and the same training window. Every row carries both."""
    oof, _, outcome = backtest_cell(
        _random_walk(200), _real_factory("naive_mean", 5), _scheme_cfg("expanding_frozen")
    )

    assert oof["yhat_stale"].notna().all()
    assert outcome.staleness_gap is not None
    # Fold 0 is the fold both arms were fit on, so at that origin they have not diverged yet.
    fold0 = oof[oof["fold_id"] == 0]
    assert np.allclose(fold0["yhat"].to_numpy(), fold0["yhat_stale"].to_numpy())
    # By the last fold the primary arm has absorbed two steps of new data and the blind one has not.
    last = oof[oof["fold_id"] == oof["fold_id"].max()]
    assert not np.allclose(last["yhat"].to_numpy(), last["yhat_stale"].to_numpy())


def test_the_gap_is_positive_when_the_series_moves_under_a_model_that_cannot_see_it() -> None:
    """The diagnostic earning its place: a level shift after the first origin costs the blind arm
    real accuracy, and `staleness_gap` is how much."""
    # Fold cutoffs are 185 / 190 / 195. The shift lands at 188: after the origin both arms were fit
    # on, and inside the last two folds' new observations — so only a model told about it follows.
    # A trailing-mean model is the clearest read, because its whole state is the recent level.
    series = _shifted(200, at=188, jump=60.0)
    _, _, outcome = backtest_cell(
        series, _real_factory("naive_moving_average", 5), _scheme_cfg("expanding_frozen")
    )

    assert outcome.staleness_gap is not None
    assert outcome.staleness_gap > 0.0


def test_a_model_without_the_seam_refits_and_records_that_it_did() -> None:
    """`theta` re-estimates on every fit and has no way to absorb an observation, so asking for
    `expanding_frozen` gets an honest refit rather than an approximation wearing the name. The
    control arm still runs — the blind fit is already paid for and costs only a forecast."""
    oof, _, outcome = backtest_cell(
        _random_walk(200), _real_factory("theta", 5), _scheme_cfg("expanding_frozen")
    )

    assert outcome.refit_mode == "unsupported"
    assert oof["yhat_stale"].notna().all()
    assert outcome.staleness_gap is not None


def test_a_model_that_cannot_even_extrapolate_degrades_instead_of_raising() -> None:
    """`_LastValue` is a local stub that opts into neither seam, standing in for an out-of-tree
    model written before the frozen schemes existed. A frozen run must fall back to refitting, not
    fail the cell — scoring is never allowed to cost the forecast."""
    oof, fold_metrics, outcome = backtest_cell(
        _series(200), _factory(), _scheme_cfg("expanding_frozen")
    )

    assert outcome.refit_mode == "unsupported"
    assert len(fold_metrics) == 3
    assert oof["yhat_stale"].isna().all()
    assert outcome.staleness_gap is None


@pytest.mark.parametrize(
    ("scheme", "mode"),
    [
        ("expanding", "per_fold"),
        ("sliding", "per_fold"),
        ("expanding_frozen", "recondition"),
        ("expanding_stale", "extrapolate"),
    ],
)
def test_a_series_too_short_to_score_still_names_the_scheme_it_would_have_used(
    scheme: str, mode: str
) -> None:
    """No fold ran, so nothing was carried anywhere. The column reports the intent; the row's
    `backtest_status` is what says nothing was scored."""
    oof, fold_metrics, outcome = backtest_cell(_series(8), _factory(), _scheme_cfg(scheme))

    assert fold_metrics == []
    assert list(oof.columns) == list(OOF_COLUMNS)
    assert outcome.refit_mode == mode
    assert outcome.staleness_gap is None


def test_fold_dataclass_helpers() -> None:
    f = Fold(fold_id=0, train_start=0, train_end=30, val_start=30, val_end=35)
    assert f.train_size == 30
    assert f.val_size == 5


# --- the shared scale denominator (`training_window`) ---------------------------------------
# The Python path slices `y[fold.train_start:fold.train_end]` and hands that to `compute_metrics`
# as the MASE/RMSSE denominator. The native engine and the ensemble scorer have no folds — they
# hold a whole history and one `cutoff_date` — so `training_window` is how they reconstruct the
# same slice. If the two ever disagree, native MASE and Python MASE stop being comparable, and
# nothing in a live run would say so: both columns would still hold plausible floats. That
# equivalence is the assertion below, and it is the exit gate for standardizing the denominator.


def test_the_cutoff_rebuilds_exactly_the_slice_the_fold_trained_on() -> None:
    for scheme in ("expanding", "sliding"):
        cfg = _cfg({"n_folds": 3, "horizon": 4, "step": 4, "min_train": 10, "scheme": scheme})
        series = _series(40)
        ds = series["ds"].to_numpy()
        y = series["y"].to_numpy()
        for fold in make_folds(len(y), cfg):
            # What the Python cell fits and scales by, and the cutoff it writes onto its OOF rows.
            expected = y[fold.train_start : fold.train_end]
            cutoff = ds[fold.train_end - 1]
            assert np.array_equal(training_window(ds, y, cutoff, cfg), expected), (
                f"{scheme} fold {fold.fold_id}"
            )


def test_the_two_schemes_disagree_so_the_test_above_is_not_vacuous() -> None:
    # `sliding` caps the window at `min_train`; `expanding` grows it. A `training_window` that
    # ignored the scheme would still pass the equivalence test for expanding runs only, so the
    # difference is asserted rather than assumed.
    series = _series(40)
    ds, y = series["ds"].to_numpy(), series["y"].to_numpy()
    cutoff = ds[31]
    grown = training_window(ds, y, cutoff, _cfg({"min_train": 10, "scheme": "expanding"}))
    fixed = training_window(ds, y, cutoff, _cfg({"min_train": 10, "scheme": "sliding"}))
    assert len(grown) == 32
    assert len(fixed) == 10
    assert np.array_equal(fixed, grown[-10:])


def test_a_missing_cutoff_keeps_the_whole_history() -> None:
    # An OOF frame written before the cutoff was recorded has nothing to cut at. Falling back to
    # the full series is what those runs already did — the wrong denominator, but a stable one,
    # and better than scoring nothing.
    series = _series(20)
    ds, y = series["ds"].to_numpy(), series["y"].to_numpy()
    cfg = _cfg({"min_train": 5, "scheme": "expanding"})
    assert np.array_equal(training_window(ds, y, None, cfg), y)
    assert np.array_equal(training_window(ds, y, pd.NaT, cfg), y)


def test_the_window_excludes_the_scored_observations() -> None:
    # The defect this closes: the native engine and the ensemble scorer scaled by a history that
    # contained the very window they were being judged on.
    series = _series(20)
    ds, y = series["ds"].to_numpy(), series["y"].to_numpy()
    window = training_window(ds, y, ds[11], _cfg({"scheme": "expanding"}))
    assert window[-1] == y[11]
    assert y[12] not in set(window.tolist())


# --- fit_rows / suggest_min_train: the arithmetic behind the workload estimate --


def test_fit_rows_is_the_final_fit_plus_every_fold_window() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    rows = fit_rows(100, cfg)
    assert rows[0] == 100  # the full-history fit the shipped forecast comes from
    assert rows[1:] == [f.train_size for f in make_folds(100, cfg)]


def test_fit_rows_shrinks_with_the_folds_a_short_series_loses() -> None:
    cfg = _cfg({"n_folds": 4, "horizon": 5, "step": 5, "min_train": 10})
    assert len(fit_rows(100, cfg)) == 5  # all four folds achieved
    assert len(fit_rows(20, cfg)) == 3  # only two of the four folds achieved
    assert fit_rows(10, cfg) == [10]  # none: the cell still fits and forecasts


def test_fit_rows_is_one_fit_when_backtesting_is_off() -> None:
    assert fit_rows(500, _cfg()) == [500]


def test_n_fits_equals_the_factory_calls_the_cell_actually_makes() -> None:
    # The definitive cross-check: `estimate_workload` counts fits from the geometry, and
    # `backtest_cell` constructs one model per fold. Run the cell with a counting factory and the
    # two numbers must reconcile — the folds it ran, plus the one final full-history fit that
    # happens in `worker.run_cell` rather than here.
    from scale_forecasting.config import estimate_workload

    n = 120
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    calls = 0
    inner = _factory()

    def counting() -> BaseModel:
        nonlocal calls
        calls += 1
        return inner()

    backtest_cell(_series(n), counting, cfg)
    assert estimate_workload(cfg, obs_counts=[n]).n_fits == calls + 1


def test_suggest_min_train_reports_the_ceiling_the_marginal_series_sets() -> None:
    # Full folds need min_train <= n - horizon - (n_folds-1)*step. At n=100, horizon 5, step 5,
    # 3 folds that is 85; the 60-row series caps at 45. Nine of ten series clear 85.
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    counts = [100] * 9 + [60]
    assert suggest_min_train(counts, cfg, target_share=0.9) == 85
    assert suggest_min_train(counts, cfg, target_share=1.0) == 45


def test_suggest_min_train_is_none_when_the_panel_cannot_reach_the_target() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    # 12 observations: horizon 5 + two steps of 5 already consumes all of it.
    assert suggest_min_train([12, 12], cfg) is None
    assert suggest_min_train([], cfg) is None
