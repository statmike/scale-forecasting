"""Tests for feature engineering.

Covers: log1p round-trips (apply→invert is identity), holiday parity to the `holidays`
package, exog pass-through, lag/Fourier/holiday-flag columns, the (y, X) shape/index,
level-shift detection, and the forecast-horizon design frame (`build_future_features`) —
whose whole job is that deterministic columns are recomputed at the *future* dates.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.features import (
    _fourier_terms,
    apply_transform,
    build_features,
    build_future_features,
    fit_transform_lambda,
    holiday_frame,
    invert_transform,
    level_shift_step,
)


def _cfg(features: dict[str, Any] | None = None, data: dict[str, Any] | None = None) -> RunConfig:
    base_data = {"source_table": "p.d.s"}
    if data:
        base_data.update(data)
    kw: dict[str, Any] = {"run_name": "r", "data": base_data, "models": ["theta"]}
    if features is not None:
        kw["features"] = features
    return RunConfig(**kw)


def _series(n: int = 10, with_exog: bool = False) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=n, freq="D")
    frame = {"ds": ds, "y": np.arange(1.0, n + 1.0)}
    if with_exog:
        frame["price_index"] = np.arange(100.0, 100.0 + n)
    return pd.DataFrame(frame)


# --- transforms ----------------------------------------------------------------


def test_log1p_roundtrips() -> None:
    y = pd.Series([0.0, 1.0, 9.0, 99.0])
    fwd = apply_transform(y, "log1p")
    back = invert_transform(fwd.to_numpy(), "log1p")
    assert np.allclose(back, y.to_numpy())


def test_none_transform_is_identity() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert apply_transform(y, "none") is y
    assert np.allclose(invert_transform(y.to_numpy(), "none"), y.to_numpy())


def test_log1p_rejects_below_neg_one() -> None:
    with pytest.raises(ConfigError, match="log1p"):
        apply_transform(pd.Series([-2.0, 0.0]), "log1p")


def test_boxcox_roundtrips_with_fitted_lambda() -> None:
    # λ fit on the series drives both directions; apply→invert is identity.
    y = pd.Series([10.0, 12.0, 15.0, 11.0, 14.0, 20.0, 25.0, 18.0, 22.0, 30.0])
    lam = fit_transform_lambda(y, "boxcox")
    assert lam is not None
    fwd = apply_transform(y, "boxcox", lam)
    back = invert_transform(fwd.to_numpy(), "boxcox", lam)
    assert np.allclose(back, y.to_numpy())


def test_boxcox_lambda_is_deterministic() -> None:
    y = pd.Series([3.0, 7.0, 2.0, 9.0, 5.0, 11.0, 4.0])
    assert fit_transform_lambda(y, "boxcox") == fit_transform_lambda(y, "boxcox")


def test_fit_transform_lambda_none_for_stateless() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert fit_transform_lambda(y, "none") is None
    assert fit_transform_lambda(y, "log1p") is None


def test_boxcox_requires_positive_y() -> None:
    with pytest.raises(ConfigError, match="strictly positive"):
        fit_transform_lambda(pd.Series([1.0, 0.0, 3.0]), "boxcox")
    with pytest.raises(ConfigError, match="strictly positive"):
        fit_transform_lambda(pd.Series([1.0, -2.0]), "boxcox")


def test_boxcox_without_lambda_raises() -> None:
    # A caller that forgets to fit λ gets a clear error, not a silent mis-transform.
    with pytest.raises(ConfigError, match="fitted lambda"):
        apply_transform(pd.Series([1.0, 2.0]), "boxcox")
    with pytest.raises(ConfigError, match="fitted lambda"):
        invert_transform(np.array([1.0, 2.0]), "boxcox")


def test_build_features_applies_boxcox_with_lambda() -> None:
    s = _series(8)  # y = 1..8, strictly positive
    y_raw = s["y"].astype(float)
    lam = fit_transform_lambda(y_raw, "boxcox")
    y, _ = build_features(s, _cfg(features={"transform": "boxcox"}), lam)
    # forward-transformed values differ from raw but invert back to raw.
    assert not np.allclose(y.to_numpy(), y_raw.to_numpy())
    assert np.allclose(invert_transform(y.to_numpy(), "boxcox", lam), y_raw.to_numpy())


def test_unknown_transform_raises() -> None:
    with pytest.raises(ConfigError, match="unknown transform"):
        apply_transform(pd.Series([1.0]), "sqrt")


# --- holidays ------------------------------------------------------------------


def test_holiday_frame_empty_when_unconfigured() -> None:
    hf = holiday_frame(_cfg())
    assert list(hf.columns) == ["ds", "holiday"]
    assert len(hf) == 0


def test_holiday_frame_matches_holidays_package() -> None:
    import holidays as holidays_pkg

    hf = holiday_frame(_cfg(features={"holidays": ["US"]}))
    days = set(hf["ds"].dt.date)
    us = holidays_pkg.country_holidays("US", years=range(2015, 2036))
    # July 4th 2026 and New Year 2026 are US holidays and must be present.
    assert dt.date(2026, 7, 4) in days
    assert dt.date(2026, 1, 1) in days
    # exact parity of the set within the window
    assert days == set(us.keys())


def test_holiday_frame_unknown_code_raises() -> None:
    with pytest.raises(ConfigError, match="unknown holiday country code"):
        holiday_frame(_cfg(features={"holidays": ["ZZ"]}))


# --- build_features ------------------------------------------------------------


def test_build_features_bare_returns_y_and_none() -> None:
    y, X = build_features(_series(), _cfg())
    assert X is None
    assert y.name == "y"
    assert y.index.name == "ds"
    assert y.index.dtype == np.dtype("datetime64[ns]")
    assert len(y) == 10


def test_build_features_sorts_by_date() -> None:
    s = _series(5).iloc[::-1]  # reversed
    y, _ = build_features(s, _cfg())
    assert y.index.is_monotonic_increasing


def test_build_features_applies_transform_to_y() -> None:
    y, _ = build_features(_series(4), _cfg(features={"transform": "log1p"}))
    assert np.allclose(y.to_numpy(), np.log1p(np.arange(1.0, 5.0)))


def test_build_features_exog_passthrough() -> None:
    y, X = build_features(_series(6, with_exog=True), _cfg(features={"exog": ["price_index"]}))
    assert X is not None
    assert "price_index" in X.columns
    assert np.allclose(X["price_index"].to_numpy(), np.arange(100.0, 106.0))


def test_build_features_missing_exog_raises() -> None:
    with pytest.raises(ConfigError, match="exog column 'nope'"):
        build_features(_series(), _cfg(features={"exog": ["nope"]}))


def test_build_features_holiday_flag() -> None:
    # series spanning US New Year's Day 2026-01-01
    y, X = build_features(_series(10), _cfg(features={"holidays": ["US"]}))
    assert X is not None
    assert "is_holiday" in X.columns
    # 2026-01-01 is a holiday, 2026-01-02 is not
    assert X["is_holiday"].iloc[0] == 1.0
    assert X["is_holiday"].iloc[1] == 0.0


def test_build_features_exog_lags() -> None:
    """Lags are built from the *covariate*, and the undefined head leaves the frame."""
    cfg = _cfg(features={"exog": ["price_index"], "exog_lags": {"price_index": [1, 2]}})
    y, X = build_features(_series(6, with_exog=True), cfg)
    assert X is not None
    assert {"price_index_lag_1", "price_index_lag_2"} <= set(X.columns)

    # A lag of k is undefined for the first k rows. Those rows are dropped rather than
    # filled, so six observations at a maximum lag of two leave four — and nothing in the
    # frame a model fits on is a fabricated value.
    assert len(X) == len(y) == 4
    assert not X.isna().to_numpy().any()

    raw = _series(6, with_exog=True)["price_index"].to_numpy(dtype=float)
    assert X["price_index_lag_1"].to_numpy() == pytest.approx(raw[1:5])
    assert X["price_index_lag_2"].to_numpy() == pytest.approx(raw[0:4])


def test_build_features_skips_exog_lags_for_a_model_that_owns_them() -> None:
    """Precedence: a model that lags covariates internally is handed the unlagged columns.

    The config does not get a second opinion, so nothing is lagged twice — and the head is
    not dropped either, because no column in the frame is undefined at the start.
    """
    cfg = _cfg(features={"exog": ["price_index"], "exog_lags": {"price_index": [1, 2]}})
    y, X = build_features(_series(6, with_exog=True), cfg, owns_covariate_lags=True)
    assert X is not None
    assert list(X.columns) == ["price_index"]
    assert len(X) == len(y) == 6


def test_build_features_fourier_terms() -> None:
    y, X = build_features(_series(8), _cfg(features={"fourier": True}))
    assert X is not None
    fcols = [c for c in X.columns if c.startswith("fourier_")]
    assert len(fcols) == 6  # order 3 → sin+cos × 3
    assert (X[fcols].abs() <= 1.0 + 1e-9).all().all()


def test_build_features_X_aligned_to_y() -> None:
    cfg = _cfg(features={"exog": ["price_index"], "exog_lags": {"price_index": [1]}})
    y, X = build_features(_series(7, with_exog=True), cfg)
    assert X is not None
    assert X.index.equals(y.index), "the head-drop must cut y and X together, never one of them"


def test_build_features_missing_target_raises() -> None:
    bad = _series().rename(columns={"y": "value"})
    with pytest.raises(ConfigError, match="missing required columns"):
        build_features(bad, _cfg())


# --- level shift ----------------------------------------------------------------


def _shifted(n: int = 60, cut: int = 30, jump: float = 50.0) -> pd.Series:
    """A flat, low-noise series with one abrupt additive jump at ``cut`` — the shape
    `data_gen.generator` plants via ``level_shift_prob``."""
    rng = np.random.default_rng(0)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    values[cut:] += jump
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=n, freq="D"))


def test_level_shift_step_finds_the_planted_changepoint() -> None:
    step = level_shift_step(_shifted(cut=30))
    assert step[:30].sum() == 0.0, "nothing flagged before the jump"
    assert step[30:].all(), "every observation from the jump onward is in the new regime"


def test_level_shift_step_is_a_step_not_a_spike() -> None:
    """The whole point of the encoding: a regime change persists, an outlier does not."""
    step = level_shift_step(_shifted())
    transitions = np.flatnonzero(np.diff(step) != 0)
    assert transitions.size == 1, f"a step changes value exactly once, saw {transitions.size}"


def test_level_shift_step_stays_silent_on_pure_noise() -> None:
    """A false positive hands a model a spurious regressor on a forecast nobody reviews."""
    rng = np.random.default_rng(7)
    quiet = pd.Series(100.0 + rng.normal(0.0, 1.0, 200))
    assert not level_shift_step(quiet).any()


def test_level_shift_step_zero_for_series_too_short_to_split() -> None:
    assert level_shift_step(pd.Series([1.0, 2.0, 3.0, 4.0])).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_level_shift_step_zero_when_series_is_constant() -> None:
    """Zero noise scale must degrade to 'no shift', not divide by zero."""
    assert not level_shift_step(pd.Series(np.full(40, 5.0))).any()


def test_build_features_level_shift_column_is_opt_in() -> None:
    frame = pd.DataFrame({"ds": _shifted().index, "y": _shifted().to_numpy()})
    _, off = build_features(frame, _cfg(features={"fourier": True}))
    assert off is not None and "level_shift" not in off.columns
    _, on = build_features(frame, _cfg(features={"level_shift": True}))
    assert on is not None and on["level_shift"].tolist() == level_shift_step(_shifted()).tolist()


# --- the forecast-horizon design frame -------------------------------------------


def _future_cfg(features: dict[str, Any], horizon: int = 5) -> RunConfig:
    return _cfg(features=features, data={"horizon": horizon})


def test_build_future_features_none_when_no_features_configured() -> None:
    cfg = _future_cfg({})
    y, X = build_features(_series(20), cfg)
    assert X is None
    assert build_future_features(y, X, cfg) is None


def test_build_future_features_matches_training_columns_exactly() -> None:
    """Column *order* is load-bearing: `_lag_forecaster.recursive_predict` reads exog
    positionally, so a reordered frame feeds the wrong column to the wrong coefficient."""
    cfg = _future_cfg(
        {
            "exog": ["price_index"],
            "holidays": ["US"],
            "fourier": True,
            "exog_lags": {"price_index": [1, 3]},
        }
    )
    y, X = build_features(_series(30, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert list(future.columns) == list(X.columns)


def test_build_future_features_is_indexed_by_the_future() -> None:
    cfg = _future_cfg({"fourier": True}, horizon=5)
    y, X = build_features(_series(30), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert len(future) == 5
    assert future.index[0] == y.index[-1] + pd.Timedelta(days=1)
    assert (future.index > y.index[-1]).all()


def test_build_future_features_continues_the_fourier_phase() -> None:
    """The bug this frame exists to fix: handing a model the *first* horizon rows of history
    gave it the seasonal phase of four years ago for the dates it is forecasting."""
    cfg = _future_cfg({"fourier": True}, horizon=5)
    y, X = build_features(_series(400), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    expected = _fourier_terms(pd.DatetimeIndex(future.index), cfg.data.freq, order=3)
    for name, values in expected.items():
        assert future[name].to_numpy() == pytest.approx(values)
    assert not np.allclose(future["fourier_sin_1"].to_numpy(), X["fourier_sin_1"].to_numpy()[:5])


def test_build_future_features_recomputes_holidays_at_the_future_dates() -> None:
    # A history ending 2025-12-30 puts New Year's Day inside a 5-day horizon.
    ds = pd.date_range("2025-11-01", periods=60, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": np.arange(1.0, 61.0)})
    cfg = _future_cfg({"holidays": ["US"]}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert future.loc[pd.Timestamp("2026-01-01"), "is_holiday"] == 1.0


def test_build_future_features_carries_the_level_shift_forward() -> None:
    """A regime change is still in force over the horizon — that is what makes it a shift."""
    series = _shifted(n=60, cut=30)
    frame = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
    cfg = _future_cfg({"level_shift": True}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert (future["level_shift"] == 1.0).all()


def test_build_future_features_level_shift_stays_zero_when_none_detected() -> None:
    rng = np.random.default_rng(3)
    ds = pd.date_range("2026-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": 100.0 + rng.normal(0.0, 1.0, 80)})
    cfg = _future_cfg({"level_shift": True}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert not X["level_shift"].any()
    assert not future["level_shift"].any()


def test_build_future_features_exog_lags_read_the_covariates_own_timeline() -> None:
    """The soundness property: a lagged covariate is exactly as knowable as its source.

    Nothing is invented here that the covariate did not already carry. The first ``k`` future
    steps read genuine observations, and from there the column reads whatever the *unlagged*
    covariate itself resolves to. That self-consistency is the whole reason lagging a covariate
    is sound where lagging the target was not: no one knows the future target, so the removed
    ``features.lags`` had to fill the horizon with a flat line and hand it to a coefficient
    fitted on real history.
    """
    cfg = _future_cfg({"exog": ["price_index"], "exog_lags": {"price_index": [3]}}, horizon=5)
    y, X = build_features(_series(20, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None

    lag3 = future["price_index_lag_3"].to_numpy()
    # Steps 0..2 look back into real history: the last three observed covariate values.
    assert lag3[:3] == pytest.approx(X["price_index"].to_numpy()[-3:])
    # Steps 3..4 look back into the horizon, so they read exactly what the unlagged column
    # resolves to at steps 0..1 — never a value the covariate did not itself take.
    assert lag3[3:] == pytest.approx(future["price_index"].to_numpy()[:2])


def test_build_future_features_exog_falls_back_to_the_most_recent_rows() -> None:
    """True exog is genuinely unknown; the stand-in should reflect the current regime, not
    the oldest one in the history."""
    cfg = _future_cfg({"exog": ["price_index"]}, horizon=5)
    y, X = build_features(_series(20, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert future["price_index"].to_numpy() == pytest.approx(X["price_index"].to_numpy()[-5:])


def test_build_future_features_handles_history_shorter_than_the_horizon() -> None:
    cfg = _future_cfg({"exog": ["price_index"]}, horizon=10)
    y, X = build_features(_series(4, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert len(future) == 10, "length follows the horizon, never the history"


def test_extract_static_covariates_validates_constancy() -> None:
    from scale_forecasting.features import extract_static_covariates

    s = _series(6)
    s["region"] = "NA"
    s["category"] = "enterprise"
    cfg = _cfg(features={"static_covariates": ["region", "category"]})
    static = extract_static_covariates(s, cfg)
    assert static == {"region": "NA", "category": "enterprise"}

    # Static covariates do not pollute the local single-series X matrix with constant columns.
    y, X = build_features(s, cfg)
    assert X is None

    # A time-varying value in a static covariate column is rejected.
    s_bad = s.copy()
    s_bad.loc[3, "region"] = "EMEA"
    with pytest.raises(ConfigError, match="must be constant within a series"):
        extract_static_covariates(s_bad, cfg)


def test_future_vs_past_covariates_lookahead_isolation() -> None:
    s = _series(20)
    s["promo_flag"] = np.zeros(20)
    s["temperature"] = np.arange(10.0, 30.0)
    cfg = _future_cfg(
        {
            "future_covariates": ["promo_flag"],
            "past_covariates": ["temperature"],
            "exog_lags": {"temperature": [2]},
        },
        horizon=4,
    )
    y, X = build_features(s, cfg)
    assert X is not None
    assert list(X.columns) == ["promo_flag", "temperature", "temperature_lag_2"]

    # Suppose we have a future_covariates_df where promo_flag=1.0 and temperature=999.0 (a future
    # validation window): future_covariates reads 1.0, while past_covariates ignores 999.0!
    future_window = pd.DataFrame(
        {"promo_flag": [1.0, 1.0, 0.0, 1.0], "temperature": [999.0, 999.0, 999.0, 999.0]}
    )
    future = build_future_features(y, X, cfg, future_covariates_df=future_window)
    assert future is not None
    assert future["promo_flag"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert (future["temperature"] < 100.0).all()
    assert (future["temperature_lag_2"] < 100.0).all()
    # And the first 2 steps of temperature_lag_2 read the last 2 real historical temperatures:
    assert future["temperature_lag_2"].to_numpy()[:2] == pytest.approx(
        X["temperature"].to_numpy()[-2:]
    )


def test_every_model_declares_explicit_three_tier_covariate_support() -> None:
    from scale_forecasting.models import get_model, list_models
    from scale_forecasting.playground import model_catalog

    catalog = model_catalog().set_index("model")
    assert len(catalog) == 30

    static_models = {"tide", "tft", "tsmixer"}
    for name in list_models():
        cls = get_model(name)
        assert isinstance(cls.supports_future_covariates, bool)
        assert isinstance(cls.supports_past_covariates, bool)
        assert isinstance(cls.supports_static_covariates, bool)
        assert cls.supports_future_covariates == cls.supports_exog
        assert cls.supports_past_covariates == cls.supports_exog
        assert cls.supports_static_covariates == (name in static_models)

        row = catalog.loc[name]
        assert bool(row["future_covariates"]) == cls.supports_future_covariates
        assert bool(row["past_covariates"]) == cls.supports_past_covariates
        assert bool(row["static_covariates"]) == cls.supports_static_covariates


def test_on_unsupported_covariates_fallback_strips_tiers_without_dropping_lags() -> None:
    from scale_forecasting.dag import covariate_support_report, preflight
    from scale_forecasting.features import effective_config_for_model
    from scale_forecasting.models import get_model
    from scale_forecasting.registry.ids import make_run_id
    from scale_forecasting.worker import run_cell

    base_cfg = RunConfig(
        run_name="cov policy test",
        data={"source_table": "src", "horizon": 4},
        models=["theta", "xgboost", "tide"],
        features={
            "future_covariates": ["promo_flag"],
            "past_covariates": ["temperature"],
            "static_covariates": ["base_price"],
            "exog_lags": {"promo_flag": [3], "temperature": [2]},
            "holidays": ["US"],
            "fourier": True,
            "level_shift": True,
        },
    )
    # Default "fallback" elides cleanly so existing run_id digests are unchanged.
    explicit_fallback = RunConfig(
        run_name="cov policy test",
        data={"source_table": "src", "horizon": 4},
        models=["theta", "xgboost", "tide"],
        features={
            **base_cfg.features.model_dump(),
            "on_unsupported_covariates": "fallback",
        },
    )
    assert make_run_id(base_cfg) == make_run_id(explicit_fallback)

    # Report identifies univariate fallback on theta and partial tier fallback on xgboost.
    notes = covariate_support_report(base_cfg)
    assert len(notes) == 2
    assert "theta" in notes[0] and "fall back to univariate" in notes[0]
    assert "xgboost" in notes[1] and "static_covariates" in notes[1]
    assert preflight(base_cfg) is not None

    # theta (univariate) strips dynamic/static covariate columns and exog_lags so y keeps all rows.
    theta_cfg = effective_config_for_model(base_cfg, get_model("theta"))
    assert theta_cfg.features.future_covariates == []
    assert theta_cfg.features.past_covariates == []
    assert theta_cfg.features.static_covariates == []
    assert theta_cfg.features.exog_lags == {}

    # xgboost keeps future & past covariates and their lags, but drops static_covariates.
    xgb_cfg = effective_config_for_model(base_cfg, get_model("xgboost"))
    assert xgb_cfg.features.future_covariates == ["promo_flag"]
    assert xgb_cfg.features.past_covariates == ["temperature"]
    assert xgb_cfg.features.static_covariates == []
    assert xgb_cfg.features.exog_lags == {"promo_flag": [3], "temperature": [2]}

    # tide supports all three tiers, so its config is returned untouched.
    tide_cfg = effective_config_for_model(base_cfg, get_model("tide"))
    assert tide_cfg is base_cfg

    # Running theta via run_cell on a series with promo_flag/temperature/base_price succeeds
    # and does not drop the first 3 observations onto exog_lags.
    s = _series(40)
    s["base_price"] = 19.99
    res = run_cell(
        s,
        "theta",
        RunConfig(
            run_name="theta fallback cell",
            data={"source_table": "src", "horizon": 4},
            models=["theta"],
            features=base_cfg.features.model_dump(),
        ),
    )
    assert res.status == "ok"
    assert len(res.predictions) == 4


def test_on_unsupported_covariates_error_refuses_in_preflight_and_effective_config() -> None:
    from scale_forecasting.dag import check_covariate_support, preflight
    from scale_forecasting.features import effective_config_for_model
    from scale_forecasting.models import get_model

    # Supported model with on_unsupported_covariates="error" passes preflight cleanly.
    valid_strict = RunConfig(
        run_name="strict valid",
        data={"source_table": "src", "horizon": 4},
        models=["tide", "tft"],
        features={
            "future_covariates": ["promo_flag"],
            "past_covariates": ["temperature"],
            "static_covariates": ["base_price"],
            "on_unsupported_covariates": "error",
        },
    )
    check_covariate_support(valid_strict)
    assert effective_config_for_model(valid_strict, get_model("tide")) is valid_strict

    # Univariate or partially-supported model with "error" raises ConfigError.
    invalid_strict = RunConfig(
        run_name="strict invalid",
        data={"source_table": "src", "horizon": 4},
        models=["theta", "xgboost"],
        features={
            "future_covariates": ["promo_flag"],
            "static_covariates": ["base_price"],
            "on_unsupported_covariates": "error",
        },
    )
    with pytest.raises(ConfigError, match="features.on_unsupported_covariates='error'"):
        preflight(invalid_strict)
    with pytest.raises(ConfigError, match="does not support configured covariate tier"):
        effective_config_for_model(invalid_strict, get_model("xgboost"))


def test_preflight_rejects_hierarchy_with_native_and_per_series_hpo_with_global_models() -> None:
    from scale_forecasting.dag import check_model_params

    hier_native = RunConfig(
        run_name="hier native",
        data={"source_table": "src", "horizon": 4},
        models=["theta", "arima_plus"],
        hierarchy={"enabled": True, "levels": [["region"]]},
    )
    with pytest.raises(
        ConfigError, match="hierarchy.enabled=True is not supported with BigQuery-native"
    ):
        check_model_params(hier_native)

    per_series_global = RunConfig(
        run_name="hpo global",
        data={"source_table": "src", "horizon": 4},
        models=["tide"],
        model_params={"tide": {"training_mode": "global"}},
        backtest={"enabled": True, "n_folds": 2, "horizon": 4, "step": 4, "min_train": 20},
        hpo={"enabled": True, "granularity": "per_series", "n_trials": 2},
    )
    with pytest.raises(ConfigError, match="cannot be combined with hpo.granularity='per_series'"):
        check_model_params(per_series_global)
