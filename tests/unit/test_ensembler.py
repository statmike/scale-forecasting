"""Tests for the ensembler — calculated combine math, learned meta-learners, pandas blend.

Covers mean/median exact, inverse-error weights sum to
1, NNLS weights ≥ 0, the leakage guard (learned strategies refuse to run without backtest),
multi-strategy dispatch, and the pandas :func:`combine_calculated` blend that replaced the retired
``INSERT…SELECT`` SQL (every append-only cell write now goes through the Write API).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.ensembler import (
    _fold_key,
    _pivot_oof,
    combine_calculated,
    combine_oof,
    fit_learned,
    inner_fold_mask,
    inverse_error_weights,
    mean_combine,
    median_combine,
)
from scale_forecasting.errors import ConfigError


def _cfg(strategies: list[str], *, backtest: bool = True, prune: float = 0.0) -> RunConfig:
    over: dict[str, Any] = {
        "run_name": "ens test",
        "data": {"source_table": "t"},
        "models": ["theta", "sarimax", "xgboost"],
        "ensemble": {"enabled": True, "strategies": strategies, "prune_threshold": prune},
        "backtest": {"enabled": backtest, "n_folds": 2, "decision_metric": "wape"},
    }
    return RunConfig(**over)


def _oof(models: list[str], n_per: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic long-format OOF: truth + each model's noisy forecast of it."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_per, freq="D")
    truth = np.linspace(10, 30, n_per) + rng.normal(0, 1, n_per)
    rows = []
    # model quality decreasing: theta best, then sarimax, then xgboost noisiest.
    noise = {"theta": 0.5, "sarimax": 1.5, "xgboost": 3.0}
    for m in models:
        yhat = truth + rng.normal(0, noise.get(m, 1.0), n_per)
        for d, yt, yh in zip(dates, truth, yhat, strict=True):
            rows.append(
                {"ts_id": "s1", "model_type": m, "fold_id": 0, "ds": d, "y_true": yt, "yhat": yh}
            )
    return pd.DataFrame(rows)


# --- calculated combine math ---------------------------------------------------


def test_mean_combine_exact() -> None:
    yhats = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert np.allclose(mean_combine(yhats), [2.0, 3.0, 4.0])


def test_median_combine_exact_and_robust() -> None:
    # third model is a wild outlier; median ignores it, mean would not.
    yhats = np.array([[10.0, 10.0], [12.0, 12.0], [1000.0, 1000.0]])
    assert np.allclose(median_combine(yhats), [12.0, 12.0])


def test_inverse_error_weights_sum_to_one() -> None:
    w = inverse_error_weights(np.array([1.0, 2.0, 4.0]))
    assert w.sum() == pytest.approx(1.0)
    # smaller error → larger weight
    assert w[0] > w[1] > w[2]


def test_inverse_error_zero_error_dominates() -> None:
    w = inverse_error_weights(np.array([0.0, 2.0, 4.0]))
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_inverse_error_all_nonfinite_is_uniform() -> None:
    w = inverse_error_weights(np.array([np.nan, np.inf]))
    assert np.allclose(w, [0.5, 0.5])


def test_inverse_error_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        inverse_error_weights(np.array([]))


# --- learned meta-learners -----------------------------------------------------


def test_nnls_weights_are_nonnegative() -> None:
    cfg = _cfg(["nnls"])
    weights, artifacts, _ = fit_learned(_oof(cfg.models), cfg)
    assert set(weights) == {"nnls"}
    assert all(w >= 0.0 for w in weights["nnls"].values())
    assert "nnls" in artifacts and isinstance(artifacts["nnls"], bytes)


def test_learned_trusts_the_better_model_more() -> None:
    # theta is the least-noisy base model → should earn the largest nnls weight.
    cfg = _cfg(["nnls"])
    weights, _, _ = fit_learned(_oof(cfg.models, seed=7), cfg)
    w = weights["nnls"]
    assert w["theta"] >= w["sarimax"]
    assert w["theta"] >= w["xgboost"]


def test_ridge_returns_weight_per_model() -> None:
    cfg = _cfg(["ridge"])
    weights, _, _ = fit_learned(_oof(cfg.models), cfg)
    assert set(weights["ridge"]) == set(cfg.models)


def test_xgb_weights_are_normalized_importances_not_coefficients() -> None:
    """The third learned strategy, and the one whose weights mean something different.

    ``nnls`` and ``ridge`` return regression coefficients; ``xgb`` returns the meta-learner's
    **normalized feature importances**. That distinction is why it is asserted separately:
    importances are non-negative and sum to one by construction, so a blend built from them is
    always a convex combination, where a ridge coefficient can legitimately be negative. A change
    that started returning raw importances, or coefficients, would still produce a plausible
    weight per model and quietly change what the ensemble is.
    """
    cfg = _cfg(["xgb"])
    weights, artifacts, _ = fit_learned(_oof(cfg.models), cfg)

    w = weights["xgb"]
    assert set(w) == set(cfg.models)
    assert all(v >= 0.0 for v in w.values())
    assert sum(w.values()) == pytest.approx(1.0)
    assert artifacts["xgb"]


def test_xgbs_artifact_is_the_fitted_model_where_the_others_are_a_dict() -> None:
    """An asymmetry in the payload, pinned because a reader would not expect it.

    ``nnls`` and ``ridge`` pickle a plain ``{"strategy", "models", "weights"}`` dict — the weights
    are everything needed to re-apply them. ``xgb`` pickles the estimator itself, because its
    importances summarise the model rather than being the model. Anything consuming these
    artifacts has to handle both shapes, so both shapes are asserted rather than assumed.
    """
    import pickle

    cfg = _cfg(["ridge", "xgb"])
    weights, artifacts, _ = fit_learned(_oof(cfg.models), cfg)

    assert isinstance(pickle.loads(artifacts["ridge"]), dict)
    revived = pickle.loads(artifacts["xgb"])
    assert not isinstance(revived, dict)
    assert hasattr(revived, "predict"), "the xgb artifact should be the fitted estimator"
    assert set(weights) == {"ridge", "xgb"}


def test_the_learned_dispatch_has_a_branch_for_every_learned_strategy() -> None:
    """`fit_learned` ends in a bare ``else:  # xgb``, so an unhandled strategy is fit as xgb.

    That is correct for three known strategies and silently wrong for a fourth. There would be no
    error to catch — a new learned strategy would produce weights, an artifact and a leaderboard
    entry under its own name, all of them xgb's. This is the tripwire: extending
    `LEARNED_STRATEGIES` without extending the dispatch fails here rather than shipping a
    mislabelled ensemble.
    """
    from scale_forecasting.config import LEARNED_STRATEGIES

    assert LEARNED_STRATEGIES == {"nnls", "ridge", "xgb"}


def test_multi_strategy_fits_each_learned() -> None:
    cfg = _cfg(["nnls", "ridge"])
    weights, artifacts, _ = fit_learned(_oof(cfg.models), cfg)
    assert set(weights) == {"nnls", "ridge"}
    assert set(artifacts) == {"nnls", "ridge"}


def test_calculated_only_config_fits_nothing() -> None:
    cfg = _cfg(["mean", "median"])
    weights, artifacts, _ = fit_learned(_oof(cfg.models), cfg)
    assert weights == {} and artifacts == {}


# --- leakage guard -------------------------------------------------------------


def test_learned_without_backtest_is_rejected() -> None:
    # NOTE: RunConfig drops learned strategies when backtest is off, so construct the guard
    # scenario by asking fit_learned directly with a backtest-off config that still lists one.
    cfg = _cfg(["nnls"], backtest=True)
    off = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"enabled": False})})
    with pytest.raises(ConfigError, match="require backtest"):
        fit_learned(_oof(cfg.models), off)


def test_learned_with_empty_oof_is_rejected() -> None:
    cfg = _cfg(["nnls"])
    empty = pd.DataFrame(columns=["ts_id", "model_type", "fold_id", "ds", "y_true", "yhat"])
    with pytest.raises(ConfigError, match="empty"):
        fit_learned(empty, cfg)


def test_learned_missing_a_base_model_is_rejected() -> None:
    cfg = _cfg(["nnls"])
    partial = _oof(["theta", "sarimax"])  # xgboost absent from OOF
    with pytest.raises(ConfigError, match="missing base models"):
        fit_learned(partial, cfg)


# --- the newest fold is never fit on ------------------------------------------


def _two_fold_oof(n_per: int = 200) -> pd.DataFrame:
    """OOF over two folds where ``xgboost`` is *exact* on the newest fold and the worst on the old.

    This is the shape the invariant exists for, and the arithmetic is deliberate. Averaged over
    every row xgboost looks like the best base model — half its residuals are zero. Averaged over
    the inner fold alone it is the worst of the three. So a fit that sees all the folds prefers it
    and is then scored on the very rows that earned it the preference; a fit that reserves fold 1
    prefers theta. The noise levels sit inside the window where those two answers disagree
    (``theta_sd < xgboost_sd < theta_sd * sqrt(2)``), which is what makes the test a test.
    """
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=n_per, freq="D")
    truth = np.linspace(10, 60, n_per) + rng.normal(0, 1, n_per)
    folds = np.where(np.arange(n_per) < n_per // 2, 0, 1)
    noise = {"theta": 2.0, "sarimax": 2.8, "xgboost": 2.5}
    rows = []
    for m in ("theta", "sarimax", "xgboost"):
        yhat = truth + rng.normal(0, noise[m], n_per)
        if m == "xgboost":
            yhat = np.where(folds == 1, truth, yhat)
        for f, d, yt, yh in zip(folds, dates, truth, yhat, strict=True):
            rows.append(
                {
                    "ts_id": "s1",
                    "model_type": m,
                    "fold_id": int(f),
                    "ds": d,
                    "y_true": yt,
                    "yhat": yh,
                }
            )
    return pd.DataFrame(rows)


def test_a_model_that_is_perfect_only_on_the_newest_fold_does_not_win_the_stacker() -> None:
    from scipy.optimize import nnls

    cfg = _cfg(["nnls"])  # n_folds=2 → fold 1 is the holdout
    oof = _two_fold_oof()
    # What fitting on everything would have done — the same NNLS on the unfiltered pivot. It is
    # here so the test fails loudly if the fixture ever stops posing the question.
    naive_x, naive_y = _pivot_oof(oof, cfg.models)
    naive = dict(zip(cfg.models, nnls(naive_x, naive_y)[0], strict=True))
    assert naive["xgboost"] > naive["theta"]

    weights, _artifacts, basis = fit_learned(oof, cfg)
    assert basis == "holdout"
    assert weights["nnls"]["theta"] > weights["nnls"]["xgboost"]


def test_inverse_error_weights_are_earned_on_the_inner_folds_only() -> None:
    cfg = _cfg(["inverse_error"])
    oof = _two_fold_oof()
    # The same call with the holdout pointed at a fold that does not exist: every row becomes an
    # inner row, so this is the leaking behaviour reproduced through the shipping code path rather
    # than re-implemented in the test.
    leaky = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"n_folds": 3})})

    def _holdout_error(config: RunConfig) -> float:
        blended = combine_oof(oof, config, learned_weights={})
        hold = blended[blended["fold_id"] == 1]
        assert not hold.empty
        return float(np.abs(hold["yhat"].to_numpy() - hold["y_true"].to_numpy()).mean())

    # Leaked weights flatter themselves on the holdout — they were partly chosen by it. The point
    # is not that the honest blend is *better*; it is that its number is not self-congratulatory.
    assert _holdout_error(cfg) > _holdout_error(leaky) * 1.1


def test_a_single_fold_run_falls_back_and_says_so() -> None:
    cfg = _cfg(["nnls"])
    one = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"n_folds": 1})})
    oof = _oof(one.models)  # every row is fold 0, which is now the holdout
    mask, basis = inner_fold_mask(oof, one)
    assert basis == "in_sample" and mask.all()
    weights, _artifacts, learned_basis = fit_learned(oof, one)
    # It still produces an ensemble — refusing would cost the run its whole learned blend over a
    # config choice — but the run records that the number is in-sample rather than held out.
    assert set(weights) == {"nnls"} and learned_basis == "in_sample"


def test_an_oof_frame_without_fold_ids_cannot_claim_a_holdout() -> None:
    cfg = _cfg(["nnls"])
    oof = _oof(cfg.models).drop(columns=["fold_id"])
    mask, basis = inner_fold_mask(oof, cfg)
    assert basis == "in_sample" and mask.all()


# --- combine_calculated: the pandas blend (Write-API path) ---------------------


def _base_df(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Long-format base predictions (ts_id, model_type, forecast_date, yhat); bounds mirror yhat."""
    return pd.DataFrame(
        [
            {
                "ts_id": t,
                "model_type": m,
                "forecast_date": d,
                "yhat": y,
                "yhat_lower": y - 1.0,
                "yhat_upper": y + 1.0,
            }
            for (t, m, d, y) in rows
        ]
    )


def _metric_df(rows: list[tuple[str, float]], *, metric: str = "wape") -> pd.DataFrame:
    """Per-model run-level metric rows (model_type, <metric>) — the forecast_metadata subset."""
    return pd.DataFrame([{"model_type": m, metric: v} for (m, v) in rows])


def _series_metric_df(rows: list[tuple[str, str, float]], *, metric: str = "wape") -> pd.DataFrame:
    """Per-series metric rows (ts_id, model_type, <metric>) — the shape production reads."""
    return pd.DataFrame([{"ts_id": t, "model_type": m, metric: v} for (t, m, v) in rows])


def test_no_calculated_strategy_yields_no_rows() -> None:
    # a learned-only config produces nothing from the calculated blender.
    assert combine_calculated(_base_df([("s1", "theta", "d1", 10.0)]), _cfg(["nnls"])) == []


def test_empty_base_yields_no_rows() -> None:
    assert combine_calculated(_base_df([]), _cfg(["mean"])) == []


def test_mean_blends_base_predictions() -> None:
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    rows = combine_calculated(base, _cfg(["mean"]))
    assert len(rows) == 1
    r = rows[0]
    assert r["model_type"] == "ensemble_mean"
    assert r["yhat"] == pytest.approx(15.0)  # (10 + 20) / 2
    assert r["yhat_lower"] == pytest.approx(14.0)  # bounds blend the same way (yhat ± 1)
    assert r["yhat_upper"] == pytest.approx(16.0)
    # run_id / ensemble_id are stamped by the orchestrator, not the pure blender.
    assert "run_id" not in r and "ensemble_id" not in r


def test_median_is_robust_to_a_wild_base_forecast() -> None:
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 12.0),
            ("s1", "xgboost", "d1", 1000.0),
        ]
    )
    rows = combine_calculated(base, _cfg(["median"]))
    assert rows[0]["yhat"] == pytest.approx(12.0)  # median ignores the 1000 outlier


def test_inverse_error_weights_by_run_metric() -> None:
    # theta far better than sarimax (lower wape) → blend pulled toward theta's 10.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    metric = _metric_df([("theta", 0.1), ("sarimax", 0.9)])
    rows = combine_calculated(base, _cfg(["inverse_error"]), metric)
    # weights ∝ 1/0.1 : 1/0.9 = 9 : 1 → (9*10 + 1*30)/10 = 12.0
    assert rows[0]["yhat"] == pytest.approx(12.0)


def test_inverse_error_weights_are_estimated_per_series() -> None:
    # Two series that disagree about which model is better. Pooled weights would hand both the same
    # blend; per-series weights pull each toward its own winner.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 30.0),
            ("s2", "theta", "d1", 10.0),
            ("s2", "sarimax", "d1", 30.0),
        ]
    )
    metric = _series_metric_df(
        [("s1", "theta", 0.1), ("s1", "sarimax", 0.9), ("s2", "theta", 0.9), ("s2", "sarimax", 0.1)]
    )
    by_series = {
        r["ts_id"]: r["yhat"] for r in combine_calculated(base, _cfg(["inverse_error"]), metric)
    }
    # s1 trusts theta 9:1 → (9*10 + 1*30)/10 = 12.0; s2 trusts sarimax 9:1 → the mirror image.
    assert by_series["s1"] == pytest.approx(12.0)
    assert by_series["s2"] == pytest.approx(28.0)


def test_inverse_error_is_unchanged_by_how_the_series_are_batched() -> None:
    """The 2026-09-11 live defect, pinned.

    Microbatch hands `combine_calculated` one ready-batch of series at a time; barrier hands it all
    of them at once. Those have to agree, and for ``inverse_error`` they did not: weights came from
    a run-wide ``groupby("model_type").mean()`` over whatever metric rows the call was given, so
    filtering to a batch's series changed them. Smokes 11 and 12 differed on every one of 2,800
    prediction rows while their leaderboards matched, because the OOF-scored counterpart really was
    per-series and hid it. Split the input two ways and demand the same numbers.
    """
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 30.0),
            ("s2", "theta", "d1", 50.0),
            ("s2", "sarimax", "d1", 70.0),
        ]
    )
    metric = _series_metric_df(
        [("s1", "theta", 0.1), ("s1", "sarimax", 0.9), ("s2", "theta", 0.9), ("s2", "sarimax", 0.2)]
    )
    cfg = _cfg(["inverse_error"])

    whole = {r["ts_id"]: r["yhat"] for r in combine_calculated(base, cfg, metric)}
    batched: dict[str, float] = {}
    for ts_id in ("s1", "s2"):
        rows = combine_calculated(
            base[base["ts_id"] == ts_id], cfg, metric[metric["ts_id"] == ts_id]
        )
        batched.update({r["ts_id"]: r["yhat"] for r in rows})

    assert whole == pytest.approx(batched), "batching the series changed the forecast"


def test_inverse_error_series_without_metadata_falls_back_to_mean() -> None:
    # s2 has no metric rows of its own. Falling back to the *pooled* weights would reintroduce the
    # batch dependence the per-series estimate exists to remove, so it degrades to uniform instead.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 30.0),
            ("s2", "theta", "d1", 10.0),
            ("s2", "sarimax", "d1", 30.0),
        ]
    )
    metric = _series_metric_df([("s1", "theta", 0.1), ("s1", "sarimax", 0.9)])
    by_series = {
        r["ts_id"]: r["yhat"] for r in combine_calculated(base, _cfg(["inverse_error"]), metric)
    }
    assert by_series["s1"] == pytest.approx(12.0)
    assert by_series["s2"] == pytest.approx(20.0)  # (10 + 30) / 2


def test_inverse_error_without_metric_frame_degrades_to_mean() -> None:
    # no metadata → uniform weights (the old SQL's NULL-tolerant SAFE_DIVIDE behavior).
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    rows = combine_calculated(base, _cfg(["inverse_error"]), None)
    assert rows[0]["yhat"] == pytest.approx(20.0)  # (10 + 30) / 2


def test_prune_threshold_drops_weak_base_models_fleetwide() -> None:
    # sarimax's mean wape (0.8) exceeds the 0.5 threshold → dropped; mean is theta alone.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    metric = _metric_df([("theta", 0.1), ("sarimax", 0.8)])
    rows = combine_calculated(base, _cfg(["mean"], prune=0.5), metric)
    assert rows[0]["yhat"] == pytest.approx(10.0)  # sarimax pruned → theta only


def test_blend_renormalizes_over_present_models_per_key() -> None:
    # d2 has only theta present → its mean blend is theta alone.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 20.0),
            ("s1", "theta", "d2", 40.0),
        ]
    )
    rows = combine_calculated(base, _cfg(["mean"]))
    by_date = {r["forecast_date"]: r["yhat"] for r in rows if r["model_type"] == "ensemble_mean"}
    assert by_date["d1"] == pytest.approx(15.0)
    assert by_date["d2"] == pytest.approx(40.0)


def test_multi_strategy_emits_each_calculated_family() -> None:
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    metric = _metric_df([("theta", 0.5), ("sarimax", 0.5)])
    rows = combine_calculated(base, _cfg(["mean", "median", "inverse_error"]), metric)
    assert {r["model_type"] for r in rows} == {
        "ensemble_mean",
        "ensemble_median",
        "ensemble_inverse_error",
    }


def test_learned_strategies_are_ignored_by_calculated_blender() -> None:
    # a mixed config blends only the calculated members; learned ones are fit_learned's job.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    rows = combine_calculated(base, _cfg(["mean", "nnls"]))
    assert {r["model_type"] for r in rows} == {"ensemble_mean"}


# --- fold identity on a ragged panel (plan item 2.6) ---------------------------------
#
# `fold_id` is an ordinal within one series' own plan, and the two engines number from different
# anchors: `backtest.make_folds` counts back from each series' last observation, while
# `bigquery_sql.fold_plan` counts back from one global `MAX(ds)`. Where series end on different
# dates those disagree, and the disagreement is invisible — the join simply finds nothing, and
# `_pivot_oof`'s `dropna()` deletes the rows a meta-learner would have learned from.

_HORIZON, _N_FOLDS = 7, 3
_GLOBAL_END = pd.Timestamp("2026-03-31")
# The short series stops one whole fold-step earlier than the long one. That offset is the entire
# defect: it is what makes "fold 1" name a different training window on each engine.
_SERIES_END = {"long": _GLOBAL_END, "short": _GLOBAL_END - pd.Timedelta(days=_HORIZON)}


def _ragged_two_engine_oof() -> pd.DataFrame:
    """Two series ending a fold apart, scored by one Python model and one BigQuery-native model.

    Each row is written the way its engine writes it: `theta` anchors each series on that series'
    own last observation, `arima_plus` anchors both on the panel's last observation and loses the
    rows with no actual to join to. Both record the cutoff they actually trained to.
    """
    rows = []
    for model, per_series_anchor in (("theta", True), ("arima_plus", False)):
        for ts_id, end in _SERIES_END.items():
            anchor = end if per_series_anchor else _GLOBAL_END
            for fold_id in range(_N_FOLDS):
                cutoff = anchor - pd.Timedelta(days=_HORIZON * (_N_FOLDS - fold_id))
                for step in range(1, _HORIZON + 1):
                    date = cutoff + pd.Timedelta(days=step)
                    if date > end:
                        continue  # no actual to join to; this engine writes no row
                    rows.append(
                        {
                            "ts_id": ts_id,
                            "model_type": model,
                            "fold_id": fold_id,
                            "forecast_date": date,
                            "cutoff_date": cutoff,
                            "horizon_step": step,
                            "y_true": 100.0 + date.day,
                            "yhat": 100.0 + date.day + (0.5 if model == "theta" else -0.5),
                        }
                    )
    return pd.DataFrame(rows)


def test_the_cutoff_is_the_join_key_and_the_ordinal_is_the_fallback() -> None:
    oof = _ragged_two_engine_oof()
    assert _fold_key(oof) == ["ts_id", "cutoff_date", "forecast_date"]
    # An OOF written before the native path projected its cutoff — the ordinal is all there is,
    # which is what the product did for its whole history.
    assert _fold_key(oof.drop(columns=["cutoff_date"])) == ["ts_id", "fold_id", "forecast_date"]
    # Partially populated counts as absent: half a key is worse than the old one, because the
    # rows that do have it would pair while the rest silently would not.
    partial = oof.copy()
    partial.loc[partial.index[0], "cutoff_date"] = pd.NaT
    assert _fold_key(partial) == ["ts_id", "fold_id", "forecast_date"]


def test_a_ragged_panel_keeps_the_short_series_when_folds_are_keyed_on_the_cutoff() -> None:
    """The exit gate. On the ordinal the short series contributes nothing at all."""
    oof = _ragged_two_engine_oof()
    models = ["theta", "arima_plus"]

    x_cut, y_cut = _pivot_oof(oof, models)
    x_ord, y_ord = _pivot_oof(oof.drop(columns=["cutoff_date"]), models)

    # The long series is anchored the same way by both engines, so it pairs either way: three
    # folds of seven dates. That is the whole of what the ordinal key recovers.
    assert len(x_ord) == _N_FOLDS * _HORIZON
    # The short series shares two of its windows with the native path — same cutoff, same dates,
    # same information — and the ordinal names them differently on each engine, so none of them
    # pair. Keyed on the cutoff, all fourteen do.
    assert len(x_cut) == _N_FOLDS * _HORIZON + 2 * _HORIZON
    assert len(y_cut) == len(x_cut) and len(y_ord) == len(x_ord)
    assert np.isfinite(x_cut).all(), "a paired row must have both models, not a filled NaN"


def test_the_ensemble_blends_both_models_on_the_short_series_of_a_ragged_panel() -> None:
    """Same defect one layer up, and here it is worse than a dropped row — it is a wrong number.

    `combine_oof` blends over whichever models are present per key, so an unpaired row does not
    disappear: it produces a two-model ensemble's row from one model's forecast. The short series
    still shows up on the leaderboard, still labelled `ensemble_mean`, and nothing about the row
    says it was a consensus of one.
    """
    cfg = RunConfig(
        **{
            "run_name": "ens ragged",
            "data": {"source_table": "t"},
            "models": ["theta", "arima_plus"],
            "ensemble": {"enabled": True, "strategies": ["mean"]},
            "backtest": {"enabled": True, "n_folds": _N_FOLDS, "decision_metric": "wape"},
        }
    )
    oof = _ragged_two_engine_oof()

    blended = combine_oof(oof, cfg)
    on_ordinal = combine_oof(oof.drop(columns=["cutoff_date"]), cfg)

    # The two models straddle the truth by ±0.5, so a row that blended both is the truth exactly
    # and a row that blended one is half a unit off. That makes "how many models entered this
    # blend" readable straight off the number.
    def _two_model_rows(df: pd.DataFrame, ts_id: str) -> int:
        rows = df[df["ts_id"] == ts_id]
        return int(np.isclose(rows["yhat"], rows["y_true"]).sum())

    # The long series is anchored identically by both engines, so it is unaffected either way.
    assert _two_model_rows(blended, "long") == _N_FOLDS * _HORIZON
    assert _two_model_rows(on_ordinal, "long") == _N_FOLDS * _HORIZON

    # The short series shares two windows with the native path. Keyed on the cutoff, both blend
    # two models; keyed on the ordinal, **not one row does** — and the rows are still there,
    # wearing an ensemble's name over a single model's forecast. A dropped row would at least be
    # visible.
    assert _two_model_rows(blended, "short") == 2 * _HORIZON
    assert _two_model_rows(on_ordinal, "short") == 0, "the defect this item fixes"
    assert not on_ordinal[on_ordinal["ts_id"] == "short"].empty

    # The short series' oldest window has no native counterpart at all — the native path never
    # trained a fold that far back. That row is a one-model blend under either key, and honestly
    # so: this item aligns the folds that exist, it does not invent one.
    assert len(blended[blended["ts_id"] == "short"]) == _N_FOLDS * _HORIZON

    # `fold_id` is still written, because it is the ordinal a reader recognises and the ensemble
    # rows sit in the same table as the base rows. It is carried, not joined on.
    assert blended["fold_id"].notna().all()
