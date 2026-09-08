"""Offline tests for the ensemble orchestrator (``scale_forecasting.ensemble_run``).

The GCP path (``run_ensembles`` executing SQL + Write API) is the ``@gcp`` smoke in
``tests/integration/test_ensemble_smoke.py``; here we cover what is offline-testable:

* :func:`ensemble_run._apply_weights` — the pure learned-blend core: ``yhat = Σ wₘ·yhatₘ``
  renormalized over the base models *present* per ``(ts_id, forecast_date)``, with bound handling
  and row-dropping when nothing is present.
* :func:`ensemble_run.run_ensembles` short-circuits to a no-op when ``ensemble.enabled`` is false,
  without touching any GCP seam (it returns before importing ``google.cloud``).
* :func:`ensembler.combine_oof` — the pure OOF-consensus scoring core (ensembles earn their
  leaderboard metric on the backtest OOF window, not by joining true-future predictions to actuals).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.ensemble_run import (
    OOF_READ_COLUMNS,
    _apply_weights,
    _override_ensemble,
    base_read_sql,
    ensemble_scoring_basis,
    run_ensemble_scoring,
    run_ensembles,
)
from scale_forecasting.ensembler import _OOF_BLEND_COLS, _fold_key, combine_oof
from scale_forecasting.registry.ids import make_ensemble_id
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "ens run test",
        "data": {"source_table": "source_series_native", "horizon": 7},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return RunConfig(**base)


_BASE_COLS = ["ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper"]


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
        ],
        columns=_BASE_COLS,
    )


# --- _apply_weights: the pure learned-blend core -------------------------------


def test_apply_weights_is_weighted_mean_when_all_present() -> None:
    # weights already sum to 1 → blend is the plain weighted mean.
    base = _base_df(
        [
            ("s1", "theta", "2024-01-01", 10.0),
            ("s1", "arima_plus", "2024-01-01", 20.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 0.75, "arima_plus": 0.25}, "rid", "nnls")
    assert len(rows) == 1
    r = rows[0]
    assert r["model_type"] == "ensemble_nnls"
    assert r["compute_engine"] == "ensemble"
    assert r["run_id"] == "rid"
    assert r["yhat"] == pytest.approx(0.75 * 10.0 + 0.25 * 20.0)  # 12.5
    # bounds blend the same way (each base bound is yhat ± 1)
    assert r["yhat_lower"] == pytest.approx(11.5)
    assert r["yhat_upper"] == pytest.approx(13.5)


def test_apply_weights_renormalizes_unnormalized_weights() -> None:
    # raw weights need not sum to 1; the blend renormalizes over present models.
    base = _base_df(
        [
            ("s1", "theta", "2024-01-01", 10.0),
            ("s1", "arima_plus", "2024-01-01", 30.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 3.0, "arima_plus": 1.0}, "rid", "ridge")
    # (3*10 + 1*30) / 4 = 15.0
    assert rows[0]["yhat"] == pytest.approx(15.0)


def test_apply_weights_renormalizes_over_present_subset_per_date() -> None:
    # date d1 has both models; date d2 has only theta → d2 blend is theta alone (weight renormed).
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "arima_plus", "d1", 20.0),
            ("s1", "theta", "d2", 40.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 0.5, "arima_plus": 0.5}, "rid", "nnls")
    by_date = {r["forecast_date"]: r["yhat"] for r in rows}
    assert by_date["d1"] == pytest.approx(15.0)
    assert by_date["d2"] == pytest.approx(40.0)  # only theta present → its weight renormed to 1


def test_apply_weights_drops_rows_with_no_present_weighted_model() -> None:
    # only arima_plus forecasts this date, but it has zero weight → nothing to blend, drop the row.
    base = _base_df([("s1", "arima_plus", "d1", 20.0)])
    rows = _apply_weights(base, {"theta": 1.0, "arima_plus": 0.0}, "rid", "nnls")
    assert rows == []


def test_apply_weights_ignores_models_not_in_weight_map() -> None:
    # a base model with no learned weight is simply not part of the blend.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "other", "d1", 999.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 1.0}, "rid", "nnls")
    assert rows[0]["yhat"] == pytest.approx(10.0)


def test_apply_weights_empty_base_is_empty() -> None:
    base = _base_df([])
    assert _apply_weights(base, {"theta": 1.0}, "rid", "nnls") == []


# --- combine_oof: the pure OOF-consensus scoring core --------------------------

_OOF_COLS = ["ts_id", "model_type", "fold_id", "forecast_date", "y_true", "yhat"]


def _oof_df(rows: list[tuple[str, str, int, str, float, float]]) -> pd.DataFrame:
    """Long-format base OOF (ts_id, model_type, fold_id, forecast_date, y_true, yhat)."""
    return pd.DataFrame(
        [
            {
                "ts_id": t,
                "model_type": m,
                "fold_id": f,
                "forecast_date": d,
                "y_true": yt,
                "yhat": yh,
            }
            for (t, m, f, d, yt, yh) in rows
        ],
        columns=_OOF_COLS,
    )


def test_combine_oof_mean_is_unweighted_mean_per_key() -> None:
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 12.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 12.0, 20.0),
        ]
    )
    out = combine_oof(oof, cfg)
    assert list(out["model_type"].unique()) == ["ensemble_mean"]
    r = out.iloc[0]
    assert r["yhat"] == pytest.approx(15.0)  # (10 + 20) / 2
    assert r["y_true"] == pytest.approx(12.0)  # truth carried through for scoring
    assert r["fold_id"] == 0


def test_combine_oof_median_and_multiple_strategies() -> None:
    cfg = _cfg(
        models=["a", "b", "c"],
        ensemble={"enabled": True, "strategies": ["mean", "median"]},
    )
    oof = _oof_df(
        [
            ("s1", "a", 0, "d1", 5.0, 10.0),
            ("s1", "b", 0, "d1", 5.0, 20.0),
            ("s1", "c", 0, "d1", 5.0, 90.0),
        ]
    )
    out = combine_oof(oof, cfg)
    by = {m: g["yhat"].iloc[0] for m, g in out.groupby("model_type")}
    assert by["ensemble_mean"] == pytest.approx(40.0)  # (10+20+90)/3
    assert by["ensemble_median"] == pytest.approx(20.0)  # median robust to the 90 outlier


def test_combine_oof_renormalizes_over_present_models_per_key() -> None:
    # d2 has only theta present → its blend is theta alone (mean over the present subset).
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 1.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 1.0, 20.0),
            ("s1", "theta", 0, "d2", 1.0, 40.0),
        ]
    )
    out = combine_oof(oof, cfg).set_index("forecast_date")
    assert out.loc["d1", "yhat"] == pytest.approx(15.0)
    assert out.loc["d2", "yhat"] == pytest.approx(40.0)


def test_combine_oof_inverse_error_downweights_the_worse_model() -> None:
    # theta tracks truth exactly (WAPE 0), arima_plus is badly off → inverse_error ≈ theta.
    cfg = _cfg(
        models=["theta", "arima_plus"],
        ensemble={"enabled": True, "strategies": ["inverse_error"]},
    )
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 10.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 10.0, 50.0),
            ("s1", "theta", 0, "d2", 20.0, 20.0),
            ("s1", "arima_plus", 0, "d2", 20.0, 80.0),
        ]
    )
    out = combine_oof(oof, cfg).set_index("forecast_date")
    # theta has zero error → inverse_error_weights gives it all the weight → blend == theta.
    assert out.loc["d1", "yhat"] == pytest.approx(10.0)
    assert out.loc["d2", "yhat"] == pytest.approx(20.0)


def test_combine_oof_learned_applies_given_weights_and_skips_unfitted() -> None:
    # Learned strategies require backtest ON (leakage guard in config validation).
    cfg = _cfg(
        models=["theta", "arima_plus"],
        backtest={"enabled": True, "n_folds": 2, "horizon": 7, "step": 7},
        ensemble={"enabled": True, "strategies": ["nnls", "ridge"]},
    )
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 1.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 1.0, 30.0),
        ]
    )
    # Only nnls has fitted weights → ridge is skipped (no rows), nnls applies its blend.
    out = combine_oof(oof, cfg, {"nnls": {"theta": 3.0, "arima_plus": 1.0}})
    assert set(out["model_type"].unique()) == {"ensemble_nnls"}
    assert out["yhat"].iloc[0] == pytest.approx(15.0)  # (3*10 + 1*30)/4


def test_combine_oof_empty_input_is_empty() -> None:
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    empty = pd.DataFrame(columns=_OOF_COLS)
    out = combine_oof(empty, cfg)
    assert out.empty
    # The empty frame has to carry the same columns as a populated one — the scorer reads
    # `cutoff_date` off it, and a column that only exists in the non-empty case is a KeyError
    # that appears exactly once, on the run where nothing blended.
    assert list(out.columns) == list(_OOF_BLEND_COLS)


def test_combine_oof_preserves_fold_ids_for_rollup() -> None:
    # Two folds → the ensemble OOF keeps both fold_ids so the scorer can roll up per fold.
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 1.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 1.0, 20.0),
            ("s1", "theta", 1, "d2", 1.0, 30.0),
            ("s1", "arima_plus", 1, "d2", 1.0, 40.0),
        ]
    )
    out = combine_oof(oof, cfg)
    assert set(out["fold_id"].unique()) == {0, 1}
    assert np.isnan(out["yhat"]).sum() == 0


def test_combine_oof_carries_the_cutoff_for_the_scorers_denominator() -> None:
    # The scorer cuts each series' history at the fold's cutoff to get the MASE/RMSSE denominator.
    # If `combine_oof` dropped the column the scorer would fall back to the whole history — the
    # exact defect standardizing the denominator closed, silently reintroduced for ensembles only.
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 1.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 1.0, 20.0),
            ("s1", "theta", 1, "d2", 1.0, 30.0),
            ("s1", "arima_plus", 1, "d2", 1.0, 40.0),
        ]
    ).assign(cutoff_date=["c0", "c0", "c1", "c1"])
    out = combine_oof(oof, cfg)
    assert dict(zip(out["fold_id"], out["cutoff_date"], strict=True)) == {0: "c0", 1: "c1"}


def test_combine_oof_carries_the_cutoff_on_the_ordinal_fallback_too() -> None:
    # Here the cutoff is present but not fully populated, so `_fold_key` joins on the ordinal and
    # the cutoff is an ordinary column rather than an index level. It still has to come out — read
    # from the wrong side it would be all-NaN, and the scorer would quietly stop cutting.
    cfg = _cfg(models=["theta", "arima_plus"], ensemble={"enabled": True, "strategies": ["mean"]})
    oof = _oof_df(
        [
            ("s1", "theta", 0, "d1", 1.0, 10.0),
            ("s1", "arima_plus", 0, "d1", 1.0, 20.0),
        ]
    ).assign(cutoff_date=["c0", None])
    assert _fold_key(oof) == ["ts_id", "fold_id", "forecast_date"]
    out = combine_oof(oof, cfg)
    assert out["cutoff_date"].tolist() == ["c0"]


# --- run_ensembles: disabled is a no-op that never touches GCP ------------------


def test_run_ensembles_disabled_is_noop() -> None:
    # ensemble.enabled defaults to False; run_ensembles must return before importing any GCP client.
    cfg = _cfg()
    assert cfg.ensemble.enabled is False
    # No monkeypatching of google.cloud needed: a GCP touch here would raise, so a clean return
    # proves the short-circuit.
    run_ensembles(cfg, "rid", settings=_SETTINGS)


# --- ensemble_id: config-keyed coexistence -------------------------------------


def test_ensemble_id_is_stable_for_identical_config() -> None:
    a = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    b = _cfg(ensemble={"enabled": True, "strategies": ["median", "mean"]})  # order-independent
    assert make_ensemble_id(a.ensemble) == make_ensemble_id(b.ensemble)


def test_ensemble_id_differs_for_different_strategies() -> None:
    a = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    b = _cfg(ensemble={"enabled": True, "strategies": ["median"]})
    # distinct configs → distinct ids → they coexist under one run_id (never collide).
    assert make_ensemble_id(a.ensemble) != make_ensemble_id(b.ensemble)


def test_ensemble_id_ignores_enabled_toggle() -> None:
    on = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    off = _cfg(ensemble={"enabled": False, "strategies": ["mean"]})
    assert make_ensemble_id(on.ensemble) == make_ensemble_id(off.ensemble)


# --- _override_ensemble: the CLI strategy override -----------------------------


def test_override_ensemble_none_leaves_config_untouched() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    assert _override_ensemble(cfg, None) is cfg


def test_override_ensemble_rebuilds_and_enables() -> None:
    cfg = _cfg(ensemble={"enabled": False, "strategies": ["mean"], "prune_threshold": 0.3})
    out = _override_ensemble(cfg, ["median", "nnls"])
    assert out.ensemble.enabled is True
    assert set(out.ensemble.strategies) == {"median", "nnls"}
    assert out.ensemble.prune_threshold == 0.3  # preserved from the original block
    # a distinct override yields a distinct ensemble_id under the same run_id.
    assert make_ensemble_id(out.ensemble) != make_ensemble_id(cfg.ensemble)


# --- _drain_ready: the microbatch loop driver (pure; I/O injected) -------------


def test_drain_ready_processes_all_ready_in_one_pass_when_done() -> None:
    from scale_forecasting.ensemble_run import _drain_ready

    processed_batches: list[list[str]] = []
    out = _drain_ready(
        ready_fn=lambda: {"s1", "s2", "s3"},
        process_fn=processed_batches.append,
        done_fn=lambda: True,  # post-join trigger: upstream already finished
        sleep_fn=lambda: (_ for _ in ()).throw(AssertionError("must not sleep when done")),
        max_polls=10,
    )
    assert out == {"s1", "s2", "s3"}
    # One batch, series sorted deterministically; then a poll finds nothing new and exits.
    assert processed_batches == [["s1", "s2", "s3"]]


def test_drain_ready_only_processes_newly_ready_series() -> None:
    from scale_forecasting.ensemble_run import _drain_ready

    # Series arrive over three polls; each already-processed series is never re-handed.
    waves = iter([{"s1"}, {"s1", "s2"}, {"s1", "s2", "s3"}])
    ready: set[str] = set()

    def _ready() -> set[str]:
        nonlocal ready
        ready = ready | next(waves, ready)
        return ready

    batches: list[list[str]] = []

    def _done() -> bool:
        # Upstream reports done only after the third wave has surfaced.
        return len(ready) == 3

    _drain_ready(
        ready_fn=_ready,
        process_fn=batches.append,
        done_fn=_done,
        sleep_fn=lambda: None,
        max_polls=20,
    )
    assert batches == [["s1"], ["s2"], ["s3"]]  # each series handed exactly once


def test_drain_ready_waits_when_nothing_ready_but_upstream_running() -> None:
    from scale_forecasting.ensemble_run import _drain_ready

    sleeps = {"n": 0}
    polls = {"n": 0}

    def _ready() -> set[str]:
        polls["n"] += 1
        return {"s1"} if polls["n"] >= 3 else set()  # nothing ready for the first two polls

    def _done() -> bool:
        return polls["n"] >= 3  # upstream finishes as the first series lands

    def _sleep() -> None:
        sleeps["n"] += 1

    out = _drain_ready(
        ready_fn=_ready,
        process_fn=lambda _b: None,
        done_fn=_done,
        sleep_fn=_sleep,
        max_polls=20,
    )
    assert out == {"s1"}
    assert sleeps["n"] == 2  # waited across the two empty-but-not-done polls


def test_drain_ready_bounded_by_max_polls() -> None:
    from scale_forecasting.ensemble_run import _drain_ready

    # Upstream never finishes and never produces a series: the loop must still terminate.
    out = _drain_ready(
        ready_fn=lambda: set(),
        process_fn=lambda _b: None,
        done_fn=lambda: False,
        sleep_fn=lambda: None,
        max_polls=5,
    )
    assert out == set()


def test_the_oof_read_carries_enough_to_join_folds_on_the_cutoff() -> None:
    # The blend's join key is chosen from the columns present on the frame, and the frame is
    # whatever the OOF read returned. A SELECT missing `cutoff_date` therefore does not raise --
    # it quietly demotes every ensemble in the product to the fold ordinal, which is exactly the
    # key that pairs nothing across engines on a ragged panel. So the read list is asserted
    # against the chooser rather than eyeballed.
    frame = pd.DataFrame(
        {c: ["x"] if c != "y_true" and c != "yhat" else [1.0] for c in OOF_READ_COLUMNS}
    )
    assert _fold_key(frame) == ["ts_id", "cutoff_date", "forecast_date"]


def test_the_oof_read_carries_the_horizon_step_out_to_the_written_rows() -> None:
    # The blended rows land in `backtest_oof` beside the base rows, and a per-horizon read
    # (`reads.read_coverage_by_step`) excludes a NULL step — so an ensemble whose step was never
    # read back is simply missing from every per-horizon answer rather than visibly empty in one.
    assert "horizon_step" in OOF_READ_COLUMNS
    assert "horizon_step" in _OOF_BLEND_COLS


# --- base_read_sql: every read is scoped to the base models --------------------


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("forecast_predictions", "ts_id, model_type, yhat"),
        ("backtest_oof", ", ".join(OOF_READ_COLUMNS)),
        ("forecast_metadata", "ts_id, model_type, wape"),
    ],
)
def test_every_ensemble_source_read_filters_to_the_base_models(table: str, columns: str) -> None:
    # The ensemble now writes into two of the three tables it reads. Without the model filter, a
    # second pass over the same run — a microbatch drain, a `--force` re-ensemble — reads
    # `ensemble_mean` back as if it were a base model and blends consensuses of consensuses. The
    # filter is in the shared builder precisely so it cannot be forgotten on one call site.
    sql = base_read_sql("p.ds", table, columns, "'theta', 'xgboost'", "")
    assert "WHERE run_id = @run_id AND model_type IN ('theta', 'xgboost')" in sql
    assert f"FROM `p.ds.{table}`" in sql


def test_the_base_read_scopes_to_a_microbatch_and_keeps_the_extra_clause() -> None:
    sql = base_read_sql(
        "p.ds",
        "forecast_metadata",
        "ts_id, model_type, wape",
        "'theta'",
        " AND ts_id IN UNNEST(@ts_ids)",
        extra=" AND fold_id IS NULL",
    )
    # Both clauses land, and the model filter still precedes them — one WHERE, no stray AND.
    assert sql.endswith(
        "WHERE run_id = @run_id AND model_type IN ('theta') "
        "AND fold_id IS NULL AND ts_id IN UNNEST(@ts_ids)"
    )
    assert sql.count("WHERE") == 1


# --- what the ensemble rows say about how they were scored ---------------------


def _oof_two_folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_id": ["s1"] * 4,
            "model_type": ["theta", "theta", "arima_plus", "arima_plus"],
            "fold_id": [0, 1, 0, 1],
            "cutoff_date": ["2026-01-01", "2026-01-08"] * 2,
            "forecast_date": ["2026-01-02", "2026-01-09"] * 2,
            "y_true": [10.0, 11.0, 10.0, 11.0],
            "yhat": [9.0, 10.5, 11.0, 11.5],
        }
    )


def test_a_strategy_that_fits_nothing_makes_no_scoring_claim() -> None:
    # mean and median have no parameters, so no fold was reserved from anything and the honest
    # value is NULL — not "holdout", which would read as a guarantee about a fit that never ran.
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    oof = _oof_two_folds()
    assert ensemble_scoring_basis("mean", oof, cfg, "holdout") is None
    assert ensemble_scoring_basis("median", oof, cfg, "holdout") is None


def test_inverse_error_and_the_learned_strategies_each_report_their_own_basis() -> None:
    cfg = _cfg(
        backtest={"enabled": True, "n_folds": 2},
        ensemble={"enabled": True, "strategies": ["inverse_error", "nnls"]},
    )
    oof = _oof_two_folds()
    assert ensemble_scoring_basis("inverse_error", oof, cfg, "holdout") == "holdout"
    # The learned answer comes from the fit that actually happened, not from the config: a
    # coverage gap can push a stacker onto every fold even when the geometry allowed a split.
    assert ensemble_scoring_basis("nnls", oof, cfg, "in_sample") == "in_sample"


def test_a_single_fold_run_reports_in_sample_for_the_weighted_strategies() -> None:
    cfg = _cfg(
        backtest={"enabled": True, "n_folds": 1},
        ensemble={"enabled": True, "strategies": ["inverse_error"]},
    )
    oof = _oof_two_folds().query("fold_id == 0")
    assert ensemble_scoring_basis("inverse_error", oof, cfg, "in_sample") == "in_sample"


def test_the_runs_ensemble_claim_takes_the_weakest_answer() -> None:
    # One strategy fitting in-sample makes the run's ensemble metrics partly in-sample; the
    # header must not report the best case. All-None means nothing fit, so there is no claim.
    assert run_ensemble_scoring([None, None]) is None
    assert run_ensemble_scoring([None, "holdout"]) == "holdout"
    assert run_ensemble_scoring(["holdout", "in_sample"]) == "in_sample"
