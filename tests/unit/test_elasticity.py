"""Unit tests for Track J Capability 2: Double-ML Promo & Price Elasticity (`elasticity.py`)."""

from __future__ import annotations

import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.diagnostics import (
    PROMO_ELASTICITY_COLUMNS,
    compute_promo_elasticity_decomposition,
    make_confounded_promo_dgp,
    plot_promo_elasticity,
    summarize_promo_elasticity,
)
from scale_forecasting.sdk import Forecaster


def test_double_ml_recovers_true_promo_lift_and_price_elasticity_under_confounding() -> None:
    """Acceptance Criterion 2: On a confounded DGP where promotions coincide with seasonal/holiday
    peaks, Double-ML partialing-out recovers true promo lift within +/- 15% of ground truth while
    naive correlation is upward-biased by > 25%.
    """
    true_lift = 45.0
    panel, truth = make_confounded_promo_dgp(
        n_series=3,
        n_days=365,
        true_promo_lift=true_lift,
        seasonal_amplitude=40.0,
        pull_forward_fraction=0.25,
        cannibalization_fraction=0.15,
        seed=42,
    )
    assert truth["true_promo_lift"] == true_lift

    decomp = compute_promo_elasticity_decomposition(
        panel,
        promo_col="promo_flag",
        freq="D",
    )
    assert list(decomp.columns) == list(PROMO_ELASTICITY_COLUMNS)
    assert len(decomp) == len(panel)

    summary = summarize_promo_elasticity(decomp)
    assert len(summary) == 3

    # Verify naive correlation is upward-biased due to seasonal peak confounding
    mean_naive_lift = float(summary["mean_naive_ols_lift"].mean())
    naive_rel_err = abs(mean_naive_lift - true_lift) / true_lift
    assert naive_rel_err > 0.25, (
        f"Expected naive promo lift to be confounded by >25%, got {naive_rel_err:.2%}"
    )

    # Verify Double-ML orthogonalized promo lift is within +/- 15% of ground truth
    mean_dml_lift = float(summary["mean_causal_promo_lift"].mean())
    dml_lift_rel_err = abs(mean_dml_lift - true_lift) / true_lift
    assert dml_lift_rel_err <= 0.15, (
        f"Double-ML promo lift {mean_dml_lift:.2f} outside +/-15% of true {true_lift:.2f} "
        f"(rel_err={dml_lift_rel_err:.2%})"
    )

    # Verify post-promo dip (pantry loading) and net incremental lift are accurately separated
    for _, row in summary.iterrows():
        assert float(row["total_pull_forward_dip"]) < 0.0
        assert float(row["net_to_gross_lift_ratio"]) < 1.0
        assert float(row["total_net_incremental_units"]) < float(row["total_gross_causal_lift"])


def test_forecaster_promo_elasticity_df_and_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel, _ = make_confounded_promo_dgp(n_series=2, n_days=180, seed=99)
    cfg = RunConfig.model_validate(
        {
            "run_name": "promo-elast-sdk",
            "data": {"source_table": "playground", "freq": "D", "horizon": 14},
            "models": ["regression_lags"],
            "features": {"future_covariates": ["promo_flag"]},
        }
    )
    f = Forecaster(cfg)
    decomp = f.promo_elasticity_df(panel)
    assert set(PROMO_ELASTICITY_COLUMNS).issubset(decomp.columns)
    summ = f.promo_elasticity_df(panel, summary=True)
    assert len(summ) == 2

    ax = f.plot_promo_elasticity(decomp, ts_id="sku_001")
    assert ax is not None
    ax_fallback = plot_promo_elasticity(decomp, ts_id="NON_EXISTENT_SKU")
    assert ax_fallback is not None
    ax2 = plot_promo_elasticity(pd.DataFrame())
    assert ax2 is not None
    plt.close("all")


def test_holiday_confounding_deconfounded_by_holidays_parameter() -> None:
    """C2-1 regression: passing `holidays=['US']` partials out sharp US holiday demand spikes
    from `incremental_promo_lift`.
    """
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    us_holidays = pd.to_datetime(["2025-01-01", "2025-07-04", "2025-11-27", "2025-12-25"])
    is_hol = np.asarray(dates.isin(us_holidays), dtype=float)
    promo = is_hol.copy()
    promo[rng.choice(np.where(is_hol == 0)[0], size=8, replace=False)] = 1.0
    df_hol = pd.DataFrame(
        {
            "ts_id": "sku_hol",
            "ds": dates,
            "y": 100.0 + 80.0 * is_hol + 15.0 * promo,
            "promo_flag": promo,
        }
    )
    r_hol = compute_promo_elasticity_decomposition(df_hol, holidays=["US"])
    est_lift = float(
        r_hol.loc[r_hol["incremental_promo_lift"] > 0, "incremental_promo_lift"].mean()
    )
    assert 12.0 <= est_lift <= 18.0


def test_zero_baseline_trough_does_not_explode_elasticity_ratios() -> None:
    """C2-2 regression: when `baseline_demand` touches 0.0 on deep troughs, `net_causal_elasticity`
    and `promo_lift_pct` remain bounded via series-scale regularization.
    """
    import numpy as np

    dgp_1, _ = make_confounded_promo_dgp(n_series=1, n_days=120, seed=11)
    dgp_trough = dgp_1.assign(y=np.maximum(0.0, dgp_1["y"] - 95.0))
    r_trough = compute_promo_elasticity_decomposition(dgp_trough)
    assert float(r_trough["net_causal_elasticity"].abs().max()) < 100.0
    assert float(r_trough["promo_lift_pct"].abs().max()) < 10000.0


def test_all_zero_promo_col_with_active_price_discounts_recovers_dips() -> None:
    """C2-3 regression: when `promo_col` is all 0.0 but `price_col` has active markdowns,
    `pull_forward_dip` and `cannibalization_effect` are recovered from price discounts.
    """
    import numpy as np

    dgp_3, _ = make_confounded_promo_dgp(n_series=3, n_days=120, seed=42)
    dgp_p = dgp_3.assign(
        price=np.where(dgp_3["promo_flag"] > 0, 8.0, 10.0),
        promo_flag=0.0,
    )
    r_p = compute_promo_elasticity_decomposition(dgp_p, promo_col="promo_flag", price_col="price")
    assert float(r_p["pull_forward_dip"].min()) < -1.0
    assert float(r_p["cannibalization_effect"].min()) < -1.0


def test_similarity_weighted_cannibalization_priors_and_top_cannibal_columns() -> None:
    """Verify F8 (`use_similarity_priors=True` in `compute_promo_elasticity_decomposition` /
    `estimate_promo_elasticity`): populates `top_cannibal_ts_id` and `cannibal_similarity_score`
    and weights cross-SKU cannibalization pressure by normalized trajectory similarity.
    """
    import numpy as np

    from scale_forecasting.elasticity import estimate_promo_elasticity

    assert estimate_promo_elasticity is compute_promo_elasticity_decomposition

    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    t_idx = np.arange(len(dates), dtype=float)
    # sku_A and sku_B share an identical seasonal phase portrait; sku_C is an unrelated step trend
    base_ab = 100.0 + 25.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
    base_c = 60.0 + 0.4 * t_idx

    promo_a = np.zeros(len(dates), dtype=float)
    promo_b = np.zeros(len(dates), dtype=float)
    promo_c = np.zeros(len(dates), dtype=float)
    promo_a[[10, 24, 38, 52, 66, 80, 94]] = 1.0
    promo_b[[17, 31, 45, 59, 73, 87, 101]] = 1.0
    promo_c[[12, 26, 40, 54, 68, 82, 96]] = 1.0

    y_a = base_ab + 35.0 * promo_a - 14.0 * promo_b
    y_b = base_ab + 35.0 * promo_b - 14.0 * promo_a
    y_c = base_c + 20.0 * promo_c

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "sku_A", "ds": dates, "y": y_a, "promo_flag": promo_a}),
            pd.DataFrame({"ts_id": "sku_B", "ds": dates, "y": y_b, "promo_flag": promo_b}),
            pd.DataFrame({"ts_id": "sku_C", "ds": dates, "y": y_c, "promo_flag": promo_c}),
        ],
        ignore_index=True,
    )

    decomp = estimate_promo_elasticity(panel, promo_col="promo_flag", use_similarity_priors=True)
    assert "top_cannibal_ts_id" in decomp.columns
    assert "cannibal_similarity_score" in decomp.columns

    row_a = decomp[decomp["ts_id"] == "sku_A"].iloc[0]
    row_b = decomp[decomp["ts_id"] == "sku_B"].iloc[0]
    assert row_a["top_cannibal_ts_id"] == "sku_B"
    assert row_b["top_cannibal_ts_id"] == "sku_A"
    assert float(row_a["cannibal_similarity_score"]) > 0.50

    w_mat = np.asarray(decomp.attrs["similarity_weights"], dtype=float)
    assert w_mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(w_mat), 0.0, atol=1e-7)
    np.testing.assert_allclose(w_mat.sum(axis=1), 1.0, atol=1e-6)
    # Weight between twin SKUs (0, 1) exceeds weight toward orthogonal SKU (0, 2)
    assert float(w_mat[0, 1]) > float(w_mat[0, 2])

    summary = summarize_promo_elasticity(decomp)
    assert "top_cannibal_ts_id" in summary.columns
    assert "cannibal_similarity_score" in summary.columns
    summ_a = summary[summary["ts_id"] == "sku_A"].iloc[0]
    assert summ_a["top_cannibal_ts_id"] == "sku_B"
