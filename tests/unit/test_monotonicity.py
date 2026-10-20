"""Unit tests for Track J Capability 1: Monotonicity Constraints (`monotonicity.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.auto import build_config
from scale_forecasting.config import RunConfig
from scale_forecasting.features import (
    expand_model_variants,
    resolve_model_variant,
)
from scale_forecasting.models import get_model, list_models
from scale_forecasting.monotonicity import (
    MONOTONE_SUPPORTED_MODELS,
    audit_scenario_monotonicity,
    build_monotonic_constraint_vector,
    fit_sign_constrained_ridge,
    infer_monotonic_constraints,
    list_monotonic_models,
    resolve_model_monotonic_vector,
    resolve_monotonic_constraints,
    validate_monotonic_constraints_map,
)
from scale_forecasting.registry.ids import make_model_spec_hash, make_run_id
from scale_forecasting.scenarios import simulate_scenarios


def _make_confounded_price_panel(n_series: int = 2, n_days: int = 120) -> pd.DataFrame:
    """Create a panel where price increases coincide with high-demand peak periods so unconstrained
    learners fit a spurious positive price slope, testing whether monotonic constraints eliminate
    upward demand responses under +5%..+30% price hikes.
    """
    rng = np.random.default_rng(123)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows: list[dict[str, object]] = []
    for s in range(n_series):
        tid = f"sku_{s}"
        for t, d in enumerate(dates):
            # High demand peak every 10 days where price is ALSO raised (+25%), creating positive
            # observational correlation between price_index and y without aligning to 7/14/28 lags.
            peak = 1.0 if (t % 10 in (7, 8)) else 0.0
            promo_flag = 1.0 if (t % 21 == 5) else 0.0
            price_index = 1.0 + 0.25 * peak - 0.15 * promo_flag + rng.normal(0.0, 0.01)
            y = 100.0 + 80.0 * peak + 35.0 * promo_flag + rng.normal(0.0, 2.0)
            rows.append(
                {
                    "ts_id": tid,
                    "ds": d,
                    "y": float(max(1.0, y)),
                    "price_index": float(price_index),
                    "promo_flag": float(promo_flag),
                }
            )
    return pd.DataFrame(rows)


def test_validate_and_infer_monotonic_constraints() -> None:
    cleaned = validate_monotonic_constraints_map(
        {"price_index": -1, "promo_flag": 1, "neutral_col": 0}
    )
    assert cleaned == {"price_index": -1, "promo_flag": 1, "neutral_col": 0}

    with pytest.raises(ValueError, match="must be -1, 0, or 1"):
        validate_monotonic_constraints_map({"price_index": -2})

    inferred = infer_monotonic_constraints(
        ["price_index", "unit_price", "promo_flag", "discount_depth", "temperature"]
    )
    assert inferred["price_index"] == -1
    assert inferred["unit_price"] == -1
    assert inferred["promo_flag"] == 1
    assert inferred["discount_depth"] == 1
    assert "temperature" not in inferred


def test_resolve_monotonic_vector_and_sign_constrained_ridge() -> None:
    feature_cols = ["lag_1", "lag_7", "dow_sin", "price_index", "promo_flag"]
    vec = resolve_model_monotonic_vector(
        feature_cols,
        params=None,
        ctx_constraints={"price_index": -1, "promo_flag": 1},
    )
    assert vec == [0, 0, 0, -1, 1]
    assert build_monotonic_constraint_vector(feature_cols, {"price_index": -1}) == [
        0,
        0,
        0,
        -1,
        0,
    ]
    resolved_map = resolve_monotonic_constraints(
        {"price_index": -1},
        candidate_columns=["price_index", "promo_flag"],
        infer_if_empty=True,
    )
    assert resolved_map == {"price_index": -1}
    resolved_inferred = resolve_monotonic_constraints(
        {},
        candidate_columns=["price_index", "promo_flag"],
        infer_if_empty=True,
    )
    assert resolved_inferred == {"price_index": -1, "promo_flag": 1}

    # Construct synthetic linear data where y is positively correlated with x0 (spurious price)
    # and negatively correlated with x1 (spurious promo), then verify sign-constrained Ridge
    # enforces beta_0 <= 0 and beta_1 >= 0.
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 2))
    y = 50.0 + 4.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0.0, 0.5, size=80)
    w = fit_sign_constrained_ridge(X, y, alpha=1.0, constraint_vec=[-1, 1])
    intercept, coef = float(w[0]), w[1:]
    assert coef[0] <= 1e-10
    assert coef[1] >= -1e-10
    assert np.isfinite(intercept)


def test_canonical_model_count_and_ridge_lags_alias() -> None:
    # Canonical model count must remain 37 while ridge_lags resolves cleanly as an alias.
    assert len(list_models()) == 37
    assert "regression_lags" in MONOTONE_SUPPORTED_MODELS
    m_cls = get_model("ridge_lags")
    assert m_cls.name == "regression_lags"

    cfg = RunConfig.model_validate(
        {
            "run_name": "test-ridge-alias",
            "data": {"source_table": "t", "horizon": 7},
            "models": ["ridge_lags", "ridge_lags@monotone"],
            "model_params": {"ridge_lags": {"alpha": 2.5}},
            "features": {"future_covariates": ["price_index", "promo_flag"]},
        }
    )
    assert cfg.models == ["regression_lags", "regression_lags@monotone"]
    assert cfg.model_params["regression_lags"]["alpha"] == 2.5


def test_run_id_invariance_and_model_spec_hash_with_monotonic_constraints() -> None:
    base_raw = {
        "run_name": "monotone-hash-check",
        "data": {"source_table": "t", "horizon": 7},
        "models": ["lightgbm", "regression_lags"],
        "features": {"future_covariates": ["price_index", "promo_flag"]},
    }
    cfg_default = RunConfig.model_validate(base_raw)
    cfg_explicit_empty = RunConfig.model_validate(
        {
            **base_raw,
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
                "monotonic_constraints": {},
            },
        }
    )
    assert make_run_id(cfg_default) == make_run_id(cfg_explicit_empty)

    cfg_constrained = RunConfig.model_validate(
        {
            **base_raw,
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
                "monotonic_constraints": {"price_index": -1, "promo_flag": 1},
            },
        }
    )
    assert make_run_id(cfg_default) != make_run_id(cfg_constrained)
    assert make_model_spec_hash("lightgbm", cfg_default) != make_model_spec_hash(
        "lightgbm", cfg_constrained
    )


def test_monotone_variant_parsing_and_ablation_expansion() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "ablation-monotone",
            "data": {"source_table": "t", "horizon": 7},
            "models": ["lightgbm", "lightgbm@monotone", "regression_lags@monotone"],
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
                "ablation_modes": ["univariate"],
            },
        }
    )
    expanded, _ = expand_model_variants(cfg)
    assert "lightgbm@monotone" in expanded
    assert "regression_lags@monotone" in expanded

    v_spec = resolve_model_variant(cfg, "lightgbm@monotone")
    assert v_spec.base_model == "lightgbm"
    assert v_spec.effective_cfg.features.monotonic_constraints == {
        "price_index": -1,
        "promo_flag": 1,
    }

    v_direct = resolve_model_variant(cfg, "xgboost@direct_monotone")
    assert v_direct.base_model == "xgboost"
    assert v_direct.clean_params.get("strategy") == "direct"
    assert v_direct.effective_cfg.features.monotonic_constraints == {
        "price_index": -1,
        "promo_flag": 1,
    }


def test_zero_upward_price_shock_violations_across_monotone_models() -> None:
    """Acceptance Criterion 1: Sweeping price shocks (+5%, +10%, +20%, +30%) across constrained
    models (`lightgbm@monotone`, `xgboost@monotone`, `regression_lags@monotone`, and
    `catboost@monotone` if available) produces 0 upward demand responses (`delta_yhat <= 0`).
    """
    panel = _make_confounded_price_panel(n_series=2, n_days=100)
    candidate_models = ["regression_lags@monotone", "xgboost@monotone", "lightgbm@monotone"]
    if get_model("catboost").is_available():
        candidate_models.append("catboost@monotone")
    active_models = [m for m in candidate_models if get_model(m.split("@", 1)[0]).is_available()]
    assert len(active_models) >= 2

    cfg = RunConfig.model_validate(
        {
            "run_name": "price-shock-sweep",
            "data": {"source_table": "playground", "freq": "D", "horizon": 7},
            "models": active_models,
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
                "monotonic_constraints": {"price_index": -1, "promo_flag": 1},
            },
            "backtest": {"enabled": False},
        }
    )
    shocks = {
        "hike_5pct": {"price_index": "*1.05"},
        "hike_10pct": {"price_index": "*1.10"},
        "hike_20pct": {"price_index": "*1.20"},
        "hike_30pct": {"price_index": "*1.30"},
    }
    sim_df = simulate_scenarios(panel, cfg, shocks, price_col="price_index")
    assert not sim_df.empty
    for scen_name in shocks:
        sub = sim_df[sim_df["scenario"] == scen_name]
        assert not sub.empty
        max_delta = float(sub["delta_yhat"].max())
        assert max_delta <= 1e-6, (
            f"Upward price-shock violation in {scen_name}: max delta_yhat={max_delta}"
        )


def test_auto_forecaster_compiles_monotonic_constraints_for_price_and_promo() -> None:
    panel = _make_confounded_price_panel(n_series=2, n_days=90)
    bp = build_config(panel, tier="wide_field", where="local", horizon=7)
    assert bp.config.features.monotonic_constraints.get("price_index") == -1
    assert bp.config.features.monotonic_constraints.get("promo_flag") == 1


def test_list_monotonic_models_and_random_forest_monotone() -> None:
    mono_models = list_models(supports_monotonic_constraints=True)
    assert mono_models == [
        "catboost",
        "lightgbm",
        "random_forest",
        "regression_lags",
        "xgboost",
    ]
    assert list_monotonic_models() == mono_models
    assert "random_forest" in MONOTONE_SUPPORTED_MODELS

    panel = _make_confounded_price_panel(n_series=1, n_days=100)
    cfg = RunConfig.model_validate(
        {
            "run_name": "rf-monotone-check",
            "data": {"source_table": "playground", "freq": "D", "horizon": 7},
            "models": ["random_forest@monotone"],
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
            },
            "backtest": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        panel,
        cfg,
        {"hike_20pct": {"price_index": "*1.20"}},
        price_col="price_index",
    )
    hike_sub = sim_df[sim_df["scenario"] == "hike_20pct"]
    assert not hike_sub.empty
    assert float(hike_sub["delta_yhat"].max()) <= 1e-6


def test_hpo_propagates_monotone_variant_constraints_and_audit_monotonicity() -> None:
    from scale_forecasting.hpo import tune_model
    from scale_forecasting.sdk import Forecaster

    panel = _make_confounded_price_panel(n_series=2, n_days=100)
    # Note: top-level monotonic_constraints is empty; @monotone variant infers constraints.
    cfg = RunConfig.model_validate(
        {
            "run_name": "hpo-monotone-audit",
            "data": {"source_table": "playground", "freq": "D", "horizon": 7},
            "models": ["regression_lags", "regression_lags@monotone"],
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7},
            "hpo": {"enabled": True, "n_trials": 2},
        }
    )
    sample = [grp for _, grp in panel.groupby("ts_id")]
    best_params = tune_model("regression_lags@monotone", sample, cfg)
    assert "alpha" in best_params

    f = Forecaster(cfg)
    shocks = {
        "hike_20pct": {"price_index": "*1.20"},
        "cut_15pct": {"price_index": "*0.85"},
    }
    sim_df = f.simulate_scenarios(shocks, data=panel, price_col="price_index")
    audit_df = f.audit_monotonicity_df(sim_df, scenarios=shocks)
    assert not audit_df.empty
    mono_rows = audit_df[audit_df["model_type"] == "regression_lags@monotone"]
    assert (mono_rows["n_violations"] == 0).all()
    assert (mono_rows["status"] == "PASS").all()
    unconstrained_hike = audit_df[
        (audit_df["model_type"] == "regression_lags") & (audit_df["scenario"] == "hike_20pct")
    ]
    assert int(unconstrained_hike["n_violations"].iloc[0]) > 0
    assert str(unconstrained_hike["status"].iloc[0]) == "VIOLATION"
    assert audit_scenario_monotonicity(pd.DataFrame()).empty
