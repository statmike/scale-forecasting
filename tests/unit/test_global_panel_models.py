"""Global and hybrid multi-series panel forecasting across NeuralForecast and NeuralProphet.

Verifies that:
1. ``is_panel_model`` dispatches on ``model_params.<model>.training_mode`` in
   ``{"global", "hybrid"}``.
2. ``run_panel`` fits a single cross-series model per backtest fold and once on the full panel
   for ``tide``, ``tft``, ``tsmixer``, ``patchtst``, and ``neuralprophet``.
3. Three-tier covariates (``static_covariates``, ``future_covariates``, ``past_covariates``) flow
   cleanly into ``NeuralForecast`` (``futr_exog_list``, ``hist_exog_list``, ``stat_exog_list``)
   and ``NeuralProphet`` (``future_regressors``, ``lagged_regressors``).
4. ``ray_io.chunk_cells`` packs all series for a panel model into a single chunk so one Ray task
   receives the full panel, and ``spark_io.run_group`` routes multi-series frames to ``run_panel``.
5. ``dag.check_model_params`` enforces runtime and model capability boundaries for
   ``training_mode``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.dag import check_model_params
from scale_forecasting.engines.ray_io import chunk_cells
from scale_forecasting.engines.spark_io import _MODEL_COL, run_group
from scale_forecasting.errors import ConfigError
from scale_forecasting.worker import is_panel_model, run_panel

HORIZON = 7


def _panel_df(n: int = 120) -> pd.DataFrame:
    """Two-series synthetic panel with static, future, and past covariates."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frames: list[pd.DataFrame] = []
    for i, (ts_id, region, base) in enumerate([("s1", "US", 20.0), ("s2", "EU", 50.0)]):
        t = np.arange(n, dtype=float)
        promo = ((t % 14) == 0).astype(float)
        temp = 15.0 + 8.0 * np.sin(2.0 * np.pi * t / 30.0) + i * 3.0
        y = (
            base
            + 0.1 * t
            + 3.0 * np.sin(2.0 * np.pi * t / 7.0)
            + 4.0 * promo
            + 0.1 * temp
            + rng.normal(0.0, 0.4, size=n)
        )
        frames.append(
            pd.DataFrame(
                {
                    "ts_id": ts_id,
                    "ds": idx,
                    "y": y,
                    "region": region,
                    "promo_flag": promo,
                    "temperature": temp,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_is_panel_model_dispatch() -> None:
    cfg = RunConfig(
        run_name="panel dispatch",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide", "neuralprophet", "theta"],
        model_params={
            "tide": {"training_mode": "global", "max_steps": 5},
            "neuralprophet": {"training_mode": "hybrid", "epochs": 2},
        },
    )
    assert is_panel_model("tide", cfg) is True
    assert is_panel_model("neuralprophet", cfg) is True
    assert is_panel_model("theta", cfg) is False


@pytest.mark.parametrize("model_name", ["tide", "tft", "tsmixer", "patchtst"])
def test_neuralforecast_global_panel_with_three_tier_covariates(model_name: str) -> None:
    panel = _panel_df(120)
    cfg = RunConfig(
        run_name=f"global {model_name}",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        features={
            "static_covariates": ["region"],
            "future_covariates": ["promo_flag"],
            "past_covariates": ["temperature"],
        },
        models=[model_name],
        model_params={
            model_name: {
                "training_mode": "global",
                "max_steps": 5,
                "input_size": 14,
            }
        },
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
    )
    results = run_panel(panel, model_name, cfg)
    assert len(results) == 2
    by_id = {r.ts_id: r for r in results}
    assert set(by_id) == {"s1", "s2"}
    for ts_id, cell in by_id.items():
        assert cell.status == "ok", f"{model_name} ({ts_id}) failed: {cell.error}"
        assert cell.n_folds_achieved == 2
        assert len(cell.predictions) == HORIZON
        assert cell.predictions["yhat"].notna().all()
        assert np.isfinite(cell.metrics["mae"])


@pytest.mark.parametrize("mode", ["global", "hybrid"])
def test_neuralprophet_global_and_hybrid_panel(mode: str) -> None:
    panel = _panel_df(120)
    cfg = RunConfig(
        run_name=f"np {mode}",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        features={
            "future_covariates": ["promo_flag"],
            "past_covariates": ["temperature"],
        },
        models=["neuralprophet"],
        model_params={
            "neuralprophet": {
                "training_mode": mode,
                "epochs": 3,
                "n_lags": 7,
                "n_forecasts": HORIZON,
            }
        },
        backtest={
            "enabled": True,
            "n_folds": 1,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
    )
    results = run_panel(panel, "neuralprophet", cfg)
    assert len(results) == 2
    for cell in results:
        assert cell.status == "ok", f"neuralprophet ({mode}, {cell.ts_id}) failed: {cell.error}"
        assert cell.n_folds_achieved == 1
        assert len(cell.predictions) == HORIZON
        assert cell.predictions["yhat"].notna().all()


def test_ray_chunk_cells_packs_global_panel_into_single_chunk() -> None:
    panel = _panel_df(80)
    cfg = RunConfig(
        run_name="ray chunking",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide", "theta"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5}},
    )
    chunks = chunk_cells(panel, cfg, ["tide", "theta"], n_chunks=4)
    # tide (global) -> 1 chunk containing both s1 and s2; theta (local) -> sharded chunks
    tide_chunks = [c for c in chunks if (c[_MODEL_COL] == "tide").all()]
    assert len(tide_chunks) == 1
    assert set(tide_chunks[0]["ts_id"]) == {"s1", "s2"}


def test_spark_run_group_routes_panel_to_run_panel() -> None:
    panel = _panel_df(90)
    cfg = RunConfig(
        run_name="spark group panel",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5, "input_size": 14}},
        backtest={"enabled": False},
    )
    results, status = run_group(panel, cfg, ["tide"])
    assert len(results) == 2
    assert set(status["ts_id"]) == {"s1", "s2"}
    assert (status["status"] == "ok").all()
    for cell in results:
        assert len(cell.predictions) == HORIZON


def test_dag_check_model_params_validates_training_mode() -> None:
    # 1. Valid global and hybrid on Ray, Vertex CustomJob, GCE, and GKE
    for runtime in ("ray", "vertex", "gce", "gke"):
        cfg_ok = RunConfig(
            run_name="ok",
            python_runtime=runtime,
            data={"source_table": "t", "freq": "D", "horizon": HORIZON},
            models=["tide", "neuralprophet"],
            model_params={
                "tide": {"training_mode": "global"},
                "neuralprophet": {"training_mode": "hybrid"},
            },
        )
        check_model_params(cfg_ok)

    # 2. Global on Spark is rejected
    cfg_spark = RunConfig(
        run_name="bad spark",
        python_runtime="spark",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide"],
        model_params={"tide": {"training_mode": "global"}},
    )
    with pytest.raises(ConfigError, match="requires a Ray, Vertex, GCE, or GKE runtime"):
        check_model_params(cfg_spark)

    # 3. Global on local-only model is rejected
    cfg_local_only = RunConfig(
        run_name="bad local",
        python_runtime="ray",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["theta"],
        model_params={"theta": {"training_mode": "global"}},
    )
    with pytest.raises(ConfigError, match="training_mode='global' is not supported by 'theta'"):
        check_model_params(cfg_local_only)

    # 4. Hybrid on global-only model (tide) is rejected
    cfg_no_hybrid = RunConfig(
        run_name="bad hybrid",
        python_runtime="ray",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide"],
        model_params={"tide": {"training_mode": "hybrid"}},
    )
    with pytest.raises(ConfigError, match="training_mode='hybrid' is not supported by 'tide'"):
        check_model_params(cfg_no_hybrid)

    # 5. Unknown training_mode is rejected
    cfg_unknown = RunConfig(
        run_name="bad mode",
        python_runtime="ray",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=["tide"],
        model_params={"tide": {"training_mode": "federated"}},
    )
    with pytest.raises(ConfigError, match=r"supported modes: \['local', 'global'\]"):
        check_model_params(cfg_unknown)
