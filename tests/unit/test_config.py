"""Tests for config loading, validation, normalization, and fanout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scale_forecasting.config import (
    LEARNED_STRATEGIES,
    RunConfig,
    estimate_fanout,
    load_config,
)
from scale_forecasting.errors import ConfigError


def _minimal_dict(**over: Any) -> dict[str, Any]:
    """A minimal valid config; override any top-level key via kwargs."""
    base: dict[str, Any] = {
        "run_name": "t",
        "data": {"source_table": "p.d.source_series_native"},
        "models": ["theta"],
    }
    base.update(over)
    return base


# --- valid round-trip ----------------------------------------------------------


def test_minimal_config_is_valid_and_frozen() -> None:
    cfg = RunConfig(**_minimal_dict())
    assert cfg.run_name == "t"
    assert cfg.data.horizon == 28  # default
    with pytest.raises(ValidationError):  # frozen model rejects mutation
        cfg.run_name = "x"  # type: ignore[misc]


def test_full_config_round_trips_from_design_example() -> None:
    # Mirrors the documented example config.
    cfg = RunConfig(
        **_minimal_dict(
            run_name="example_daily_v1",
            data={
                "source_table": "project.scale_forecasting.source_series_native",
                "freq": "D",
                "horizon": 28,
                "series_limit": 1000,
            },
            python_runtime="spark",
            models=["sarimax", "theta", "prophet", "xgboost", "arima_plus", "timesfm"],
            features={"holidays": ["US"], "transform": "log1p"},
            backtest={"enabled": True, "n_folds": 3, "decision_metric": "wape"},
            ensemble={"enabled": True, "strategies": ["median", "inverse_error", "nnls"]},
            compute={"use_gpu": False, "gpu_type": "T4", "gpu_fraction": "auto"},
        )
    )
    assert cfg.data.series_limit == 1000
    assert cfg.backtest.decision_metric == "wape"
    assert set(cfg.ensemble.strategies) == {"median", "inverse_error", "nnls"}


# --- invalid fields fail fast --------------------------------------------------


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"models": []}, "at least 1"),  # empty model list
        ({"data": {"source_table": "t", "horizon": 0}}, "horizon"),  # non-positive horizon
        ({"data": {"source_table": "t", "series_limit": 0}}, "series_limit"),  # non-positive limit
        ({"python_runtime": "dask"}, "python_runtime"),  # unknown runtime
        ({"models": ["a", "a"]}, "duplicate"),  # duplicate models
        ({"backtest": {"decision_metric": "r2"}}, "decision_metric"),  # unknown metric
        ({"features": {"transform": "sqrt"}}, "transform"),  # unknown transform
        ({"unknown_key": 1}, "unknown_key"),  # extra key forbidden
    ],
)
def test_invalid_config_raises(over: dict[str, Any], needle: str) -> None:
    with pytest.raises(Exception) as exc:
        RunConfig(**_minimal_dict(**over))
    assert needle.lower() in str(exc.value).lower()


def test_gpu_fraction_out_of_range_rejected() -> None:
    with pytest.raises(Exception) as exc:
        RunConfig(**_minimal_dict(compute={"gpu_fraction": 1.5}))
    assert "gpu_fraction" in str(exc.value)


def test_gpu_fraction_float_in_range_ok() -> None:
    cfg = RunConfig(**_minimal_dict(compute={"gpu_fraction": 0.25}))
    assert cfg.compute.gpu_fraction == 0.25


# --- ray autoscaling bounds ----------------------------------------------------


def test_ray_autoscale_defaults_on() -> None:
    cfg = RunConfig(**_minimal_dict(python_runtime="ray"))
    assert cfg.compute.ray_autoscale is True
    assert cfg.compute.ray_cpu_min_nodes == 1
    assert cfg.compute.ray_gpu_min_nodes == 1
    assert cfg.compute.ray_cpu_max_nodes is None  # defers to ray_max_nodes at plan time
    assert cfg.compute.ray_gpu_max_nodes is None


def test_ray_pool_min_above_explicit_max_rejected() -> None:
    with pytest.raises(Exception) as exc:
        RunConfig(**_minimal_dict(compute={"ray_cpu_min_nodes": 5, "ray_cpu_max_nodes": 4}))
    assert "ray_cpu_min_nodes" in str(exc.value)


def test_ray_pool_min_above_shared_max_rejected() -> None:
    # An unset per-pool max defers to ray_max_nodes, which the min must still respect.
    with pytest.raises(Exception) as exc:
        RunConfig(**_minimal_dict(compute={"ray_gpu_min_nodes": 10, "ray_max_nodes": 4}))
    assert "ray_gpu_min_nodes" in str(exc.value)


def test_ray_pool_min_equal_max_ok() -> None:
    cfg = RunConfig(**_minimal_dict(compute={"ray_cpu_min_nodes": 4, "ray_cpu_max_nodes": 4}))
    assert cfg.compute.ray_cpu_min_nodes == 4


# --- normalization -------------------------------------------------------------


def test_singular_strategy_shorthand_becomes_list() -> None:
    # Use a calculated strategy so the shorthand is what's under test, not the
    # learned-without-backtest drop rule.
    cfg = RunConfig(**_minimal_dict(ensemble={"enabled": True, "strategy": "median"}))
    assert cfg.ensemble.strategies == ["median"]


def test_learned_strategy_without_backtest_is_dropped_not_error() -> None:
    # Learned strategies need backtest ON; without it they're dropped
    # (logged), not fatal. Calculated strategies survive.
    cfg = RunConfig(
        **_minimal_dict(
            backtest={"enabled": False},
            ensemble={"enabled": True, "strategies": ["median", "nnls", "ridge"]},
        )
    )
    assert cfg.ensemble.strategies == ["median"]
    assert not any(s in LEARNED_STRATEGIES for s in cfg.ensemble.strategies)


def test_learned_strategy_with_backtest_survives() -> None:
    cfg = RunConfig(
        **_minimal_dict(
            backtest={"enabled": True},
            ensemble={"enabled": True, "strategies": ["median", "nnls"]},
        )
    )
    assert cfg.ensemble.strategies == ["median", "nnls"]


# --- HPO config ----------------------------------------------------------------


def test_hpo_defaults_are_off_and_fleetwide() -> None:
    cfg = RunConfig(**_minimal_dict())
    assert cfg.hpo.enabled is False
    assert cfg.hpo.granularity == "fleetwide"
    assert cfg.hpo.sample_size == 20
    assert cfg.hpo.n_trials == 20


def test_hpo_enabled_requires_backtest() -> None:
    # HPO tunes on the backtest folds, so enabling it with backtest off fails fast at load.
    with pytest.raises((ValidationError, ValueError), match="requires backtest"):
        RunConfig(**_minimal_dict(hpo={"enabled": True}, backtest={"enabled": False}))


def test_hpo_enabled_with_backtest_is_valid() -> None:
    cfg = RunConfig(
        **_minimal_dict(
            hpo={"enabled": True, "granularity": "per_series", "n_trials": 10, "sample_size": 5},
            backtest={"enabled": True},
        )
    )
    assert cfg.hpo.enabled is True
    assert cfg.hpo.granularity == "per_series"


def test_hpo_rejects_unknown_granularity() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**_minimal_dict(hpo={"enabled": True, "granularity": "nonsense"}))


# --- fanout --------------------------------------------------------------------


def test_fanout_with_series_limit_and_backtest() -> None:
    cfg = RunConfig(
        **_minimal_dict(
            data={"source_table": "t", "series_limit": 100},
            models=["theta", "sarimax"],
            backtest={"enabled": True, "n_folds": 3},
        )
    )
    fo = estimate_fanout(cfg)
    assert (fo.n_series, fo.n_models, fo.n_folds, fo.n_cells) == (100, 2, 3, 600)


def test_fanout_no_backtest_uses_one_fold() -> None:
    cfg = RunConfig(**_minimal_dict(data={"source_table": "t", "series_limit": 10}))
    fo = estimate_fanout(cfg)
    assert fo.n_folds == 1 and fo.n_cells == 10


def test_fanout_unlimited_series_is_none() -> None:
    cfg = RunConfig(**_minimal_dict())  # no series_limit
    fo = estimate_fanout(cfg)
    assert fo.n_series is None and fo.n_cells is None


# --- load_config ---------------------------------------------------------------


def test_load_config_reads_valid_file(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(_minimal_dict(run_name="from_file")))
    cfg = load_config(p)
    assert cfg.run_name == "from_file"


def test_load_config_missing_file_raises_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.json")
    assert "cannot read" in str(exc.value)


def test_load_config_bad_json_raises_configerror(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError) as exc:
        load_config(p)
    assert "not valid json" in str(exc.value).lower()


def test_load_config_invalid_schema_raises_configerror(tmp_path: Path) -> None:
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps({"run_name": "x"}))  # missing data + models
    with pytest.raises(ConfigError) as exc:
        load_config(p)
    assert "invalid config" in str(exc.value).lower()
