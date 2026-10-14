"""Unit tests for open-weight foundation models (chronos_bolt, chronos2, timesfm_torch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import BacktestConfig, DataConfig, FeaturesConfig, RunConfig
from scale_forecasting.errors import ConfigError, ModelError
from scale_forecasting.features import resolve_model_variant
from scale_forecasting.models import base_model_name, get_model, list_models
from scale_forecasting.models._foundation_base import (
    DECILE_GRID,
    FoundationSeriesInput,
    clear_foundation_cache,
    encode_static_carriers,
    get_or_load_foundation_pipeline,
    map_deciles_to_quantiles,
)
from scale_forecasting.models.base_model import ModelContext
from scale_forecasting.models.chronos2 import Chronos2Model
from scale_forecasting.models.chronos_bolt import ChronosBoltModel
from scale_forecasting.models.timesfm_torch import TimesFMTorchModel, _fit_predict_ridge_xreg
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.worker import run_cell, run_panel


def test_foundation_models_registered_and_contract_flags() -> None:
    models = list_models()
    assert len(models) == 37
    for name in ("chronos_bolt", "chronos2", "timesfm_torch"):
        assert name in models
        cls = get_model(name)
        assert cls.family == "deep_learning"
        assert cls.runtime == "python"
        assert cls.optional_extra == "models-foundation"
        assert cls.gpu_capable is True
        assert cls.supports_global is True
        assert cls.supports_hybrid is False
        assert cls.supports_recondition is True
        assert cls.supports_extrapolate is True
        assert cls.supports_native_intervals is True
        assert cls.gpu_useful({}) is True
        inst = cls({}, ModelContext(freq="D", horizon=7))
        assert inst.serialize() is None

    assert ChronosBoltModel.supports_exog is False
    assert ChronosBoltModel.supports_future_covariates is False
    assert ChronosBoltModel.supports_past_covariates is False
    assert ChronosBoltModel.supports_static_covariates is False

    for cov_cls in (Chronos2Model, TimesFMTorchModel):
        assert cov_cls.supports_exog is True
        assert cov_cls.supports_future_covariates is True
        assert cov_cls.supports_past_covariates is True
        assert cov_cls.supports_static_covariates is True


def test_bq_timesfm_alias_normalization_and_run_id_parity() -> None:
    assert base_model_name("bq_timesfm") == "timesfm"
    assert base_model_name("bq_timesfm@fast") == "timesfm"
    assert get_model("bq_timesfm").name == "timesfm"

    cfg_canonical = RunConfig(
        run_name="alias-check",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["timesfm"],
        model_params={"timesfm": {"horizon": 7}},
    )
    cfg_alias = RunConfig(
        run_name="alias-check",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["bq_timesfm"],
        model_params={"bq_timesfm": {"horizon": 7}},
    )
    assert cfg_alias.models == ["timesfm"]
    assert cfg_alias.model_params == {"timesfm": {"horizon": 7}}
    assert make_run_id(cfg_canonical) == make_run_id(cfg_alias)

    # Tagged alias normalization and model_params merging.
    cfg_tagged = RunConfig(
        run_name="alias-tagged",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["bq_timesfm", "bq_timesfm@v1"],
        model_params={
            "bq_timesfm": {"a": 1, "b": 2},
            "timesfm": {"b": 99},
            "bq_timesfm@v1": {"c": 3},
        },
    )
    assert cfg_tagged.models == ["timesfm", "timesfm@v1"]
    assert cfg_tagged.model_params["timesfm"] == {"a": 1, "b": 99}
    assert cfg_tagged.model_params["timesfm@v1"] == {"c": 3}

    with pytest.raises(ValueError, match="contains duplicates after normalization"):
        RunConfig(
            run_name="alias-dup",
            data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
            models=["bq_timesfm", "timesfm"],
        )


def test_foundation_variant_tags_in_execution_plan() -> None:
    cfg = RunConfig(
        run_name="tag-test",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=[
            "chronos_bolt@tiny",
            "chronos_bolt@small",
            "timesfm_torch@v2.5",
            "timesfm_torch@v3.0",
            "chronos2@global",
        ],
        features=FeaturesConfig(
            future_covariates=["promo"],
            past_covariates=["temp"],
            static_covariates=["region"],
            on_unsupported_covariates="fallback",
        ),
    )
    plan_tiny = resolve_model_variant(cfg, "chronos_bolt@tiny")
    assert plan_tiny.base_model == "chronos_bolt"
    assert plan_tiny.clean_params["size"] == "tiny"
    assert plan_tiny.training_mode == "local"

    plan_small = resolve_model_variant(cfg, "chronos_bolt@small")
    assert plan_small.clean_params["size"] == "small"

    plan_tfm25 = resolve_model_variant(cfg, "timesfm_torch@v2.5")
    assert plan_tfm25.base_model == "timesfm_torch"
    assert plan_tfm25.clean_params["version"] == "2.5"
    assert set(plan_tfm25.covariates_used) == {"promo", "temp", "region"}

    plan_tfm30 = resolve_model_variant(cfg, "timesfm_torch@v3.0")
    assert plan_tfm30.clean_params["version"] == "3.0"

    plan_c2 = resolve_model_variant(cfg, "chronos2@global")
    assert plan_c2.training_mode == "global"
    assert set(plan_c2.covariates_used) == {"promo", "temp", "region"}


def test_foundation_validate_params() -> None:
    ChronosBoltModel.validate_params(
        {"size": "tiny", "context_length": 128, "batch_size": 16}, max_horizon=7
    )
    with pytest.raises(ConfigError, match="size"):
        ChronosBoltModel.validate_params({"size": "xxl"}, max_horizon=7)
    with pytest.raises(ConfigError, match="context_length"):
        ChronosBoltModel.validate_params({"context_length": 0}, max_horizon=7)
    with pytest.raises(ConfigError, match="batch_size"):
        ChronosBoltModel.validate_params({"batch_size": 0}, max_horizon=7)
    with pytest.raises(ConfigError, match="quantiles"):
        ChronosBoltModel.validate_params({"quantiles": [0.0, 1.0]}, max_horizon=7)

    TimesFMTorchModel.validate_params({"version": "3.0"}, max_horizon=7)
    TimesFMTorchModel.validate_params({"version": "2.5"}, max_horizon=7)
    with pytest.raises(ConfigError, match="version"):
        TimesFMTorchModel.validate_params({"version": "1.0"}, max_horizon=7)


def test_encode_static_carriers_preserves_distinction_under_instance_norm() -> None:
    assert encode_static_carriers(None, hist_len=20, futr_len=5) == {}
    c_east = encode_static_carriers(
        {"region": "East", "tier": 1, "active": True},
        hist_len=30,
        futr_len=7,
    )
    c_west = encode_static_carriers(
        {"region": "West", "tier": 2, "active": False},
        hist_len=30,
        futr_len=7,
    )
    assert set(c_east.keys()) == {"stat__active", "stat__region", "stat__tier"}
    for key in c_east:
        p_e, f_e = c_east[key]
        p_w, _f_w = c_west[key]
        assert len(p_e) == 30 and len(f_e) == 7
        # Verify non-zero variance so RevIN / InstanceNorm never divides by zero.
        assert float(np.std(p_e)) > 0.1
        assert float(np.std(p_w)) > 0.1
        # Verify distinct trajectories after per-series z-score normalization.
        norm_e = (p_e - np.mean(p_e)) / np.std(p_e)
        norm_w = (p_w - np.mean(p_w)) / np.std(p_w)
        assert not np.allclose(norm_e, norm_w)


def test_map_deciles_to_quantiles_interpolation_and_tails() -> None:
    horizon = 5
    # Construct deciles [10, 20, ..., 90] at each horizon step.
    deciles = np.tile(np.linspace(10.0, 90.0, 9), (horizon, 1))
    qs = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    mapped = map_deciles_to_quantiles(deciles, qs)

    np.testing.assert_allclose(mapped[0.1], 10.0)
    np.testing.assert_allclose(mapped[0.25], 25.0)
    np.testing.assert_allclose(mapped[0.5], 50.0)
    np.testing.assert_allclose(mapped[0.75], 75.0)
    np.testing.assert_allclose(mapped[0.9], 90.0)
    assert (mapped[0.025] < mapped[0.1]).all()
    assert (mapped[0.975] > mapped[0.9]).all()

    with pytest.raises(ModelError, match="expected decile matrix"):
        map_deciles_to_quantiles(np.zeros((horizon, 5)), (0.1, 0.9))


def test_fit_predict_ridge_xreg() -> None:
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(60, 2))
    X_te = rng.normal(size=(7, 2))
    y = 3.0 * X_tr[:, 0] - 2.0 * X_tr[:, 1] + 10.0
    hat_tr, hat_te = _fit_predict_ridge_xreg(y, X_tr, X_te, ridge=1e-3)
    expected_te = 3.0 * X_te[:, 0] - 2.0 * X_te[:, 1] + 10.0
    np.testing.assert_allclose(hat_tr, y, atol=0.05)
    np.testing.assert_allclose(hat_te, expected_te, atol=0.05)

    # Degenerate short series returns zeros cleanly.
    z_tr, z_te = _fit_predict_ridge_xreg(np.array([1.0]), np.ones((1, 1)), np.ones((3, 1)))
    assert np.all(z_tr == 0.0) and np.all(z_te == 0.0)


def test_foundation_lru_cache_and_oom_recovery() -> None:
    clear_foundation_cache()
    calls: list[str] = []

    def make_loader(label: str) -> Any:
        def _load() -> dict[str, str]:
            calls.append(label)
            return {"model": label}

        return _load

    a1 = get_or_load_foundation_pipeline(("k1",), make_loader("m1"), max_entries=2)
    a2 = get_or_load_foundation_pipeline(("k1",), make_loader("m1"), max_entries=2)
    assert a1 is a2
    assert calls == ["m1"]

    get_or_load_foundation_pipeline(("k2",), make_loader("m2"), max_entries=2)
    # Touch k1 so k2 becomes least-recently-used.
    get_or_load_foundation_pipeline(("k1",), make_loader("m1"), max_entries=2)
    # Loading k3 with max_entries=2 evicts k2, keeping k1 cached.
    get_or_load_foundation_pipeline(("k3",), make_loader("m3"), max_entries=2)
    get_or_load_foundation_pipeline(("k1",), make_loader("m1"), max_entries=2)
    assert calls == ["m1", "m2", "m3"]

    # Simulated CUDA OOM clears cache and retries once.
    attempts = 0

    def oom_once() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("CUDA out of memory")
        return "recovered"

    assert get_or_load_foundation_pipeline(("k_oom",), oom_once, max_entries=2) == "recovered"
    assert attempts == 2
    clear_foundation_cache()


def test_foundation_recondition_and_advance_origin_with_stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fit, predict, recondition, and advance_origin mechanics with a fast stub pipeline."""
    clear_foundation_cache()
    try:
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        y = pd.Series(np.linspace(100.0, 139.0, 40), index=idx, name="y")
        X = pd.DataFrame(
            {"promo": np.arange(40, dtype=float), "temp": np.sin(np.arange(40, dtype=float))},
            index=idx,
        )
        ctx = ModelContext(
            freq="D",
            horizon=5,
            past_covariates=("temp",),
            static_covariates={"store_type": "flagship"},
        )
        model = Chronos2Model({}, ctx)
        monkeypatch.setattr(model, "_load_pipeline", lambda repo_id, dev: "stub_pipeline")

        seen_items: list[FoundationSeriesInput] = []

        def fake_predict_batch(
            pipeline: Any,
            items: list[FoundationSeriesInput],
            steps: int,
            *,
            is_panel: bool,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            seen_items.extend(items)
            out = []
            for it in items:
                last_v = float(it.y[-1])
                pt = np.full(steps, last_v + 1.0)
                dec = np.tile(np.linspace(last_v - 4.0, last_v + 6.0, len(DECILE_GRID)), (steps, 1))
                out.append((pt, dec))
            return out

        monkeypatch.setattr(model, "_predict_batch_raw", fake_predict_batch)
        model.fit(y, X)
        fut_idx = pd.date_range("2025-02-10", periods=5, freq="D")
        X_fut = pd.DataFrame({"promo": np.ones(5)}, index=fut_idx)
        pred1 = model.predict(5, X_fut)
        assert len(pred1) == 5
        assert pred1["yhat"].iloc[0] == pytest.approx(140.0)
        assert model.device_used() == "cpu"

        # 1. Recondition with 2 new observations and verify context updates without refitting.
        new_idx = pd.date_range("2025-02-10", periods=2, freq="D")
        y_new = pd.Series([200.0, 210.0], index=new_idx, name="y")
        X_new = pd.DataFrame({"promo": [1.0, 1.0], "temp": [0.5, 0.6]}, index=new_idx)
        model.recondition(y_new, X_new)
        fut_idx2 = pd.date_range("2025-02-12", periods=5, freq="D")
        X_fut2 = pd.DataFrame({"promo": np.ones(5)}, index=fut_idx2)
        pred2 = model.predict(5, X_fut2)
        assert pred2["yhat"].iloc[0] == pytest.approx(211.0)
        assert pred2["ds"].iloc[0] == pd.Timestamp("2025-02-12")

        # 2. Advance origin blind by 3 steps and verify forecast dates advance by 3 days.
        gap_idx = pd.date_range("2025-02-12", periods=3, freq="D")
        X_gap = pd.DataFrame({"promo": np.ones(3)}, index=gap_idx)
        model.advance_origin(3, X_gap)
        fut_idx3 = pd.date_range("2025-02-15", periods=5, freq="D")
        X_fut3 = pd.DataFrame({"promo": np.ones(5)}, index=fut_idx3)
        pred3 = model.predict(5, X_fut3)
        assert len(pred3) == 5
        assert pred3["ds"].iloc[0] == pd.Timestamp("2025-02-15")
    finally:
        clear_foundation_cache()


def test_real_weight_foundation_local_and_global_panel() -> None:
    """End-to-end verification with real checkpoints across chronos_bolt, chronos2, timesfm."""
    idx = pd.date_range("2025-01-01", periods=64, freq="D")
    rows = []
    for s_idx, uid in enumerate(("series_0", "series_1")):
        base = 50.0 + 20.0 * s_idx + 5.0 * np.sin(np.arange(64) * 2.0 * np.pi / 7.0)
        for t_i, ts in enumerate(idx):
            rows.append(
                {
                    "ts_id": uid,
                    "ds": ts,
                    "y": float(base[t_i] + 0.2 * t_i),
                    "promo": float(t_i % 7 == 0),
                    "temp": float(np.cos(t_i / 10.0)),
                    "region": "North" if s_idx == 0 else "South",
                }
            )
    panel = pd.DataFrame(rows)
    s0 = panel[panel["ts_id"] == "series_0"].reset_index(drop=True)

    cfg = RunConfig(
        run_name="foundation-unit",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=[
            "chronos_bolt@tiny",
            "chronos2@global",
            "timesfm_torch@global",
            "timesfm_torch@v2.5",
        ],
        features=FeaturesConfig(
            future_covariates=["promo"],
            past_covariates=["temp"],
            static_covariates=["region"],
            on_unsupported_covariates="fallback",
        ),
        backtest=BacktestConfig(enabled=True, n_folds=1, horizon=7, step=7, min_train=40),
        model_params={"timesfm_torch@v2.5": {"training_mode": "global"}},
    )

    c_bolt = run_cell(s0, "chronos_bolt@tiny", cfg)
    assert c_bolt.status == "ok", c_bolt.error
    assert len(c_bolt.predictions) == 7
    assert c_bolt.diagnostics["size"] == "tiny"

    p_c2 = run_panel(panel, "chronos2@global", cfg)
    assert all(r.status == "ok" for r in p_c2)
    assert p_c2[0].diagnostics["cross_learning"] is True

    p_tfm3 = run_panel(panel, "timesfm_torch@global", cfg)
    assert all(r.status == "ok" for r in p_tfm3)
    assert p_tfm3[0].diagnostics["version"] == "3.0"

    p_tfm25 = run_panel(panel, "timesfm_torch@v2.5", cfg)
    assert all(r.status == "ok" for r in p_tfm25)
    assert p_tfm25[0].diagnostics["version"] == "2.5"


def test_foundation_models_shared_vm_default_and_explicit_worker_scaling() -> None:
    """Zero-shot foundation models default to 1 shared VM unless workers > 1 or paired with DL."""
    from scale_forecasting.engines.vertex_engine import effective_worker_count

    cfg_fnd_local = RunConfig(
        run_name="fnd-local",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["chronos_bolt", "chronos2", "timesfm_torch"],
    )
    assert effective_worker_count(cfg_fnd_local, cfg_fnd_local.models, 1, n_series=10) == 1
    assert effective_worker_count(cfg_fnd_local, cfg_fnd_local.models, 3, n_series=10) == 3

    cfg_fnd_global = RunConfig(
        run_name="fnd-global",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["chronos2@global", "timesfm_torch@global"],
    )
    assert effective_worker_count(cfg_fnd_global, cfg_fnd_global.models, 1, n_series=10) == 1
    assert effective_worker_count(cfg_fnd_global, cfg_fnd_global.models, 2, n_series=10) == 2

    cfg_mixed_dl = RunConfig(
        run_name="mixed-dl",
        data=DataConfig(source_table="p.d.t", freq="D", horizon=7),
        models=["tide", "chronos2"],
    )
    assert effective_worker_count(cfg_mixed_dl, cfg_mixed_dl.models, 1, n_series=10) == 2


def test_foundation_runtime_guards_hf_cache_fallback_and_triton_deregister(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Verify unwritable HF_HOME fallback and no-compiler Triton override deregistration."""
    import sys
    import types

    from scale_forecasting.models import _foundation_base as fb

    fake_hf_constants = types.SimpleNamespace(
        HF_HOME="/unwritable",
        HF_HUB_CACHE="/unwritable/hub",
        HUGGINGFACE_HUB_CACHE="/unwritable/hub",
        HF_ASSETS_CACHE="/unwritable/assets",
        HUGGINGFACE_ASSETS_CACHE="/unwritable/assets",
        HF_XET_CACHE="/unwritable/xet",
    )
    fake_tf_hub = types.SimpleNamespace(HF_MODULES_CACHE="/unwritable/modules")
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_hf_constants)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", fake_tf_hub)
    monkeypatch.setenv("HF_HOME", "/proc/nonexistent_unwritable_dir")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    resolved = fb._ensure_writable_hf_cache()
    assert resolved == str(tmp_path / "scale_forecasting_hf_cache")
    assert fake_hf_constants.__dict__["HF_HUB_CACHE"] == str(
        tmp_path / "scale_forecasting_hf_cache" / "hub"
    )
    assert fake_tf_hub.__dict__["HF_MODULES_CACHE"] == str(
        tmp_path / "scale_forecasting_hf_cache" / "modules"
    )

    with fb._checkpoint_load_lock(resolved):
        assert (tmp_path / "scale_forecasting_hf_cache" / ".scale_forecasting_load.lock").exists()

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    assert fb._has_c_compiler() is False
    fb._ensure_torch_runtime_ready()
    assert fb.os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1"
