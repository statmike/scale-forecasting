"""Offline tests for the run orchestrator (BUILD Arc B, ``scale_forecasting.main``).

No GCP: the pure plan (:func:`main._plan`) — run_id parity, the per-runtime model split, and the
ray/multi rejections — plus the ``dry_run`` path and the CLI's dispatch of it. The live parallel
launch (Spark batch + BigQuery engine under one run_id) is the ``@gcp`` smoke in
``tests/integration/test_main_orchestration_smoke.py``; here the GCP seams are never reached because
``dry_run`` returns before them and the rejection tests raise first.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import main
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

# Model names by runtime: theta is a Python/Spark model; arima_plus / arima_plus_xreg / timesfm are
# the BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "arima_plus_xreg", "timesfm"]

# A resolved Settings for the dispatch tests (never used to touch GCP — the submit fns are faked).
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "main test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_SPARK, *_NATIVE],
        "features": {"exog": ["price_index"]},
    }
    base.update(over)
    return RunConfig(**base)


# --- _plan: run_id parity + the per-runtime split ------------------------------


def test_plan_run_id_matches_full_config_digest() -> None:
    # Both engines must derive the same id, so the plan's run_id is the digest over the WHOLE cfg
    # (incl. every model), not over either executed subset.
    cfg = _cfg()
    assert main._plan(cfg).run_id == make_run_id(cfg)


def test_plan_splits_models_by_runtime() -> None:
    plan = main._plan(_cfg())
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE
    assert plan.spark_method == "explode"  # normalized default


def test_plan_all_bigquery_has_no_python_models() -> None:
    plan = main._plan(_cfg(models=_NATIVE))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


def test_plan_all_python_has_no_bq_models() -> None:
    plan = main._plan(_cfg(models=[_SPARK, "holtwinters"]))
    assert plan.python_models == [_SPARK, "holtwinters"]
    assert plan.bq_models == []


# --- _plan: ray is accepted; the out-of-scope multi shape is rejected ----------


def test_plan_accepts_ray_when_python_models_present() -> None:
    # B4: the Ray engine is built, so main.run now dispatches ray — _plan must NOT reject it.
    plan = main._plan(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE


def test_plan_rejects_multi_when_python_models_present() -> None:
    with pytest.raises(ConfigError, match="submit --engine multi"):
        main._plan(_cfg(models=[_SPARK, *_NATIVE], spark_method="multi"))


def test_ray_runtime_cannot_carry_spark_method() -> None:
    # multi is a Spark-only method, so main._plan's multi guard is gated on python_runtime="spark".
    # It never has to fire for a ray config because the config layer forbids ray + any spark_method
    # outright — so a ray config that names one fails to construct, well before _plan sees it.
    with pytest.raises(ValueError, match="spark_method is only valid"):
        _cfg(models=[_SPARK], python_runtime="ray", spark_method="multi")


def test_plan_allows_ray_config_when_only_bigquery_models() -> None:
    # An all-native config never uses the Python runtime, so runtime choice doesn't apply.
    plan = main._plan(_cfg(models=_NATIVE, python_runtime="ray"))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


# --- run(dry_run=True): offline, no GCP ----------------------------------------


def test_dry_run_returns_run_id_and_estimates_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    # dry_run must return the shared run_id and call estimate_fanout, without touching any GCP seam.
    import scale_forecasting.config as config_mod

    called: dict[str, Any] = {}
    real_estimate = config_mod.estimate_fanout

    def _spy(cfg: RunConfig) -> Any:
        called["cfg"] = cfg
        return real_estimate(cfg)

    monkeypatch.setattr(config_mod, "estimate_fanout", _spy)

    cfg = _cfg()
    run_id = main.run(cfg, dry_run=True)
    assert run_id == make_run_id(cfg)
    assert called["cfg"] is cfg


def test_dry_run_still_rejects_multi() -> None:
    # The plan (and its rejection) runs before the dry_run short-circuit, so bad shapes fail fast.
    with pytest.raises(ConfigError, match="submit --engine multi"):
        main.run(_cfg(models=[_SPARK, *_NATIVE], spark_method="multi"), dry_run=True)


def test_dry_run_allows_ray() -> None:
    # Ray is a supported runtime now; a ray config plans + dry-runs like any other.
    run_id = main.run(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), dry_run=True)
    assert run_id == make_run_id(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))


# --- _launch_python_runtime: dispatch by python_runtime ------------------------


def test_launch_python_runtime_dispatches_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_batch(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        seen["cfg"] = cfg
        return "batch-1"

    monkeypatch.setattr(submit_mod, "submit_batch", _fake_submit_batch)

    cfg = _cfg(models=[_SPARK], python_runtime="spark")
    plan = main._plan(cfg)
    main._launch_python_runtime(cfg, plan, _SETTINGS)
    assert seen["engine"] == "explode"
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False


def test_launch_python_runtime_dispatches_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_ray(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        seen["cfg"] = cfg
        return "job-1"

    monkeypatch.setattr(ray_submit_mod, "submit_ray", _fake_submit_ray)

    cfg = _cfg(models=[_SPARK], python_runtime="ray")
    plan = main._plan(cfg)
    main._launch_python_runtime(cfg, plan, _SETTINGS)
    assert "engine" not in seen  # ray takes no spark engine arg
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False


# --- run(): ensemble orchestration after the engine join -----------------------


def _patch_run_seams(
    monkeypatch: pytest.MonkeyPatch, *, bq_error: Exception | None = None
) -> dict[str, Any]:
    """Fake every GCP seam main.run touches so the ensemble gating is exercised offline.

    Records what happened in the returned dict: header status finalized, whether the BigQuery engine
    and the Spark launch ran. The Python-runtime launch is faked to a no-op success. ``bq_error``
    makes the BigQuery engine raise, to prove ensembles are skipped when an engine fails.
    """
    import scale_forecasting.ensemble_run as ensemble_mod
    from scale_forecasting.engines import bigquery_engine
    from scale_forecasting.registry import bq

    seen: dict[str, Any] = {"ensemble_called": False}

    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: _SETTINGS))
    monkeypatch.setattr(bq, "ensure_tables", lambda *a, **k: None)
    monkeypatch.setattr(bq, "write_header", lambda *a, **k: None)

    def _fake_update(run_id: str, *, settings: Any = None, **fields: Any) -> None:
        seen["status"] = fields.get("status")

    monkeypatch.setattr(bq, "update_header", _fake_update)

    def _fake_bq_run(cfg: RunConfig, models: list[str], **kw: Any) -> Any:
        seen["bq_ran"] = True
        if bq_error is not None:
            raise bq_error
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=models)

    monkeypatch.setattr(bigquery_engine, "run", _fake_bq_run)
    monkeypatch.setattr(
        main, "_launch_python_runtime", lambda *a, **k: seen.__setitem__("spark_ran", True)
    )

    def _fake_ensembles(cfg: RunConfig, run_id: str, *, settings: Any) -> None:
        seen["ensemble_called"] = True
        seen["ensemble_run_id"] = run_id

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _fake_ensembles)
    return seen


def test_run_invokes_ensembles_when_enabled_and_engines_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_run_seams(monkeypatch)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    run_id = main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["ensemble_run_id"] == run_id
    assert seen["status"] == "COMPLETED"


def test_run_skips_ensembles_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_run_seams(monkeypatch)
    main.run(_cfg())  # ensemble.enabled defaults False
    assert seen["ensemble_called"] is False
    assert seen["status"] == "COMPLETED"


def test_run_skips_ensembles_when_engine_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A BigQuery engine failure must skip ensembles (they'd read incomplete predictions) and the
    # header finalizes FAILED.
    seen = _patch_run_seams(monkeypatch, bq_error=RuntimeError("bq boom"))
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    with pytest.raises(RuntimeError, match="bq boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is False
    assert seen["status"] == "FAILED"


def test_run_ensemble_failure_finalizes_header_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ensemble_run as ensemble_mod

    seen = _patch_run_seams(monkeypatch)

    def _boom(cfg: RunConfig, run_id: str, *, settings: Any) -> None:
        seen["ensemble_called"] = True
        raise RuntimeError("ensemble boom")

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _boom)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    with pytest.raises(RuntimeError, match="ensemble boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["status"] == "FAILED"


# --- CLI: dispatches dry_run ---------------------------------------------------


def test_cli_dispatches_dry_run(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    seen: dict[str, Any] = {}

    def _fake_run(cfg: RunConfig, *, dry_run: bool = False) -> str:
        seen["dry_run"] = dry_run
        seen["run_name"] = cfg.run_name
        return "rid-123"

    monkeypatch.setattr(main, "run", _fake_run)

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli main test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config", str(path), "--dry-run"])
    assert seen == {"dry_run": True, "run_name": "cli main test"}
