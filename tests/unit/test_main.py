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

# Model names by runtime: theta is a Python/Spark model; arima_plus / arima_plus_xreg / timesfm are
# the BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "arima_plus_xreg", "timesfm"]


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "main test",
        "data": {"source_table": "source_series", "horizon": 7, "series_limit": 5},
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


# --- _plan: the out-of-scope shapes it must reject -----------------------------


def test_plan_rejects_ray_when_python_models_present() -> None:
    with pytest.raises(ConfigError, match="ray"):
        main._plan(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))


def test_plan_rejects_multi_when_python_models_present() -> None:
    with pytest.raises(ConfigError, match="submit --engine multi"):
        main._plan(_cfg(models=[_SPARK, *_NATIVE], spark_method="multi"))


def test_plan_allows_ray_config_when_only_bigquery_models() -> None:
    # An all-native config never uses the Python runtime, so ray/multi don't apply — it must plan.
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


def test_dry_run_still_rejects_ray() -> None:
    # The plan (and its rejections) runs before the dry_run short-circuit, so bad shapes fail fast.
    with pytest.raises(ConfigError, match="ray"):
        main.run(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), dry_run=True)


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
                "data": {"source_table": "source_series", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config", str(path), "--dry-run"])
    assert seen == {"dry_run": True, "run_name": "cli main test"}
