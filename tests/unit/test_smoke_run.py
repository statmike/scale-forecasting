"""Offline tests for the smoke-forecast launcher (Arc B, ``smoke_run``).

No Spark, no GCP: the launcher parses ``--config-uri`` + the ``--sf-*`` infra args (exporting them
to ``os.environ`` so env-based ``Settings`` resolves), loads a local config, and calls
:func:`scale_forecasting.main.run` with the batch's :class:`SparkSession` injected. Here both the
Spark session builder and ``main.run`` are monkeypatched, so the actual pyspark / GCP path (covered
by the live smoke) never runs — we assert the config round-trips and the injected-session call shape
is what ``main.run`` expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting import smoke_run
from scale_forecasting._infra_args import INFRA_ARG_ENV

_CONFIG: dict[str, Any] = {
    "run_name": "smoke run test",
    "python_runtime": "spark",
    "spark_method": "explode",
    "data": {"source_table": "source_series_native", "horizon": 14, "series_limit": 20},
    "models": ["theta", "holtwinters", "arima_plus"],
    "features": {"holidays": ["US"]},
}


def _write_config(tmp_path: Path) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_CONFIG))
    return str(path)


class _FakeSession:
    """Stand-in for a SparkSession — records that stop() was called."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _patch_spark(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    """Point ``SparkSession.builder...getOrCreate()`` at ``session`` (no real Spark)."""
    import pyspark.sql as pyspark_sql

    class _Builder:
        def appName(self, _name: str) -> _Builder:  # noqa: N802 - mirror Spark's API
            return self

        def getOrCreate(self) -> _FakeSession:  # noqa: N802 - mirror Spark's API
            return session

    monkeypatch.setattr(pyspark_sql.SparkSession, "builder", _Builder())


def test_parse_args_exports_infra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The batch delivers SF_* as --sf-* args; parsing them must populate os.environ so the existing
    # env-based Settings.resolve() seam works unchanged (Dataproc rejects driver-env).
    for _, env_name in INFRA_ARG_ENV:
        monkeypatch.delenv(env_name, raising=False)
    smoke_run._parse_args(
        [
            "--config-uri",
            "gs://bkt/smoke/run_config.json",
            "--sf-project-id",
            "proj-x",
            "--sf-connection",
            "proj-x.us-central1.conn",
            "--sf-warehouse-uri",
            "gs://bkt/warehouse",
            "--sf-dataset-id",
            "ds_x",
            "--sf-region",
            "us-central1",
        ]
    )
    import os

    assert os.environ["SF_PROJECT_ID"] == "proj-x"
    assert os.environ["SF_CONNECTION"] == "proj-x.us-central1.conn"
    assert os.environ["SF_WAREHOUSE_URI"] == "gs://bkt/warehouse"
    assert os.environ["SF_DATASET_ID"] == "ds_x"
    assert os.environ["SF_REGION"] == "us-central1"


def test_main_loads_config_and_injects_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole contract: load the staged config and call main.run with the batch's session injected
    # (the injectable-session seam — Spark in-process ∥ BigQuery inline, one run_id).
    import scale_forecasting.main as main_mod

    captured: dict[str, Any] = {}
    session = _FakeSession()
    _patch_spark(monkeypatch, session)

    def _fake_run(cfg: Any, *, dry_run: bool = False, spark: Any = None) -> str:
        captured["cfg"] = cfg
        captured["spark"] = spark
        return "rid-smoke"

    monkeypatch.setattr(main_mod, "run", _fake_run)

    smoke_run.main(["--config-uri", _write_config(tmp_path)])

    assert captured["cfg"].run_name == "smoke run test"
    assert captured["cfg"].models == ["theta", "holtwinters", "arima_plus"]
    assert captured["spark"] is session  # the injected session, not a remote batch
    assert session.stopped is True  # stopped in the finally


def test_main_stops_session_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A forecast failure must still stop the session (finally). The batch goes FAILED and the
    # Terraform layer's on_failure=continue keeps the apply tolerant — that tolerance is not here.
    import scale_forecasting.main as main_mod

    session = _FakeSession()
    _patch_spark(monkeypatch, session)

    def _boom(cfg: Any, *, dry_run: bool = False, spark: Any = None) -> str:
        raise RuntimeError("forecast boom")

    monkeypatch.setattr(main_mod, "run", _boom)

    with pytest.raises(RuntimeError, match="forecast boom"):
        smoke_run.main(["--config-uri", _write_config(tmp_path)])
    assert session.stopped is True
