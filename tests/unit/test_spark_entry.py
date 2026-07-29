"""Offline tests for the Spark launcher dispatch (BUILD B2 / Arc B, ``spark_entry``).

No Spark, no GCP: the launcher parses ``--engine`` + ``--config-uri`` + the Arc B ``--models`` /
``--manage-header`` flags, loads a local config, and forwards to the engine's
``run(cfg, models=..., manage_header=...)``. The engine module's ``run`` is monkeypatched to capture
the forwarded call, so the actual pyspark path (covered by ``@spark``/``@gcp``) never runs here. The
CSV parser (:func:`_parse_models`) is exercised directly as the pure unit it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting import spark_entry
from scale_forecasting.engines import spark_explode, spark_naive

_CONFIG: dict[str, Any] = {
    "run_name": "spark entry test",
    "data": {"source_table": "source_series", "horizon": 7},
    "models": ["theta", "holtwinters"],
    "spark_method": "explode",
}


def _write_config(tmp_path: Path) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_CONFIG))
    return str(path)


# --- _parse_models: the CSV subset parser --------------------------------------


def test_parse_models_none_when_absent() -> None:
    # No --models → None → the engine runs its full cfg.models (standalone behavior).
    assert spark_entry._parse_models(None) is None


def test_parse_models_splits_and_trims() -> None:
    assert spark_entry._parse_models("theta, holtwinters ,xgboost") == [
        "theta",
        "holtwinters",
        "xgboost",
    ]


def test_parse_models_all_empty_collapses_to_none() -> None:
    # A stray comma / whitespace must not become an empty subset (which would run nothing).
    assert spark_entry._parse_models(" , ") is None
    assert spark_entry._parse_models("theta,,") == ["theta"]


# --- main: parse + dispatch + forward ------------------------------------------


def test_main_defaults_forward_standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No Arc B flags: forward models=None, manage_header=True — the pre-Arc-B dispatch, unchanged.
    captured: dict[str, Any] = {}

    def _fake_run(cfg: Any, models: Any = None, *, manage_header: bool = True) -> None:
        captured["cfg"] = cfg
        captured["models"] = models
        captured["manage_header"] = manage_header

    monkeypatch.setattr(spark_explode, "run", _fake_run)
    spark_entry.main(["--engine", "explode", "--config-uri", _write_config(tmp_path)])

    assert captured["models"] is None
    assert captured["manage_header"] is True
    assert captured["cfg"].run_name == "spark entry test"


def test_main_forwards_arc_b_contributor_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # main.run's launch: a subset + contributor mode reach the engine as parsed values.
    captured: dict[str, Any] = {}

    def _fake_run(cfg: Any, models: Any = None, *, manage_header: bool = True) -> None:
        captured["models"] = models
        captured["manage_header"] = manage_header

    monkeypatch.setattr(spark_explode, "run", _fake_run)
    spark_entry.main(
        [
            "--engine",
            "explode",
            "--config-uri",
            _write_config(tmp_path),
            "--models",
            "theta,holtwinters",
            "--manage-header",
            "false",
        ]
    )

    assert captured["models"] == ["theta", "holtwinters"]
    assert captured["manage_header"] is False


def test_main_dispatches_to_selected_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --engine naive must reach spark_naive.run, not spark_explode.run.
    seen: dict[str, Any] = {}
    monkeypatch.setattr(spark_naive, "run", lambda *a, **k: seen.setdefault("naive", True))
    monkeypatch.setattr(spark_explode, "run", lambda *a, **k: seen.setdefault("explode", True))
    spark_entry.main(["--engine", "naive", "--config-uri", _write_config(tmp_path)])

    assert seen == {"naive": True}
