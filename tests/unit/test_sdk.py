"""Offline tests for the public SDK front door (``scale_forecasting`` package + ``sdk.py``).

No GCP, no compute: the :class:`Forecaster` facade only wraps :func:`scale_forecasting.main.run`, so
here we prove the wrapper's contract — construction parity, the offline ``dry_run``/``review``
pointers, and that ``run`` delegates to a spied ``main.run`` — plus the load-bearing **import-cost
contract**: ``import scale_forecasting`` must not pull the heavy model modules; only touching a
heavy public name (``run_cell``/``run``/``Forecaster``…) may. That last one runs in a subprocess
because the rest of the suite has already imported ``worker`` into this process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scale_forecasting as sf
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.registry.views import VIEW_NAMES
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg_dict(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_name": "sdk test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return base


# --- construction --------------------------------------------------------------


def test_from_dict_and_from_file_build_equal_config(tmp_path: Path) -> None:
    import json

    data = _cfg_dict()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))

    from_dict = sf.Forecaster.from_dict(data)
    from_file = sf.Forecaster.from_file(path)
    assert isinstance(from_dict.config, RunConfig)
    assert from_dict.config == from_file.config


def test_from_dict_rejects_bad_config() -> None:
    with pytest.raises(ConfigError):
        sf.Forecaster.from_dict({"not_a_field": True})


def test_from_file_rejects_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json ")
    with pytest.raises(ConfigError):
        sf.Forecaster.from_file(bad)


def test_run_id_is_deterministic_config_hash() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict())
    assert f.run_id == make_run_id(f.config)


# --- dry_run: offline plan -----------------------------------------------------


def test_dry_run_reports_id_fanout_and_runtime_split() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict())
    dr = f.dry_run()
    assert dr.run_id == f.run_id  # single source of truth (delegates to main.run dry_run)
    assert dr.python_models == ["theta"]
    assert dr.bq_models == ["arima_plus"]
    assert dr.fanout.n_series == 5
    assert dr.fanout.n_models == 2
    assert dr.fanout.n_cells == 5 * 2  # no folds → cells = series × models


# --- run: delegates to main.run, wraps in a RunResult --------------------------


def test_run_delegates_to_main_run_and_wraps_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import main

    calls: dict[str, Any] = {}

    def _spy(cfg: RunConfig, *, dry_run: bool = False, spark: object | None = None,
             settings: Settings | None = None) -> str:
        calls["cfg"] = cfg
        calls["spark"] = spark
        calls["settings"] = settings
        return make_run_id(cfg)

    monkeypatch.setattr(main, "run", _spy)

    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    result = f.run(spark="fake-session")

    assert calls["cfg"] == f.config
    assert calls["spark"] == "fake-session"
    assert calls["settings"] is _SETTINGS  # injected identity is threaded through
    assert result.run_id == f.run_id
    assert result.dataset_ref == _SETTINGS.dataset_ref
    assert result.views == VIEW_NAMES


# --- review: offline pointer ---------------------------------------------------


def test_review_is_offline_pointer_with_injected_settings() -> None:
    f = sf.Forecaster.from_dict(_cfg_dict(), settings=_SETTINGS)
    rr = f.review()
    assert rr.run_id == f.run_id
    assert rr.dataset_ref == _SETTINGS.dataset_ref
    assert rr.views == VIEW_NAMES


def test_review_is_graceful_when_identity_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SF_PROJECT_ID", "SF_CONNECTION", "SF_WAREHOUSE_URI"):
        monkeypatch.delenv(var, raising=False)
    f = sf.Forecaster.from_dict(_cfg_dict())  # no injected settings
    rr = f.review()
    assert rr.run_id == f.run_id
    assert rr.dataset_ref is None  # unresolved infra → None, not an exception
    assert rr.views == VIEW_NAMES


# --- public surface + import-cost contract -------------------------------------


def test_every_public_name_resolves() -> None:
    for name in sf.__all__:
        assert getattr(sf, name) is not None, name
    assert dir(sf) == sorted(sf.__all__)


def test_import_does_not_pull_heavy_modules_until_needed() -> None:
    # PEP 562 lazy contract, checked in a clean subprocess (this process already loaded worker).
    script = (
        "import sys, scale_forecasting as sf\n"
        "assert 'scale_forecasting.worker' not in sys.modules, 'import pulled worker'\n"
        "_ = sf.RunConfig, sf.Settings, sf.ConfigError, sf.Fanout\n"
        "assert 'scale_forecasting.worker' not in sys.modules, 'light names pulled worker'\n"
        "_ = sf.run_cell\n"
        "assert 'scale_forecasting.worker' in sys.modules, 'run_cell did not load worker'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
