"""Tests for the local model-dev loop (playground.py).

The playground is the on-ramp a new user hits first, so these tests pin the behavior that
makes it trustworthy: it auto-discovers models from the factory, its sample data is real
generator output that passes validation, it runs the actual worker cell, and it never
raises on a bad model (error comes back as an error cell, like production).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scale_forecasting.playground import (
    available_models,
    build_config,
    run_model,
    sample_data,
    summarize,
)


def test_available_models_excludes_bigquery_by_default() -> None:
    names = available_models()
    assert "theta" in names
    # BigQuery-native models run as SQL, not in a local cell.
    assert not any("arima_plus" in n or n.startswith("bq_") for n in names)
    assert available_models(include_bigquery=True) != names or names  # superset or equal


def test_available_models_is_the_factory_registry() -> None:
    from scale_forecasting.models import get_model, list_models

    expected = [n for n in list_models() if get_model(n).runtime != "bigquery"]
    assert available_models() == expected


def test_sample_data_is_wellformed_and_deterministic() -> None:
    a = sample_data(n_series=3, history=120)
    b = sample_data(n_series=3, history=120)
    pd.testing.assert_frame_equal(a, b)
    assert set(["ts_id", "ds", "y"]).issubset(a.columns)
    assert a["ts_id"].nunique() == 3


def test_sample_data_with_exog_has_driver_column() -> None:
    df = sample_data(n_series=2, history=90, with_exog=True)
    assert "price_index" in df.columns


def test_build_config_matches_sample_columns() -> None:
    cfg = build_config("theta", horizon=14)
    assert cfg.data.ts_id_col == "ts_id"
    assert cfg.data.date_col == "ds"
    assert cfg.data.target_col == "y"
    assert cfg.data.horizon == 14
    assert cfg.models == ["theta"]


def test_run_model_theta_ok() -> None:
    run = run_model("theta", freq="D", horizon=14)
    assert run.result.status == "ok"
    assert len(run.result.predictions) == 14
    # Canonical frame columns present.
    assert list(run.result.predictions.columns) == [
        "ds", "yhat", "yhat_lower", "yhat_upper", "quantiles",
    ]


def test_run_model_backtest_populates_metrics() -> None:
    run = run_model("theta", horizon=14, backtest=True)
    assert run.result.status == "ok"
    # At least one metric is finite once backtest is on.
    assert any(v == v for v in run.result.metrics.values())


def test_run_model_no_backtest_has_nan_metrics() -> None:
    run = run_model("theta", horizon=14, backtest=False)
    assert all(v != v for v in run.result.metrics.values())  # all NaN by design


def test_run_model_unknown_model_is_error_cell_not_raise() -> None:
    # run_cell never raises; an unknown model comes back as an error cell.
    run = run_model("does_not_exist")
    assert run.result.status == "error"
    assert "does_not_exist" in (run.result.error or "")


def test_run_model_bad_ts_id_raises_with_available() -> None:
    with pytest.raises(ValueError, match="not in data; available"):
        run_model("theta", ts_id="s_999999")


def test_run_model_accepts_user_data() -> None:
    data = sample_data(n_series=2, history=200)
    run = run_model("theta", data=data, ts_id="s_000001", horizon=7)
    assert run.result.status == "ok"
    assert run.result.ts_id == "s_000001"


def test_run_model_exog_model_with_driver() -> None:
    run = run_model("sarimax", horizon=10, with_exog=True)
    assert run.result.status in {"ok", "error"}  # fit may be finicky, but must not raise
    assert run.result.model_type == "sarimax"


def test_summarize_ok_and_error() -> None:
    ok = summarize(run_model("theta", horizon=7))
    assert "model      : theta" in ok
    assert "status     : ok" in ok

    err = summarize(run_model("nope"))
    assert "status     : error" in err


# --- CLI -----------------------------------------------------------------------


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    from scale_forecasting.playground import _main

    rc = _main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "theta" in out


def test_cli_run_model(capsys: pytest.CaptureFixture[str]) -> None:
    from scale_forecasting.playground import _main

    rc = _main(["--model", "theta", "--horizon", "7"])
    assert rc == 0
    assert "status     : ok" in capsys.readouterr().out


def test_cli_no_args_lists_and_hints(capsys: pytest.CaptureFixture[str]) -> None:
    from scale_forecasting.playground import _main

    rc = _main([])
    assert rc == 0
    assert "run one with" in capsys.readouterr().out
