"""Offline tests for the B5 ensemble orchestrator (``scale_forecasting.ensemble_run``).

The GCP path (``run_ensembles`` executing SQL + Write API) is the ``@gcp`` smoke in
``tests/integration/test_ensemble_smoke.py``; here we cover what is offline-testable:

* :func:`ensemble_run._apply_weights` — the pure learned-blend core: ``yhat = Σ wₘ·yhatₘ``
  renormalized over the base models *present* per ``(ts_id, forecast_date)``, with bound handling
  and row-dropping when nothing is present.
* :func:`ensemble_run.run_ensembles` short-circuits to a no-op when ``ensemble.enabled`` is false,
  without touching any GCP seam (it returns before importing ``google.cloud``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.ensemble_run import _apply_weights, run_ensembles
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "ens run test",
        "data": {"source_table": "source_series_native", "horizon": 7},
        "models": ["theta", "arima_plus"],
    }
    base.update(over)
    return RunConfig(**base)


_BASE_COLS = ["ts_id", "model_type", "forecast_date", "yhat", "yhat_lower", "yhat_upper"]


def _base_df(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Long-format base predictions (ts_id, model_type, forecast_date, yhat); bounds mirror yhat."""
    return pd.DataFrame(
        [
            {
                "ts_id": t,
                "model_type": m,
                "forecast_date": d,
                "yhat": y,
                "yhat_lower": y - 1.0,
                "yhat_upper": y + 1.0,
            }
            for (t, m, d, y) in rows
        ],
        columns=_BASE_COLS,
    )


# --- _apply_weights: the pure learned-blend core -------------------------------


def test_apply_weights_is_weighted_mean_when_all_present() -> None:
    # weights already sum to 1 → blend is the plain weighted mean.
    base = _base_df(
        [
            ("s1", "theta", "2024-01-01", 10.0),
            ("s1", "arima_plus", "2024-01-01", 20.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 0.75, "arima_plus": 0.25}, "rid", "nnls")
    assert len(rows) == 1
    r = rows[0]
    assert r["model_type"] == "ensemble_nnls"
    assert r["compute_engine"] == "ensemble"
    assert r["run_id"] == "rid"
    assert r["yhat"] == pytest.approx(0.75 * 10.0 + 0.25 * 20.0)  # 12.5
    # bounds blend the same way (each base bound is yhat ± 1)
    assert r["yhat_lower"] == pytest.approx(11.5)
    assert r["yhat_upper"] == pytest.approx(13.5)


def test_apply_weights_renormalizes_unnormalized_weights() -> None:
    # raw weights need not sum to 1; the blend renormalizes over present models.
    base = _base_df(
        [
            ("s1", "theta", "2024-01-01", 10.0),
            ("s1", "arima_plus", "2024-01-01", 30.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 3.0, "arima_plus": 1.0}, "rid", "ridge")
    # (3*10 + 1*30) / 4 = 15.0
    assert rows[0]["yhat"] == pytest.approx(15.0)


def test_apply_weights_renormalizes_over_present_subset_per_date() -> None:
    # date d1 has both models; date d2 has only theta → d2 blend is theta alone (weight renormed).
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "arima_plus", "d1", 20.0),
            ("s1", "theta", "d2", 40.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 0.5, "arima_plus": 0.5}, "rid", "nnls")
    by_date = {r["forecast_date"]: r["yhat"] for r in rows}
    assert by_date["d1"] == pytest.approx(15.0)
    assert by_date["d2"] == pytest.approx(40.0)  # only theta present → its weight renormed to 1


def test_apply_weights_drops_rows_with_no_present_weighted_model() -> None:
    # only arima_plus forecasts this date, but it has zero weight → nothing to blend, drop the row.
    base = _base_df([("s1", "arima_plus", "d1", 20.0)])
    rows = _apply_weights(base, {"theta": 1.0, "arima_plus": 0.0}, "rid", "nnls")
    assert rows == []


def test_apply_weights_ignores_models_not_in_weight_map() -> None:
    # a base model with no learned weight is simply not part of the blend.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "other", "d1", 999.0),
        ]
    )
    rows = _apply_weights(base, {"theta": 1.0}, "rid", "nnls")
    assert rows[0]["yhat"] == pytest.approx(10.0)


def test_apply_weights_empty_base_is_empty() -> None:
    base = _base_df([])
    assert _apply_weights(base, {"theta": 1.0}, "rid", "nnls") == []


# --- run_ensembles: disabled is a no-op that never touches GCP ------------------


def test_run_ensembles_disabled_is_noop() -> None:
    # ensemble.enabled defaults to False; run_ensembles must return before importing any GCP client.
    cfg = _cfg()
    assert cfg.ensemble.enabled is False
    # No monkeypatching of google.cloud needed: a GCP touch here would raise, so a clean return
    # proves the short-circuit.
    run_ensembles(cfg, "rid", settings=_SETTINGS)
