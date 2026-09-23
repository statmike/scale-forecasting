"""Every registered Python model, driven through ``run_cell`` and out into registry rows.

``test_models_contract`` already fits and predicts every model, so the model layer is covered. What
it cannot cover is everything ``run_cell`` adds on top — the backtest fold loop, metric computation,
calibration, arm selection — and then the encoding step that turns the result into rows BigQuery
will accept. Those stages are shared code, but what flows *through* them is per-model, and only
seven of the registered models have ever appeared in a shipped config. For the other nine the fit is
proven and everything downstream of it is not.

The failure this is built to catch is quiet rather than loud. A model whose out-of-fold predictions
come back unusable does not raise: the fold loop completes, the metrics compute to NaN, `_as_float`
maps NaN to NULL exactly as it is supposed to, and the cell lands in the registry as a well-formed
row of nothing. Nobody sees a stack trace. The model simply never wins a leaderboard, and a
leaderboard with a silently dead entry looks identical to one where that model was merely worse. So
the assertion that matters here is not "it didn't crash" — it is **the cell is scorable**: folds
were achieved, and the headline metrics came back finite.

Each model is run once, in a module-scoped fixture. Fitting sixteen models across three folds is the
expensive part, and the four assertions below all interrogate the same `CellResult`.
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.models import get_model, list_models
from scale_forecasting.registry.rows import (
    METRIC_COLUMNS,
    assemble_metadata_row,
    assemble_oof_rows,
    assemble_prediction_rows,
)
from scale_forecasting.worker import CellResult, run_cell

HORIZON = 7
_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

# Optional third-party dep required by each model (None = core-only). Same map as the model
# contract test: a model whose optional extra is absent registers and skips rather than failing.
_MODEL_DEP: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "prophet": "prophet",
    "neuralprophet": "neuralprophet",
}

# The metrics a well-behaved model on a well-behaved series has no excuse for. The other twelve are
# allowed to be NULL here — `coverage` and the interval metrics depend on bounds the model may not
# produce, and the scaled errors need a seasonal denominator a short fold can fail to supply.
_MUST_BE_FINITE = ("mae", "rmse", "wape")


def _python_models() -> list[str]:
    """Registered models that execute in-process. The BigQuery-native ones run as SQL."""
    return [name for name in list_models() if get_model(name).runtime == "python"]


def _series(n: int = 400) -> pd.DataFrame:
    """Deterministic trend + weekly seasonality + mild noise — long enough for three folds.

    Noise is present on purpose. Several models special-case a perfectly smooth series (a zero
    residual variance makes an interval degenerate and a scaled error divide by zero), so a
    noiseless panel would let a model pass here and produce NULL metrics on real data.
    """
    rng = np.random.default_rng(1234)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 40.0, n)
    weekly = 4.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    noise = rng.normal(0, 0.5, n)
    return pd.DataFrame({"ts_id": "series-a", "ds": idx, "y": trend + weekly + noise})


def _cfg(model_name: str) -> RunConfig:
    return RunConfig(
        run_name="worker model matrix",
        data={"source_table": "t", "freq": "D", "horizon": HORIZON},
        models=[model_name],
        backtest={
            "enabled": True,
            "n_folds": 3,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 180,
        },
    )


@pytest.fixture(params=_python_models(), scope="module")
def cell(request: pytest.FixtureRequest) -> CellResult:
    """One scored cell per model, computed once and shared by every assertion below."""
    name = str(request.param)
    dep = _MODEL_DEP.get(name)
    if dep is not None and importlib.util.find_spec(dep) is None:
        pytest.skip(f"optional dependency '{dep}' not installed for model '{name}'")
    return run_cell(_series(), name, _cfg(name))


# --- the cell is scorable, not merely successful --------------------------------


def test_every_model_produces_a_scored_cell(cell: CellResult) -> None:
    """Folds achieved and out-of-fold predictions returned — the precondition for every metric.

    A cell can be ``status="ok"`` with zero folds (a series too short to score still ships its
    forecast, by design), so ``ok`` alone would pass for a model that never scored anything. On a
    400-point series with a 180-point floor there is no legitimate reason to achieve fewer than
    three.
    """
    assert cell.status == "ok", cell.error
    assert cell.n_folds_achieved == 3
    assert cell.oof is not None and not cell.oof.empty


def test_every_models_headline_metrics_are_finite(cell: CellResult) -> None:
    """The quiet failure: a fold loop that completed and scored nothing.

    NaN here is not an error anywhere downstream — it encodes to NULL, the row is valid, and the
    model just never places on a leaderboard. This is the only place that distinction is visible.
    """
    missing = [name for name in METRIC_COLUMNS if name not in cell.metrics]
    assert missing == [], f"metrics panel is incomplete: {missing}"

    dead = [m for m in _MUST_BE_FINITE if not math.isfinite(float(cell.metrics[m]))]
    assert dead == [], f"scored three folds but {dead} came back non-finite"


# --- and it survives the encoding boundary --------------------------------------


def test_every_models_metadata_row_encodes(cell: CellResult) -> None:
    """Every metric reaches the row as a Python float or NULL, never a numpy scalar.

    ``_as_float`` calls ``float(value)`` on anything non-None, so a metric that arrived as something
    unnumeric raises here rather than at the write. That is the point of running it per model: the
    encoder is shared, but what it is handed is not.
    """
    row = assemble_metadata_row(cell, _CREATED)
    for name in METRIC_COLUMNS:
        value = row[name]
        assert value is None or type(value) is float, f"{name} encoded as {type(value).__name__}"


def test_every_models_prediction_rows_encode(cell: CellResult) -> None:
    """Horizon rows out, every numeric field finite-or-NULL, quantiles a JSON string.

    The Storage Write API rejects NaN and ±Inf on a FLOAT64 column and fails the *whole*
    ``append_rows`` request, which in a worker takes the task and cascades. So the assertion is on
    the encoded row, not on the frame it came from.
    """
    rows = assemble_prediction_rows(cell, _CREATED)
    assert len(rows) == HORIZON

    for row in rows:
        for field in ("yhat", "yhat_raw", "yhat_adjusted", "yhat_lower", "yhat_upper"):
            value = row[field]
            assert value is None or (type(value) is float and math.isfinite(value)), (
                f"{field} encoded as {value!r}"
            )
        if row["quantiles"] is not None:
            assert isinstance(json.loads(row["quantiles"]), dict)


def test_every_models_oof_rows_encode(cell: CellResult) -> None:
    """The backtest half of the same boundary — one row per scored observation, all encodable."""
    rows = assemble_oof_rows(cell, _CREATED)
    assert rows, "a cell that achieved three folds must produce out-of-fold rows"

    for row in rows:
        for field, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{field} is non-finite: {value!r}"
            assert not isinstance(value, np.generic), f"{field} kept a numpy type: {type(value)}"
