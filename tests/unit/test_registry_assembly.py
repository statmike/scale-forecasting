"""Tests for the pure registry row-assembly layer (CONTRACTS §3.4, §4, BUILD 1.4).

Offline only: a :class:`CellResult` (plus a :class:`RunConfig` for the header) maps to the
exact row dicts each table expects. No BigQuery client — the I/O writers are Arc B (B1).
Covers frame→rows mapping, run/series/model/engine stamping, dtype/NaN coercion, JSON
serialization, and the model_hash idempotency key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry import artifacts, bq
from scale_forecasting.worker import CellResult

_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "my run",
        "data": {"source_table": "p.d.source_series_native", "series_limit": 5},
        "models": ["theta", "sarimax"],
    }
    base.update(over)
    return RunConfig(**base)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "yhat": [10.0, 11.5],
            "yhat_lower": [8.0, 9.0],
            "yhat_upper": [12.0, 14.0],
            "quantiles": [{"0.5": 10.0}, None],
        }
    )


def _oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [0, 0, 1],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-15"]),
            "y_true": [9.0, 10.0, 11.0],
            "yhat": [8.5, 10.5, 10.0],
        }
    )


def _result(**over: Any) -> CellResult:
    base: dict[str, Any] = {
        "run_id": "my-run-abc123def456",
        "ts_id": "series-7",
        "model_type": "theta",
        "compute_engine": "spark",
        "model_hash": "f" * 64,
        "status": "ok",
        "error": None,
        "predictions": _predictions(),
        "oof": _oof(),
        "metrics": {"wape": 0.12, "mae": 3.4},
        "best_params": {"alpha": 0.5},
        "fit_seconds": 1.25,
    }
    base.update(over)
    return CellResult(**base)


# --- prediction rows -----------------------------------------------------------


def test_prediction_rows_map_and_stamp() -> None:
    rows = bq.assemble_prediction_rows(_result())
    assert len(rows) == 2
    first = rows[0]
    # stamped identity on every row
    assert first["run_id"] == "my-run-abc123def456"
    assert first["ts_id"] == "series-7"
    assert first["model_type"] == "theta"
    assert first["compute_engine"] == "spark"
    # ds -> forecast_date, as a date (not timestamp)
    assert first["forecast_date"].isoformat() == "2026-02-01"
    assert first["yhat"] == 10.0


def test_prediction_quantiles_serialized_to_json_or_none() -> None:
    rows = bq.assemble_prediction_rows(_result())
    assert rows[0]["quantiles"] == '{"0.5": 10.0}'
    assert rows[1]["quantiles"] is None


def test_prediction_quantiles_drop_non_finite_values() -> None:
    # Same runaway-series pathology as the scalar columns, but on the JSON `quantiles` field:
    # json.dumps emits the bare literal `NaN`/`Infinity` (invalid JSON), and BigQuery's JSON
    # column parser rejects it ("syntax error while parsing value - invalid literal"), failing
    # the whole Storage Write API append. Non-finite quantile entries must be dropped; a dict
    # that is entirely non-finite collapses to NULL.
    preds = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "yhat": [10.0, 11.0],
            "yhat_lower": [8.0, 9.0],
            "yhat_upper": [12.0, 13.0],
            "quantiles": [
                {"0.1": float("nan"), "0.5": 10.0, "0.9": float("inf")},
                {"0.5": float("nan")},
            ],
        }
    )
    rows = bq.assemble_prediction_rows(_result(predictions=preds))
    # finite entries survive; NaN/Inf keys are dropped
    assert rows[0]["quantiles"] == '{"0.5": 10.0}'
    # an all-non-finite dict becomes NULL, not invalid JSON
    assert rows[1]["quantiles"] is None


def test_non_finite_forecast_values_coerced_to_none() -> None:
    # A runaway series (e.g. log1p's expm1 inverse overflowing) can yield +Inf/-Inf/NaN. The
    # BigQuery Storage Write API rejects those for a FLOAT64 column, and one rejected row fails
    # the whole append — which killed a 100k-series run mid-flight. They must land as NULL so the
    # pathological cell is a missing value, not a fleet-wide failure.
    preds = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
            "yhat": [float("inf"), float("-inf"), float("nan")],
            "yhat_lower": [1.0, float("inf"), 2.0],
            "yhat_upper": [float("inf"), 3.0, 4.0],
            "quantiles": [None, None, None],
        }
    )
    rows = bq.assemble_prediction_rows(_result(predictions=preds))
    assert [r["yhat"] for r in rows] == [None, None, None]
    assert rows[0]["yhat_upper"] is None
    assert rows[1]["yhat_lower"] is None
    # finite values on the same rows are untouched
    assert rows[0]["yhat_lower"] == 1.0
    assert rows[2]["yhat_upper"] == 4.0


# --- oof rows ------------------------------------------------------------------


def test_oof_rows_carry_fold_and_truth() -> None:
    rows = bq.assemble_oof_rows(_result())
    assert len(rows) == 3
    assert [r["fold_id"] for r in rows] == [0, 0, 1]
    assert all(isinstance(r["fold_id"], int) for r in rows)
    assert rows[0]["y_true"] == 9.0
    assert rows[0]["forecast_date"].isoformat() == "2026-01-01"


def test_oof_rows_empty_when_no_backtest() -> None:
    assert bq.assemble_oof_rows(_result(oof=None)) == []


# --- metadata row --------------------------------------------------------------


def test_metadata_row_has_full_metric_panel() -> None:
    row = bq.assemble_metadata_row(_result(), _CREATED)
    # every declared metric column is present...
    for name in bq.METRIC_COLUMNS:
        assert name in row
    # ...populated where provided, NULL (None) where absent
    assert row["wape"] == 0.12
    assert row["mae"] == 3.4
    assert row["rmse"] is None
    assert row["fold_id"] is None  # full-fit summary row
    assert row["best_params"] == '{"alpha": 0.5}'
    assert row["fit_seconds"] == 1.25
    assert row["created_at"] is _CREATED
    assert row["model_hash"] == "f" * 64


def test_metadata_row_carries_artifact_link() -> None:
    row = bq.assemble_metadata_row(_result(), _CREATED, model_artifact="gs://wh/artifacts/x/m.pkl")
    assert row["model_artifact"] == "gs://wh/artifacts/x/m.pkl"


def test_metadata_metric_columns_match_ddl() -> None:
    # The assembled metric keys must be exactly the metric columns in the DDL (§4).
    ddl_metrics = {
        "mae",
        "rmse",
        "mse",
        "mape",
        "smape",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "coverage",
        "pinball",
    }
    assert set(bq.METRIC_COLUMNS) == ddl_metrics


# --- coercion ------------------------------------------------------------------


def test_nan_metric_coerced_to_none() -> None:
    row = bq.assemble_metadata_row(_result(metrics={"wape": float("nan")}), _CREATED)
    assert row["wape"] is None


def test_empty_best_params_serialized_to_none() -> None:
    row = bq.assemble_metadata_row(_result(best_params={}), _CREATED)
    assert row["best_params"] is None


# --- idempotency key -----------------------------------------------------------


def test_dedup_key_is_run_scoped() -> None:
    # Idempotency is enforced at the run grain (run-level clear + append), so the key is
    # run_id alone — not model_hash (B0.3: can't DELETE the streaming buffer per-cell).
    key = bq.cell_dedup_key(_result())
    assert key == {"run_id": "my-run-abc123def456"}


# --- header row ----------------------------------------------------------------


def test_header_row_snapshots_config() -> None:
    cfg = _cfg()
    row = bq.assemble_header_row(cfg, "my-run-abc123def456", _CREATED)
    assert row["run_id"] == "my-run-abc123def456"
    assert row["status"] == "RUNNING"
    assert row["python_runtime"] == "spark"
    assert row["spark_method"] == "explode"  # normalized default
    assert row["n_models"] == 2
    assert row["n_series"] == 5
    assert row["backtest_on"] is False
    assert row["created_at"] is _CREATED
    # raw_config is the verbatim validated config as a dict (the native JSON column's query
    # parameter serializes it; a pre-serialized string would double-encode — D19).
    assert isinstance(row["raw_config"], dict)
    assert row["raw_config"]["run_name"] == "my run"
    assert row["raw_config"]["models"] == ["theta", "sarimax"]


def test_header_ensemble_strategies_empty_when_disabled() -> None:
    row = bq.assemble_header_row(_cfg(), "rid", _CREATED)
    assert row["ensemble_strategies"] == []


def test_header_ensemble_strategies_listed_when_enabled() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    row = bq.assemble_header_row(cfg, "rid", _CREATED)
    assert row["ensemble_strategies"] == ["mean", "median"]


# --- artifact uri --------------------------------------------------------------


def test_artifact_uri_is_run_scoped_and_deterministic() -> None:
    uri = artifacts.artifact_gcs_uri("/tmp/model.pkl", "my-run-abc123", "gs://bucket/warehouse/")
    assert uri == "gs://bucket/warehouse/artifacts/my-run-abc123/model.pkl"
    # deterministic
    assert uri == artifacts.artifact_gcs_uri(
        "/tmp/model.pkl", "my-run-abc123", "gs://bucket/warehouse"
    )


# --- Storage Write API retry-on-transient (_append_via_write_api) ---------------
#
# The append path is the single chokepoint every engine's writes funnel through (Ray, all Spark
# methods, BigQuery-native). A transient service 500/503/429 must be retried — not fail an
# otherwise-complete multi-hour run — while a permanent error still fails fast. These tests drive
# a fake write_client so they stay offline; time.sleep is neutralized so backoff is instant.


class _FakeWriteClient:
    """Minimal stand-in for BigQueryWriteClient.

    ``append_rows`` consumes the request generator (so the lazy iterator is exercised) and then
    replays a scripted outcome per call: raise a supplied exception, or return an iterable of
    fake responses. ``table_path`` just formats a path like the real client.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def table_path(self, project: str, dataset: str, table: str) -> str:
        return f"projects/{project}/datasets/{dataset}/tables/{table}"

    def append_rows(self, requests: Any) -> Any:
        list(requests)  # drain the generator, matching the real bidi call
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _OkResponse:
    class _Err:
        code = 0
        message = ""

    error = _Err()
    row_errors: list[Any] = []


def _append(client: Any) -> None:
    # Build a real descriptor + serialized row so the request generator (which the fake client
    # drains) constructs a valid AppendRowsRequest — same machinery write_cells uses.
    _msg, descriptor = bq._proto_for("backtest_oof", bq._OOF_SPEC)
    serialized = bq._encode_rows(
        _msg, bq._OOF_SPEC, [{"run_id": "r", "ts_id": "s", "model_type": "theta", "fold_id": 0}]
    )
    bq._append_via_write_api(client, "proj", "ds", "backtest_oof", descriptor, serialized)


def test_append_retries_transient_then_succeeds(monkeypatch: Any) -> None:
    from google.api_core.exceptions import InternalServerError

    monkeypatch.setattr(bq.time, "sleep", lambda *_a, **_k: None)
    # two transient 500s, then a clean response
    client = _FakeWriteClient(
        [
            InternalServerError("500 An error occurred while verifying authorization"),
            InternalServerError("500 transient"),
            [_OkResponse()],
        ]
    )
    _append(client)  # must not raise
    assert client.calls == 3


def test_append_transient_exhausts_attempts_and_raises(monkeypatch: Any) -> None:
    from google.api_core.exceptions import ServiceUnavailable

    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(bq.time, "sleep", lambda *_a, **_k: None)
    client = _FakeWriteClient(
        [ServiceUnavailable("503") for _ in range(bq._WRITE_RETRY_ATTEMPTS)]
    )
    with pytest.raises(RegistryError, match="after 5 attempts"):
        _append(client)
    assert client.calls == bq._WRITE_RETRY_ATTEMPTS


def test_append_permanent_api_error_fails_fast(monkeypatch: Any) -> None:
    from google.api_core.exceptions import Forbidden

    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(bq.time, "sleep", lambda *_a, **_k: None)
    # a non-transient GoogleAPICallError (e.g. real 403) is not retried
    client = _FakeWriteClient([Forbidden("403 permission denied")])
    with pytest.raises(RegistryError, match="failed:"):
        _append(client)
    assert client.calls == 1


def test_append_response_level_error_fails_fast(monkeypatch: Any) -> None:
    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(bq.time, "sleep", lambda *_a, **_k: None)

    class _BadResponse:
        class _Err:
            code = 7
            message = "N Errors found"

        error = _Err()
        row_errors: list[Any] = []

    # a response-level error is a data/schema problem — fail on the first call, no retry
    client = _FakeWriteClient([[_BadResponse()]])
    with pytest.raises(RegistryError, match="backtest_oof failed: 7"):
        _append(client)
    assert client.calls == 1
