"""End-to-end registry round-trip against live BigQuery (@gcp).

Exercises the real writers
(``ensure_tables`` → ``write_header`` → ``write_cells`` → ``update_header``) against a live
managed-Iceberg dataset and asserts the contract — every write route lands the right rows,
a re-run appends but the dedupe-on-read logical count is stable (append-only idempotency),
an artifact uploads to a readable GCS object, and an error cell yields a PARTIAL header.

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). Run manually::

    SF_PROJECT_ID=… SF_CONNECTION=… SF_WAREHOUSE_URI=… SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_registry_roundtrip.py

Each invocation uses a unique ``run_id`` for isolation: append-only writes accumulate and can't
be DELETE-d while buffered, so the test never reuses a run_id across invocations.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry import bq
from scale_forecasting.registry.ids import make_model_hash, make_run_id
from scale_forecasting.settings import Settings
from scale_forecasting.worker import CellResult

pytestmark = pytest.mark.gcp


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


def _cfg() -> RunConfig:
    return RunConfig(
        run_name="b1 roundtrip test",
        data={"source_table": "p.d.source_series", "series_limit": 2},
        models=["theta"],
    )


def _predictions(base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
            "yhat": [base, base + 1, base + 2],
            "yhat_lower": [base - 1, base, base + 1],
            "yhat_upper": [base + 1, base + 2, base + 3],
            "quantiles": [{"0.5": base}, None, {"0.5": base + 2}],
        }
    )


def _oof(base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [0, 0, 1],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-15"]),
            "y_true": [base, base + 1, base + 2],
            "yhat": [base - 0.5, base + 0.5, base + 1.5],
        }
    )


def _ok_cell(
    run_id: str, ts_id: str, cfg: RunConfig, base: float, artifact_bytes: bytes | None = None
) -> CellResult:
    return CellResult(
        run_id=run_id,
        ts_id=ts_id,
        model_type="theta",
        compute_engine="local",
        model_hash=make_model_hash(run_id, ts_id, "theta", cfg),
        status="ok",
        error=None,
        predictions=_predictions(base),
        oof=_oof(base),
        metrics={"wape": 0.1, "mae": base},
        best_params={"alpha": 0.5},
        fit_seconds=1.0,
        artifact_bytes=artifact_bytes,
    )


def _error_cell(run_id: str, ts_id: str, cfg: RunConfig) -> CellResult:
    return CellResult(
        run_id=run_id,
        ts_id=ts_id,
        model_type="theta",
        compute_engine="local",
        model_hash=make_model_hash(run_id, ts_id, "theta", cfg),
        status="error",
        error="model blew up",
        predictions=pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "quantiles"]),
        oof=None,
        metrics={},
        best_params={},
        fit_seconds=0.0,
    )


def _count(client: Any, table_ref: str, run_id: str) -> int:
    rows = client.query(
        f"SELECT COUNT(*) c FROM `{table_ref}` WHERE run_id=@run_id",
        job_config=_run_id_params(run_id),
    ).result()
    return int(next(iter(rows)).c)


def _run_id_params(run_id: str) -> Any:
    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )


def _poll_count(client: Any, table_ref: str, run_id: str, expected: int) -> int:
    """Write API rows are async-visible; poll briefly for the expected count."""
    n = 0
    for _ in range(8):
        n = _count(client, table_ref, run_id)
        if n == expected:
            return n
        time.sleep(2)
    return n


# The natural cell keys serving views dedupe on (append-only + dedupe-on-read).
_DEDUP_KEYS: dict[str, str] = {
    "forecast_predictions": "ts_id, model_type, forecast_date",
    "backtest_oof": "ts_id, model_type, fold_id, forecast_date",
    "forecast_metadata": "ts_id, model_type",
}


def _distinct_count(client: Any, table: str, table_ref: str, run_id: str) -> int:
    """The dedupe-on-read logical count: distinct cell keys for a run (re-run stable)."""
    keys = _DEDUP_KEYS[table]
    rows = client.query(
        f"SELECT COUNT(*) c FROM (SELECT {keys} FROM `{table_ref}` "
        f"WHERE run_id=@run_id GROUP BY {keys})",
        job_config=_run_id_params(run_id),
    ).result()
    return int(next(iter(rows)).c)


def test_registry_roundtrip(settings: Settings) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg()
    # Append-only means rows for a run_id accumulate and cannot be DELETE-d while buffered, so
    # each invocation uses a UNIQUE run_id to stay isolated. (make_run_id's deterministic digest
    # is unit-tested separately; here we exercise the write/dedupe contract.)
    run_id = f"{make_run_id(cfg)}-{int(time.time())}"

    preds = settings.table_ref("forecast_predictions")
    oof = settings.table_ref("backtest_oof")
    meta = settings.table_ref("forecast_metadata")

    # ensure_tables is idempotent — safe to call on an existing dataset.
    bq.ensure_tables(cfg, settings=settings)
    bq.ensure_tables(cfg, settings=settings)  # twice: proves IF NOT EXISTS

    # Write a fresh RUNNING header. run_registry is a single-row-per-run header written by
    # query-INSERT (no streaming buffer), so a defensive pre-clean is safe here (unlike the
    # cell tables); with a unique run_id it matches nothing.
    client.query(
        f"DELETE FROM `{settings.table_ref('run_registry')}` WHERE run_id=@run_id",
        job_config=_run_id_params(run_id),
    ).result()
    bq.write_header(cfg, run_id, settings=settings)
    header = next(
        iter(
            client.query(
                f"SELECT status, n_models FROM `{settings.table_ref('run_registry')}` "
                "WHERE run_id=@run_id",
                job_config=_run_id_params(run_id),
            ).result()
        )
    )
    assert header.status == "RUNNING"
    assert header.n_models == 1

    # write_cells: 2 ok cells + 1 error cell (empty predictions/oof). One ok cell carries a
    # model artifact (in-memory bytes), to exercise upload → GCS → model_artifact link.
    results = [
        _ok_cell(run_id, "series-0", cfg, base=10.0, artifact_bytes=b"fake-fitted-model-bytes"),
        _ok_cell(run_id, "series-1", cfg, base=20.0),
        _error_cell(run_id, "series-2", cfg),
    ]
    bq.write_cells(results, settings=settings)

    # 2 ok cells × 3 prediction rows = 6; error cell contributes none.
    assert _poll_count(client, preds, run_id, 6) == 6
    # 2 ok cells × 3 oof rows = 6; error cell has no oof.
    assert _poll_count(client, oof, run_id, 6) == 6
    # one metadata row per cell (including the error cell) = 3.
    assert _poll_count(client, meta, run_id, 3) == 3

    # The artifact-carrying cell's metadata row links a readable GCS object.
    artifact_uri = next(
        iter(
            client.query(
                f"SELECT model_artifact FROM `{meta}` WHERE run_id=@run_id AND ts_id='series-0'",
                job_config=_run_id_params(run_id),
            ).result()
        )
    ).model_artifact
    assert artifact_uri and artifact_uri.startswith("gs://")
    expected_hash = make_model_hash(run_id, "series-0", "theta", cfg)
    # The artifact path carries the registry key (project/registry-dataset), which is what makes a
    # per-registry orphan sweep unambiguous — assert the whole root, not just the run segment.
    assert artifact_uri == f"{settings.artifact_root}/{run_id}/{expected_hash}.pkl"
    from google.cloud import storage

    without_scheme = artifact_uri[len("gs://") :]
    bucket_name, _, blob_path = without_scheme.partition("/")
    blob = storage.Client(project=settings.project_id).bucket(bucket_name).blob(blob_path)
    assert blob.download_as_bytes() == b"fake-fitted-model-bytes"

    # Re-run: append-only (no DELETE — the Write API buffer forbids it). Raw rows may double,
    # but the dedupe-on-read logical count is stable — the idempotency contract.
    bq.write_cells(results, settings=settings)
    assert _poll_count(client, preds, run_id, 12) == 12  # raw rows appended
    assert _distinct_count(client, "forecast_predictions", preds, run_id) == 6
    assert _distinct_count(client, "backtest_oof", oof, run_id) == 6
    assert _distinct_count(client, "forecast_metadata", meta, run_id) == 3

    # update_header closes the run: an error cell present → PARTIAL. job_telemetry is the JSON
    # overlay the submitter stamps post-batch (here a stand-in) — proves the new column round-trips.
    status = "PARTIAL" if any(r.status == "error" for r in results) else "COMPLETED"
    telemetry_json = json.dumps(
        {"total_wall_s": 20.0, "executor_instances": 2, "dcu_milli_seconds": 123456},
        sort_keys=True,
    )
    bq.update_header(
        run_id,
        settings=settings,
        status=status,
        runtime_seconds=12.5,
        n_series=len(results),
        job_telemetry=telemetry_json,
    )
    closed = next(
        iter(
            client.query(
                f"SELECT status, runtime_seconds, n_series, job_telemetry FROM "
                f"`{settings.table_ref('run_registry')}` WHERE run_id=@run_id",
                job_config=_run_id_params(run_id),
            ).result()
        )
    )
    assert closed.status == "PARTIAL"
    assert closed.runtime_seconds == 12.5
    assert closed.n_series == 3
    # The telemetry column stores JSON as STRING (Iceberg rejects native JSON); parse to compare.
    assert json.loads(closed.job_telemetry)["executor_instances"] == 2
