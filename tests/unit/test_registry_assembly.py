"""Tests for the pure registry row-assembly layer.

Offline only: a :class:`CellResult` (plus a :class:`RunConfig` for the header) maps to the
exact row dicts each table expects. No BigQuery client — the I/O writers are covered elsewhere.
Covers frame→rows mapping, run/series/model/engine stamping, dtype/NaN coercion, JSON
serialization, and the model_hash idempotency key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry import artifacts, jobs, write_api
from scale_forecasting.registry.harvest import rank_harvest_candidates
from scale_forecasting.registry.header import (
    merge_header_telemetry,
    render_header_telemetry_merge,
    sizing_telemetry_path,
)
from scale_forecasting.registry.lifecycle import run_job
from scale_forecasting.registry.params import (
    _HEADER_PARAM_TYPES,
    _JOB_PARAM_TYPES,
    render_status_guard,
)
from scale_forecasting.registry.rows import (
    METRIC_COLUMNS,
    assemble_header_row,
    assemble_job_row,
    assemble_metadata_row,
    assemble_oof_rows,
    assemble_prediction_rows,
    cell_dedup_key,
)
from scale_forecasting.registry.write_api import (
    _META_SPEC,
    _OOF_SPEC,
    _WRITE_RETRY_ATTEMPTS,
    _append_via_write_api,
    _encode_rows,
    _proto_for,
)
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
    rows = assemble_prediction_rows(_result())
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
    rows = assemble_prediction_rows(_result())
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
    rows = assemble_prediction_rows(_result(predictions=preds))
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
    rows = assemble_prediction_rows(_result(predictions=preds))
    assert [r["yhat"] for r in rows] == [None, None, None]
    assert rows[0]["yhat_upper"] is None
    assert rows[1]["yhat_lower"] is None
    # finite values on the same rows are untouched
    assert rows[0]["yhat_lower"] == 1.0
    assert rows[2]["yhat_upper"] == 4.0


# --- oof rows ------------------------------------------------------------------


def test_oof_rows_carry_fold_and_truth() -> None:
    rows = assemble_oof_rows(_result())
    assert len(rows) == 3
    assert [r["fold_id"] for r in rows] == [0, 0, 1]
    assert all(isinstance(r["fold_id"], int) for r in rows)
    assert rows[0]["y_true"] == 9.0
    assert rows[0]["forecast_date"].isoformat() == "2026-01-01"


def test_oof_rows_empty_when_no_backtest() -> None:
    assert assemble_oof_rows(_result(oof=None)) == []


# --- metadata row --------------------------------------------------------------


def test_metadata_row_has_full_metric_panel() -> None:
    row = assemble_metadata_row(_result(), _CREATED)
    # every declared metric column is present...
    for name in METRIC_COLUMNS:
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


def test_metadata_row_carries_cell_timing_and_worker() -> None:
    # The per-cell wall-clock bracket + worker identity flow off the cell onto the row (the trace
    # lane). A cell without them (native/ensemble) leaves the three columns NULL.
    started = datetime(2026, 1, 2, 3, 4, 0, tzinfo=UTC)
    ended = datetime(2026, 1, 2, 3, 4, 2, tzinfo=UTC)
    row = assemble_metadata_row(
        _result(worker_id="host-9:4242", cell_started_at=started, cell_ended_at=ended), _CREATED
    )
    assert row["worker_id"] == "host-9:4242"
    assert row["cell_started_at"] is started
    assert row["cell_ended_at"] is ended

    bare = assemble_metadata_row(_result(), _CREATED)
    assert bare["worker_id"] is None
    assert bare["cell_started_at"] is None and bare["cell_ended_at"] is None


def test_metadata_row_carries_the_harvested_measurement() -> None:
    # A completed run *is* a profile (`profiling.harvest_profile`), which only works if the cost
    # every cell already paid lands on the row next to the forecast it produced.
    row = assemble_metadata_row(
        _result(
            cpu_seconds=4.5,
            process_rss_bytes=2 * 1024**3,
            peak_gpu_bytes=1024**3,
            intraop_threads=2,
            n_obs=730,
        ),
        _CREATED,
    )
    assert row["cpu_seconds"] == 4.5
    assert row["process_rss_bytes"] == 2 * 1024**3
    assert row["peak_gpu_bytes"] == 1024**3
    assert row["intraop_threads"] == 2
    assert row["n_obs"] == 730


def test_an_unmeasured_cell_writes_nulls_which_is_also_how_older_rows_read_back() -> None:
    # This is why the harvest reader needs no version check: "measurement was off" and "this row
    # predates the columns" are the same NULLs, and both mean "no evidence", not "zero".
    row = assemble_metadata_row(_result(), _CREATED)
    measured = ("cpu_seconds", "process_rss_bytes", "peak_gpu_bytes", "intraop_threads", "n_obs")
    for column in measured:
        assert row[column] is None, column


def test_every_measurement_column_is_declared_in_the_write_api_spec() -> None:
    # A key the assembler emits that `_META_SPEC` does not type is dropped silently by the
    # Storage Write API — the measurement would vanish between the worker and the table.
    typed = {name for name, _ in _META_SPEC}
    assert set(assemble_metadata_row(_result(), _CREATED)) <= typed


def test_metadata_row_carries_artifact_link() -> None:
    row = assemble_metadata_row(_result(), _CREATED, model_artifact="gs://wh/artifacts/x/m.pkl")
    assert row["model_artifact"] == "gs://wh/artifacts/x/m.pkl"


def test_metadata_metric_columns_match_ddl() -> None:
    # The assembled metric keys must be exactly the metric columns in the DDL.
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
    assert set(METRIC_COLUMNS) == ddl_metrics


# --- coercion ------------------------------------------------------------------


def test_nan_metric_coerced_to_none() -> None:
    row = assemble_metadata_row(_result(metrics={"wape": float("nan")}), _CREATED)
    assert row["wape"] is None


def test_empty_best_params_serialized_to_none() -> None:
    row = assemble_metadata_row(_result(best_params={}), _CREATED)
    assert row["best_params"] is None


# --- idempotency key -----------------------------------------------------------


def test_dedup_key_is_run_scoped() -> None:
    # Idempotency is enforced at the run grain (run-level clear + append), so the key is
    # run_id alone — not model_hash (can't DELETE the streaming buffer per-cell).
    key = cell_dedup_key(_result())
    assert key == {"run_id": "my-run-abc123def456"}


# --- header row ----------------------------------------------------------------


def test_header_row_snapshots_config() -> None:
    cfg = _cfg()
    row = assemble_header_row(cfg, "my-run-abc123def456", _CREATED)
    assert row["run_id"] == "my-run-abc123def456"
    assert row["status"] == "RUNNING"
    assert row["python_runtime"] == "spark"
    assert row["n_models"] == 2
    assert row["n_series"] == 5
    assert row["backtest_on"] is False
    assert row["created_at"] is _CREATED
    # Snapshot is out-of-band from the config (it must never enter the run_id hash), so it defaults
    # to None when the assembler is called without one.
    assert row["snapshot_millis"] is None
    # raw_config is the verbatim validated config as a dict (the native JSON column's query
    # parameter serializes it; a pre-serialized string would double-encode).
    assert isinstance(row["raw_config"], dict)
    assert row["raw_config"]["run_name"] == "my run"
    assert row["raw_config"]["models"] == ["theta", "sarimax"]


def test_header_row_carries_snapshot_millis_when_given() -> None:
    row = assemble_header_row(_cfg(), "rid", _CREATED, snapshot_millis=1_724_000_000_000)
    assert row["snapshot_millis"] == 1_724_000_000_000


def test_header_row_stamps_user_id_for_audit() -> None:
    # The launching principal (resolved by write_header via identity.resolve_principal) lands on the
    # header so *launch* is attributable, alongside the P5 cancel actor.
    row = assemble_header_row(
        _cfg(), "rid", _CREATED, user_id="runner@proj.iam.gserviceaccount.com"
    )
    assert row["user_id"] == "runner@proj.iam.gserviceaccount.com"


def test_header_row_user_id_defaults_null() -> None:
    # Unresolved principal (or a pre-audit call) → NULL, never a fabricated actor.
    assert assemble_header_row(_cfg(), "rid", _CREATED)["user_id"] is None


def test_header_ensemble_strategies_empty_when_disabled() -> None:
    row = assemble_header_row(_cfg(), "rid", _CREATED)
    assert row["ensemble_strategies"] == []


def test_header_ensemble_strategies_listed_when_enabled() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    row = assemble_header_row(cfg, "rid", _CREATED)
    assert row["ensemble_strategies"] == ["mean", "median"]


def test_header_row_columns_match_param_types() -> None:
    # The assembled header's keys are exactly the columns write_header knows how to bind — so a new
    # column (e.g. snapshot_millis) can't be added to one side without the other.
    row = assemble_header_row(_cfg(), "rid", _CREATED, snapshot_millis=1)
    assert set(row) == set(_HEADER_PARAM_TYPES)


# --- run_jobs row assembly -----------------------------------------------------


def test_job_row_derives_id_and_maps_resolved_compute() -> None:
    row = assemble_job_row(
        "my-run-abc123def456",
        "deep_learning",
        1,
        _CREATED,
        runtime="spark",
        spark_mode="cluster",
        hardware="gpu",
        gpu_type="T4",
        system_job_id="dp-batch-xyz",
    )
    assert row["job_id"] == "sf-my-run-abc123def456-deep_learning-a1"
    assert row["run_id"] == "my-run-abc123def456"
    assert row["family"] == "deep_learning"
    assert row["attempt"] == 1
    assert (row["runtime"], row["spark_mode"], row["hardware"], row["gpu_type"]) == (
        "spark",
        "cluster",
        "gpu",
        "T4",
    )
    assert row["system_job_id"] == "dp-batch-xyz"
    assert row["status"] == "RUNNING"  # default: RUNNING until the job finishes
    assert row["runtime_seconds"] is None and row["job_telemetry"] is None
    assert row["created_at"] is _CREATED
    # started_at defaults to created_at; ended_at is stamped later (by run_job at exit).
    assert row["started_at"] is _CREATED
    assert row["ended_at"] is None


def test_job_row_defaults_are_null_for_unset_compute() -> None:
    row = assemble_job_row("rid-0123456789ab", "native", 1, _CREATED)
    assert row["job_id"] == "sf-rid-0123456789ab-native-a1"
    for col in ("runtime", "spark_mode", "hardware", "gpu_type", "system_job_id"):
        assert row[col] is None, col


def test_job_row_id_reflects_attempt() -> None:
    row = assemble_job_row("rid-0123456789ab", "ml", 3, _CREATED)
    assert row["job_id"].endswith("-ml-a3")
    assert row["attempt"] == 3


def test_job_row_wraps_probe_handle_into_job_telemetry() -> None:
    # A probe handle passed at entry is nested under job_telemetry.$.probe_handle (the JSON path the
    # v_run_jobs view projects); absent, job_telemetry stays NULL.
    handle = {
        "runtime": "spark",
        "native_id": "dp-batch-xyz",
        "region": "us-central1",
        "id_kind": "exact",
        "spark_mode": "serverless",
    }
    row = assemble_job_row("rid-0123456789ab", "statistical", 1, _CREATED, probe_handle=handle)
    assert row["job_telemetry"] == {"probe_handle": handle}


def test_job_row_columns_match_param_types() -> None:
    # The assembled row's keys are exactly the columns write_job/update_job know how to bind.
    row = assemble_job_row("rid-0123456789ab", "statistical", 1, _CREATED)
    assert set(row) == set(_JOB_PARAM_TYPES)


# --- run_job lifecycle (context manager) ---------------------------------------
#
# run_job wraps the per-job row's RUNNING → terminal transition around a block. The write/update
# I/O is exercised against captured calls (monkeypatched write_job/update_job) so the lifecycle
# logic is covered offline; the real BigQuery writes are covered by the @gcp round-trip test.


def _capture_job_io(monkeypatch: Any) -> dict[str, Any]:
    """Redirect jobs.write_job / jobs.update_job to capture their calls; return the capture dict."""
    cap: dict[str, Any] = {"written": None, "updates": []}

    def _write(row: dict[str, Any], *, settings: Any = None) -> None:
        cap["written"] = row

    def _update(job_id: str, *, settings: Any = None, **fields: Any) -> None:
        cap["updates"].append((job_id, fields))

    monkeypatch.setattr(jobs, "write_job", _write)
    monkeypatch.setattr(jobs, "update_job", _update)
    return cap


def test_run_job_writes_running_then_completes(monkeypatch: Any) -> None:
    cap = _capture_job_io(monkeypatch)
    with run_job(
        "my-run-0123456789ab",
        "deep_learning",
        1,
        runtime="spark",
        spark_mode="cluster",
        hardware="gpu",
        gpu_type="T4",
    ) as job:
        job.finalize(system_job_id="dp-batch-xyz", job_telemetry={"total_wall_s": 12.0})

    row = cap["written"]
    assert row["job_id"] == "sf-my-run-0123456789ab-deep_learning-a1"
    assert row["status"] == "RUNNING"
    assert (row["runtime"], row["spark_mode"], row["hardware"], row["gpu_type"]) == (
        "spark",
        "cluster",
        "gpu",
        "T4",
    )
    # one terminal update: COMPLETED + measured runtime + the finalizer's extras
    assert len(cap["updates"]) == 1
    job_id, fields = cap["updates"][0]
    assert job_id == "sf-my-run-0123456789ab-deep_learning-a1"
    assert fields["status"] == "COMPLETED"
    assert "runtime_seconds" in fields
    assert fields["ended_at"] is not None  # wall-clock end stamped at exit
    assert fields["system_job_id"] == "dp-batch-xyz"
    assert fields["job_telemetry"] == {"total_wall_s": 12.0}


def test_run_job_records_failed_and_reraises(monkeypatch: Any) -> None:
    cap = _capture_job_io(monkeypatch)
    with pytest.raises(ValueError, match="boom"):
        with run_job("rid-0123456789ab", "ml", 2):
            raise ValueError("boom")

    assert cap["written"]["status"] == "RUNNING"
    assert len(cap["updates"]) == 1
    job_id, fields = cap["updates"][0]
    assert job_id == "sf-rid-0123456789ab-ml-a2"  # attempt reflected in the id
    assert fields["status"] == "FAILED"
    assert "runtime_seconds" in fields
    assert fields["ended_at"] is not None  # a crashed job still records its wall-clock end


def test_job_row_started_at_overrides_created_at() -> None:
    started = datetime(2026, 1, 2, 3, 5, 0, tzinfo=UTC)
    row = assemble_job_row("rid-0123456789ab", "ml", 1, _CREATED, started_at=started)
    assert row["started_at"] is started
    assert row["created_at"] is _CREATED


def test_run_job_contributor_mode_touches_nothing(monkeypatch: Any) -> None:
    cap = _capture_job_io(monkeypatch)
    with run_job("rid-0123456789ab", "statistical", 1, manage=False) as job:
        job.finalize(status="COMPLETED")
    assert cap["written"] is None
    assert cap["updates"] == []


# --- a cancellation is sticky against the failure it caused ---------------------
#
# Live 2026-09-02: `--cancel` stopped a Ray job and wrote CANCELLED, then the launching process saw
# its own job go STOPPED, read that as death, and finalized FAILED seventeen seconds later. These
# assert the guard travels with every non-green finalize; that it *works* is BigQuery's job, and
# the SQL it renders is asserted separately below.


def test_a_failing_job_will_not_overwrite_a_cancellation(monkeypatch: Any) -> None:
    cap = _capture_job_io(monkeypatch)
    with pytest.raises(ValueError, match="stopped"):
        with run_job("rid-0123456789ab", "deep_learning", 1):
            raise ValueError("ray job stopped")

    _, fields = cap["updates"][0]
    assert fields["status"] == "FAILED"
    assert fields["unless_status_in"] == ("CANCELLED",)


@pytest.mark.parametrize(
    ("status", "guarded"),
    [("FAILED", True), ("PARTIAL", True), ("COMPLETED", False)],
)
def test_only_a_non_green_finalize_carries_the_guard(
    monkeypatch: Any, status: str, guarded: bool
) -> None:
    # PARTIAL is guarded too: a run whose other families finished is finalized PARTIAL, not FAILED,
    # and that would erase a cancellation just as completely. COMPLETED is not — if the work
    # actually finished, the cancellation lost a race it had no claim to win.
    cap = _capture_job_io(monkeypatch)
    with run_job("rid-0123456789ab", "ml", 1) as job:
        job.finalize(status=status)

    _, fields = cap["updates"][0]
    assert fields["unless_status_in"] == (("CANCELLED",) if guarded else ())


def test_the_status_guard_renders_only_when_something_is_protected() -> None:
    assert render_status_guard(()) == ""
    guard = render_status_guard(("CANCELLED",))
    assert guard.startswith(" AND ")
    assert "@unless_status_in" in guard
    # NULL status must not silently drop the row from the update — SQL says NULL NOT IN (…) is
    # unknown, and a row with no status has nothing worth protecting.
    assert "status IS NULL" in guard


# --- artifact layout: composing the path, and reading it back -------------------


def test_artifact_uri_is_run_scoped_and_deterministic() -> None:
    root = "gs://bucket/warehouse/artifacts/proj/scale_forecasting"
    uri = artifacts.artifact_gcs_uri("/tmp/model.pkl", "my-run-abc123", root + "/")
    assert uri == f"{root}/my-run-abc123/model.pkl"
    # deterministic, and a trailing slash on the root makes no difference
    assert uri == artifacts.artifact_gcs_uri("/tmp/model.pkl", "my-run-abc123", root)


# --- split_gcs_uri ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("gs://bucket/warehouse/artifacts/proj/ds", ("bucket", "warehouse/artifacts/proj/ds")),
        ("gs://bucket/warehouse/", ("bucket", "warehouse")),
        ("gs://bucket", ("bucket", "")),
        ("gs://bucket/", ("bucket", "")),
    ],
)
def test_split_gcs_uri(uri, expected):
    assert artifacts.split_gcs_uri(uri) == expected


@pytest.mark.parametrize("bad", ["s3://bucket/x", "/local/path", "gs://", "bucket/x"])
def test_split_gcs_uri_rejects_non_gcs(bad):
    """A misconfigured warehouse root fails here, not as a confusing 404 from the client."""
    with pytest.raises(ValueError):
        artifacts.split_gcs_uri(bad)


# --- run_id_from_blob -------------------------------------------------------------


def test_run_id_is_the_first_segment_under_the_root():
    root = "warehouse/artifacts/proj/registry_ds"
    assert artifacts.run_id_from_blob(f"{root}/run_abc/model.pkl", root) == "run_abc"
    # Nested paths under a run still attribute to that run.
    assert artifacts.run_id_from_blob(f"{root}/run_abc/sub/dir/model.pkl", root) == "run_abc"


def test_objects_outside_the_root_are_not_attributed():
    """The scope guard: another registry's artifacts in the same bucket are invisible here."""
    root = "warehouse/artifacts/proj/registry_a"
    other = "warehouse/artifacts/proj/registry_b/run_abc/model.pkl"
    assert artifacts.run_id_from_blob(other, root) is None
    # A near-miss prefix must not match either.
    assert artifacts.run_id_from_blob("warehouse/artifacts/proj/registry_a2/r/m.pkl", root) is None


def test_an_object_directly_in_the_root_has_no_run():
    """No run segment ⇒ unattributable ⇒ never swept. Guessing here would delete real data."""
    root = "warehouse/artifacts/proj/ds"
    assert artifacts.run_id_from_blob(f"{root}/stray.txt", root) is None
    assert artifacts.run_id_from_blob(f"{root}/", root) is None


def test_an_empty_root_treats_the_first_segment_as_the_run():
    assert artifacts.run_id_from_blob("run_abc/model.pkl", "") == "run_abc"


# --- orphan_run_ids / blocking_runs -----------------------------------------------


def test_orphans_are_the_gcs_ids_the_registry_does_not_know():
    seen = ["run_c", "run_a", "run_b", "run_a"]
    known = ["run_a"]
    assert artifacts.orphan_run_ids(seen, known) == ("run_b", "run_c")


def test_no_orphans_when_every_prefix_has_a_row():
    assert artifacts.orphan_run_ids(["r1", "r2"], ["r1", "r2", "r3"]) == ()


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
    _msg, descriptor = _proto_for("backtest_oof", _OOF_SPEC)
    serialized = _encode_rows(
        _msg, _OOF_SPEC, [{"run_id": "r", "ts_id": "s", "model_type": "theta", "fold_id": 0}]
    )
    _append_via_write_api(client, "proj", "ds", "backtest_oof", descriptor, serialized)


def test_append_retries_transient_then_succeeds(monkeypatch: Any) -> None:
    from google.api_core.exceptions import InternalServerError

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)
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

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)
    client = _FakeWriteClient([ServiceUnavailable("503") for _ in range(_WRITE_RETRY_ATTEMPTS)])
    with pytest.raises(RegistryError, match="after 5 attempts"):
        _append(client)
    assert client.calls == _WRITE_RETRY_ATTEMPTS


def test_append_retries_routing_400_then_succeeds(monkeypatch: Any) -> None:
    from google.api_core.exceptions import BadRequest

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)
    # the intermittent Storage Write API routing glitch: a 400 that IS retryable despite being a 400
    client = _FakeWriteClient(
        [
            BadRequest("400 Cannot route on empty project id ''"),
            [_OkResponse()],
        ]
    )
    _append(client)  # must not raise — the routing 400 is retried
    assert client.calls == 2


def test_append_permanent_api_error_fails_fast(monkeypatch: Any) -> None:
    from google.api_core.exceptions import Forbidden

    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)
    # a non-transient GoogleAPICallError (e.g. real 403) is not retried
    client = _FakeWriteClient([Forbidden("403 permission denied")])
    with pytest.raises(RegistryError, match="failed:"):
        _append(client)
    assert client.calls == 1


def test_append_genuine_400_fails_fast(monkeypatch: Any) -> None:
    from google.api_core.exceptions import BadRequest

    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)
    # a real 400 (bad schema/proto) has no routing text → not retryable, fail on first call
    client = _FakeWriteClient([BadRequest("400 The proto field is incompatible with the column")])
    with pytest.raises(RegistryError, match="failed:"):
        _append(client)
    assert client.calls == 1


def test_append_response_level_error_fails_fast(monkeypatch: Any) -> None:
    from scale_forecasting.errors import RegistryError

    monkeypatch.setattr(write_api.time, "sleep", lambda *_a, **_k: None)

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


# --- accreting job_telemetry writes --------------------------------------------


def test_the_telemetry_merge_sets_each_path_against_the_existing_document() -> None:
    sql = render_header_telemetry_merge("p.d.run_registry", ["total_wall_s", "sizing.ml"])
    # IFNULL, so the first writer on a header with no telemetry yet merges into {} rather than
    # into NULL (which would swallow the write).
    assert sql.startswith("UPDATE `p.d.run_registry` SET job_telemetry = JSON_SET(")
    assert "IFNULL(job_telemetry, JSON '{}')" in sql
    assert "'$.total_wall_s', @t0" in sql
    assert "'$.sizing.ml', @t1" in sql
    assert sql.endswith("WHERE run_id=@run_id")


def test_each_family_files_its_sizing_under_its_own_path() -> None:
    # The whole point of the merge: two families of one run write different paths, so neither
    # overwrites the other.
    assert sizing_telemetry_path({"family": "deep_learning"}) == "sizing.deep_learning"
    assert sizing_telemetry_path({"family": "statistical"}) == "sizing.statistical"
    # A Ray CPU pool's merged label is "+"-joined; slugified so it is a legal JSON path.
    assert sizing_telemetry_path({"family": "statistical+ml"}) == "sizing.statistical_ml"
    # No family to file under (nothing was planned) still lands somewhere readable.
    assert sizing_telemetry_path({"family": None}) == "sizing.run"
    assert sizing_telemetry_path({}) == "sizing.run"


def test_an_illegal_telemetry_path_is_a_caller_bug_not_an_escaped_string() -> None:
    from scale_forecasting.errors import RegistryError

    # Every path is our own code's constant, so a path needing quotes means the caller is wrong.
    with pytest.raises(RegistryError, match="illegal telemetry path"):
        merge_header_telemetry("rid", {"sizing.'; DROP": {"x": 1}})


def test_merging_nothing_touches_nothing() -> None:
    # No client is constructed, so this would raise if the empty patch weren't short-circuited.
    merge_header_telemetry("rid", {})


# --- the native engine writes into the same specs, from its own builders --------
#
# `_encode_rows` walks the column *spec*, so a key a row builder emits that the spec does not name
# is dropped with no error -- the column simply reads NULL, and only for the engine that emitted it.
# The BigQuery-native family assembles its rows in `engines.bigquery_engine` rather than through
# `registry.rows`, so it is the one path where that can happen without any Python-worker test
# noticing. These pin both native builders against the specs they feed.


def _spec_columns(spec: tuple[tuple[str, str], ...]) -> set[str]:
    return {col for col, _ in spec}


def test_the_native_metadata_row_emits_no_column_the_spec_would_drop() -> None:
    from scale_forecasting.engines.bigquery_engine import _meta_row

    row = _meta_row(
        "my-run-abc123def456",
        "series-7",
        "arima_plus",
        dict.fromkeys(METRIC_COLUMNS, 0.5),
        '{"horizon": 28}',
        _CREATED,
        _cfg(),
    )
    assert set(row) <= _spec_columns(_META_SPEC), set(row) - _spec_columns(_META_SPEC)
    # The identity columns the leaderboard joins on have to be present *and* set, not merely legal.
    for col in ("run_id", "ts_id", "model_type", "compute_engine", "model_hash", "created_at"):
        assert row[col] is not None, col
    assert row["compute_engine"] == "bigquery"
    assert row["fold_id"] is None  # the rolled-up panel, not a per-fold row


def test_the_native_metadata_row_carries_every_metric_the_table_has() -> None:
    # The native engine fills the panel from `metrics.METRIC_NAMES` while the spec is generated from
    # `registry.rows.METRIC_COLUMNS`; a metric in one and not the other is a silently-NULL column.
    from scale_forecasting.engines.bigquery_engine import _meta_row

    panel = {name: float(i) for i, name in enumerate(METRIC_COLUMNS)}
    row = _meta_row("r", "s", "timesfm", panel, "{}", _CREATED, _cfg())
    assert {m: row[m] for m in METRIC_COLUMNS} == panel


def test_the_two_metric_vocabularies_are_the_same_vocabulary() -> None:
    # `metrics.METRIC_NAMES` is hand-written; `METRIC_COLUMNS` is derived from the config's
    # DecisionMetric literal. Nothing forces them to agree, and the native path spans both.
    from scale_forecasting.metrics import METRIC_NAMES

    assert METRIC_NAMES == METRIC_COLUMNS


def test_the_native_oof_row_emits_no_column_the_spec_would_drop() -> None:
    from scale_forecasting.engines.bigquery_engine import _oof_row

    row = _oof_row(
        "my-run-abc123def456",
        "series-7",
        "arima_plus",
        3,
        {"forecast_date": "2026-02-01", "y_true": 9.0, "yhat": 8.5, "yhat_lower": 7.0},
    )
    assert set(row) <= _spec_columns(_OOF_SPEC), set(row) - _spec_columns(_OOF_SPEC)
    assert row["fold_id"] == 3  # a real fold id, not the metadata row's NULL
    # The eval query returns interval columns too; backtest_oof does not carry them.
    assert "yhat_lower" not in row


def test_the_ensemble_metadata_row_emits_no_column_the_spec_would_drop() -> None:
    # The third independent producer of forecast_metadata rows, after registry.rows (Python worker)
    # and bigquery_engine (native). Same encoder, same silent-drop exposure.
    from scale_forecasting.ensemble_run import _ensemble_meta_row

    row = _ensemble_meta_row(
        run_id="my-run-abc123def456",
        ts_id="series-7",
        model_type="ensemble_nnls",
        panel=dict.fromkeys(METRIC_COLUMNS, 0.25),
        ensemble_id="ens-abc123",
        weights={"theta": 0.6, "sarimax": 0.4},
        artifact_uri="gs://bkt/warehouse/artifacts/p/d/r/ensemble_ens-abc123_nnls.pkl",
        created_at=_CREATED,
        cfg=_cfg(),
    )
    assert set(row) <= _spec_columns(_META_SPEC), set(row) - _spec_columns(_META_SPEC)
    assert row["compute_engine"] == "ensemble"
    assert row["ensemble_id"] == "ens-abc123"
    assert row["best_params"] == '{"sarimax": 0.4, "theta": 0.6}'  # sorted, so it is comparable
    assert {m: row[m] for m in METRIC_COLUMNS} == dict.fromkeys(METRIC_COLUMNS, 0.25)


def test_a_calculated_ensemble_is_distinguishable_from_one_that_fitted_nothing() -> None:
    # mean/median/inverse_error have no fitted parameters (weights=None -> best_params NULL); a
    # learned strategy that fitted an empty weight map is a different thing and must not read the
    # same. The truthiness shortcut would collapse them.
    from scale_forecasting.ensemble_run import _ensemble_meta_row

    common: dict[str, Any] = {
        "run_id": "r",
        "ts_id": "s",
        "model_type": "ensemble_mean",
        "panel": dict.fromkeys(METRIC_COLUMNS, 1.0),
        "ensemble_id": "e",
        "artifact_uri": None,
        "created_at": _CREATED,
        "cfg": _cfg(),
    }
    assert _ensemble_meta_row(weights=None, **common)["best_params"] is None
    assert _ensemble_meta_row(weights={}, **common)["best_params"] == "{}"


def test_all_three_metadata_producers_agree_on_the_identity_columns() -> None:
    # forecast_metadata is one table fed by three engines. The columns the leaderboard groups and
    # joins on have to be set by every one of them, or a whole engine drops off the comparison.
    from scale_forecasting.engines.bigquery_engine import _meta_row
    from scale_forecasting.ensemble_run import _ensemble_meta_row

    panel = dict.fromkeys(METRIC_COLUMNS, 1.0)
    rows = [
        assemble_metadata_row(_result(), _CREATED),
        _meta_row("r", "s", "arima_plus", panel, "{}", _CREATED, _cfg()),
        _ensemble_meta_row(
            run_id="r",
            ts_id="s",
            model_type="ensemble_mean",
            panel=panel,
            ensemble_id="e",
            weights=None,
            artifact_uri=None,
            created_at=_CREATED,
            cfg=_cfg(),
        ),
    ]
    for row in rows:
        for col in ("run_id", "ts_id", "model_type", "compute_engine", "model_hash", "created_at"):
            assert row.get(col) is not None, (col, row.get("compute_engine"))
        assert row["fold_id"] is None  # every producer's summary row is the fold_id=NULL one
    assert {r["compute_engine"] for r in rows} == {"spark", "bigquery", "ensemble"}


def test_both_native_builders_survive_the_encoder_they_feed() -> None:
    # The end of the contract: assemble a row, encode it against its spec, and get bytes back --
    # which is where a wrong *type* (as opposed to a wrong key) would surface.
    from scale_forecasting.engines.bigquery_engine import _meta_row, _oof_row

    meta = _meta_row(
        "r", "s", "arima_plus", dict.fromkeys(METRIC_COLUMNS, 1.0), "{}", _CREATED, _cfg()
    )
    oof = _oof_row(
        "r", "s", "arima_plus", 0, {"forecast_date": "2026-02-01", "y_true": 1.0, "yhat": 2.0}
    )
    for table, spec, rows in (
        ("forecast_metadata", _META_SPEC, [meta]),
        ("backtest_oof", _OOF_SPEC, [oof]),
    ):
        msg_cls, _ = _proto_for(table, spec)
        assert len(_encode_rows(msg_cls, spec, rows)) == 1


# --- choosing which prior harvest to size from ---------------------------------
#
# Live 2026-09-02, three times in one campaign: `source="auto"` resolved a 100k-series plan to a
# harvest measured on six series, because the six-series run was the most recent one. Small runs
# are frequent and large runs are rare, so recency-first systematically picks the worst evidence
# available.


def _cand(run_id: str, day: int, n_series: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "measured_at": datetime(2026, 9, day, tzinfo=UTC),
        "n_series": n_series,
    }


def test_a_closer_scale_beats_a_newer_run() -> None:
    candidates = [_cand("tiny-and-newest", 2, 6), _cand("big-and-older", 1, 100_000)]
    assert rank_harvest_candidates(candidates, target_series=100_000) == "big-and-older"


def test_recency_still_decides_between_equally_close_runs() -> None:
    candidates = [_cand("older", 1, 1_000), _cand("newer", 3, 1_000), _cand("middle", 2, 1_000)]
    assert rank_harvest_candidates(candidates, target_series=100_000) == "newer"


def test_distance_is_measured_in_log_space() -> None:
    # Linearly, 1k and 6 are both "about 100k away" from a 100k target; multiplicatively they are
    # two orders of magnitude apart, and that is the gap that matters for sizing.
    candidates = [_cand("six", 3, 6), _cand("one-thousand", 1, 1_000)]
    assert rank_harvest_candidates(candidates, target_series=100_000) == "one-thousand"


def test_over_large_and_over_small_are_trusted_equally() -> None:
    # 100 and 10_000 are both 10x from 1_000. Neither is preferred, so recency breaks the tie.
    candidates = [_cand("under", 1, 100), _cand("over", 2, 10_000)]
    assert rank_harvest_candidates(candidates, target_series=1_000) == "over"
    candidates = [_cand("under", 2, 100), _cand("over", 1, 10_000)]
    assert rank_harvest_candidates(candidates, target_series=1_000) == "under"


def test_an_exact_match_wins_outright() -> None:
    candidates = [_cand("newest", 5, 10), _cand("exact", 1, 500), _cand("close", 4, 400)]
    assert rank_harvest_candidates(candidates, target_series=500) == "exact"


@pytest.mark.parametrize("target", [None, 0])
def test_without_a_target_it_degrades_to_recency(target: int | None) -> None:
    """A config with no `series_limit` does not know its scale until the source is read."""
    candidates = [_cand("older", 1, 100_000), _cand("newer", 3, 6)]
    assert rank_harvest_candidates(candidates, target_series=target) == "newer"


def test_a_candidate_with_no_measured_count_is_not_ranked_out() -> None:
    # Distance is unknowable, so it sorts as zero rather than as infinitely far: excluding it would
    # be a harder judgement than the data supports.
    candidates = [_cand("unknown", 1, 0), _cand("far", 3, 6)]
    assert rank_harvest_candidates(candidates, target_series=100_000) == "unknown"


def test_no_candidates_means_no_pick() -> None:
    assert rank_harvest_candidates([], target_series=100_000) is None
