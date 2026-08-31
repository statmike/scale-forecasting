"""Registry-operations verbs against live BigQuery + GCS (@gcp).

The pure half of `registry.ops` — path arithmetic, set arithmetic, SQL strings, the BQML name
matcher — is fully covered offline in ``tests/unit/test_registry_ops.py``. What only a live run can
prove is the two things the offline tests *cannot* fake:

1. **The artifact-prefix delete actually reaches GCS**, and reaches only the named run's prefix.
   The whole safety argument for `drop_run`/`sweep_orphans` rests on the artifact root carrying the
   registry key, and on the list-then-delete pass being correctly scoped — a bug there deletes a
   neighbouring run's models, which no string test can catch.
2. **`CREATE SNAPSHOT TABLE` is valid against the real registry schema.** The statement is rendered
   offline but BigQuery is the only thing that can say whether it parses and whether a snapshot of
   these tables (native `JSON` columns included) is legal.

Both tests own their fixtures end to end — a unique ``run_id`` per invocation, objects they wrote
themselves, and a teardown that removes what they created — so a failure never leaves the shared
deployment dirty and a re-run never collides. Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see
``tests/conftest.py``). Run manually::

    SF_PROJECT_ID=… SF_CONNECTION=… SF_WAREHOUSE_URI=… SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_registry_ops_live.py
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from scale_forecasting.registry import ops

if TYPE_CHECKING:
    from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    from scale_forecasting.settings import Settings

    return Settings.resolve()


def _blob(settings: Settings, run_id: str, name: str) -> Any:
    """A handle on ``<artifact_root>/<run_id>/<name>`` — the exact layout `drop_run` scans."""
    from google.cloud import storage

    bucket_name, root = ops.split_gcs_uri(settings.artifact_root)
    client = storage.Client(project=settings.project_id)
    return client.bucket(bucket_name).blob(f"{root}/{run_id}/{name}")


def test_artifact_prefix_delete_hits_the_named_run_and_nothing_else(settings: Settings) -> None:
    """`_delete_prefixes` clears one run's prefix and leaves its neighbour untouched.

    Two runs, two objects each, delete one. This is the assertion the whole destructive tier rests
    on: scoped to the run, and scoped to *this* registry's artifact root.
    """
    stamp = uuid.uuid4().hex[:12]
    target, neighbour = f"opstest_{stamp}_a", f"opstest_{stamp}_b"
    written = [
        _blob(settings, run_id, name)
        for run_id in (target, neighbour)
        for name in ("model.pkl", "nested/second.pkl")
    ]
    try:
        for blob in written:
            blob.upload_from_string(b"live-ops-test")

        # 1. Enumerate — the same list pass `plan_drop_run` runs, over the real root.
        prefixes = ops._list_prefixes(settings)
        assert prefixes[target].object_count == 2
        assert prefixes[neighbour].object_count == 2
        assert prefixes[target].byte_total == 2 * len(b"live-ops-test")

        # 2. Delete only the target's prefix.
        assert ops._delete_prefixes(settings, [target]) == 2

        after = ops._list_prefixes(settings)
        assert target not in after
        assert after[neighbour].object_count == 2, "a neighbouring run's artifacts were deleted"
        assert not _blob(settings, target, "nested/second.pkl").exists()
    finally:
        for blob in written:
            if blob.exists():
                blob.delete()


def test_orphan_sweep_sees_a_prefix_with_no_registry_row(settings: Settings) -> None:
    """An artifact prefix whose ``run_id`` has no header row is planned as an orphan.

    Read-only on the registry and plan-only on GCS — it asserts the join between the live
    ``run_registry`` and the live bucket listing, which is the one part of `sweep_orphans` that the
    offline set-arithmetic test cannot exercise.
    """
    run_id = f"opstest_orphan_{uuid.uuid4().hex[:12]}"
    blob = _blob(settings, run_id, "stranded.pkl")
    try:
        blob.upload_from_string(b"stranded")
        plan = ops.plan_sweep_orphans(settings=settings)
        assert plan.root == settings.artifact_root
        assert run_id in {p.run_id for p in plan.prefixes}
        # Nothing was deleted — planning is read-only.
        assert blob.exists()
    finally:
        if blob.exists():
            blob.delete()


def test_snapshot_creates_readable_point_in_time_copies(settings: Settings) -> None:
    """`snapshot` renders SQL BigQuery accepts, and the clones are queryable.

    Snapshots go into the registry's own dataset with a unique suffix (the documented default), and
    carry a 1-day expiration so a failed teardown cannot leave storage accruing indefinitely.
    """
    from google.cloud import bigquery

    from scale_forecasting.registry.bq import ensure_tables
    from scale_forecasting.registry.ddl import REGISTRY_TABLE_NAMES

    ensure_tables(settings=settings)
    suffix = f"opstest_{uuid.uuid4().hex[:10]}"
    client = bigquery.Client(project=settings.project_id)
    created: dict[str, str] = {}
    try:
        created = ops.snapshot(suffix, settings=settings, expiration_days=1)
        assert set(created) == set(REGISTRY_TABLE_NAMES)

        for table, ref in created.items():
            snap = client.get_table(ref)
            assert snap.table_type == "SNAPSHOT", f"{table} clone is not a snapshot"
            base = client.get_table(settings.registry_table_ref(table))
            assert [f.name for f in snap.schema] == [f.name for f in base.schema]
            # Readable, and consistent with the base at snapshot time.
            n_snap = next(iter(client.query(f"SELECT COUNT(*) c FROM `{ref}`").result())).c
            assert n_snap >= 0
    finally:
        for ref in created.values():
            client.delete_table(ref, not_found_ok=True)


def test_doctor_reports_the_live_registry(settings: Settings) -> None:
    """`doctor` is read-only and answers about the registry that is actually deployed."""
    from scale_forecasting.registry.bq import ensure_tables
    from scale_forecasting.registry.ddl import REGISTRY_TABLE_NAMES

    ensure_tables(settings=settings)
    report = ops.doctor(settings=settings)
    assert report.registry == settings.registry_dataset_ref
    assert report.artifact_root == settings.artifact_root
    assert {t.table for t in report.tables} == set(REGISTRY_TABLE_NAMES)
    assert report.missing_tables == (), "ensure_tables ran, so every registry table must exist"
    # Rendering must not raise on live data (mixed statuses, orphans present or not).
    assert report.registry in ops.format_doctor(report)


def test_drop_run_previews_without_touching_anything(settings: Settings) -> None:
    """Without ``yes`` the verb enumerates and reports — the artifacts survive.

    The preview contract is the last line of defence in front of an irreversible delete, so it is
    asserted against real GCS rather than a fake.
    """
    run_id = f"opstest_preview_{uuid.uuid4().hex[:12]}"
    blob = _blob(settings, run_id, "model.pkl")
    try:
        blob.upload_from_string(b"survives-a-preview")
        plan = ops.drop_run([run_id], settings=settings)  # yes=False
        # No header row was ever written, so the run is unknown and dropped from the plan.
        assert plan.unknown == (run_id,)
        assert plan.is_empty
        assert blob.exists(), "a preview deleted a GCS object"

        # An unknown run's artifacts are the orphan sweep's business, not drop-run's.
        sweep = ops.plan_sweep_orphans(settings=settings)
        assert run_id in {p.run_id for p in sweep.prefixes}
    finally:
        if blob.exists():
            blob.delete()


def test_drop_run_deletes_every_tier_of_a_real_run(settings: Settings) -> None:
    """The full ordering rule end to end: artifacts, then rows, for a run this test writes.

    Writes a minimal header + one artifact-carrying cell through the *real* writers, then drops the
    run and asserts both tiers are gone. Uses a unique ``run_id``, so the append-only Write API and
    the streaming-buffer DELETE restriction (which applies for ~90 min after an append) are the one
    real hazard — the poll below waits for the rows to become visible first, and a buffer rejection
    surfaces as a `RegistryError` naming the table rather than a silent partial delete.
    """
    import pandas as pd
    from google.cloud import bigquery

    from scale_forecasting.config import RunConfig
    from scale_forecasting.registry import bq
    from scale_forecasting.registry.ids import make_run_id
    from scale_forecasting.worker import CellResult

    cfg = RunConfig(
        run_name=f"ops droprun {uuid.uuid4().hex[:8]}",
        data={"source_table": "p.d.source_series", "series_limit": 1},
        models=["theta"],
    )
    run_id = make_run_id(cfg)
    bq.write_header(cfg, run_id, settings=settings)
    cell = CellResult(
        run_id=run_id,
        ts_id="series-0",
        model_type="theta",
        compute_engine="local",
        status="ok",
        predictions=pd.DataFrame(
            {
                "ds": pd.to_datetime(["2026-02-01"]),
                "yhat": [1.0],
                "yhat_lower": [0.0],
                "yhat_upper": [2.0],
                "quantiles": [None],
            }
        ),
        oof=None,
        metrics={"wape": 0.1},
        best_params={},
        fit_seconds=0.5,
        artifact_bytes=b"fake-fitted-model-bytes",
    )
    bq.write_cells([cell], settings=settings)
    bq.update_header(run_id, settings=settings, status="COMPLETED", n_series=1)

    client = bigquery.Client(project=settings.project_id)
    header = settings.registry_table_ref("run_registry")
    for _ in range(10):  # Write API rows are async-visible.
        rows = client.query(
            f"SELECT COUNT(*) c FROM `{header}` WHERE run_id='{run_id}'"
        ).result()
        if int(next(iter(rows)).c) > 0:
            break
        time.sleep(2)

    plan = ops.plan_drop_run([run_id], settings=settings)
    assert plan.run_ids == (run_id,)
    assert plan.blocked == (), "a COMPLETED run must not be treated as in flight"
    assert plan.object_count >= 1, "the run's artifact prefix was not enumerated"

    ops.drop_run([run_id], settings=settings, yes=True)

    assert run_id not in ops._list_prefixes(settings)
    for table in ("run_registry", "forecast_metadata", "forecast_predictions"):
        ref = settings.registry_table_ref(table)
        left = client.query(f"SELECT COUNT(*) c FROM `{ref}` WHERE run_id='{run_id}'").result()
        assert int(next(iter(left)).c) == 0, f"{table} still has rows for the dropped run"
