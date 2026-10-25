"""Reach a running or finished Dataproc batch, and read what it says.

The other half of the batch story from `submit`: not "launch one" but "what is it doing, what did
it cost, and can I stop it". Two callers want exactly that, which is why it is its own module —
`submit.submit_batch` stamps the record once the batch is terminal, and `probes.runtimes` reads (and
cancels) one that is still in flight. Before this split the prober had to import two private names
out of the submitter to do it.

`_batch_client` is the handle, `extract_job_telemetry` is the pure read off the object that handle
returns, and `_stamp_job_telemetry` files the result on the run header. The extractor is pure and
the stamper does I/O, but they are one capability and are kept together on purpose: the extractor
has no other reason to exist, and separating them by testability would put a function and its only
writer in different files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import get_logger

if TYPE_CHECKING:
    from google.cloud.dataproc_v1 import BatchControllerClient

    from .settings import Settings

_log = get_logger(__name__)


def _batch_client(region: str) -> BatchControllerClient:
    """A regional `BatchControllerClient` (Dataproc batches are a regional resource)."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.BatchControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


def _to_epoch_seconds(val: object) -> float | None:
    """Coerce a protobuf ``Timestamp``, ``datetime``, epoch int/float, or ISO-8601 string."""
    if val is None or isinstance(val, bool):
        return None
    ts_fn = getattr(val, "timestamp", None)
    if callable(ts_fn):
        try:
            res = ts_fn()
            return float(res) if isinstance(res, int | float) else None
        except Exception:  # noqa: BLE001 - telemetry is best-effort, never fatal
            return None
    if isinstance(val, int | float):
        fval = float(val)
        # Ray JobInfo timestamps are epoch milliseconds (> 1e11 for modern dates).
        return fval / 1000.0 if fval > 1e11 else fval
    if isinstance(val, str) and val.strip():
        from datetime import datetime

        raw = val.strip()
        try:
            return (
                float(raw)
                if raw.replace(".", "", 1).isdigit()
                else datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            )
        except Exception:  # noqa: BLE001 - malformed timestamp string degrades cleanly
            return None
    return None


def _rfc3339_seconds(a: object, b: object) -> float | None:
    """Seconds between two timestamps (``b - a``), or None.

    Accepts ``google.protobuf.Timestamp`` / ``datetime`` (``.timestamp()``), epoch seconds or
    milliseconds (``int`` / ``float``), and ISO-8601 / RFC-3339 strings (such as Kubernetes
    ``creationTimestamp`` / ``startTime`` / ``completionTime``). Returns None if either is missing
    so a partial job object degrades cleanly rather than raising.
    """
    ea = _to_epoch_seconds(a)
    eb = _to_epoch_seconds(b)
    if ea is None or eb is None:
        return None
    try:
        return round(max(0.0, eb - ea), 1)
    except Exception:  # noqa: BLE001 - telemetry is best-effort, never fatal
        return None


def _running_start(batch: object) -> object | None:
    """When the batch entered ``RUNNING``, from ``state_history``; None if it never did.

    Dataproc keeps the *previous* states on ``state_history`` (``state`` / ``state_time`` hold the
    current one), each with a ``state_start_time``. A batch that failed in ``PENDING`` has no
    ``RUNNING`` entry, and then there is no provisioning figure to report — None, not zero.
    """
    for entry in getattr(batch, "state_history", None) or ():
        state = getattr(entry, "state", None)
        if (getattr(state, "name", None) or str(state)) == "RUNNING":
            return getattr(entry, "state_start_time", None)
    return None


def platform_phases(
    create_time: object, start_time: object, end_time: object
) -> dict[str, float | None]:
    """The provision / execute split of one platform job, from its three timestamps.

    ``provision_s`` is ``start_time − create_time`` — what the platform spent finding and
    starting capacity before any of our code ran; ``execute_s`` is ``end_time − start_time`` — the
    part the forecasting work itself took. Together they explain ``total_wall_s``, which is the
    figure the ``run_jobs`` bracket already had; the split is what turns "this job took 11 minutes"
    into "8 of them were waiting for a GPU", which is the question that decides whether a family
    should move runtime, region, or machine shape. Each figure is None when its timestamps are
    missing, so a job that never started reports no execute time rather than a negative one.

    Pure, and shared by every runtime that exposes the three moments: a Dataproc ``Batch``
    (``create_time`` / the ``RUNNING`` ``state_history`` entry / ``state_time``) and a Vertex
    ``CustomJob`` (``create_time`` / ``start_time`` / ``end_time``). Stamped per job under
    ``run_jobs.job_telemetry.$.platform`` by `job_launch.launch_family_job` and read back as
    ``provision_seconds`` / ``execute_seconds`` on ``v_run_jobs``.
    """
    return {
        "provision_s": _rfc3339_seconds(create_time, start_time),
        "execute_s": _rfc3339_seconds(start_time, end_time),
        "total_wall_s": _rfc3339_seconds(create_time, end_time),
    }


PLATFORM_PHASE_KEYS = ("provision_s", "execute_s", "total_wall_s")


def extract_job_telemetry(batch: object) -> dict[str, Any]:
    """Flatten a Dataproc ``Batch`` into the JSON-able telemetry dict stamped on the run header.

    Pure (no network): reads only fields already on the ``batch`` object that ``get_batch`` returns.
    Answers the operability questions the registry couldn't before — *how big was the cluster, did
    it autoscale, how much did it cost, and how long did the platform hold it*:

    - ``total_wall_s`` — ``state_time − create_time``: this batch's full provision→terminal
      wall-clock as Dataproc saw it. It is one family's figure and it lands on the run **header**
      (last batch to finish wins), so it is a per-batch fact, not a run-level one; the run's
      overhead is derived from the ``run_jobs`` brackets in ``v_run_summary`` instead
      (`registry.views`).
    - ``provision_s`` / ``execute_s`` — the same wall split at the ``RUNNING`` transition
      (`platform_phases`): how long Dataproc took to stand the batch up, and how long the work ran
      once it had. None when the batch never reached ``RUNNING``.
    - ``dcu_milli_seconds`` / ``shuffle_storage_gb_seconds`` — approximate usage (billing proxy +
      shuffle pressure).
    - ``driver_cores`` / ``executor_cores`` / ``task_cpus`` / ``executor_instances`` /
      ``max_executors`` / ``executor_memory`` / ``executor_memory_overhead`` — the resolved cluster
      sizing and the autoscaling cap (the executor throttle shows up here). This is the *echoed*
      shape — what Dataproc says it ran — as against the ``sizing`` record, which is what we asked
      for and why; the two disagreeing is a finding, so both are kept.

      ``spark.sql.shuffle.partitions`` is deliberately **not** among them, even though it is the
      number that decides how many tasks the fan-out actually runs. The driver sets it on the live
      session (`engines.spark_explode._widen_fanout`) after the batch is created, so the batch's
      ``runtime_config`` never carries it and reading it here would report ``None`` on every run
      that pinned it — a field that is always empty reads as "not set" rather than "asked
      elsewhere". The executed fan-out is stamped from the driver instead, where it is known.
    - ``runtime_version`` / ``container_image`` — what actually ran (reproducibility).
    - ``service_account`` / ``subnetwork_uri`` — the identity + network the batch had access to.

    Every field is individually optional: a missing sub-message yields None for its keys, never a
    raise, so this is safe to call on any batch object the API returns.
    """
    tel: dict[str, Any] = {}

    # `total_wall_s` is create→terminal whether or not the batch ever ran; the split beside it is
    # None on a batch that failed in PENDING.
    tel.update(
        platform_phases(
            getattr(batch, "create_time", None),
            _running_start(batch),
            getattr(batch, "state_time", None),
        )
    )

    runtime_info = getattr(batch, "runtime_info", None)
    usage = getattr(runtime_info, "approximate_usage", None) if runtime_info else None
    tel["dcu_milli_seconds"] = (
        int(getattr(usage, "milli_dcu_seconds", 0)) or None if usage else None
    )
    tel["shuffle_storage_gb_seconds"] = (
        int(getattr(usage, "shuffle_storage_gb_seconds", 0)) or None if usage else None
    )

    runtime_config = getattr(batch, "runtime_config", None)
    props = dict(getattr(runtime_config, "properties", {}) or {}) if runtime_config else {}

    def _prop_int(key: str) -> int | None:
        raw = props.get(key)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    tel["driver_cores"] = _prop_int("spark.driver.cores")
    tel["executor_cores"] = _prop_int("spark.executor.cores")
    tel["executor_instances"] = _prop_int("spark.executor.instances")
    tel["max_executors"] = _prop_int("spark.dynamicAllocation.maxExecutors")
    # `executor_cores` alone does not say how many cells ran at once. An executor runs
    # ``executor.cores / task.cpus`` tasks concurrently, and the thread pins are exported at
    # ``task.cpus`` — so without this number the echoed shape cannot be turned back into either a
    # concurrency or a thread budget, and 16 cores at 1 cpu-per-task reads identically to 16 cores
    # at 4. It is also the divisor `reachable_bucket_count` used, so it is what makes the bucket
    # count on the driver side reproducible from the record.
    tel["task_cpus"] = _prop_int("spark.task.cpus")
    # The memory half of the resolved shape, and the only half a profile actually moves (§3.10):
    # cores and the executor counts follow from fan-out with or without evidence. Strings, because
    # Spark spells them ``"8g"`` / ``"3891m"`` — kept verbatim rather than parsed to bytes so the
    # record shows what the platform was told, not our reading of it. Absent on a batch that left
    # Serverless' own defaults standing, which is itself the answer to "what sized this".
    tel["executor_memory"] = props.get("spark.executor.memory") or None
    tel["executor_memory_overhead"] = props.get("spark.executor.memoryOverhead") or None
    tel["runtime_version"] = (
        getattr(runtime_config, "version", None) or None if runtime_config else None
    )
    tel["container_image"] = (
        getattr(runtime_config, "container_image", None) or None if runtime_config else None
    )
    # The other way a batch can get its dependencies (`serverless_dep_properties`). Exactly one of
    # these two is set on any batch, so the pair reads as "which envelope delivered the env" — and a
    # batch with neither is one running against the stock runtime's Python, which is a finding.
    tel["venv_archive"] = props.get("spark.archives") or None

    env = getattr(batch, "environment_config", None)
    exec_cfg = getattr(env, "execution_config", None) if env else None
    tel["service_account"] = (
        getattr(exec_cfg, "service_account", None) or None if exec_cfg else None
    )
    tel["subnetwork_uri"] = getattr(exec_cfg, "subnetwork_uri", None) or None if exec_cfg else None

    return tel


def _stamp_job_telemetry(
    client: Any,
    parent: str,
    batch_id: str,
    run_id: str,
    settings: Settings,
    *,
    sizing: dict[str, Any] | None = None,
) -> None:
    """Read the finished batch's telemetry and write it to the run header (best-effort).

    A fresh ``get_batch`` (the LRO result can carry incomplete ``approximate_usage``) → the pure
    `extract_job_telemetry` → the header's ``job_telemetry``. The header column is a
    native ``JSON`` type whose query parameter serializes the value itself, so we pass the telemetry
    **dict** (not a pre-serialized string, which would double-encode). Wrapped so any failure (API
    error, missing field, header not yet written) is logged and swallowed: telemetry is a
    nice-to-have overlay on an already-complete run, never a reason to fail it.

    ``sizing`` (`plan_sizing`'s second half) rides along under ``$.sizing.<family>``. It is decided
    at *submit* and stamped at *finish* so one write carries both halves, and a batch that never
    reaches terminal has no telemetry worth reading anyway.

    The write **merges** (`registry.header.merge_header_telemetry`) rather than replacing the
    column: several family jobs of one run each land here, and a whole-column write would leave only
    whichever finished last.
    """
    from .registry.header import merge_header_telemetry, sizing_telemetry_path

    try:
        fetched = client.get_batch(name=f"{parent}/batches/{batch_id}")
        telemetry = extract_job_telemetry(fetched)
        if sizing:
            telemetry[sizing_telemetry_path(sizing)] = sizing
        merge_header_telemetry(run_id, telemetry, settings=settings)
        _log.info("batch %s telemetry stamped: %s", batch_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("batch %s telemetry capture failed (non-fatal): %r", batch_id, exc)


def batch_client(*args: Any, **kwargs: Any) -> Any:
    """Shared internal helper delegating to `_batch_client`."""
    return _batch_client(*args, **kwargs)


def stamp_job_telemetry(*args: Any, **kwargs: Any) -> Any:
    """Shared internal helper delegating to `_stamp_job_telemetry`."""
    return _stamp_job_telemetry(*args, **kwargs)
