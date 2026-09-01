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
    from .settings import Settings

_log = get_logger(__name__)


def _batch_client(region: str) -> object:
    """A regional `BatchControllerClient` (Dataproc batches are a regional resource)."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.BatchControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


def _rfc3339_seconds(a: object, b: object) -> float | None:
    """Whole seconds between two Dataproc timestamp fields (``b - a``), or None.

    Dataproc stamps ``create_time``/``state_time`` as ``google.protobuf.Timestamp``; both expose
    ``.timestamp()`` (via the proto's datetime helper). Returns None if either is missing so a
    partial batch object degrades cleanly rather than raising.
    """
    ts_a = getattr(a, "timestamp", None)
    ts_b = getattr(b, "timestamp", None)
    if not callable(ts_a) or not callable(ts_b):
        return None
    try:
        return round(ts_b() - ts_a(), 1)
    except Exception:  # noqa: BLE001 - telemetry is best-effort, never fatal
        return None


def extract_job_telemetry(batch: object) -> dict[str, Any]:
    """Flatten a Dataproc ``Batch`` into the JSON-able telemetry dict stamped on the run header.

    Pure (no network): reads only fields already on the ``batch`` object that ``get_batch`` returns.
    Answers the operability questions the registry couldn't before — *how big was the cluster, did
    it autoscale, how much did it cost, and where did the wall-clock go* (provision + startup +
    closeout vs. our own ``runtime_seconds``):

    - ``total_wall_s`` — ``state_time − create_time``: the full provision→terminal wall-clock. The
      gap between this and the engine's ``runtime_seconds`` is Dataproc overhead (autoscaling
      warm-up + teardown), which amortizes as scale grows — the efficiency half of the scale story.
    - ``dcu_milli_seconds`` / ``shuffle_storage_gb_seconds`` — approximate usage (billing proxy +
      shuffle pressure).
    - ``driver_cores`` / ``executor_cores`` / ``executor_instances`` / ``max_executors`` /
      ``executor_memory`` / ``executor_memory_overhead`` — the resolved cluster sizing and the
      autoscaling cap (the executor throttle shows up here). This is the *echoed* shape — what
      Dataproc says it ran — as against the ``sizing`` record, which is what we asked for and why;
      the two disagreeing is a finding, so both are kept.
    - ``runtime_version`` / ``container_image`` — what actually ran (reproducibility).
    - ``service_account`` / ``subnetwork_uri`` — the identity + network the batch had access to.

    Every field is individually optional: a missing sub-message yields None for its keys, never a
    raise, so this is safe to call on any batch object the API returns.
    """
    tel: dict[str, Any] = {}

    tel["total_wall_s"] = _rfc3339_seconds(
        getattr(batch, "create_time", None), getattr(batch, "state_time", None)
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
