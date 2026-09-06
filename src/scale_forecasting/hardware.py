"""What hardware a job was actually **provisioned** onto — the one carrier, driver to worker.

A run can ask for a GPU two ways (the flat ``compute.use_gpu`` or ``compute.families.<f>.hardware``)
and the submitter resolves both into one answer before it buys anything. This module carries *that
answer* — the provisioned fact, not the config intent — to the process that fits a cell, so a model
can select its device explicitly instead of asking a library to guess.

**Why the provisioned fact and not the config.** ``resolve_family_compute`` is environment-blind: it
returns ``hardware="gpu"`` for a GPU config on a laptop with no card. Keying device selection on it
would break ``main.run configs/ray_gpu_demo.json`` on a workstation. Only the submitter knows what
was actually bought, so only the submitter may say so.

**Why an environment variable.** ``worker.run_cell`` runs on a Spark executor or a Ray task worker,
and a driver's ``os.environ`` does not reach either. The fleet already solves this exact problem for
the native-thread pin: `resources` writes ``spark.executorEnv.OMP_NUM_THREADS`` into the batch and
Ray exports it through ``runtime_env``, and `worker._intraop_threads` reads it back out of the
environment "because that is where the fleet actually sets it". Device provisioning travels the same
road for the same reason. One variable, one read site (`worker._model_context`).

**Job-level, not family-level, and that is sufficient.** The variable says only "this job has
devices". Which *model* may use one is then resolved per family by
``RunConfig.resolve_family_compute``, so a Ray job whose GPU pool serves ``deep_learning`` still
tells its statistical cells ``cpu``. A Spark batch is single-hardware anyway.

**Absent means auto**, which is byte-identical to every run made before this existed: a local run,
an SDK call, a notebook and a CPU batch all leave it unset and every model keeps asking its library
to choose. Only a GPU job sets it.

**The other half of the module is what is actually here.** ``provisioned_hardware`` answers what
was bought; `visible_device` answers what the process can see. Keeping them together is the point —
the whole GPU contract is about the gap between those two answers, and a run where they disagree is
exactly the run nobody noticed for twenty-one jobs.

Public surface: ``PROVISIONED_HARDWARE_ENV``, ``add_hardware_arg``, ``export_hardware_env``,
``hardware_args``, ``provisioned_hardware``, ``spark_executor_env``, ``ray_env_vars``,
``driver_fit_scope``, ``visible_device``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

# The single environment variable this module owns. Deliberately NOT in `_infra_args.INFRA_ARG_ENV`:
# that tuple maps `Settings` fields, and provisioned hardware is not a setting — it is a fact about
# one job, different for two jobs of the same run.
PROVISIONED_HARDWARE_ENV = "SF_PROVISIONED_HARDWARE"

# The only value that changes anything. "cpu" and absent are the same state (device selection stays
# "auto"), so a CPU job emits nothing and its command stays byte-identical to today's.
_GPU = "gpu"


def add_hardware_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--provisioned-hardware {cpu,gpu}`` on a launcher parser (default ``None``).

    Optional everywhere. A driver launched without it — every local run, every CPU batch, every
    command emitted before this flag existed — parses exactly as it did before.
    """
    parser.add_argument(
        "--provisioned-hardware",
        type=str,
        default=None,
        choices=("cpu", _GPU),
        help=f"hardware this job was provisioned onto; sets {PROVISIONED_HARDWARE_ENV}",
    )


def export_hardware_env(ns: argparse.Namespace) -> None:
    """Promote a parsed ``--provisioned-hardware`` into ``os.environ`` (only when set).

    The driver half of the carrier. The executor half is `spark_executor_env` / `ray_env_vars`,
    written into the job spec by the submitter — a driver's environment does not propagate.
    """
    value = getattr(ns, "provisioned_hardware", None)
    if value:
        os.environ[PROVISIONED_HARDWARE_ENV] = value


def hardware_args(hardware: str | None) -> list[str]:
    """The driver-arg fragment for a job's provisioned hardware — ``[]`` unless it is a GPU job.

    Emitting nothing for CPU is the point: every existing command is unchanged, so the only diff a
    reviewer sees in the emitted-command snapshots is on GPU jobs.
    """
    return ["--provisioned-hardware", _GPU] if hardware == _GPU else []


def provisioned_hardware() -> str | None:
    """What this process was told it is running on: ``"gpu"``, or ``None`` when nothing said.

    ``None`` is the honest answer for a local run, an SDK call and any job submitted before the
    flag existed — not ``"cpu"``, which would claim knowledge nobody supplied.
    """
    return os.environ.get(PROVISIONED_HARDWARE_ENV) or None


def spark_executor_env(hardware: str | None) -> dict[str, str]:
    """Spark properties that carry the fact to executors — ``{}`` unless it is a GPU job.

    ``spark.executorEnv.*`` is the same seam `resources.serverless` and `resources.cluster` already
    use for the thread pin, so this adds a key to an existing mechanism rather than a mechanism.
    """
    if hardware != _GPU:
        return {}
    return {f"spark.executorEnv.{PROVISIONED_HARDWARE_ENV}": _GPU}


def ray_env_vars(hardware: str | None) -> dict[str, str]:
    """Ray ``runtime_env.env_vars`` that carry the fact to task workers — ``{}`` on a CPU job.

    Job-level, so the head-node driver sees it too. That is why the two driver-side fit paths
    (`profiling.measure.measure_fit`, `hpo._score_params`) force their device back to ``auto``: the
    Ray head has no card, and a driver that honoured this would fail every profiled GPU run at
    submit.
    """
    if hardware != _GPU:
        return {}
    return {PROVISIONED_HARDWARE_ENV: _GPU}


_visible: tuple[str, str | None] | None = None  # memoized; a device does not appear mid-process


def visible_device() -> tuple[str, str | None]:
    """What this process can actually see: ``(availability, device name)``.

    Availability is one of three words and the third one is load-bearing. ``"cuda"`` means a CUDA
    device is present and named. ``"cpu"`` means torch is here and reports no device — a positive
    finding. ``"unknown"`` means torch could not be imported at all, so nobody asked anything and
    the honest answer is that we do not know; a CPU-only worker with no tensor library is not
    evidence that a GPU job lost its card.

    This exists because ``peak_gpu_bytes`` cannot answer the question. Its ``None`` is overloaded
    across four different causes — no torch, no CUDA build, no device, and profiling off — so a
    reader cannot tell "the accelerator never attached" from "nobody looked". Layer 4 records the
    device, it does not infer it.

    Memoized for the same reason `worker._peak_gpu_bytes` is: a *failed* ``import torch`` is not
    cached in ``sys.modules``, so at a hundred thousand cells an unmemoized probe would re-walk
    ``sys.path`` a hundred thousand times.
    """
    global _visible
    if _visible is None:
        _visible = _probe_visible_device()
    return _visible


def _probe_visible_device() -> tuple[str, str | None]:
    """The uncached probe (see `visible_device`). Torch is imported lazily and never required.

    ``hardware`` is on the lean launch path — `commands` and `_entry` import it — so a top-level
    tensor-library import here would put torch in front of every submit.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - no tensor library is not an error, it is an answer
        return ("unknown", None)
    try:
        if not torch.cuda.is_available():
            return ("cpu", None)
        return ("cuda", str(torch.cuda.get_device_name(0)))
    except Exception:  # noqa: BLE001 - a broken CUDA build must not sink a cell
        return ("unknown", None)


@contextmanager
def driver_fit_scope() -> Iterator[None]:
    """Run a fit as if nothing were provisioned, then restore — for driver-side fits.

    The sizing pre-pass fits on the *driver*: a Ray head node, a Spark driver, an Airflow worker.
    None of them has an accelerator, and Lightning raises
    ``MisconfigurationException: No supported gpu backend found!`` rather than degrading, so a
    driver that inherited the job's ``gpu`` would sink every profiled GPU run before a single cell
    ran. Restoring on exit matters for the same reason `profiling.measure._pinned_threads` restores
    the thread pin: the driver goes on to do real work afterwards.
    """
    previous = os.environ.pop(PROVISIONED_HARDWARE_ENV, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[PROVISIONED_HARDWARE_ENV] = previous
