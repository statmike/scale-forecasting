"""Layer 3 of the GPU contract: the cell fits on the device it was told to, and says which.

Layers 1–2 (`test_gpu_preflight`) settle what a *config* may ask for. This module covers what
happens at runtime, which is a different question with a different failure: a job can be
provisioned onto accelerators and still fit every cell on the CPU, because ``accelerator="auto"``
can never fail. Across the Phase 0 measurement — 31,356 NeuralProphet fits on live T4s — peak
device memory was 50–78 KB against a 17 GB card and ``cpu_seconds / fit_seconds`` was 0.93–0.996.
Every GPU run in the ledger was in substance a CPU run, and nothing reported it.

Layer 3 replaces the guess with a statement, and the statement needs two facts that live in
different places. What the job was **provisioned** onto is known only to the submitter and travels
in one environment variable (`hardware`), because `worker.run_cell` runs on Spark executors and Ray
task workers where a driver's ``os.environ`` does not reach. Which **family** may use a device is
config, resolved by ``RunConfig.resolve_family_compute``. `worker._resolve_device` is where the two
meet.

Three properties are load-bearing and each has its own section below:

* **Absent means auto.** A local run, an SDK call, a notebook and every CPU job set nothing, and
  every command, batch, job and ``runtime_env`` they build is byte-identical to the pre-Layer-3
  one. Only a GPU job differs, and the diff is exactly two things.
* **The two facts must agree.** Either alone breaks something: without the job half a GPU config
  crashes on a laptop, without the family half a mixed-hardware Dataproc cluster hands a
  statistical cell the card its executor happens to expose.
* **The driver is not a worker.** The sizing pre-pass and every HPO trial fit on a Ray head node, a
  Spark driver or an Airflow worker — none of which has an accelerator — so both force their way
  back to ``auto`` or a profiled GPU run dies before a single cell runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting import hardware
from scale_forecasting.code_delivery import build_runtime_env
from scale_forecasting.commands import build_driver_args, build_spark_commands
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.models import get_model
from scale_forecasting.models.base_model import BaseModel, ModelContext
from scale_forecasting.ray_submit import build_entrypoint
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings
from scale_forecasting.worker import (
    _model_context,
    _require_device,
    _resolve_device,
    run_cell,
)

HORIZON = 7

# A run that asks for accelerators: T4 is Ray-only (Serverless is L4-only), which is also the
# shape the deep-learning family actually runs in.
_GPU_COMPUTE: dict[str, Any] = {
    "python_runtime": "ray",
    "compute": {"use_gpu": True, "gpu_type": "T4"},
}


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "device selection test",
        "data": {"source_table": "t", "freq": "D", "horizon": HORIZON},
        "models": ["neuralprophet"],
    }
    base.update(over)
    return RunConfig(**base)


def _settings() -> Settings:
    return Settings(
        project_id="proj-x",
        connection="proj-x.us-central1.conn",
        warehouse_uri="gs://bkt/warehouse",
        dataset_id="ds_x",
        region="us-central1",
    )


def _series(n: int = 140, ts_id: str = "s0") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    y = np.linspace(10.0, 30.0, n) + 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": y})


@pytest.fixture(autouse=True)
def _unprovisioned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from "nothing said" — the state a laptop and a CPU job are in.

    Autouse because the variable is process-global: a test that set it and a test that reads it
    could otherwise pass or fail on collection order. The fault-injection switch is cleared for
    the same reason — a developer with it exported would otherwise see a different suite.
    """
    monkeypatch.delenv(hardware.PROVISIONED_HARDWARE_ENV, raising=False)
    monkeypatch.delenv(hardware.HIDE_DEVICES_ENV, raising=False)


# --- the carrier: what a job puts on the wire ----------------------------------


@pytest.mark.parametrize("value", [None, "cpu"])
def test_a_job_that_is_not_on_gpu_hardware_says_nothing_at_all(value: str | None) -> None:
    """ "cpu" and absent are one state, so neither may add a byte to a command or a job spec."""
    assert hardware.hardware_args(value) == []
    assert hardware.spark_executor_env(value) == {}
    assert hardware.ray_env_vars(value) == {}


def test_a_gpu_job_says_so_on_all_three_seams() -> None:
    """The driver arg, the Spark executor property and the Ray env var carry one fact."""
    assert hardware.hardware_args("gpu") == ["--provisioned-hardware", "gpu"]
    assert hardware.spark_executor_env("gpu") == {
        f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}": "gpu"
    }
    assert hardware.ray_env_vars("gpu") == {hardware.PROVISIONED_HARDWARE_ENV: "gpu"}


def test_nothing_in_the_environment_reads_as_nothing_known() -> None:
    """``None``, not ``"cpu"`` — claiming CPU would assert knowledge nobody supplied."""
    assert hardware.provisioned_hardware() is None


def test_the_flag_reaches_the_engine_through_the_driver_entry(tmp_path: Any) -> None:
    """End-to-end through ``run_entry``: what the submitter passed is what the engine runs under."""
    from scale_forecasting import _entry

    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(_cfg().model_dump(mode="json")))
    seen: list[str | None] = []

    def _resolve(ns: Any) -> tuple[Any, str]:
        def _run(cfg: RunConfig, **_: Any) -> None:
            seen.append(hardware.provisioned_hardware())

        return _run, "stub"

    _entry.run_entry(
        ["--config-uri", str(config_path), "--provisioned-hardware", "gpu"],
        prog="stub",
        description="stub",
        resolve_engine=_resolve,
    )
    assert seen == ["gpu"]


def test_a_driver_launched_without_the_flag_parses_as_it_always_did(tmp_path: Any) -> None:
    """Every command emitted before this flag existed still runs, and stays unprovisioned."""
    from scale_forecasting import _entry

    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(_cfg().model_dump(mode="json")))
    seen: list[str | None] = []

    def _resolve(ns: Any) -> tuple[Any, str]:
        def _run(cfg: RunConfig, **_: Any) -> None:
            seen.append(hardware.provisioned_hardware())

        return _run, "stub"

    _entry.run_entry(
        ["--config-uri", str(config_path)],
        prog="stub",
        description="stub",
        resolve_engine=_resolve,
    )
    assert seen == [None]


def test_an_unknown_hardware_word_is_refused_at_the_parser() -> None:
    """The vocabulary is closed: a typo must not silently read as "not gpu"."""
    from scale_forecasting._entry import build_parser

    parser = build_parser("stub", "stub")
    with pytest.raises(SystemExit):
        parser.parse_args(["--config-uri", "gs://x", "--provisioned-hardware", "tpu"])


# --- absent means auto: the emitted artifacts are unchanged --------------------


def test_a_cpu_job_builds_the_exact_driver_args_it_built_before() -> None:
    baseline = build_driver_args("gs://b/c.json", _settings())
    for value in (None, "cpu"):
        assert (
            build_driver_args("gs://b/c.json", _settings(), provisioned_hardware=value) == baseline
        )


def test_a_gpu_job_adds_the_flag_and_nothing_else() -> None:
    baseline = build_driver_args("gs://b/c.json", _settings())
    args = build_driver_args("gs://b/c.json", _settings(), provisioned_hardware="gpu")
    assert args == [*baseline, "--provisioned-hardware", "gpu"]


def _spark_commands(**over: Any) -> Any:
    pytest.importorskip("google.cloud.dataproc_v1")
    from scale_forecasting.batch_infra import BatchInfra

    return build_spark_commands(
        settings=_settings(),
        infra=BatchInfra(
            code_bucket="code-bkt",
            container_image="us-docker.pkg.dev/proj-x/repo/runtime:latest",
            compute_sa="compute@proj-x.iam.gserviceaccount.com",
            subnetwork_uri="projects/proj-x/regions/us-central1/subnetworks/sf",
        ),
        batch_id="sf-abc",
        package_uri="gs://code-bkt/pkg.zip",
        launcher_uri="gs://code-bkt/spark_main.py",
        config_uri="gs://code-bkt/cfg.json",
        **over,
    )


def test_a_cpu_batch_emits_the_command_it_emitted_before() -> None:
    baseline = _spark_commands().native
    assert _spark_commands(provisioned_hardware=None).native == baseline
    assert _spark_commands(provisioned_hardware="cpu").native == baseline
    assert "provisioned-hardware" not in (baseline or "")


def test_a_gpu_batch_emits_both_halves_of_the_carrier() -> None:
    """A driver and an executor are two processes; a command that showed only one would lie."""
    native = _spark_commands(provisioned_hardware="gpu").native or ""
    assert "--provisioned-hardware gpu" in native
    assert f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}=gpu" in native


def test_the_serverless_batch_carries_the_same_two_halves() -> None:
    pytest.importorskip("google.cloud.dataproc_v1")
    from scale_forecasting.batch_infra import BatchInfra
    from scale_forecasting.submit import build_batch

    kwargs: dict[str, Any] = {
        "infra": BatchInfra(
            code_bucket="code-bkt",
            container_image="us-docker.pkg.dev/proj-x/repo/runtime:latest",
            compute_sa="compute@proj-x.iam.gserviceaccount.com",
            subnetwork_uri="projects/proj-x/regions/us-central1/subnetworks/sf",
        ),
        "settings": _settings(),
        "package_uri": "gs://code-bkt/pkg.zip",
        "launcher_uri": "gs://code-bkt/spark_main.py",
        "config_uri": "gs://code-bkt/cfg.json",
    }
    cpu = build_batch(**kwargs)
    gpu = build_batch(**kwargs, hardware="gpu", gpu_type="L4")
    env_key = f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}"

    assert "--provisioned-hardware" not in list(cpu.pyspark_batch.args)  # type: ignore[attr-defined]
    assert env_key not in dict(cpu.runtime_config.properties)  # type: ignore[attr-defined]
    assert list(gpu.pyspark_batch.args)[-2:] == ["--provisioned-hardware", "gpu"]  # type: ignore[attr-defined]
    assert dict(gpu.runtime_config.properties)[env_key] == "gpu"  # type: ignore[attr-defined]


def test_a_cluster_job_carries_the_same_two_halves() -> None:
    """A Dataproc cluster is the one surface that can be mixed, so it needs the fact most."""
    pytest.importorskip("google.cloud.dataproc_v1")
    from scale_forecasting.cluster_submit import build_job

    kwargs: dict[str, Any] = {
        "cluster": "sf-cluster",
        "launcher_uri": "gs://code-bkt/spark_main.py",
        "package_uri": "gs://code-bkt/pkg.zip",
        "config_uri": "gs://code-bkt/cfg.json",
        "settings": _settings(),
    }
    cpu = build_job(**kwargs)
    gpu = build_job(**kwargs, provisioned_hardware="gpu")
    env_key = f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}"

    assert "--provisioned-hardware" not in list(cpu.pyspark_job.args)  # type: ignore[attr-defined]
    assert env_key not in dict(cpu.pyspark_job.properties)  # type: ignore[attr-defined]
    assert list(gpu.pyspark_job.args)[-2:] == ["--provisioned-hardware", "gpu"]  # type: ignore[attr-defined]
    assert dict(gpu.pyspark_job.properties)[env_key] == "gpu"  # type: ignore[attr-defined]


def test_a_cpu_ray_job_has_no_env_vars_key_at_all() -> None:
    """Not an empty dict — a missing key, so the ``runtime_env`` is the one Ray already accepted."""
    assert "env_vars" not in build_runtime_env()
    assert "env_vars" not in build_runtime_env(provisioned_hardware="cpu")
    assert build_runtime_env(provisioned_hardware="gpu")["env_vars"] == {
        hardware.PROVISIONED_HARDWARE_ENV: "gpu"
    }


def test_the_ray_entrypoint_only_grows_on_a_gpu_job() -> None:
    baseline = build_entrypoint("gs://b/c.json", _settings())
    assert build_entrypoint("gs://b/c.json", _settings(), provisioned_hardware="cpu") == baseline
    assert (
        build_entrypoint("gs://b/c.json", _settings(), provisioned_hardware="gpu")
        == f"{baseline} --provisioned-hardware gpu"
    )


# --- the two facts have to agree ------------------------------------------------


@pytest.mark.parametrize("family", ["deep_learning", "statistical", "ml"])
def test_a_run_that_was_never_told_anything_lets_the_library_choose(family: str) -> None:
    """Including a GPU config: on a workstation ``resolve_family_compute`` still says ``gpu``, and
    honouring that would crash a local run of ``configs/ray_gpu_demo.json`` that works today."""
    assert _resolve_device(_cfg(**_GPU_COMPUTE), family) == "auto"


def test_a_gpu_job_hands_the_device_to_the_family_that_asked_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    assert _resolve_device(_cfg(**_GPU_COMPUTE), "deep_learning") == "gpu"


@pytest.mark.parametrize("family", ["statistical", "ml"])
def test_a_gpu_job_pushes_every_other_family_off_the_card(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """The mixed-cluster case: the executor exposes a device the family must not use, and only an
    explicit ``cpu`` keeps it off — this is what ``accelerator="auto"`` could never do."""
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    assert _resolve_device(_cfg(**_GPU_COMPUTE), family) == "cpu"


def test_a_gpu_job_running_a_cpu_config_still_fits_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared hardware is not permission: the config decides, the environment only enables."""
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    assert _resolve_device(_cfg(), "deep_learning") == "cpu"


def test_a_context_built_without_a_family_stays_on_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driver-side contexts (`hpo._context`, sizing) name no family, so they never force a
    device."""
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    assert _resolve_device(_cfg(**_GPU_COMPUTE), None) == "auto"
    assert _model_context(_cfg(**_GPU_COMPUTE)).device == "auto"


def test_only_the_word_gpu_means_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU job may set the variable outright; it selects what not setting it selects."""
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "cpu")
    assert _resolve_device(_cfg(**_GPU_COMPUTE), "deep_learning") == "auto"


def test_the_resolved_device_lands_on_the_context_the_model_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    ctx = _model_context(_cfg(**_GPU_COMPUTE), family="deep_learning")
    assert ctx.device == "gpu"
    assert _model_context(_cfg(**_GPU_COMPUTE), family="statistical").device == "cpu"


# --- the fast failure when the device is not actually there ---------------------


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_a_cell_that_asked_for_no_device_cannot_be_short_of_one(device: str) -> None:
    _require_device(device, "statistical", "spark")


def test_a_gpu_cell_on_a_host_with_no_device_fails_before_it_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovering this after the fleet-hour is spent leaves forecast rows under a failed job."""
    monkeypatch.setattr("scale_forecasting.worker._peak_gpu_bytes", lambda: None)
    with pytest.raises(ConfigError) as excinfo:
        _require_device("gpu", "deep_learning", "ray")
    message = str(excinfo.value)
    # The message has to name the family, the service, and the field that turns it off — the
    # reader is looking at a failed job, not at this source file.
    assert "deep_learning" in message
    assert "ray" in message
    assert "compute.families.deep_learning.hardware" in message


def test_a_gpu_cell_with_a_visible_device_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scale_forecasting.worker._peak_gpu_bytes", lambda: 0)
    _require_device("gpu", "deep_learning", "ray")


@pytest.mark.parametrize(
    ("available", "phrase"),
    [
        ("cpu", "reports no CUDA device"),
        ("unknown", "could not be imported"),
    ],
)
def test_the_failure_says_which_of_the_two_causes_it_found(
    monkeypatch: pytest.MonkeyPatch, available: str, phrase: str
) -> None:
    """No card attached and no tensor library here need different fixes, and look identical.

    The first is an infrastructure problem — the accelerator was bought and did not arrive, or
    something hid it. The second is a packaging problem: this worker's environment has no torch,
    so it was never going to run on a device whatever hardware it sat on. A message that said only
    "no device" would send a reader to the wrong half of the system.
    """
    monkeypatch.setattr("scale_forecasting.worker._peak_gpu_bytes", lambda: None)
    monkeypatch.setattr("scale_forecasting.worker.visible_device", lambda: (available, None))
    with pytest.raises(ConfigError) as excinfo:
        _require_device("gpu", "deep_learning", "spark")
    assert phrase in str(excinfo.value)


# --- the negative arm: taking the card away on purpose --------------------------
#
# Three live GPU rungs will each report a device. Three green lights prove nothing by themselves,
# so each service also gets a run where the accelerator is provisioned and then hidden, and the
# job has to stop instead of quietly finishing on CPU. `SF_HIDE_DEVICES` is how the card is taken
# away — it rides the executor-env seams that already carry the provisioned fact.


def test_the_switch_is_off_by_default_and_changes_not_one_byte() -> None:
    """Every job ever submitted stays byte-identical; the arm is opt-in from the environment."""
    assert hardware.spark_executor_env("gpu") == {
        f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}": "gpu"
    }
    assert hardware.ray_env_vars("gpu") == {hardware.PROVISIONED_HARDWARE_ENV: "gpu"}


def test_arming_it_hides_the_card_on_both_worker_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spark under the ``spark.executorEnv.`` prefix, Ray plain — the same two seams as the fact."""
    monkeypatch.setenv(hardware.HIDE_DEVICES_ENV, "1")
    assert hardware.spark_executor_env("gpu") == {
        f"spark.executorEnv.{hardware.PROVISIONED_HARDWARE_ENV}": "gpu",
        "spark.executorEnv.CUDA_VISIBLE_DEVICES": "",
    }
    assert hardware.ray_env_vars("gpu") == {
        hardware.PROVISIONED_HARDWARE_ENV: "gpu",
        "CUDA_VISIBLE_DEVICES": "",
    }


@pytest.mark.parametrize("value", [None, "cpu"])
def test_arming_it_does_nothing_to_a_job_that_was_never_given_a_card(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Hiding a device from a CPU job tests nothing, so a CPU job still emits nothing at all."""
    monkeypatch.setenv(hardware.HIDE_DEVICES_ENV, "1")
    assert hardware.spark_executor_env(value) == {}
    assert hardware.ray_env_vars(value) == {}


def test_the_switch_stays_out_of_the_config_and_therefore_out_of_the_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infra, like ``SF_SERVERLESS_DEPS``: a fault injected around a run must not rename it.

    If it entered ``ComputeConfig`` the negative arm would carry a different ``run_id`` from the
    positive one, and the two rungs would no longer be the same job with one thing changed.
    """
    before = make_run_id(_cfg(**_GPU_COMPUTE))
    monkeypatch.setenv(hardware.HIDE_DEVICES_ENV, "1")
    assert make_run_id(_cfg(**_GPU_COMPUTE)) == before


# --- the model states the device instead of asking for a guess ------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", {"accelerator": "auto"}),
        ("cpu", {"accelerator": "cpu"}),
        ("gpu", {"accelerator": "gpu", "devices": 1}),
    ],
)
def test_the_trainer_is_told_which_device_to_use(device: str, expected: dict[str, Any]) -> None:
    """``devices=1`` is stated on the GPU branch even though NeuralProphet overwrites it with -1."""
    model_cls = get_model("neuralprophet")
    ctx = ModelContext(freq="D", horizon=HORIZON, device=device)  # type: ignore[arg-type]
    assert model_cls({}, ctx)._trainer_config() == expected  # type: ignore[attr-defined]


@pytest.mark.parametrize("device", ["auto", "cpu", "gpu"])
def test_no_callback_is_handed_to_a_library_that_cannot_take_one(device: str) -> None:
    """A Lightning callback is the natural way to read the fit's device before teardown moves it,
    and it is unavailable here: NeuralProphet 0.9.0's custom-callbacks branch dereferences
    ``pl.callbacks.ProgressBarBase``, gone from the Lightning we pin, so the fit dies with an
    ``AttributeError`` before training starts. `device_used` reads the trainer instead."""
    model_cls = get_model("neuralprophet")
    ctx = ModelContext(freq="D", horizon=HORIZON, device=device)  # type: ignore[arg-type]
    assert "callbacks" not in model_cls({}, ctx)._trainer_config()  # type: ignore[attr-defined]


def test_the_device_is_read_off_the_trainer_that_survives_teardown() -> None:
    """Lightning ends a fit with ``lightning_module.cpu()``, so the weights say ``"cpu"`` however
    the fit ran — which is how all three live services reported ``device_used="cpu"`` on cells with
    50–68 KB allocated on a real card. ``strategy.root_device`` is the resolved placement and
    outlives the move, so the parameter read is only the fallback."""

    class _Strategy:
        root_device = SimpleNamespace(type="cuda")

    class _Module:
        def parameters(self) -> Any:
            return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    model_cls = get_model("neuralprophet")
    model = model_cls({}, ModelContext(freq="D", horizon=HORIZON, device="gpu"))  # type: ignore[arg-type]
    model._model = SimpleNamespace(  # type: ignore[attr-defined]
        trainer=SimpleNamespace(strategy=_Strategy()), model=_Module()
    )
    assert model.device_used() == "cuda"


def test_the_weights_still_answer_when_there_is_no_trainer_to_ask() -> None:
    """Nothing moves a CPU fit's weights, so the old probe is right there — and it is what a
    Lightning version that reshaped the trainer would fall back to."""

    class _Module:
        def parameters(self) -> Any:
            return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    model_cls = get_model("neuralprophet")
    model = model_cls({}, ModelContext(freq="D", horizon=HORIZON, device="cpu"))  # type: ignore[arg-type]
    model._model = SimpleNamespace(model=_Module())  # type: ignore[attr-defined]
    assert model.device_used() == "cpu"


def test_an_unfitted_model_says_unknown_rather_than_guessing() -> None:
    model_cls = get_model("neuralprophet")
    model = model_cls({}, ModelContext(freq="D", horizon=HORIZON, device="gpu"))  # type: ignore[arg-type]
    assert model.device_used() is None


def test_auto_is_the_default_a_context_is_born_with() -> None:
    """The pre-Layer-3 behaviour is the default, so nothing that never sets it changes."""
    assert ModelContext(freq="D", horizon=HORIZON).device == "auto"


# --- the driver is not a worker -------------------------------------------------


def test_the_scope_hides_the_provisioned_fact_and_puts_it_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    with hardware.driver_fit_scope():
        assert hardware.provisioned_hardware() is None
    assert hardware.provisioned_hardware() == "gpu"


def test_the_scope_puts_it_back_even_when_the_fit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver goes on to submit the real run afterwards — the same reason the thread pin
    restores in a ``finally``."""
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    with pytest.raises(RuntimeError), hardware.driver_fit_scope():
        raise RuntimeError("a fit blew up")
    assert hardware.provisioned_hardware() == "gpu"


def test_the_scope_is_a_no_op_when_nothing_was_provisioned() -> None:
    with hardware.driver_fit_scope():
        assert hardware.provisioned_hardware() is None
    assert hardware.PROVISIONED_HARDWARE_ENV not in __import__("os").environ


class _DeviceSpy(BaseModel):
    """Records the device on every context it is built with, at every fit site."""

    name = "_device_spy"
    runtime = "python"
    family = "deep_learning"
    gpu_capable = True
    seen: list[str] = []

    def __init__(self, params: dict[str, Any], ctx: Any) -> None:
        super().__init__(params, ctx)
        type(self).seen.append(ctx.device)

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._mean = float(y.mean())
        self._last = y.index[-1]

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        yhat = np.full(horizon, self._mean)
        return self._assemble_frame(
            self._future_index(self._last, horizon), {q: yhat for q in quantiles}
        )

    @classmethod
    def search_space(cls, trial: Any) -> dict[str, Any]:
        return {"alpha": trial.suggest_categorical("alpha", [0.25, 0.75])}


@pytest.fixture
def spy() -> Any:
    """Register the spy for one test and take it back out of the global registry after."""
    from scale_forecasting.models import base_model

    _DeviceSpy.seen = []
    base_model._REGISTRY[_DeviceSpy.name] = _DeviceSpy
    try:
        yield _DeviceSpy
    finally:
        base_model._REGISTRY.pop(_DeviceSpy.name, None)


def _spy_cfg(**over: Any) -> RunConfig:
    return _cfg(models=[_DeviceSpy.name], **over)


def test_a_cell_on_a_gpu_job_really_is_told_gpu(spy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the two driver-path tests below: through ``run_cell`` the device is forced.

    Without this, a driver test that saw ``"auto"`` would prove nothing — ``"auto"`` is also what a
    broken carrier produces.
    """
    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    monkeypatch.setattr("scale_forecasting.worker._peak_gpu_bytes", lambda: 0)
    result = run_cell(_series(), _DeviceSpy.name, _spy_cfg(**_GPU_COMPUTE))
    assert result.status == "ok"
    assert spy.seen and set(spy.seen) == {"gpu"}


def test_the_sizing_pre_pass_fits_as_if_nothing_were_provisioned(
    spy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``measure_fit`` runs on the driver — a Ray head node has no card, and Lightning raises
    ``No supported gpu backend found!`` rather than degrading, so an inherited ``gpu`` would sink
    every profiled GPU run at submit."""
    from scale_forecasting.profiling.measure import measure_fit

    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    monkeypatch.setattr("scale_forecasting.worker._peak_gpu_bytes", lambda: 0)
    measure_fit(_series(), _DeviceSpy.name, _spy_cfg(**_GPU_COMPUTE))
    assert spy.seen and set(spy.seen) == {"auto"}
    # And the driver is left exactly as it was found, because it submits the real run next.
    assert hardware.provisioned_hardware() == "gpu"


def test_an_hpo_trial_never_inherits_the_jobs_device(
    spy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-series study runs inside the cell, where the context already says ``gpu``; a fleetwide
    one runs on the driver. Forcing ``auto`` in the objective covers both, and gives up only the
    ability to push a *search* off a visible card — a property of the published fit, not the
    search."""
    from scale_forecasting.hpo import _score_params

    monkeypatch.setenv(hardware.PROVISIONED_HARDWARE_ENV, "gpu")
    cfg = _spy_cfg(
        **_GPU_COMPUTE,
        backtest={
            "enabled": True,
            "n_folds": 1,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
    )
    ctx = _model_context(cfg, family=_DeviceSpy.family)
    assert ctx.device == "gpu"  # the cell-side context the study is handed

    _score_params(_DeviceSpy.name, {}, [_series()], cfg, ctx)
    assert spy.seen and set(spy.seen) == {"auto"}
