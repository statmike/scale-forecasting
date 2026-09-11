"""Layers 1-2 of the GPU contract: model capability flags, and the plan-time preflight.

Two questions a run should be able to answer before it provisions anything. *Can* any selected
model use a device (`BaseModel.gpu_capable`), and *would* it, at the hyperparameters this config
authors (`BaseModel.gpu_useful`)? They came apart when they were measured — NeuralProphet is
capable and, at the shipped defaults, not useful — and keeping them apart is the whole point:
capability is a static fact about the model, usefulness is a judgement about a config.

`dag.check_hardware_coherence` refuses; `dag.gpu_usefulness_report` only warns. That split is
asserted here in both directions, because collapsing it either way is the failure mode: refusing on
usefulness would break every shipped GPU config in favour of a setting with no green run behind it,
and merely warning on incoherence would let a run pay for idle accelerators for a fleet-hour.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import dag
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.models import get_model, list_models

_GPU = {"python_runtime": "ray", "compute": {"use_gpu": True, "gpu_type": "T4"}}


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "gpu preflight test",
        "data": {"source_table": "t", "horizon": 7, "series_limit": 5},
        "models": ["neuralprophet"],
    }
    base.update(over)
    return RunConfig(**base)


# --- Layer 1: capability vs usefulness -----------------------------------------


def test_only_the_model_with_a_tensor_library_under_it_is_gpu_capable() -> None:
    capable = [n for n in list_models() if get_model(n).gpu_capable]
    assert capable == ["neuralprophet"]


def test_capable_does_not_mean_useful_at_the_shipped_defaults() -> None:
    """The measured fact the whole layer exists for: engaged, and doing nothing."""
    np_model = get_model("neuralprophet")
    assert np_model.gpu_capable is True
    assert np_model.gpu_useful({}) is False


def test_autoregression_is_what_makes_the_device_worth_paying_for() -> None:
    np_model = get_model("neuralprophet")
    assert np_model.gpu_useful({"n_lags": 28}) is True
    # More heads on the same tiny network is not more work for the card.
    assert np_model.gpu_useful({"n_forecasts": 28}) is False


def test_a_cpu_only_model_is_neither_capable_nor_useful() -> None:
    theta = get_model("theta")
    assert theta.gpu_capable is False
    assert theta.gpu_useful({"n_lags": 28}) is False


def test_the_playground_capability_table_reads_the_flag_not_the_family() -> None:
    """One surface, one answer: a novice must not be shown a second definition of "gpu"."""
    from scale_forecasting.playground import model_catalog

    df = model_catalog().set_index("model")
    for name in list_models():
        assert bool(df.loc[name, "gpu"]) is get_model(name).gpu_capable


def test_routing_never_consults_usefulness() -> None:
    """Consuming it at the split would empty the GPU pool while the submitter still bought cards."""
    from pathlib import Path

    src = Path("src/scale_forecasting/engines/ray_io.py").read_text()
    assert "gpu_useful" not in src


# --- Layer 2: coherence refuses ------------------------------------------------


def test_a_coherent_gpu_plan_passes() -> None:
    cfg = _cfg(**_GPU)
    dag.check_hardware_coherence(cfg, dag.plan_dag(cfg).jobs)


def test_a_cpu_plan_has_nothing_to_check() -> None:
    cfg = _cfg(models=["theta"])
    dag.check_hardware_coherence(cfg, dag.plan_dag(cfg).jobs)


def test_gpu_hardware_with_nothing_that_routes_to_the_pool_is_refused() -> None:
    """The regression this stands guard over: accelerators bought, nothing scheduled onto them."""
    cfg = _cfg(**_GPU)
    jobs = dag.plan_dag(cfg).jobs
    # Simulate the state a flat-field regression would produce: a GPU-planned job whose models
    # the engine's own splitter sends to the CPU pool.
    broken = tuple(
        job if job.family != "deep_learning" else type(job)(job.family, ("theta",), job.compute)
        for job in jobs
    )
    with pytest.raises(ConfigError, match="nothing schedules onto"):
        dag.check_hardware_coherence(cfg, broken)


def test_a_device_type_mismatch_between_plan_and_route_is_refused() -> None:
    """The type sets the memory denominator the GPU fraction divides by — T4 vs L4 is 50% off."""
    from dataclasses import replace

    cfg = _cfg(**_GPU)
    jobs = dag.plan_dag(cfg).jobs
    broken = tuple(
        job
        if job.family != "deep_learning" or job.compute is None
        else type(job)(job.family, job.models, replace(job.compute, gpu_type="L4"))
        for job in jobs
    )
    with pytest.raises(ConfigError, match="mis-sizes the pool"):
        dag.check_hardware_coherence(cfg, broken)


# --- Layer 2: usefulness only warns --------------------------------------------


def test_the_shipped_gpu_config_warns_and_is_not_refused() -> None:
    """Every GPU config in the repo is in this state; refusing would break them all today."""
    cfg = _cfg(**_GPU)
    jobs = dag.plan_dag(cfg).jobs
    dag.check_hardware_coherence(cfg, jobs)  # must not raise
    report = dag.gpu_usefulness_report(cfg, jobs)
    assert any("engaged and idle" in line for line in report)


def test_the_warning_names_the_proven_remedy_and_flags_the_unproven_one() -> None:
    report = " ".join(dag.gpu_usefulness_report(_cfg(**_GPU), dag.plan_dag(_cfg(**_GPU)).jobs))
    assert "PROVEN" in report and "hardware" in report
    assert "NO green run" in report and "n_lags" in report


def test_authoring_autoregression_silences_the_usefulness_warning() -> None:
    cfg = _cfg(**_GPU, model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": 7}})
    report = dag.gpu_usefulness_report(cfg, dag.plan_dag(cfg).jobs)
    assert not any("engaged and idle" in line for line in report)


def test_a_gpu_run_says_nothing_about_models_that_are_not_on_the_gpu_job() -> None:
    """theta is on its own CPU job; it is not "an incapable model on the GPU"."""
    cfg = _cfg(**_GPU, models=["theta", "neuralprophet"])
    report = " ".join(dag.gpu_usefulness_report(cfg, dag.plan_dag(cfg).jobs))
    assert "theta" not in report


def test_asking_for_a_device_and_selecting_nothing_that_could_use_one_warns() -> None:
    cfg = _cfg(models=["theta"], **_GPU)
    report = dag.gpu_usefulness_report(cfg, dag.plan_dag(cfg).jobs)
    assert any("has no effect" in line for line in report)
    assert any("no GPU-capable model is selected" in line for line in report)


def test_a_capable_model_routed_off_the_device_is_not_reported_as_an_absent_model() -> None:
    """The A/B's CPU arm: `use_gpu` set, neuralprophet selected, family override sends it to CPU.

    Both causes end with no GPU job, so a single message for both was almost right and read as
    plainly wrong on this arm — it said no deep-learning model was selected while one was. The
    conclusion (the flag buys nothing) was correct; only the reason was invented, which is the kind
    of diagnostic that costs someone an hour looking for a bug in the wrong file.
    """
    cfg = _cfg(
        models=["neuralprophet"],
        python_runtime="ray",
        compute={
            "use_gpu": True,
            "gpu_type": "T4",
            "families": {"deep_learning": {"hardware": "cpu"}},
        },
    )
    jobs = dag.plan_dag(cfg).jobs
    assert all(j.compute is None or j.compute.hardware == "cpu" for j in jobs)

    report = " ".join(dag.gpu_usefulness_report(cfg, jobs))
    assert "neuralprophet" in report, "the warning must name the model it is talking about"
    assert "no GPU-capable model is selected" not in report, (
        "a GPU-capable model IS selected here; it is routed off the device, which is a different "
        "thing and points at a different line of the config"
    )


def test_a_plain_cpu_run_has_nothing_to_say() -> None:
    cfg = _cfg(models=["theta"])
    assert dag.gpu_usefulness_report(cfg, dag.plan_dag(cfg).jobs) == []


# --- preflight wiring ----------------------------------------------------------


def test_preflight_returns_the_same_dag_planning_does() -> None:
    cfg = _cfg(**_GPU)
    assert dag.preflight(cfg) == dag.plan_dag(cfg)


def test_preflight_logs_the_usefulness_warning(caplog: Any) -> None:
    with caplog.at_level("WARNING"):
        dag.preflight(_cfg(**_GPU))
    assert "engaged and idle" in caplog.text


def test_preflight_still_refuses_an_unhonourable_model_params_block() -> None:
    cfg = _cfg(model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": 1}})
    with pytest.raises(ConfigError, match="n_forecasts"):
        dag.preflight(cfg)


def test_planning_stays_total_where_preflight_refuses() -> None:
    """`plan_dag` is called by the SDK, notebooks and hundreds of tests; it must never raise."""
    cfg = _cfg(model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": 1}})
    assert dag.plan_dag(cfg).run_id
