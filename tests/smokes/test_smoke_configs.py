"""Offline validity + coverage checks for the smoke config library (`configs/smokes/*.json`).

No GCP: every smoke config must load, validate, and plan a DAG purely offline — so a broken config
(a typo'd model, an invalid runtime/hardware combo, a field the schema rejects) fails here in the
offline gate, long before anyone spends money submitting it. The coverage test pins that the library
still spans every runtime/hardware/ensemble combination the live campaign is meant to prove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale_forecasting.config import RunConfig, load_config
from scale_forecasting.dag import plan_dag
from scale_forecasting.router import split_by_runtime

_SMOKE_DIR = Path(__file__).resolve().parents[2] / "configs" / "smokes"
_CONFIGS = sorted(_SMOKE_DIR.glob("*.json"))


def test_smoke_dir_is_populated() -> None:
    # Guard against a path/glob regression silently collecting zero configs (which would make every
    # parametrized test below vacuously pass).
    assert len(_CONFIGS) >= 12, f"expected the smoke library under {_SMOKE_DIR}, found {_CONFIGS}"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_smoke_config_loads_and_plans(path: Path) -> None:
    cfg = load_config(str(path))
    assert isinstance(cfg, RunConfig)
    # Plans a DAG offline (resolves per-family compute + the runtime split) — the same resolution
    # the live run does, so an invalid combo surfaces here.
    dag = plan_dag(cfg)
    assert dag.families, f"{path.name} planned no families"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_smoke_config_conventions(path: Path) -> None:
    raw = json.loads(path.read_text())
    cfg = load_config(str(path))
    # run_name mirrors the file stem so a run is traceable back to its config from the registry.
    assert raw["run_name"] == f"smoke_{path.stem}", (
        f"{path.name}: run_name {raw['run_name']!r} should be 'smoke_{path.stem}'"
    )
    # Bounded scale — a smoke is ~100 series, never an unbounded (full-table) run.
    assert cfg.data.series_limit is not None and cfg.data.series_limit <= 1000, (
        f"{path.name}: set a bounded series_limit for a smoke"
    )


def test_library_covers_every_runtime_combo() -> None:
    """The library must still span the combinations the live campaign proves (coverage tripwire)."""
    seen: set[str] = set()
    for path in _CONFIGS:
        cfg = load_config(str(path))
        dag = plan_dag(cfg)
        python_models, bq_models = split_by_runtime(cfg)
        if bq_models:
            seen.add("native")
        for job in dag.python_jobs:
            rc = job.compute
            if rc.runtime == "ray":
                seen.add("ray_gpu" if rc.hardware == "gpu" else "ray_cpu")
            elif rc.runtime == "vertex":
                seen.add("vertex_gpu" if rc.hardware == "gpu" else "vertex_cpu")
            elif rc.runtime == "gce":
                seen.add("gce_gpu" if rc.hardware == "gpu" else "gce_cpu")
            elif rc.runtime == "gke":
                seen.add(f"gke_{rc.gke_mode}")
            elif rc.runtime == "spark":
                if rc.hardware == "gpu":
                    # Split by launch mode, not just by hardware. Serverless attaches an L4 through
                    # a runtime-config property; a cluster attaches a T4 through an accelerator on
                    # the worker pool. They are two different provisioning code paths that happen
                    # to share a word, and while one token covered both, either config could have
                    # been deleted and this tripwire would have stayed green.
                    seen.add(f"spark_{rc.spark_mode}_gpu")
                elif rc.spark_mode == "cluster":
                    seen.add("spark_cluster")
                else:
                    seen.add("spark_serverless")
        if cfg.ensemble.enabled:
            seen.add(f"ensemble_{cfg.compute.ensemble.mode}")

    required = {
        "spark_serverless",
        "spark_cluster",
        "spark_serverless_gpu",
        "spark_cluster_gpu",
        "ray_cpu",
        "ray_gpu",
        "vertex_cpu",
        "gce_cpu",
        "gke_job",
        "gke_ray",
        "native",
        "ensemble_barrier",
        "ensemble_microbatch",
    }
    missing = required - seen
    assert not missing, f"smoke library no longer covers: {sorted(missing)}"


def test_a_smoke_needs_two_dataproc_clusters_at_once() -> None:
    """Some smoke must force the per-hardware cluster split, or the branch ships unexercised.

    A Dataproc cluster has one worker machine type, so a run whose ephemeral cluster families span
    CPU and GPU gets one cluster each. That is a different code path from the single-cluster case —
    a second create, two distinct names, a per-cluster region, two teardowns — and *no config
    reached it* when the split was written: smoke 04 has two cluster families and both are CPU, so
    it takes the single-group path unchanged.

    This is the config-side half of that gap. It cannot prove the two clusters actually come up
    (that needs live spend), but it does guarantee a config exists that would, so the branch is
    never silently uncovered again.
    """
    from scale_forecasting.shared_clusters import shared_spark_inputs

    split = {
        path.name: sorted(groups)
        for path in _CONFIGS
        if (groups := shared_spark_inputs(plan_dag(load_config(str(path))).python_jobs) or {})
        and len(groups) > 1
    }
    assert split, (
        "no smoke config produces a multi-hardware Dataproc cluster split; add one with two "
        "ephemeral spark_mode=cluster families on different hardware"
    )


def test_a_gpu_smoke_keeps_the_instruments_on() -> None:
    """A smoke that pays for an accelerator must be able to say whether it used one.

    Two settings decide that, and both are easy to switch off by accident. ``compute.profile``
    with ``mode="off"`` stops recording ``peak_gpu_bytes``, which blinds the ``ENGAGED_IDLE`` /
    ``ENGAGED_UTILISED`` half of the device verdict. And an unset ``gpu_type`` leaves the verdict's
    denominator to a fallback, so a run on the bigger card is judged against the smaller one.

    This has to be a tripwire rather than a safer default, because ``ProfileConfig`` is inside the
    run_id digest — hardening the default would move every run_id in the ledger, which is a much
    larger act than making it impossible to author a blind GPU smoke.
    """
    for path in _CONFIGS:
        cfg = load_config(str(path))
        gpu_families = [j.family for j in plan_dag(cfg).python_jobs if j.compute.hardware == "gpu"]
        if not gpu_families:
            continue
        assert cfg.compute.profile.records_measurements, (
            f"{path.name} routes {gpu_families} onto a GPU with compute.profile recording nothing —"
            f" the run cannot report whether the device was used"
        )
        for job in plan_dag(cfg).python_jobs:
            if job.compute.hardware == "gpu":
                assert job.compute.gpu_type, (
                    f"{path.name}: family {job.family} asks for a GPU without naming a gpu_type"
                )


def test_at_least_one_native_source_format_smoke() -> None:
    # Dual-format coverage: the library must exercise both the managed-Iceberg and native BigQuery
    # source tables so a live campaign proves reads work against each.
    tables = {load_config(str(p)).data.source_table for p in _CONFIGS}
    assert "source_series_native" in tables, "add a smoke reading the native-format source table"
    assert "source_series_iceberg" in tables, "add a smoke reading the Iceberg source table"


def test_the_hpo_pair_differs_only_in_granularity() -> None:
    """Smokes 26 and 27 are one experiment in two files, and the diff between them is the result.

    Neither run proves anything alone. Both stamp `best_params` onto every cell, and on a single
    run those params look equally credible either way — a plausible `window` for
    `naive_moving_average` is a plausible `window` whether it was tuned on a ten-series sample or
    on that series. What tells the two granularities apart is whether the stamped params are
    *identical across series*: `fleetwide` tunes once and applies the winner to all fifty, so they
    must be; `per_series` tunes inside each cell, so they must not be.

    That comparison is only meaningful if the runs are otherwise the same fleet on the same data
    with the same folds, so the pin is the same one the GPU A/B arms get in
    `test_ab_preregistration`: diff the raw JSON and allow exactly the field under test. Note in
    particular that neither file sets `sample_size` — it would be read by the fleetwide arm and
    ignored by the other, which is a difference between the files that is not the difference the
    experiment names.
    """
    fleetwide = json.loads((_SMOKE_DIR / "26_hpo_fleetwide.json").read_text())
    per_series = json.loads((_SMOKE_DIR / "27_hpo_per_series.json").read_text())

    assert fleetwide.pop("run_name") == "smoke_26_hpo_fleetwide"
    assert per_series.pop("run_name") == "smoke_27_hpo_per_series"
    assert fleetwide["hpo"].pop("granularity") == "fleetwide"
    assert per_series["hpo"].pop("granularity") == "per_series"

    assert fleetwide == per_series, (
        "the HPO arms differ somewhere other than hpo.granularity, so a difference in their "
        f"best_params would not be attributable to it: {fleetwide} != {per_series}"
    )


# The features quartet: one baseline and three arms, all measured against the baseline. Arm 29
# carries `exog` as well, because `exog_lags` can only lag a declared covariate — so 31, which is
# that covariate and nothing else, is 29's nearer control: the gap between them is fourier +
# level_shift + the lagged copies of a column both arms already had.
_FEATURES_ARMS = {
    "29_features_on.json": {
        "fourier": True,
        "level_shift": True,
        "exog": ["is_holiday"],
        "exog_lags": {"is_holiday": [1, 7, 28]},
    },
    "30_features_boxcox.json": {"transform": "boxcox"},
    "31_features_exog.json": {"exog": ["is_holiday"]},
}


@pytest.mark.parametrize("arm", sorted(_FEATURES_ARMS), ids=lambda n: n[:2])
def test_each_features_arm_differs_from_the_baseline_only_in_features(arm: str) -> None:
    """Smoke 28 is the baseline the other three are read against, and nothing else may vary.

    A feature knob leaves no trace of its own in the output. There is no column recording which
    design-frame columns a cell built, so "the Fourier terms reached the model" is not something a
    single run can show — the only evidence is the out-of-fold metrics moving against a run that is
    the same fleet, the same fifty series, the same folds and the same six models with the features
    switched off. That is what 28 is for, and it only works if the arms are otherwise identical.

    Three arms rather than one because `transform` is a single field, so Box-Cox cannot ride along
    with the others, and because `exog` has to run with `holidays` unset (see the test below).

    Arm 29 is read against 31 as well as against 28. Both declare the same covariate, so their
    difference is the lagged copies of it plus the two calendar knobs, which is a sharper question
    than 29-vs-28 can ask: whether lagging a covariate is worth anything *given* the covariate.
    """
    baseline = json.loads((_SMOKE_DIR / "28_features_off.json").read_text())
    candidate = json.loads((_SMOKE_DIR / arm).read_text())

    assert baseline.pop("run_name") == "smoke_28_features_off"
    assert candidate.pop("run_name") == f"smoke_{arm.removesuffix('.json')}"
    assert "features" not in baseline, "the baseline arm must carry no features block at all"
    assert candidate.pop("features") == _FEATURES_ARMS[arm]

    assert baseline == candidate, (
        f"{arm} differs from the baseline somewhere other than `features`, so a metric difference "
        f"between them would not be attributable to the feature: {baseline} != {candidate}"
    )


def test_the_exog_arm_leaves_holidays_unset() -> None:
    """Setting both would void the proof silently, which is why this is a tripwire and not a note.

    The only numeric column the shipped source tables carry besides the target is `is_holiday`, so
    that is what the exog arm reads. But `features.holidays` *generates* a column of exactly that
    name, and `features.build_features` writes the declared exog first and the generated flag
    second — so the calendar flag overwrites the source column, the arm proves the holiday path it
    already had instead of the exog path it was written for, and every number still looks right.
    """
    raw = json.loads((_SMOKE_DIR / "31_features_exog.json").read_text())
    assert not raw["features"].get("holidays"), (
        "31_features_exog.json sets holidays as well as exog; the generated `is_holiday` column "
        "would overwrite the one read from the source table and the arm would prove nothing"
    )


def test_covariate_smokes_cover_both_iceberg_and_native_tables() -> None:
    """The covariate/hierarchy source tables ship in both Iceberg and native BigQuery formats."""
    tables = {load_config(str(p)).data.source_table for p in _CONFIGS}
    assert "source_series_covariates_iceberg" in tables
    assert "source_series_covariates_native" in tables


def test_three_tier_covariate_smoke_declares_all_three_tiers_and_lags() -> None:
    """Smoke 32 must exercise static, future, and past covariates plus lagged exog together."""
    cfg = load_config(str(_SMOKE_DIR / "32_covariates_three_tier.json"))
    assert cfg.features.static_covariates == ["region", "category"]
    assert set(cfg.features.future_covariates) >= {"promo_flag", "price_index"}
    assert cfg.features.past_covariates == ["temperature"]
    assert set(cfg.features.exog_lags) == {"promo_flag", "temperature"}


def test_global_hybrid_dl_smoke_exercises_panel_modes_on_ray() -> None:
    """Smoke 34 must exercise global NeuralForecast and hybrid NeuralProphet on Ray."""
    from scale_forecasting.worker import is_panel_model

    cfg = load_config(str(_SMOKE_DIR / "34_global_hybrid_dl.json"))
    assert cfg.python_runtime == "ray"
    for m in ("tide", "tft", "tsmixer", "patchtst"):
        assert m in cfg.models
        assert cfg.model_params[m]["training_mode"] == "global"
        assert is_panel_model(m, cfg)
    assert "neuralprophet" in cfg.models
    assert cfg.model_params["neuralprophet"]["training_mode"] == "hybrid"
    assert is_panel_model("neuralprophet", cfg)


def test_hierarchy_reconciliation_smoke_covers_all_seven_methods() -> None:
    """Smoke 35 must exercise multi-level hierarchical aggregation and all 7 FPP3 methods."""
    from scale_forecasting.config import RECONCILIATION_METHODS

    cfg = load_config(str(_SMOKE_DIR / "35_hierarchy_reconciliation.json"))
    assert cfg.hierarchy.enabled is True
    assert cfg.hierarchy.levels == [["region"], ["region", "category"]]
    assert cfg.hierarchy.middle_level == ["region"]
    assert set(cfg.hierarchy.reconciliation_methods) == RECONCILIATION_METHODS


def test_covariate_fallback_multi_runtime_smoke_spans_all_runtimes_and_tiers() -> None:
    """Smoke 36 combines 3-tier covariates, fallback across runtimes, HPO, and ensembling."""
    from scale_forecasting.dag import covariate_support_report

    cfg = load_config(str(_SMOKE_DIR / "36_covariate_fallback_multi_runtime.json"))
    dag = plan_dag(cfg)
    assert set(dag.families) == {"statistical", "ml", "deep_learning", "native"}
    assert dag.ensemble_enabled is True
    assert cfg.hpo.enabled is True and cfg.hpo.granularity == "fleetwide"
    assert cfg.features.on_unsupported_covariates == "fallback"
    report = covariate_support_report(cfg)
    assert any("theta" in line for line in report)
    assert any("patchtst" in line for line in report)
    assert any("arima_plus" in line for line in report)


def test_hierarchy_covariates_ensemble_smoke_combines_hierarchy_covariates_and_ensemble() -> None:
    """Smoke 37 combines hierarchy reconciliation, 3-tier covariates, fallback, and ensembling."""
    cfg = load_config(str(_SMOKE_DIR / "37_hierarchy_covariates_ensemble.json"))
    assert cfg.hierarchy.enabled is True
    assert cfg.ensemble.enabled is True
    assert cfg.features.static_covariates == ["region", "category"]
    assert "theta" in cfg.models and "xgboost" in cfg.models
