"""The GPU-vs-CPU A/B arms must be identical in everything except the accelerator (offline, pure).

`neuralprophet_ab_gpu.json` and `neuralprophet_ab_cpu.json` exist to answer one question — does a
T4 buy anything for NeuralProphet at scale — and an experiment only answers its question if the
arms differ in exactly the one way it names. That is not something you can check after the fact:
once two runs have landed, any difference in their fleets is indistinguishable from a difference in
their hardware, and the numbers will look like an answer either way.

So the comparison is pre-registered here, offline, and has to pass before either arm is submitted.
Everything it asserts is a pure function of the two config files, so it costs nothing and it fails
the moment somebody edits one arm without the other.

**This test earned its place by failing.** The first construction of the A/B followed the plan
literally — take `all_families_10k_full`, flip `families.deep_learning.hardware` to ``"cpu"``, run
it twice. One field. It looked airtight and it was not: on the GPU arm the deep-learning job runs
in the *GPU* pool, capped by ``ray_gpu_max_nodes``, and flipping that field moves the same job into
the *CPU* pool, capped by ``ray_cpu_max_nodes``. Each pool honours its own pinned ceiling, so the
arms got 12 nodes and 20 nodes — a 1.7x fleet difference sitting underneath a measurement of
throughput. Worse, that config puts three families on one shared cluster, so the model under test
had 12 nodes to itself on one arm and shared them with 40,000 other cells on the other. The
isolated pair this file guards has neither problem, and this file is what proves it, run after run.

What is deliberately *not* asserted: the two ``run_id``s. They must differ (a shared id would file
both arms under one run), and they are checked for that — but neither is pinned to a literal,
because ``compute.profile.source`` is excluded from the digest. The pin that matters most to this
experiment is therefore invisible in the ids, which is why it is asserted directly below.

**There are two pairs, because there are two runtimes.** The Ray pair above is the first half; the
second is `neuralprophet_ab_cluster_gpu.json` / `neuralprophet_ab_cluster_cpu.json` on a Dataproc
cluster, guarded in the second half of this file. A throughput result on one scheduler is a result
about that scheduler, so the decision rule wants both before the shipped default is treated as
settled. The cluster pair needs its own guard rather than a shared one because the confound it has
to rule out is a different mechanism entirely — not two pools with two ceilings, but one integer
(``spark.task.cpus``) that a GPU fraction silently widens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_io
from scale_forecasting.registry.ids import make_run_id

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_GPU_ARM = _CONFIGS / "neuralprophet_ab_gpu.json"
_CPU_ARM = _CONFIGS / "neuralprophet_ab_cpu.json"
_CLUSTER_GPU_ARM = _CONFIGS / "neuralprophet_ab_cluster_gpu.json"
_CLUSTER_CPU_ARM = _CONFIGS / "neuralprophet_ab_cluster_cpu.json"

# The one field the experiment is allowed to differ on, as a JSON path into the config.
_THE_ONE_DIFFERENCE = ("compute", "families")

# Pool-plan keys that must agree between the arms. These are the shape of the fleet the model under
# test runs on: how many slots each node holds, how many nodes the run derives, how many slots that
# is in total, and which resource binds. If any of them differ, the arms are measuring two fleets.
_PLAN_KEYS = (
    "slots_per_unit",
    "derived_units",
    "total_slots",
    "saturating_units",
    "slots_at_ceiling",
    "binding_axis",
    "target_cells_per_slot",
    "n_cells",
)


def _load(path: Path) -> RunConfig:
    return RunConfig(**json.loads(path.read_text(encoding="utf-8")))


def _dl_pool(cfg: RunConfig) -> dict[str, Any]:
    """The pool plan the deep-learning job actually lands in, GPU pool or CPU pool.

    Which pool that is *is* the arm: a GPU arm's NeuralProphet cells are sized into the GPU pool and
    a CPU arm's into the CPU pool. Reading the right one per arm is the whole point — comparing both
    arms' ``cpu_pool`` would compare a fleet doing the work against an empty one.
    """
    resolved = cfg.resolve_family_compute("deep_learning")
    plan = ray_io.plan_cluster(
        cfg,
        ["neuralprophet"],
        run_id=make_run_id(cfg),
        use_gpu=(resolved.hardware == "gpu"),
        gpu_type=resolved.gpu_type,
        profile=None,
    )
    pool = plan.gpu_pool if resolved.hardware == "gpu" else plan.cpu_pool
    assert pool is not None, "the arm's own pool must exist"
    return dict(pool.to_dict())


@pytest.fixture(scope="module")
def arms() -> tuple[RunConfig, RunConfig]:
    return _load(_GPU_ARM), _load(_CPU_ARM)


def test_the_arms_differ_in_exactly_one_field(arms: tuple[RunConfig, RunConfig]) -> None:
    """Diff the raw JSON. Anything beyond ``run_name`` and the DL override is a confound."""
    gpu_raw = json.loads(_GPU_ARM.read_text(encoding="utf-8"))
    cpu_raw = json.loads(_CPU_ARM.read_text(encoding="utf-8"))

    assert gpu_raw.pop("run_name") == "neuralprophet_ab_gpu"
    assert cpu_raw.pop("run_name") == "neuralprophet_ab_cpu"

    # The CPU arm carries the override; the GPU arm must not carry one at all.
    section, key = _THE_ONE_DIFFERENCE
    assert key not in gpu_raw[section], (
        "the GPU arm must take its hardware from the flat compute defaults, with no family "
        "override — otherwise 'one field differs' is not true of the files"
    )
    assert cpu_raw[section].pop(key) == {"deep_learning": {"hardware": "cpu"}}

    assert gpu_raw == cpu_raw, (
        "the A/B arms differ somewhere other than the deep-learning hardware override; "
        f"gpu-only={_only_in(gpu_raw, cpu_raw)}, cpu-only={_only_in(cpu_raw, gpu_raw)}"
    )


def _only_in(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The entries of ``a`` that ``b`` does not match, one level deep — enough to name the drift."""
    out: dict[str, Any] = {}
    for k, v in a.items():
        if k not in b:
            out[k] = v
        elif isinstance(v, dict) and isinstance(b[k], dict):
            nested = _only_in(v, b[k])
            if nested:
                out[k] = nested
        elif b[k] != v:
            out[k] = v
    return out


def test_the_arms_resolve_to_the_hardware_they_claim(arms: tuple[RunConfig, RunConfig]) -> None:
    gpu_cfg, cpu_cfg = arms
    assert gpu_cfg.resolve_family_compute("deep_learning").hardware == "gpu"
    assert gpu_cfg.resolve_family_compute("deep_learning").gpu_type == "T4"
    assert cpu_cfg.resolve_family_compute("deep_learning").hardware == "cpu"
    # Not merely absent from the plan — resolved away, so nothing downstream can pick it back up.
    assert cpu_cfg.resolve_family_compute("deep_learning").gpu_type is None


def test_both_arms_pin_the_baseline_profile(arms: tuple[RunConfig, RunConfig]) -> None:
    """``profile.source`` is digest-excluded, so this pin is not recoverable from the ``run_id``s.

    It has to be pinned, and it has to be asserted here rather than inferred later. Under ``auto`` a
    prior deep-learning harvest could resolve a two-core slot on one arm and a one-core slot on the
    other, halving that arm's concurrency for reasons that have nothing to do with the accelerator.
    The shipped baseline carries no deep-learning family at all, so ``baseline`` means *static* for
    NeuralProphet on both arms and the confound cannot occur.
    """
    for cfg in arms:
        assert cfg.compute.profile.source == "baseline", (
            f"{cfg.run_name}: the A/B requires compute.profile.source='baseline'; "
            "under 'auto' the two arms can be sized from different past runs"
        )


def test_both_arms_pin_the_same_gpu_fraction(arms: tuple[RunConfig, RunConfig]) -> None:
    """0.125, not the naive 1/7, and on *both* arms.

    At 1/7 the GPU arm reports seven slots per node against the CPU arm's eight, and the whole
    comparison inherits a one-slot handicap that looks like a hardware result.
    """
    for cfg in arms:
        assert cfg.compute.gpu_fraction == 0.125, (
            f"{cfg.run_name}: gpu_fraction must be pinned to 0.125 on both arms"
        )


@pytest.mark.parametrize("key", _PLAN_KEYS)
def test_the_two_fleets_are_the_same_shape(key: str, arms: tuple[RunConfig, RunConfig]) -> None:
    gpu_pool, cpu_pool = (_dl_pool(cfg) for cfg in arms)
    assert gpu_pool[key] == cpu_pool[key], (
        f"A/B arms disagree on {key}: gpu={gpu_pool[key]!r} cpu={cpu_pool[key]!r}. "
        "The arms would be measuring two different fleets, not two different accelerators."
    )


def test_the_two_slots_are_the_same_size(arms: tuple[RunConfig, RunConfig]) -> None:
    """One core per slot on both arms — the denominator of every per-fit number we will report."""
    gpu_pool, cpu_pool = (_dl_pool(cfg) for cfg in arms)
    assert gpu_pool["slot"]["cores"] == cpu_pool["slot"]["cores"] == 1
    assert gpu_pool["unit"]["cores"] == cpu_pool["unit"]["cores"]
    assert gpu_pool["unit"]["memory_bytes"] == cpu_pool["unit"]["memory_bytes"]


def test_task_options_differ_only_by_the_accelerator_key(arms: tuple[RunConfig, RunConfig]) -> None:
    """Ray gets ``num_gpus`` on one arm and ``num_cpus`` on the other, and nothing else moves.

    In particular the thread-pinning environment must be identical. A NeuralProphet fit that quietly
    gets three threads on one arm is not the same fit, and the resulting throughput ratio would be
    reporting a thread count as an accelerator effect.
    """
    gpu_pool, cpu_pool = (_dl_pool(cfg) for cfg in arms)
    gpu_opts = {
        k: v for k, v in gpu_pool["task_options"].items() if k not in ("num_gpus", "num_cpus")
    }
    cpu_opts = {
        k: v for k, v in cpu_pool["task_options"].items() if k not in ("num_gpus", "num_cpus")
    }
    assert gpu_opts == cpu_opts

    assert gpu_pool["task_options"].get("num_gpus") == 0.125
    assert "num_gpus" not in cpu_pool["task_options"], (
        "the CPU arm asked Ray for a GPU it never provisions — tasks that can never be scheduled, "
        "the 'use and don't buy' shape test_gpu_routing_coherence.py exists to prevent"
    )


def test_the_arms_get_distinct_run_ids(arms: tuple[RunConfig, RunConfig]) -> None:
    gpu_cfg, cpu_cfg = arms
    gpu_id, cpu_id = make_run_id(gpu_cfg), make_run_id(cpu_cfg)
    assert gpu_id != cpu_id, "both arms would file under one run_id and overwrite each other"
    assert gpu_id.startswith("neuralprophet-ab-gpu-")
    assert cpu_id.startswith("neuralprophet-ab-cpu-")


def test_the_analysis_sql_is_committed_before_the_arms_run() -> None:
    """The decision rule is pre-registered, and the query that computes it is a file in the repo.

    An analysis written after the numbers land is an analysis fitted to the numbers. This
    asserts the file exists and still queries both arms' decision inputs, so the SQL cannot be
    quietly swapped for a friendlier one between submission and write-up.
    """
    sql_path = _CONFIGS.parent / "docs" / "sql" / "neuralprophet_ab.sql"
    assert sql_path.is_file(), f"the pre-registered A/B analysis is missing at {sql_path}"

    # Comment lines are stripped first. The file explains at length *why* it does not read through
    # `read_compute_harvest`, and a naive substring check over the whole file would read that
    # explanation as the thing it forbids — which it did, the first time this test ran.
    body = "\n".join(
        line
        for line in sql_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )

    for needle in ("fit_seconds", "cpu_seconds", "peak_gpu_bytes", "neuralprophet"):
        assert needle in body, f"the A/B analysis SQL never queries {needle!r}"
    assert "read_compute_harvest" not in body, (
        "the harvest path truncates and samples by FARM_FINGERPRINT(ts_id); the A/B needs a "
        "purpose-written aggregate over every cell"
    )


# =================================================================================================
# The Dataproc-cluster pair — the same experiment, the other runtime, a different confound.
#
# On Ray the danger was two pools with two ceilings. Here there is one cluster and one executor per
# worker, so the fleets cannot diverge that way; what can diverge is how many cells fit *inside* an
# executor. `resources/cluster._cluster_task_cpus` reads the GPU fraction and widens
# ``spark.task.cpus`` until the executor holds no more cells than the device can take. Left at the
# resolved half-card default that is three cores a task, two cells a worker — against the CPU arm's
# one core and seven cells. A 3.5x concurrency gap, invisible in the config, sitting directly under
# a throughput measurement. Pinning ``gpu_fraction: 0.125`` on both arms closes it, and the tests
# below are what stop the pin from being quietly dropped.
#
# The asserts are stronger than the Ray half in one respect: the two arms' *job properties* are
# compared as whole dicts. On a cluster the properties are the fleet, so an identical property map
# is the most direct statement available that both arms run the same machine.
# =================================================================================================

# Four workers, because the deployment region's Compute Engine T4 quota is four cards. The CPU arm
# is held to the same four rather than to whatever its own arithmetic wanted — a fair comparison
# needs matching fleets more than it needs a fast CPU arm.
_CLUSTER_MAX_EXECUTORS = 4

# Four workers x seven slots. The pre-registered SQL declares this number too, and
# `test_the_sql_declares_the_slot_count_the_plan_actually_produces` holds the two together: waves
# are cells/slots, and a wave count computed against the wrong denominator would pass or fail the
# 50-wave control for arithmetic reasons.
_CLUSTER_TOTAL_SLOTS = 28


def _cluster_plan(cfg: RunConfig) -> tuple[dict[str, Any], dict[str, str]]:
    """``(fleet plan, Spark job properties)`` for the arm's deep-learning job.

    One call rather than two because a cluster decides both at once: the worker count and the
    properties come out of the same sizing pass, and reading them separately would allow a test to
    compare a plan against properties that no longer match it.
    """
    from scale_forecasting.dataproc_cluster import cluster_sizing
    from scale_forecasting.profiling.source import profile_for_run

    resolved = cfg.resolve_family_compute("deep_learning")
    assert resolved.spark_mode == "cluster", "this pair must run on a Dataproc cluster"
    _workers, properties, telemetry = cluster_sizing(
        cfg,
        ["neuralprophet"],
        hardware=resolved.hardware,
        gpu_type=resolved.gpu_type,
        profile=profile_for_run(cfg),
    )
    plans = telemetry.get("plans") or []
    assert plans, "the sizing telemetry carried no plan"
    return dict(plans[0]), dict(properties)


@pytest.fixture(scope="module")
def cluster_arms() -> tuple[RunConfig, RunConfig]:
    return _load(_CLUSTER_GPU_ARM), _load(_CLUSTER_CPU_ARM)


def test_the_cluster_arms_differ_in_exactly_one_field() -> None:
    """Diff the raw JSON. Both arms carry a family block; only ``hardware`` may differ inside it.

    Unlike the Ray pair, neither arm can leave the family block out — ``spark_mode`` lives only on
    the per-family override, so "run this on a cluster" has to be said there. That makes the one
    permitted difference a single *key within* the block rather than the block itself.
    """
    gpu_raw = json.loads(_CLUSTER_GPU_ARM.read_text(encoding="utf-8"))
    cpu_raw = json.loads(_CLUSTER_CPU_ARM.read_text(encoding="utf-8"))

    assert gpu_raw.pop("run_name") == "neuralprophet_ab_cluster_gpu"
    assert cpu_raw.pop("run_name") == "neuralprophet_ab_cluster_cpu"

    gpu_dl = gpu_raw["compute"]["families"]["deep_learning"]
    cpu_dl = cpu_raw["compute"]["families"]["deep_learning"]
    assert gpu_dl == {"spark_mode": "cluster"}, (
        "the GPU arm must take its hardware from the flat compute defaults, so that the only key "
        "distinguishing the arms is the CPU arm's override"
    )
    assert cpu_dl == {"spark_mode": "cluster", "hardware": "cpu"}

    del gpu_raw["compute"]["families"], cpu_raw["compute"]["families"]
    assert gpu_raw == cpu_raw, (
        "the cluster A/B arms differ somewhere other than the deep-learning hardware override; "
        f"gpu-only={_only_in(gpu_raw, cpu_raw)}, cpu-only={_only_in(cpu_raw, gpu_raw)}"
    )


def test_the_cluster_arms_resolve_to_the_hardware_they_claim(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    gpu_cfg, cpu_cfg = cluster_arms
    for cfg in cluster_arms:
        assert cfg.resolve_family_compute("deep_learning").spark_mode == "cluster"
    assert gpu_cfg.resolve_family_compute("deep_learning").hardware == "gpu"
    # T4 rather than L4: Serverless offers no T4, which is the whole reason this pair is on a
    # cluster and not on the batch path.
    assert gpu_cfg.resolve_family_compute("deep_learning").gpu_type == "T4"
    assert cpu_cfg.resolve_family_compute("deep_learning").hardware == "cpu"
    assert cpu_cfg.resolve_family_compute("deep_learning").gpu_type is None


def test_both_cluster_arms_pin_the_baseline_profile(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    """Digest-excluded, so this pin is not recoverable from the ``run_id``s — see the Ray twin."""
    for cfg in cluster_arms:
        assert cfg.compute.profile.source == "baseline", (
            f"{cfg.run_name}: the A/B requires compute.profile.source='baseline'; "
            "under 'auto' the two arms can be sized from different past runs"
        )


def test_both_cluster_arms_pin_the_gpu_fraction_that_equalises_the_packing(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    """0.125 on both, and on a cluster the pin does more work than it does on Ray.

    Ray asks for ``num_gpus`` per task and places what fits. A cluster has no such lever — Dataproc
    leaves YARN GPU isolation off — so the fraction is spent entirely on widening
    ``spark.task.cpus``, and an unpinned GPU arm runs two cells a worker where the CPU arm runs
    seven. Anything at or below 1/7 leaves the packing alone; 0.125 is used because it is the value
    the Ray pair already pins, and one number across the whole experiment is easier to check than
    two.
    """
    for cfg in cluster_arms:
        assert cfg.compute.gpu_fraction == 0.125, (
            f"{cfg.run_name}: gpu_fraction must be pinned to 0.125 on both arms; unpinned, the GPU "
            "arm packs two cells per worker against the CPU arm's seven"
        )


def test_both_cluster_arms_cap_the_fleet_at_the_accelerator_quota(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    """Both arms cap at four workers, which is what the region's T4 quota allows.

    Without the cap the fan-out asks for more workers than there are cards, and the GPU arm fails at
    create while the CPU arm happily takes them — the arms would not even be the same size. The cap
    has to be on *both* arms for the same reason the fraction does.
    """
    for cfg in cluster_arms:
        assert cfg.compute.max_executors == _CLUSTER_MAX_EXECUTORS, (
            f"{cfg.run_name}: max_executors must be {_CLUSTER_MAX_EXECUTORS} on both arms"
        )


@pytest.mark.parametrize("key", _PLAN_KEYS)
def test_the_two_cluster_fleets_are_the_same_shape(
    key: str, cluster_arms: tuple[RunConfig, RunConfig]
) -> None:
    gpu_plan, cpu_plan = (_cluster_plan(cfg)[0] for cfg in cluster_arms)
    assert gpu_plan[key] == cpu_plan[key], (
        f"cluster A/B arms disagree on {key}: gpu={gpu_plan[key]!r} cpu={cpu_plan[key]!r}. "
        "The arms would be measuring two different fleets, not two different accelerators."
    )


def test_the_two_cluster_slots_are_the_same_size(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    """One core per slot on both arms — the denominator of every per-fit number we will report."""
    gpu_plan, cpu_plan = (_cluster_plan(cfg)[0] for cfg in cluster_arms)
    assert gpu_plan["slot"]["cores"] == cpu_plan["slot"]["cores"] == 1
    assert gpu_plan["unit"]["cores"] == cpu_plan["unit"]["cores"]
    assert gpu_plan["unit"]["memory_bytes"] == cpu_plan["unit"]["memory_bytes"]
    # The one shape difference that is allowed, because it *is* the experiment.
    assert gpu_plan["unit"]["accelerators"] == 1
    assert cpu_plan["unit"]["accelerators"] == 0


def test_the_two_cluster_arms_submit_identical_spark_properties(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    """Byte-for-byte the same job properties, ``spark.task.cpus`` above all.

    On a cluster the properties are the fleet: executor cores, task width, the memory split, the
    thread-pin environment, and the executor count. If those maps are equal there is no room left
    for the arms to differ in anything but the card. Compared whole rather than key by key so a
    property added later is caught by this test rather than by whoever reads the results.
    """
    gpu_props, cpu_props = (_cluster_plan(cfg)[1] for cfg in cluster_arms)
    assert gpu_props == cpu_props, (
        "the cluster A/B arms would submit different Spark properties; "
        f"gpu-only={_only_in(gpu_props, cpu_props)}, cpu-only={_only_in(cpu_props, gpu_props)}"
    )
    # Spelled out, because this is the exact property the unpinned fraction moves.
    assert "spark.task.cpus" not in gpu_props, (
        "spark.task.cpus is set on the GPU arm, which means the GPU fraction widened the task and "
        "the arm is running fewer cells per worker than its CPU twin"
    )
    assert gpu_props["spark.executorEnv.OMP_NUM_THREADS"] == "1"


def test_the_planned_cluster_fleet_holds_the_slots_the_analysis_assumes(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    for cfg in cluster_arms:
        plan, _props = _cluster_plan(cfg)
        assert plan["total_slots"] == _CLUSTER_TOTAL_SLOTS, (
            f"{cfg.run_name}: planned {plan['total_slots']} slots, not {_CLUSTER_TOTAL_SLOTS}"
        )


def test_the_cluster_arms_get_distinct_run_ids(
    cluster_arms: tuple[RunConfig, RunConfig],
) -> None:
    gpu_cfg, cpu_cfg = cluster_arms
    gpu_id, cpu_id = make_run_id(gpu_cfg), make_run_id(cpu_cfg)
    assert gpu_id != cpu_id, "both arms would file under one run_id and overwrite each other"
    assert gpu_id.startswith("neuralprophet-ab-cluster-gpu-")
    assert cpu_id.startswith("neuralprophet-ab-cluster-cpu-")


def test_the_cluster_analysis_sql_is_committed_before_the_arms_run() -> None:
    """The cluster pair's decision rule is pre-registered in its own file, as the Ray pair's is."""
    sql_path = _CONFIGS.parent / "docs" / "sql" / "neuralprophet_ab_cluster.sql"
    assert sql_path.is_file(), f"the pre-registered cluster A/B analysis is missing at {sql_path}"

    body = "\n".join(
        line
        for line in sql_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    for needle in ("fit_seconds", "cpu_seconds", "peak_gpu_bytes", "neuralprophet"):
        assert needle in body, f"the cluster A/B analysis SQL never queries {needle!r}"
    assert "read_compute_harvest" not in body


def test_the_sql_declares_the_slot_count_the_plan_actually_produces() -> None:
    """Waves are cells divided by slots, so the SQL's denominator has to be the planned one.

    This is the seam where a pre-registered analysis quietly stops describing the experiment: the
    config's worker cap moves, the fleet halves, and the SQL keeps dividing by the old number — so
    the 50-wave control passes or fails for arithmetic reasons rather than for anything that
    happened on the cluster. Here rather than in the SQL because only the planner knows the answer.
    """
    sql_path = _CONFIGS.parent / "docs" / "sql" / "neuralprophet_ab_cluster.sql"
    text = sql_path.read_text(encoding="utf-8")
    assert f"DECLARE total_slots FLOAT64 DEFAULT {_CLUSTER_TOTAL_SLOTS}.0;" in text, (
        f"the cluster A/B SQL must declare total_slots = {_CLUSTER_TOTAL_SLOTS}.0, matching the "
        "fleet both arms plan"
    )


def test_the_sql_names_both_cluster_run_ids() -> None:
    """The two ids are literals in the SQL, and they are the ids these configs actually resolve to.

    Pinning them here is the opposite of what the Ray half does deliberately — there, the ids are
    left unpinned because ``profile.source`` is digest-excluded and a literal would imply the digest
    covers more than it does. The reason to pin them *here* is different and narrower: the SQL
    carries them as DECLARE defaults, so if a config is edited after the analysis is written, the
    query keeps asking about two runs that will never exist and returns empty rather than wrong.
    Empty results read as "the runs have not landed yet", which is the most expensive way to be
    misled during a live campaign.
    """
    text = (_CONFIGS.parent / "docs" / "sql" / "neuralprophet_ab_cluster.sql").read_text(
        encoding="utf-8"
    )
    for path in (_CLUSTER_GPU_ARM, _CLUSTER_CPU_ARM):
        run_id = make_run_id(_load(path))
        assert f"'{run_id}'" in text, (
            f"{path.name} resolves to {run_id}, which the pre-registered SQL does not name — "
            "the config moved after the analysis was written"
        )
