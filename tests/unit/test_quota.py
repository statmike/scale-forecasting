"""Offline tests for the quota preflight.

Everything here runs against fabricated Service Usage payloads and hand-built plans. That is not a
compromise: `quota.read_limits` is the only function in the module that touches a network, and it
takes a ``fetch`` callable precisely so the rest can be pinned exactly. The payload shapes below are
transcribed from real ``consumerQuotaMetrics`` responses — string limits, dimensionless default
buckets and an absent ``effectiveLimit`` are all things the live API actually returned.
"""

from __future__ import annotations

import pytest

from scale_forecasting import quota
from scale_forecasting.engines.ray_io import RayClusterPlan
from scale_forecasting.resources.fleet import RuntimeResourcePlan, UnitShape
from scale_forecasting.resources.slot import ResourceSlot

# --- fixtures -------------------------------------------------------------------------------------

T4 = quota.vertex_gpu_metric("T4")
CPUS = quota.vertex_cpu_metric("n1-standard-8")


def bucket(limit: str | None, region: str | None = None) -> dict:
    """One ``quotaBuckets`` entry; ``region=None`` is the dimensionless project default."""
    entry: dict = {"dimensions": {"region": region}} if region else {}
    if limit is not None:
        entry["effectiveLimit"] = limit
    return entry


def payload(*buckets: dict) -> dict:
    return {"consumerQuotaLimits": [{"quotaBuckets": list(buckets)}]}


def gpu_demand(min_units: int = 1, max_units: int = 12, per_unit: int = 1) -> quota.QuotaDemand:
    return quota.QuotaDemand(
        metric=T4,
        region="us-east1",
        pool="gpu",
        per_unit=per_unit,
        min_units=min_units,
        max_units=max_units,
    )


def reading(limit: int | None, region: str = "us-east1") -> quota.QuotaReading:
    return quota.QuotaReading(T4, region, limit, "test")


def ray_plan(**overrides) -> RayClusterPlan:
    """A two-pool autoscaling plan shaped like the one `all_families_10k` produces."""
    pool = RuntimeResourcePlan(
        runtime="ray",
        family="deep_learning",
        slot=ResourceSlot(
            family="deep_learning",
            cores=4,
            memory_bytes=None,
            gpu_fraction=0.5,
            device_bytes=None,
        ),
        unit=UnitShape(cores=8, accelerators=1),
        n_cells=10_000,
        slots_per_unit=2,
        derived_units=12,
        saturating_units=5_000,
        min_units=1,
        max_units=12,
        target_cells_per_slot=1,
    )
    fields = {
        "cluster_name": "c",
        "reuse": False,
        "head_machine_type": "n1-highmem-32",
        "cpu_machine_type": "n1-standard-8",
        "cpu_node_count": 20,
        "gpu_machine_type": "n1-standard-8",
        "gpu_node_count": 12,
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
        "sizing_gpu_fraction": 0.5,
        "n_gpu_cells": 10_000,
        "n_cpu_cells": 10_000,
        "autoscale": True,
        "cpu_min_nodes": 1,
        "cpu_max_nodes": 20,
        "gpu_min_nodes": 1,
        "gpu_max_nodes": 12,
        "cpu_pool": pool,
        "gpu_pool": pool,
    }
    fields.update(overrides)
    return RayClusterPlan(**fields)


# --- the vocabulary ---------------------------------------------------------------------------


def test_ray_and_dataproc_gpus_are_different_meters():
    """The conflation the module exists to prevent: same card, two services, two allowances."""
    vertex, compute = quota.vertex_gpu_metric("T4"), quota.compute_gpu_metric("T4")
    assert vertex.service != compute.service
    assert vertex.metric != compute.metric


def test_vertex_cpu_metric_is_machine_family_specific():
    """``custom_model_training_cpus`` is the N1/E2 meter, not the general one."""

    def suffix(machine_type: str) -> str:
        return quota.vertex_cpu_metric(machine_type).metric.rsplit("/", 1)[-1]

    assert suffix("n1-standard-8") == "custom_model_training_cpus"
    assert suffix("e2-standard-4") == "custom_model_training_cpus"
    assert suffix("g2-standard-8") == "custom_model_training_g2_cpus"
    assert suffix("a2-highgpu-1g") == "custom_model_training_a2_cpus"
    # An untabulated family falls back to the N1/E2 meter rather than inventing a metric id.
    assert suffix("z9-weird-1") == "custom_model_training_cpus"


def test_vcpu_metrics_are_shared_scope_and_device_metrics_are_not():
    """Scope is what decides clampable-vs-report-only; it must not drift."""
    assert quota.vertex_cpu_metric("n1-standard-8").scope == "shared"
    assert quota.compute_cpu_metric().scope == "shared"
    assert quota.vertex_gpu_metric("T4").scope == "pool"


def test_unknown_accelerator_has_no_metric():
    assert quota.vertex_gpu_metric("TPU_V5") is None
    assert quota.gpu_type_from_accelerator("NVIDIA_TESLA_T4") == "T4"
    assert quota.gpu_type_from_accelerator("NVIDIA_L4") == "L4"
    assert quota.gpu_type_from_accelerator("TPU_V5_LITEPOD") is None


def test_machine_family():
    assert quota.machine_family("n1-highmem-32") == "n1"
    assert quota.machine_family("g2-standard-8") == "g2"


# --- the reader -------------------------------------------------------------------------------


def test_parse_buckets_prefers_the_regional_bucket_over_the_default():
    body = payload(bucket("400"), bucket("12", "us-central1"), bucket("2", "us-east1"))
    assert quota.parse_buckets(body, T4, "us-east1").limit == 2
    assert quota.parse_buckets(body, T4, "us-central1").limit == 12


def test_parse_buckets_falls_back_to_the_project_default():
    body = payload(bucket("400"), bucket("12", "us-central1"))
    result = quota.parse_buckets(body, T4, "us-west1")
    assert result.limit == 400
    assert "project default" in result.detail


def test_parse_buckets_reads_limits_as_strings():
    """The API returns ``"12"``, not ``12``, so the int() has to be real."""
    result = quota.parse_buckets(payload(bucket("12", "us-east1")), T4, "us-east1")
    assert result.limit == 12
    assert isinstance(result.limit, int)


@pytest.mark.parametrize(
    "body",
    [
        payload(bucket(None, "us-east1")),  # absent effectiveLimit — ambiguous, never enforced
        payload(bucket("-1", "us-east1")),  # the proto's explicit "unlimited"
        payload(bucket("banana", "us-east1")),  # unparseable
        payload(),  # no buckets at all
        {},  # not a quota payload
    ],
)
def test_unreadable_limits_are_unknown_not_zero(body):
    """Every failure path reports UNKNOWN. Mistaking "cannot read" for "none allowed" would turn a
    permissions gap into a launch-blocking false BLOCKED."""
    result = quota.parse_buckets(body, T4, "us-east1")
    assert result.limit is None
    assert not result.known


def test_read_limits_never_raises_and_reports_the_failure():
    def boom(service, metric):
        raise RuntimeError("no ADC here")

    readings = quota.read_limits("p", [T4], ["us-east1"], fetch=boom)
    assert readings[(T4.metric, "us-east1")].limit is None


def test_read_limits_fetches_once_per_metric_for_all_regions():
    """One GET returns every regional bucket, so three candidates cost the same as one."""
    calls = []

    def fetch(service, metric):
        calls.append((service, metric))
        return payload(bucket("12", "us-central1"), bucket("2", "us-east1"))

    regions = ["us-central1", "us-east1", "us-west1"]
    readings = quota.read_limits("p", [T4], regions, fetch=fetch)
    assert len(calls) == 1
    assert readings[(T4.metric, "us-central1")].limit == 12
    assert readings[(T4.metric, "us-east1")].limit == 2
    assert readings[(T4.metric, "us-west1")].limit is None


# --- the reconciler ---------------------------------------------------------------------------


def test_unknown_limit_leaves_bounds_untouched():
    outcome = quota.reconcile(gpu_demand(2, 12), reading(None))
    assert outcome.status == quota.QUOTA_UNKNOWN
    assert (outcome.min_units, outcome.max_units) == (2, 12)


def test_zero_blocks():
    outcome = quota.reconcile(gpu_demand(), reading(0))
    assert outcome.status == quota.QUOTA_BLOCKED
    assert "0" in outcome.detail


def test_a_limit_below_one_node_blocks():
    """Four T4s per node against an allowance of two is not a smaller fleet, it is no fleet."""
    outcome = quota.reconcile(gpu_demand(per_unit=4), reading(2))
    assert outcome.status == quota.QUOTA_BLOCKED


def test_enough_quota_is_ok_and_changes_nothing():
    outcome = quota.reconcile(gpu_demand(1, 12), reading(12))
    assert outcome.status == quota.QUOTA_OK
    assert (outcome.min_units, outcome.max_units) == (1, 12)
    assert not outcome.clamped


def test_a_generous_allowance_never_raises_the_floor():
    """Quota is evidence about permission, not about load. This is the one-directional rule."""
    outcome = quota.reconcile(gpu_demand(1, 4), reading(400))
    assert (outcome.min_units, outcome.max_units) == (1, 4)


def test_a_short_allowance_lowers_only_the_ceiling():
    outcome = quota.reconcile(gpu_demand(1, 12), reading(2))
    assert outcome.status == quota.QUOTA_CLAMPED
    assert (outcome.min_units, outcome.max_units) == (1, 2)
    assert outcome.autoscale_viable


def test_the_floor_comes_down_when_the_ceiling_lands_on_it():
    """Vertex rejects ``min == max``, so a ceiling lowered onto the floor must push it back one."""
    outcome = quota.reconcile(gpu_demand(8, 12), reading(4))
    assert outcome.status == quota.QUOTA_CLAMPED
    assert (outcome.min_units, outcome.max_units) == (3, 4)
    assert outcome.min_units < outcome.max_units
    assert outcome.autoscale_viable


def test_a_one_node_ceiling_gives_up_autoscaling_instead_of_a_negative_floor():
    outcome = quota.reconcile(gpu_demand(4, 12), reading(1))
    assert outcome.status == quota.QUOTA_CLAMPED
    assert (outcome.min_units, outcome.max_units) == (1, 1)
    assert not outcome.autoscale_viable
    assert "fixed-size" in outcome.detail


def test_a_shared_meter_reports_a_shortfall_but_does_not_reshape_the_fleet():
    demand = quota.QuotaDemand(
        metric=CPUS, region="us-east1", pool="cpu", per_unit=8, min_units=1, max_units=20
    )
    outcome = quota.reconcile(demand, quota.QuotaReading(CPUS, "us-east1", 80, "test"))
    assert outcome.status == quota.QUOTA_OK  # report-only
    assert (outcome.min_units, outcome.max_units) == (1, 20)
    assert "not clamped" in outcome.detail


def test_a_shared_meter_still_blocks_when_the_floor_will_not_fit():
    """The one shared case that is not a judgement call: the create fails regardless of policy."""
    demand = quota.QuotaDemand(
        metric=CPUS, region="us-east1", pool="cpu", per_unit=8, min_units=10, max_units=20
    )
    outcome = quota.reconcile(demand, quota.QuotaReading(CPUS, "us-east1", 40, "test"))
    assert outcome.status == quota.QUOTA_BLOCKED


def test_a_fixed_only_demand_is_a_straight_does_it_fit():
    """A whole-cluster vCPU total has no per-node rate, so there is no ceiling to lower."""
    total = quota.QuotaDemand(
        metric=CPUS,
        region="us-east1",
        pool="cluster",
        per_unit=0,
        min_units=0,
        max_units=0,
        fixed=224,
    )
    assert total.fixed_only
    fits = quota.reconcile(total, quota.QuotaReading(CPUS, "us-east1", 2200, "test"))
    assert fits.status == quota.QUOTA_OK
    over = quota.reconcile(total, quota.QuotaReading(CPUS, "us-east1", 100, "test"))
    assert over.status == quota.QUOTA_BLOCKED


# --- the advisor ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0s"), (45.0, "45s"), (90.0, "2m"), (3600.0, "1h00m"), (4512.0, "1h15m")],
)
def test_format_duration(seconds, expected):
    assert quota.format_duration(seconds) == expected


def test_estimate_wall_seconds_halves_when_the_fleet_doubles():
    at_6 = quota.estimate_wall_seconds(1200, 2, 6, 10.0)
    at_12 = quota.estimate_wall_seconds(1200, 2, 12, 10.0)
    assert at_6 == 1000.0
    assert at_12 == 500.0


def test_estimate_wall_seconds_is_none_without_a_measured_profile():
    assert quota.estimate_wall_seconds(1200, 2, 12, None) is None
    assert quota.estimate_wall_seconds(1200, 2, 0, 10.0) is None


def test_advice_prices_the_ceiling_and_its_multiples():
    outcome = quota.reconcile(gpu_demand(1, 12), reading(12))
    advice = quota.advise(
        outcome, n_cells=1200, slots_per_unit=2, saturating_units=20, seconds_per_cell=10.0
    )
    widths = [units for units, _ in advice.projections]
    assert widths == [12, 20, 24, 48]  # ceiling, un-throttling width, 2x, 4x
    assert advice.throttled
    assert any("20" in line for line in advice.render())


def test_advice_does_not_project_an_unaskable_fleet():
    """A 5,000-node projection beside an allowance of 12 is noise, not advice."""
    outcome = quota.reconcile(gpu_demand(1, 12), reading(12))
    advice = quota.advise(
        outcome, n_cells=10_000, slots_per_unit=2, saturating_units=5_000, seconds_per_cell=90.0
    )
    assert [units for units, _ in advice.projections] == [12, 24, 48]
    assert advice.throttled  # still reported as a fact about the run


def test_advice_on_an_unthrottled_run_is_not_throttled():
    outcome = quota.reconcile(gpu_demand(1, 4), reading(400))
    advice = quota.advise(
        outcome, n_cells=8, slots_per_unit=2, saturating_units=4, seconds_per_cell=10.0
    )
    assert not advice.throttled


# --- applying it to a plan ----------------------------------------------------------------------


def _preflight_for(
    plan: RayClusterPlan, region: str, limits: dict[str, str]
) -> quota.QuotaPreflight:
    def fetch(service, metric):
        value = limits.get(metric)
        return payload(bucket(value, region)) if value is not None else payload()

    return quota.preflight_ray(plan, [region], "p", fetch=fetch)[region]


def test_a_clean_region_leaves_the_plan_identical():
    plan = ray_plan()
    pre = _preflight_for(
        plan,
        "us-central1",
        {T4.metric: "12", CPUS.metric: "2200"},
    )
    assert not pre.blocked
    assert not pre.clamped
    assert quota.apply_to_ray_plan(plan, pre) is plan


def test_a_short_region_lowers_the_gpu_pool_and_nothing_else():
    plan = ray_plan()
    pre = _preflight_for(plan, "us-east1", {T4.metric: "2", CPUS.metric: "2200"})
    assert pre.clamped
    clamped = quota.apply_to_ray_plan(plan, pre)
    assert (clamped.gpu_min_nodes, clamped.gpu_max_nodes) == (1, 2)
    assert clamped.gpu_node_count == 2
    assert (clamped.cpu_min_nodes, clamped.cpu_max_nodes) == (1, 20)
    assert clamped.autoscale


def test_a_one_node_region_drops_autoscaling():
    plan = ray_plan(gpu_min_nodes=4)
    pre = _preflight_for(plan, "us-east1", {T4.metric: "1", CPUS.metric: "2200"})
    clamped = quota.apply_to_ray_plan(plan, pre)
    assert (clamped.gpu_min_nodes, clamped.gpu_max_nodes) == (1, 1)
    assert not clamped.autoscale


def test_a_zero_region_blocks_before_any_create():
    plan = ray_plan()
    pre = _preflight_for(plan, "us-east4", {T4.metric: "0", CPUS.metric: "2200"})
    assert pre.blocked
    assert pre.block_reason


def test_an_unreadable_region_is_not_blocked():
    """No ADC, no permission, a metric the API does not know — none of those stop a launch."""
    plan = ray_plan()
    pre = _preflight_for(plan, "us-east4", {})
    assert not pre.blocked
    assert not pre.clamped
    assert quota.apply_to_ray_plan(plan, pre) is plan


def test_a_cpu_only_plan_is_not_judged_on_an_accelerator_it_never_wanted():
    plan = ray_plan(gpu_node_count=0, gpu_min_nodes=0, gpu_max_nodes=0)
    demands = quota.ray_demands(plan, "us-east1")
    assert not any(d.pool == "gpu" for d in demands)


def test_ray_demands_meters_the_whole_cluster_vcpu_total():
    """Head + both pools bill one meter at three rates; only the sum describes the ask."""
    plan = ray_plan()
    total = next(d for d in quota.ray_demands(plan, "us-east1") if d.pool == "cluster")
    assert total.fixed_only
    assert total.fixed == 32 + 20 * 8 + 12 * 8


def test_preflight_serializes_for_telemetry():
    plan = ray_plan()
    pre = _preflight_for(plan, "us-east1", {T4.metric: "2", CPUS.metric: "2200"})
    body = pre.to_dict()
    assert body["region"] == "us-east1"
    assert body["clamped"] is True


# --- regional prerequisites ---------------------------------------------------------------------

_ATTACHMENT = "projects/307701787156/regions/us-central1/networkAttachments/sf-ray-na"


def test_only_the_attachments_own_region_is_reachable():
    """The confirmed reason multi-region Ray failover has never worked: the attachment is regional
    and Terraform builds exactly one."""
    ruled_out = quota.regions_without_attachment(
        _ATTACHMENT, ["us-central1", "us-east1", "us-west1"]
    )
    assert set(ruled_out) == {"us-east1", "us-west1"}
    assert "us-central1" in ruled_out["us-east1"]


def test_attachment_region():
    assert quota.attachment_region(_ATTACHMENT) == "us-central1"
    assert quota.attachment_region("not-a-resource-name") is None


@pytest.mark.parametrize("attachment", [None, "", "garbage"])
def test_no_readable_attachment_rules_out_nothing(attachment):
    """A public or peered cluster has no such constraint, and a failed *parse* is not evidence of a
    missing prerequisite — blocking a launch on either would be the worst kind of false negative."""
    assert quota.regions_without_attachment(attachment, ["us-central1", "us-east1"]) == {}


def test_advice_says_nothing_it_cannot_measure():
    """Three rows of "~unknown" is not a projection table. The ceiling still gets reported."""
    outcome = quota.reconcile(gpu_demand(1, 12), reading(12))
    advice = quota.advise(
        outcome, n_cells=1200, slots_per_unit=2, saturating_units=20, seconds_per_cell=None
    )
    rendered = advice.render()
    assert not any("unknown" in line for line in rendered)
    assert any("usable nodes = 12" in line for line in rendered)
    assert any("saturate at 20" in line for line in rendered)


def test_describe_pools_omits_a_pool_the_job_will_never_create():
    """A CPU-only job's plan still carries gpu bounds; printing them reads as a GPU ask."""
    cpu_only = ray_plan(gpu_node_count=0)
    assert quota._describe_pools(cpu_only) == "cpu[1,20]"
    gpu_only = ray_plan(cpu_node_count=0)
    assert quota._describe_pools(gpu_only) == "gpu[1,12]"
