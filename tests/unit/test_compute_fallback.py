"""Offline tests for the Dataproc cluster capacity-failover candidate logic (pure, no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scale_forecasting import compute_fallback as cf


class _Settings:
    def __init__(self, region: str) -> None:
        self.region = region


class _Infra:
    def __init__(self, subnetwork_uri: str) -> None:
        self.subnetwork_uri = subnetwork_uri


_HOME_SUBNET = "projects/p/regions/us-central1/subnetworks/sf-compute"


# --- capacity classifier -------------------------------------------------------


# The classifier matches by class *name* (so it needs no google import); these stand in for the real
# google.api_core.exceptions types and must therefore carry their exact names.
class ServiceUnavailable(Exception):
    """Stands in for google.api_core.exceptions.ServiceUnavailable (matched by class name)."""


class ResourceExhausted(Exception):
    pass


@pytest.mark.parametrize(
    "exc",
    [
        ServiceUnavailable("UNAVAILABLE, errorSource: COMPUTE_ENGINE, Internal error"),
        ResourceExhausted("quota-ish"),
        RuntimeError("Resources are insufficient in region: us-central1"),
        RuntimeError("The zone does not have enough resources available"),
        RuntimeError("ZONE_RESOURCE_POOL_EXHAUSTED"),
    ],
)
def test_is_capacity_error_true(exc: Exception) -> None:
    assert cf.is_capacity_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Invalid machine type n1-bogus-9"),
        PermissionError("caller does not have permission"),
        ValueError("bad config"),
    ],
)
def test_is_capacity_error_false(exc: Exception) -> None:
    assert cf.is_capacity_error(exc) is False


# --- a retired image version: nowhere and never again, not somewhere-else-and-later ---
#
# Live 2026-09-02: smoke 16's GPU cluster was refused with "Selected software image version
# '2.2.86-debian12' can no longer be used to create new clusters" — the version baked into the
# custom GPU image the deploy had built nine days earlier. Its CPU sibling, on the moving
# `2.2-debian12` alias, created normally.


def test_a_retired_image_version_is_recognised_and_is_not_a_stockout() -> None:
    exc = RuntimeError(
        "400 Selected software image version '2.2.86-debian12' can no longer be used to "
        "create new clusters. Please select a more recent image."
    )
    assert cf.is_retired_image_error(exc) is True
    # It must not read as capacity: hopping zones would burn the failover walk on an image that
    # no zone has.
    assert cf.is_capacity_error(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Resources are insufficient in region us-central1"),
        RuntimeError("Invalid machine type n1-bogus-9"),
    ],
)
def test_other_failures_are_not_retired_images(exc: Exception) -> None:
    assert cf.is_retired_image_error(exc) is False


# --- catalog loading -----------------------------------------------------------


def test_load_zone_catalog_builtin_when_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv(cf._ENV_FALLBACK_FILE, raising=False)
    monkeypatch.chdir(tmp_path)  # no configs/compute_fallback.json here
    catalog = cf.load_zone_catalog()
    assert set(catalog) == set(cf.US_ZONES)
    assert catalog["us-central1"]["zones"] == cf.US_ZONES["us-central1"]
    assert all(entry["subnetwork_uri"] is None for entry in catalog.values())


def test_load_zone_catalog_reads_file(tmp_path: Any) -> None:
    path = tmp_path / "fb.json"
    path.write_text(
        json.dumps(
            {
                "regions": {
                    "us-central1": {"zones": ["us-central1-a"], "subnetwork_uri": None},
                    "us-west1": {
                        "zones": ["us-west1-a", "us-west1-b"],
                        "subnetwork_uri": "projects/p/regions/us-west1/subnetworks/s",
                    },
                }
            }
        )
    )
    catalog = cf.load_zone_catalog(str(path))
    assert catalog["us-west1"]["subnetwork_uri"] == "projects/p/regions/us-west1/subnetworks/s"
    assert catalog["us-central1"]["subnetwork_uri"] is None


def test_load_zone_catalog_falls_back_on_malformed(tmp_path: Any) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    catalog = cf.load_zone_catalog(str(path))
    assert set(catalog) == set(cf.US_ZONES)  # fell back to built-in


# --- candidate resolution ------------------------------------------------------


def test_first_candidate_is_home_auto_zone() -> None:
    """The pre-failover behavior: deploy region, auto-zone, deploy subnet — always first."""
    cands = cf.resolve_candidates(
        settings=_Settings("us-central1"),
        infra=_Infra(_HOME_SUBNET),
        catalog={"us-central1": {"zones": ["us-central1-a"], "subnetwork_uri": None}},
    )
    assert cands[0] == cf.Candidate(region="us-central1", zone=None, subnetwork_uri=_HOME_SUBNET)


def test_home_region_zones_use_home_subnet() -> None:
    cands = cf.resolve_candidates(
        settings=_Settings("us-central1"),
        infra=_Infra(_HOME_SUBNET),
        catalog={
            "us-central1": {"zones": ["us-central1-a", "us-central1-b"], "subnetwork_uri": None}
        },
    )
    assert cands == [
        cf.Candidate("us-central1", None, _HOME_SUBNET),
        cf.Candidate("us-central1", "us-central1-a", _HOME_SUBNET),
        cf.Candidate("us-central1", "us-central1-b", _HOME_SUBNET),
    ]


def test_other_region_without_subnet_is_skipped() -> None:
    cands = cf.resolve_candidates(
        settings=_Settings("us-central1"),
        infra=_Infra(_HOME_SUBNET),
        catalog={
            "us-central1": {"zones": ["us-central1-a"], "subnetwork_uri": None},
            "us-east1": {"zones": ["us-east1-b"], "subnetwork_uri": None},  # opt-in not taken
        },
    )
    assert all(c.region == "us-central1" for c in cands)


def test_other_region_with_subnet_is_included() -> None:
    east_subnet = "projects/p/regions/us-east1/subnetworks/s"
    cands = cf.resolve_candidates(
        settings=_Settings("us-central1"),
        infra=_Infra(_HOME_SUBNET),
        catalog={
            "us-central1": {"zones": ["us-central1-a"], "subnetwork_uri": None},
            "us-east1": {"zones": ["us-east1-b", "us-east1-c"], "subnetwork_uri": east_subnet},
        },
    )
    east = [c for c in cands if c.region == "us-east1"]
    assert east == [
        cf.Candidate("us-east1", "us-east1-b", east_subnet),
        cf.Candidate("us-east1", "us-east1-c", east_subnet),
    ]


def test_shipped_default_file_loads() -> None:
    """The checked-in configs/compute_fallback.json parses and lists the deploy region's zones."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    catalog = cf.load_zone_catalog(str(root / "configs" / "compute_fallback.json"))
    assert "us-central1" in catalog
    assert catalog["us-central1"]["zones"]  # non-empty
    # Shipped placeholders are subnet-less (cross-region opt-in), so nothing is enabled by default.
    assert all(entry["subnetwork_uri"] is None for entry in catalog.values())


def test_candidates_are_deduped() -> None:
    # A home-region zone that also appears verbatim shouldn't produce a duplicate candidate.
    cands = cf.resolve_candidates(
        settings=_Settings("us-central1"),
        infra=_Infra(_HOME_SUBNET),
        catalog={
            "us-central1": {"zones": ["us-central1-a", "us-central1-a"], "subnetwork_uri": None}
        },
    )
    assert len(cands) == len({(c.region, c.zone, c.subnetwork_uri) for c in cands})
