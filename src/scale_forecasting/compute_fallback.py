"""Zone/region capacity failover for Dataproc **cluster** creation.

Compute capacity is *zonal*: a cluster create can be accepted and then fail to reach RUNNING with a
transient stockout — a ``ServiceUnavailable`` from Compute Engine, or a "…does not have enough
resources" / "Resources are insufficient in region" message — even when the project's quota is fine.
This is most acute for scarce accelerators (a T4/L4 GPU cluster) but happens for CPU too. Rather
than fail the run on the first stocked-out zone, the cluster submitter walks an ordered list of
places to try; **this module builds that list**. It no longer classifies the failures — that moved
to `scale_forecasting.capacity`, which owns one verdict vocabulary for both cluster services and
the walk that consumes it. What is left here is the geography.

The order (see `resolve_candidates`) is deliberately conservative:

1. the deployment region with **auto-zone placement** — byte-for-byte the pre-failover behavior, so
   a run that would have succeeded still takes the identical first attempt;
2. the deployment region's other zones (on the deployment subnet, which is regional and already
   covers them) — the zero-config win, since a regional subnet needs no new infrastructure;
3. *other* regions, but **only** those the operator has wired a subnet for in the fallback file — a
   cross-region cluster needs a subnet in that region (with NAT + Private Google Access), which the
   deployment doesn't create, so cross-region is strictly opt-in.

The candidate catalog is a user-editable JSON file (`SF_COMPUTE_FALLBACK`, else
``configs/compute_fallback.json``) prepopulated with the US regions/zones; when absent a built-in US
zone map still gives same-region zone failover out of the box. The list is a *runtime* fallback
detail chosen at submit time — never part of the config — so it does not affect ``run_id``.

Everything here is pure + import-free so it unit-tests offline; the cluster submitter drives it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import get_logger

if TYPE_CHECKING:
    from .batch_infra import BatchInfra
    from .settings import Settings

_log = get_logger(__name__)

# Env pointing at the operator's fallback catalog file; else the checked-in default is used when it
# exists, else the built-in US zone map below.
_ENV_FALLBACK_FILE = "SF_COMPUTE_FALLBACK"
_DEFAULT_FALLBACK_FILE = "configs/compute_fallback.json"

# Built-in US region → zones map, so same-region zone failover works with no file at all. This is
# the fallback for the deployment region's zones; cross-region entries still need a subnet (file).
US_ZONES: dict[str, list[str]] = {
    "us-central1": ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"],
    "us-east1": ["us-east1-b", "us-east1-c", "us-east1-d"],
    "us-east4": ["us-east4-a", "us-east4-b", "us-east4-c"],
    "us-east5": ["us-east5-a", "us-east5-b", "us-east5-c"],
    "us-south1": ["us-south1-a", "us-south1-b", "us-south1-c"],
    "us-west1": ["us-west1-a", "us-west1-b", "us-west1-c"],
    "us-west2": ["us-west2-a", "us-west2-b", "us-west2-c"],
    "us-west3": ["us-west3-a", "us-west3-b", "us-west3-c"],
    "us-west4": ["us-west4-a", "us-west4-b", "us-west4-c"],
}

# Dataproc retires individual sub-minor image versions on its own schedule, and refuses creates from
# them. The moving `2.2-debian12` alias resolves forward, so this only bites a create pinned to one
# fixed version — in practice a *custom* image, which bakes the sub-minor it was built from into a
# label and cannot move. Not a capacity error: no zone has the retired image either.
_RETIRED_IMAGE_MARKERS: tuple[str, ...] = (
    "can no longer be used to create new clusters",
    "no longer supported",
)


def is_retired_image_error(exc: BaseException) -> bool:
    """True if a cluster-create error reads as *this image version has been retired* (pure).

    `capacity.classify` already reads this as a `CONFIG_FAULT` and stops the walk, which is right:
    a stockout is somewhere-else-and-later, while a retired image is nowhere and never again. This
    predicate exists on top of that so the caller can say what actually has to change — the version
    is baked into an image the operator never picked a version for.
    """
    low = str(exc).lower()
    return any(marker in low for marker in _RETIRED_IMAGE_MARKERS)


@dataclass(frozen=True)
class Candidate:
    """One place to try creating the cluster: a region, an optional explicit zone, and the subnet.

    ``zone=None`` means auto-placement (Dataproc picks a zone within the subnet's region) — the
    first attempt uses it to preserve the pre-failover behavior exactly. ``subnetwork_uri`` is the
    subnet to create in: the deployment subnet for same-region candidates, or the operator-supplied
    subnet for a cross-region one.
    """

    region: str
    zone: str | None
    subnetwork_uri: str

    @property
    def label(self) -> str:
        """A short human label for logs: ``region/zone`` (``region/auto`` for auto-placement)."""
        return f"{self.region}/{self.zone or 'auto'}"


def _catalog_path() -> str | None:
    """The fallback-file path: the env override, else the default file if it exists (pure)."""
    override = os.environ.get(_ENV_FALLBACK_FILE)
    if override:
        return override
    return _DEFAULT_FALLBACK_FILE if os.path.exists(_DEFAULT_FALLBACK_FILE) else None


def load_zone_catalog(path: str | None = None) -> dict[str, dict]:
    """Load the region → ``{"zones": [...], "subnetwork_uri": <str|None>}`` catalog.

    Reads the JSON fallback file (``path`` if given, else ``SF_COMPUTE_FALLBACK`` / the checked-in
    default). The file shape is ``{"regions": {"<region>": {"zones": [...],
    "subnetwork_uri": <str|null>}}}`` — a ``subnetwork_uri`` opts a *non-deployment* region in
    (it needs its own subnet), while ``null`` leaves it listed-but-skipped. When no file is present
    the built-in `US_ZONES` map is returned (subnets ``None``) so same-region zone failover still
    works with zero configuration. Malformed files log and fall back to the built-in map rather than
    breaking a run.
    """
    resolved = path or _catalog_path()
    if resolved and os.path.exists(resolved):
        try:
            with open(resolved, encoding="utf-8") as fh:
                doc = json.load(fh)
            regions = doc.get("regions", {})
            if isinstance(regions, dict) and regions:
                return {
                    r: {
                        "zones": list(entry.get("zones") or US_ZONES.get(r, [])),
                        "subnetwork_uri": entry.get("subnetwork_uri") or None,
                    }
                    for r, entry in regions.items()
                }
            _log.warning("compute fallback %s has no 'regions'; using built-in zones", resolved)
        except (OSError, ValueError) as exc:
            _log.warning(
                "could not read compute fallback file %s (%r); using built-in zones", resolved, exc
            )
    return {r: {"zones": list(zones), "subnetwork_uri": None} for r, zones in US_ZONES.items()}


def resolve_candidates(
    *, settings: Settings, infra: BatchInfra, catalog: dict[str, dict] | None = None
) -> list[Candidate]:
    """The ordered list of places to try creating the cluster (pure).

    Order: (1) the deployment region with auto-zone placement on the deployment subnet — identical
    to the pre-failover single attempt; (2) the deployment region's other zones (from the catalog,
    else `US_ZONES`), still on the deployment subnet (regional, so it covers them); (3) other
    regions from the catalog, but **only** those with a ``subnetwork_uri`` set — a cross-region
    cluster needs a subnet in that region, so a region without one is listed-but-skipped (logged).
    Deduplicated, preserving first-seen order.
    """
    catalog = catalog if catalog is not None else load_zone_catalog()
    home = settings.region
    home_subnet = infra.subnetwork_uri

    out: list[Candidate] = [Candidate(region=home, zone=None, subnetwork_uri=home_subnet)]
    home_zones = catalog.get(home, {}).get("zones") or US_ZONES.get(home, [])
    for zone in home_zones:
        out.append(Candidate(region=home, zone=zone, subnetwork_uri=home_subnet))

    for region, entry in catalog.items():
        if region == home:
            continue
        subnet = entry.get("subnetwork_uri")
        if not subnet:
            _log.info(
                "compute fallback: skipping region %s (no subnetwork_uri configured — set one in "
                "the fallback file to enable cross-region failover)",
                region,
            )
            continue
        for zone in entry.get("zones") or US_ZONES.get(region, []):
            out.append(Candidate(region=region, zone=zone, subnetwork_uri=subnet))

    seen: set[tuple[str, str | None, str]] = set()
    deduped: list[Candidate] = []
    for cand in out:
        key = (cand.region, cand.zone, cand.subnetwork_uri)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped
