"""The reference cost profile shipped with the product — measured once, versioned, committed.

The **cold-start floor** under `source.resolve_profile_source`: what
``compute.profile.source: "baseline"`` consumes outright, and what ``"auto"`` falls back to when
discovery finds no prior run of yours to learn from. It is real evidence with one honest caveat —
it was measured, genuinely, but never on your data — which is why every profile resolved from it
comes back with ``basis="reference"`` and says so in a warning.

**Where the numbers came from.** The run recorded in the validation ledger as
``ray-100k-dcc77a9d1e9b`` (2026-09-03): 100,000 daily series of 1,460 observations from
``source_series_iceberg``, fitted on Ray on Vertex across four models in two families
(``theta`` / ``holtwinters`` / ``sarimax``, and ``xgboost``). The payload below was harvested
through the ordinary `registry.harvest.read_compute_harvest` path, which caps a read at 50,000
cells — so it aggregates a deterministic ``FARM_FINGERPRINT(ts_id)`` slice of 12,500 series x 4
models out of the 400,000 fits that run performed. That cap is a feature here, not a compromise:
it makes this payload byte-identical to what pinning ``compute.profile.source:
"ray-100k-dcc77a9d1e9b"`` resolves to, so the shipped baseline is not a separate kind of thing
from a harvest — it is one, frozen.

**What it deliberately does not cover.** That run had no deep-learning family and no GPU, so
neither is in here. `ComputeProfile.for_family` returns ``None`` for an unmeasured family and
``None`` is the signal to fall back to static config, so a run with a deep-learning family gets
today's arithmetic for that family and measured numbers for the rest. A fabricated GPU bound would
be worse than an absent one, and absence is the value the type already carries.

**The axis this exists for.** ``max_effective_cores`` lands between 1.01 and 1.05 for all four
models, which is a measurement saying every one of these fits is single-threaded — hence
``slot_cores: 1``. That is a property of the libraries rather than of the panel, so it transfers to
your data in a way a memory bound does not, and it is the one number a user should never have to
spend a run measuring for themselves.

**Why a Python module and not a JSON file.** `code_delivery.build_package_zip` walks ``*.py`` and
nothing else, so a data file would silently fail to reach a Dataproc worker or a Ray
``working_dir`` — and `source.resolve_profile_source` swallows a loader failure by design, so the
miss would degrade sizing on exactly the clusters that matter and never say a word. A module also
means a change here moves the code digest by construction.

**To re-cut it:** run a representative workload with ``compute.profile.measure`` at its default,
then::

    rows, source_table = read_compute_harvest("<run_id>")
    profile = harvest_profile(rows)

stamp a `ProfileProvenance` with the new ``run_id`` / ``measured_at`` / ``baseline_version`` and
the `signature.signature_from_rows` result (filling in ``freq``, which the rows do not carry),
and paste ``profile.to_dict()`` in below. Then add a row to ``docs/validation.md``: committed
numbers are a live-proof claim, and a baseline whose provenance is a chat message is the exact
failure that ledger exists to prevent.
"""

from __future__ import annotations

from typing import Any

from .cost import ComputeProfile, profile_from_dict

# A `ComputeProfile.to_dict` payload, verbatim. The derived ``slot_*`` / ``planning_*`` values are
# kept for the reader; `profile_from_dict` ignores them and recomputes from the raw fields, and a
# test asserts the two still agree so a stale hand-edit cannot go unnoticed.
_BASELINE: dict[str, Any] = {
    "provenance": {
        "basis": "reference",
        "source": "baseline",
        "run_id": "ray-100k-dcc77a9d1e9b",
        "baseline_version": "2026.09.03",
        "measured_at": "2026-09-03 06:25:56.104982+00:00",
        "signature": {
            "source_table": "source_series_iceberg",
            "n_series": 12500,
            "median_n_obs": 1460,
            "freq": "D",
        },
        "warnings": [],
    },
    "memory_margin": 1.3,
    "time_margin": 1.2,
    "n_measurements": 50000,
    "n_ok": 50000,
    "n_failed": 0,
    "n_sample_series": 12500,
    "sample_ts_ids": [
        "s_000047",
        "s_000048",
        "s_000054",
        "s_000061",
        "s_000092",
        "s_000097",
        "s_000112",
        "s_000124",
        "s_000125",
        "s_000139",
        "s_000179",
        "s_000182",
        "s_000188",
        "s_000202",
        "s_000203",
        "s_000207",
        "s_000222",
        "s_000237",
        "s_000246",
        "s_000247",
        "s_000254",
        "s_000258",
        "s_000259",
        "s_000266",
        "s_000279",
        "s_000281",
        "s_000283",
        "s_000306",
        "s_000311",
        "s_000315",
        "s_000332",
        "s_000337",
        "s_000340",
        "s_000348",
        "s_000362",
        "s_000380",
        "s_000394",
        "s_000396",
        "s_000398",
        "s_000401",
        "s_000404",
        "s_000408",
        "s_000429",
        "s_000438",
        "s_000457",
        "s_000460",
        "s_000471",
        "s_000474",
        "s_000476",
        "s_000484",
    ],
    "dropped_models": [],
    "first_error_by_model": {},
    "sample": [],
    "models": {
        "holtwinters": {
            "model_type": "holtwinters",
            "family": "statistical",
            "n_fits": 12500,
            "n_ok": 12500,
            "max_n_obs": 1460,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1879097344,
            "max_peak_gpu_bytes": None,
            "median_wall_s": 0.4025495095011138,
            "median_cpu_s": 0.40414047400054187,
            "max_effective_cores": 1.0131170461847316,
        },
        "sarimax": {
            "model_type": "sarimax",
            "family": "statistical",
            "n_fits": 12500,
            "n_ok": 12500,
            "max_n_obs": 1460,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1877147648,
            "max_peak_gpu_bytes": None,
            "median_wall_s": 1.6104711880000195,
            "median_cpu_s": 1.6169273734999194,
            "max_effective_cores": 1.0103504039931552,
        },
        "theta": {
            "model_type": "theta",
            "family": "statistical",
            "n_fits": 12500,
            "n_ok": 12500,
            "max_n_obs": 1460,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1875443712,
            "max_peak_gpu_bytes": None,
            "median_wall_s": 0.3279619864997585,
            "median_cpu_s": 0.3294067644997085,
            "max_effective_cores": 1.0473756299626,
        },
        "xgboost": {
            "model_type": "xgboost",
            "family": "ml",
            "n_fits": 12500,
            "n_ok": 12500,
            "max_n_obs": 1460,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1658896384,
            "max_peak_gpu_bytes": None,
            "median_wall_s": 0.7417708745001619,
            "median_cpu_s": 0.7449633084999903,
            "max_effective_cores": 1.0270293680400433,
        },
    },
    "families": {
        "ml": {
            "family": "ml",
            "models": ["xgboost"],
            "n_fits": 12500,
            "n_ok": 12500,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1658896384,
            "max_peak_gpu_bytes": None,
            "max_effective_cores": 1.0270293680400433,
            "median_wall_s": 0.7417708745001619,
            "total_wall_s_per_series": 0.7417708745001619,
            "memory_margin": 1.3,
            "time_margin": 1.2,
            "slot_rss_bytes": 2156565300,
            "slot_gpu_bytes": None,
            "slot_cores": 1,
            "planning_wall_s": 0.8901250494001942,
            "planning_total_wall_s_per_series": 0.8901250494001942,
        },
        "statistical": {
            "family": "statistical",
            "models": ["holtwinters", "sarimax", "theta"],
            "n_fits": 37500,
            "n_ok": 37500,
            "max_peak_rss_bytes": None,
            "max_process_rss_bytes": 1879097344,
            "max_peak_gpu_bytes": None,
            "max_effective_cores": 1.0473756299626,
            "median_wall_s": 0.4025495095011138,
            "total_wall_s_per_series": 2.340982684000892,
            "memory_margin": 1.3,
            "time_margin": 1.2,
            "slot_rss_bytes": 2442826548,
            "slot_gpu_bytes": None,
            "slot_cores": 1,
            "planning_wall_s": 0.4830594114013365,
            "planning_total_wall_s_per_series": 2.80917922080107,
        },
    },
}


def baseline_profile() -> ComputeProfile:
    """The shipped reference profile, parsed (pure).

    Not memoized: parsing a few kilobytes is cheaper than the cache lookup that would guard it, and
    `source.profile_for_run` already memoizes one level up, where the key is the whole resolution
    rather than just this leaf.
    """
    return profile_from_dict(_BASELINE)
