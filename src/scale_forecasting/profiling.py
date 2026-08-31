"""Measured compute profiling — what one cell of a model family actually costs.

Static per-model resource guesses fail in both directions: too generous and every run
over-provisions, too tight and the task OOMs. This module replaces the guess with a small
number of instrumented fits. Every runtime here already has a working autoscaler and a
settable slot shape; none of them is ever *told* how big the work is, so a slot ends up
either a hardcoded ``num_cpus=1`` or a service default. Rather than hand-maintaining a
table of "this model needs N GiB", one instrumented fit per sampled series reports five
axes — wall time, CPU time, absolute process footprint, marginal host RSS, peak device
bytes — and ``effective_cores = cpu_s / wall_s`` answers "does this model use more than
one core" by observation instead of by assertion.

**Two of those axes are traps, and both traps were found by measuring rather than by
reasoning.** They are worth stating up front because in each case the obvious
implementation produces a plausible number that is wrong by more than an order of
magnitude — the failure mode that a sizing pre-pass can least afford, since nothing
downstream can tell a bad measurement from a good one.

*The core count measures the machine unless the fit is pinned twice.* An unpinned probe on
an idle driver measures the driver: OpenBLAS/OpenMP inside statsmodels take every free
core, so on a 32-core box theta reported **19.7** effective cores and sarimax 11.5 — the
same models report ~3 on a 4-core box, and a naive mean can score highest of all. That CPU
bought no wall-clock win. Pinning takes *both* `threadpool_limits` and the
`_INTRAOP_ENV_VARS`, because the two reach pools with different lifetimes and each alone
leaves half the threads running (`threadpoolctl` alone still gave theta 4.8 and holtwinters
7.3). Under both, every model measures 1.00. See `_pinned_intraop_threads` for why.

*The memory number must be the absolute footprint, not the per-fit delta.* The intuitive
"RSS after minus RSS before" is unusable: it swings **17x on the order the sample ran in**,
because the first fit is charged for lazily importing the shared model stack while later
fits are served from an already-warm heap and report 0.00 MB. The absolute high-water
lands within 0.6% regardless of order, and it is also the number a slot actually needs —
a slot holds the interpreter and the libraries too. See `MeasuredFit` for the measurements.

Split along the same pure/I-O seam as ``ray_io`` / ``spark_io``, and for the same reason:
the arithmetic that decides how much hardware a run gets must be testable with no
accelerator, no cluster, and no cloud.

* **Pure** (no torch, no Ray, no Spark, no BigQuery): `series_stats` (cheap per-series
  statistics, one pass, no fits), `select_profile_sample` (which series to profile, and
  why — a deterministic function of those statistics), `build_profile` (measurements ->
  per-family cost model). All three are unit-tested with injected numbers, exactly as
  ``calibrate_gpu_fraction`` is.
* **I/O** (a real fit runs, so live-only): `measure_fit` — brackets exactly one
  ``run_cell`` and reports what it consumed — and `resolve_profile`, the
  driver-side pre-pass that decides whether to measure at all, picks the sample, runs the
  fits and hands back the aggregate. The pre-pass takes its measurement function as an
  argument, so even *it* is exercised offline against a deterministic stand-in.

**Why both tails.** `build_profile` keeps the **max** of the peaks and the **median** of
the times. Max governs *safety* — how many tasks may share a device or an executor without
an OOM. Median governs *throughput* — how much work the fleet has, i.e. how wide it needs
to be. Sizing a fleet off the worst case over-provisions every run; sizing memory off the
median OOM-kills it. Using one tail for both is the mistake this split exists to prevent.

**Absence is a value.** Every aggregated axis is ``| None``, and ``None`` means "we have no
basis for this number" — never "zero". A CPU-only family reports ``peak_gpu_bytes=None``,
because ``0 * 1.3`` is still ``0`` bytes of GPU, and that is a plan with no basis behind
it. Making the absence type-level is what stops a consumer silently sizing off nothing.

**This module reports bytes, seconds and cores. It never emits a runtime knob** — no GPU
fraction, no executor cores, no node count, no autoscaling bound. Turning a
`ComputeProfile` into those is three different translations of the same numbers and lands
later as ``plan_resources``, which consumes a `ComputeProfile` plus a cell count and
nothing else. The GPU-memory denominator stays in ``ray_io.device_memory_bytes``; this
module never re-declares a device table.

Public surface: ``SeriesStats``, ``SampleSpec``, ``MeasuredFit``, ``ModelCost``,
``FamilyCost``, ``ComputeProfile``, ``series_stats``, ``select_profile_sample``,
``measure_fit``, ``build_profile``, ``should_profile``, ``resolve_profile``.
"""

from __future__ import annotations

import contextlib
import math
import os
import statistics
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from .errors import DataError, get_logger

if TYPE_CHECKING:
    import pandas as pd

    from .config import RunConfig

_log = get_logger(__name__)

__all__ = [
    "ComputeProfile",
    "DataSignature",
    "FamilyCost",
    "MeasuredFit",
    "ModelCost",
    "ProfileProvenance",
    "SampleSpec",
    "SeriesStats",
    "build_profile",
    "compare_signatures",
    "harvest_profile",
    "measure_fit",
    "resolve_profile",
    "resolve_profile_source",
    "select_profile_sample",
    "signature_from_config",
    "signature_from_rows",
    "series_stats",
    "should_profile",
]

# --- sampling budget -----------------------------------------------------------

# Series to measure in the pre-pass. Three (today's calibration sample size) is an
# alphabetical prefix of the panel: no length spread, no complexity spread, and the longest
# series — the one that bounds memory — almost never in it. Eight is the smallest budget
# that spans four length strata *and* both complexity extremes without the pre-pass cost
# mattering for statistical models.
_DEFAULT_PROFILE_SAMPLES = 8  # series

# Hard ceiling on the budget. The pre-pass is ``samples x models`` real fits, run
# sequentially on the driver; an operator asking for 500 samples has turned a sizing
# pre-pass into the run. A request above this clamps (it is a budget cap, not bad input).
_MAX_PROFILE_SAMPLES = 64  # series

# Rows below which a series cannot be differenced, fit, or meaningfully timed.
_MIN_PROFILABLE_OBS = 2  # rows

# Length quantiles that define the strata, with their labels. Length is the primary
# stratification axis because it is the one that provably bounds memory. Upper-tail
# weighted because the interesting cost lives in the tail, and the 1.00 stratum guarantees
# the longest series is representable.
_LENGTH_QUANTILES = (0.50, 0.90, 0.99, 1.00)
_LENGTH_LABELS = ("p50", "p90", "p99", "max")

# The cost axes that are *not* length: volatility, intermittency, seasonal strength.
# Equal-weighted after rank normalization — no weights to tune, and a weight table would be
# an unmeasured assumption dressed up as a constant.
_COMPLEXITY_AXES = ("diff_cv", "zero_fraction", "acf_at_season")

# --- safety margins ------------------------------------------------------------

# Headroom on measured *peaks*. Matches the existing ``compute.gpu_safety_margin`` default
# so the two safety factors do not disagree once they are unified in config.
_DEFAULT_MEMORY_MARGIN = 1.3  # ratio, applied to max()

# Headroom on measured *times*. Deliberately smaller than the memory margin: over-estimating
# time buys extra slots (money), under-estimating memory kills the task (correctness).
# Asymmetric risk, asymmetric margin.
_DEFAULT_TIME_MARGIN = 1.2  # ratio, applied to median()

# --- numeric guards ------------------------------------------------------------

# Floor on ``cpu_s / wall_s``. A fit cannot be scheduled on less than one core, and a fit
# too fast to time would otherwise report 0.
_MIN_EFFECTIVE_CORES = 1.0  # cores

# Below this a wall-clock reading is under the resolution of ``perf_counter`` on any
# platform we run, so the ratio is meaningless rather than large.
_MIN_WALL_S = 1e-6  # seconds

# Below this a ``cpu_s / wall_s`` ratio is clock noise, not a second thread. For a genuinely
# single-threaded fit the two clocks differ only by scheduling jitter, and which one lands
# higher is a coin flip — so an unguarded ``ceil()`` turns 1.0000005 into a 2-core request and
# halves fleet density. Applied before rounding up, never to the reported raw ratio.
_CORE_SNAP_TOLERANCE = 0.05  # cores

# Native thread pools are capped to this for the duration of a probe fit. One, because the
# slot being sized holds one cell out of many running concurrently on the same executor —
# letting a probe fit take the whole idle driver measures the driver's core count and calls
# it a property of the model. See the module docstring.
_PROBE_INTRAOP_THREADS = 1  # threads per native pool, during measurement only

# Set (and restored) around a probe fit, to cap native pools belonging to libraries that are
# not loaded yet — the half of the problem `threadpool_limits` structurally cannot reach,
# since it can only re-size pools that already exist. See `_pinned_intraop_threads`.
_INTRAOP_ENV_VARS = (
    "OMP_NUM_THREADS",  # OpenMP — statsmodels' late-loaded pool, the one that escaped
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
)

# Denominator guard for ``diff_cv``. A series whose mean level is 0 (all zeros) must yield
# 0.0 volatility; an inf/NaN here would propagate into a sort key and destroy determinism.
_MIN_LEVEL = 1e-12  # target units

# ``getrusage.ru_maxrss`` is KiB on Linux and bytes on macOS. This is not pedantry: getting
# it wrong is a silent 1024x error in the axis that sizes host memory — one direction on a
# dev laptop, the other on a cluster.
_RSS_UNIT_BYTES = 1 if sys.platform == "darwin" else 1024  # bytes per ru_maxrss unit


# --- pure: cheap per-series statistics -----------------------------------------


@dataclass(frozen=True)
class SeriesStats:
    """Cheap per-series statistics that predict fit cost without doing a fit (pure).

    Computable as one pandas ``groupby`` or one Spark aggregation, so it scales to the
    hero panel: `series_stats` is the pandas path, but the record is deliberately
    self-describing so a future Spark aggregation can construct these rows directly and
    feed the identical sampler with no panel in driver memory.

    Every field is finite by construction — degenerate content (all-zero, all-NaN,
    constant, single-row) yields ``0.0``/``0``, never ``inf`` or ``NaN``. A non-finite
    value here would become a sort key in `select_profile_sample` and destroy the
    determinism the whole module rests on.
    """

    ts_id: str  # the series key, from cfg.data.ts_id_col, coerced with str()
    n_obs: int  # rows the fit will see (duplicates and NaN gaps included), >= 1
    zero_fraction: float  # [0,1] share of non-null target values == 0; intermittency
    distinct_values: int  # >= 0 over non-null values; 0 == all-null, 1 == flat/degenerate
    diff_cv: float  # >= 0, finite; std(diff(y), ddof=0) / max(|mean(y)|, _MIN_LEVEL)
    acf_at_season: float  # [-1,1]; lag-m Pearson autocorrelation, m = seasonal_period(freq)
    n_exog: int  # declared exog columns present in the panel; feature-matrix width


def series_stats(panel: pd.DataFrame, cfg: RunConfig) -> list[SeriesStats]:
    """Cheap per-series statistics over a driver-side panel — one pass, no fits (pure).

    One ``groupby`` over the whole panel, never a Python loop over per-series frames, so it
    stays O(rows) at hero scale. Reads only ``data.{ts_id_col, date_col, target_col, freq}``
    and ``features.exog`` — no clock, no RNG, no environment, no I/O.

    **Structural violations raise; degenerate content degrades.** A missing id/date/target
    column means the config points at the wrong table and every number downstream would be
    computed from the wrong bytes, so that raises `DataError` with the same wording the
    pre-flight validator uses. A missing *declared exog* column does **not** raise: this is a
    sizing pre-pass and it must never change whether a run starts, and understating the
    feature width is bounded and absorbed by the memory margin. Likewise an unparseable
    timestamp or a non-numeric target is coerced (``errors="coerce"``) rather than rejected —
    neither changes ``n_obs``, and ``n_obs`` is the axis that bounds memory. Non-finite target
    values are treated as gaps, so every returned field is finite.

    Rows are stable-sorted by ``(ts_id, timestamp, target)`` before any order-dependent
    statistic, so the result does not depend on the order the reader happened to emit.
    Timestamps parse with ``utc=True`` because the only use here is ordering, and a mixed-offset
    column must still yield one consistent instant order.

    The target is the **third** sort key so that no residual dependency on arrival order
    survives. It is a no-op on the contracted input — one row per ``(ts_id, timestamp)`` on a
    regular grid, which `validation.validate_panel` enforces — and it matters on input that
    never reached that validator: duplicate ``(ts_id, timestamp)`` rows from a restatement or a
    re-landed partition, and a date column that fails to parse at all, which makes *every* row
    ``NaT`` and hands the whole series' ordering to the reader. ``diff_cv`` and
    ``acf_at_season`` are computed over this order and feed a sort key in the sampler, so
    without the third key the same table read twice could size two different fleets. No engine
    here promises row order, so paying one extra sort key to be exactly reproducible is the
    cheap side of that trade.

    Duplicate ``(ts_id, timestamp)`` rows are **counted, not deduped** — ``run_cell`` is handed
    exactly these rows, and the cost driver is what the fit sees, not what a clean panel would
    have had. An empty panel returns ``[]`` (before any column check: zero rows carry zero
    information either way). Output is ordered by ``ts_id`` ascending, never panel order.
    """
    import pandas as pd

    # Local import so this module's own import surface stays stdlib-only; ``seasonality`` owns
    # the freq -> lag-m mapping and already falls back for an unrecognised freq.
    from .seasonality import seasonal_period

    if panel.empty:
        return []

    d = cfg.data
    have = list(panel.columns)
    for col in (d.ts_id_col, d.date_col, d.target_col):
        if col not in panel.columns:
            raise DataError(f"missing column '{col}'; panel has {have}")

    # Feature-matrix width is the count of declared exog columns *present*: ``build_features``
    # builds one column per declared exog, and a declared-but-absent one contributes nothing.
    n_exog = sum(1 for col in cfg.features.exog if col in panel.columns)

    target = pd.to_numeric(panel[d.target_col], errors="coerce").astype("float64")
    frame = pd.DataFrame(
        {
            "ts_id": panel[d.ts_id_col].astype(str),
            "ds": pd.to_datetime(panel[d.date_col], errors="coerce", utc=True),
            # inf is a gap, not a level: it would otherwise poison mean/std and, through them,
            # a sort key.
            "y": target.where(target.abs() < math.inf),
        }
    )
    # NaT sorts last so an unparseable timestamp degrades the *order* of one series, not the run.
    # "y" is the tie-break that makes the order a function of the row multiset (see docstring).
    frame = frame.sort_values(["ts_id", "ds", "y"], kind="stable", na_position="last")

    n_obs = frame.groupby("ts_id", sort=True).size()
    valid = frame.dropna(subset=["y"])
    by_valid = valid.groupby("ts_id", sort=True)["y"]

    n_valid = by_valid.size().reindex(n_obs.index, fill_value=0)
    n_zero = (valid["y"] == 0.0).groupby(valid["ts_id"], sort=True).sum().reindex(n_obs.index)
    distinct = by_valid.nunique().reindex(n_obs.index, fill_value=0)

    # Volatility: sigma of the first differences over the non-null values, scaled by the level
    # so it is comparable across series. Fewer than two non-null values -> no differences -> NaN,
    # zeroed below. The level is floored so an all-zero series yields 0.0 rather than inf/NaN.
    diffs = by_valid.diff()
    diff_std = diffs.groupby(valid["ts_id"], sort=True).std(ddof=0).reindex(n_obs.index)
    level = by_valid.mean().abs().reindex(n_obs.index).clip(lower=_MIN_LEVEL)

    table = pd.DataFrame(
        {
            "n_obs": n_obs,
            "zero_fraction": n_zero / n_valid.where(n_valid > 0),
            "distinct_values": distinct,
            "diff_cv": diff_std / level,
            "acf_at_season": _acf_at_season(valid, seasonal_period(d.freq)).reindex(n_obs.index),
        }
    )
    for col in ("zero_fraction", "diff_cv", "acf_at_season"):
        # One sweep to the post-condition: every float finite. NaN and inf both fail the
        # comparison, so both land on 0.0 ("no evidence"), which is the only honest reading.
        table[col] = table[col].where(table[col].abs() < math.inf, 0.0)

    return [_one_series_stats(row, n_exog) for row in table.itertuples()]


def _one_series_stats(row: Any, n_exog: int) -> SeriesStats:
    """Build one `SeriesStats` from an aggregated row (pure; plain Python scalars only).

    The casts are load-bearing, not cosmetic: a numpy scalar leaking into the record would
    survive dataclass equality but break ``json.dumps`` on anything derived from it, and
    ``numpy.str_`` ids compare differently from ``str`` ids in the sampler's tie-breaks.
    """
    return SeriesStats(
        ts_id=str(row.Index),
        n_obs=int(row.n_obs),
        zero_fraction=float(row.zero_fraction),
        distinct_values=int(row.distinct_values),
        diff_cv=float(row.diff_cv),
        acf_at_season=float(row.acf_at_season),
        n_exog=int(n_exog),
    )


def _acf_at_season(valid: pd.DataFrame, period: int) -> pd.Series:
    """Lag-``period`` Pearson autocorrelation per series, in one vectorized pass (pure).

    Computed from grouped sums rather than a per-series ``corr`` call so the whole panel is
    one aggregation. Signed, and clipped into ``[-1, 1]``: a *negative* lag-m correlation is a
    real and different phenomenon from no correlation, so it is preserved here and the sampler
    ranks its magnitude. A series shorter than the season, or with zero variance on either
    side of the shift, has no correlation to estimate and yields ``0.0`` — "no evidence of
    seasonality", which is true — rather than a ``NaN`` that would become a sort key.

    **Values are centered per series before the sums are formed.** The one-pass form
    ``n*sum(x^2) - (sum x)^2`` subtracts two nearly equal large numbers, so it loses all
    significance once the coefficient of variation falls below about ``sqrt(eps)`` ~ 1e-8 —
    the regime of a high-baseline, low-variance series (a revenue line near 1e9 that moves by
    tens). The failure mode is not a ``NaN`` the estimability guard would catch; it is a
    plausible wrong number, decided by which way the rounding fell, that then becomes a sort
    key. Centering costs one extra grouped pass and removes the cancellation entirely.

    **At least three lagged pairs are required.** A Pearson correlation over exactly two points
    is ``+/-1`` by construction — two points always lie on a line — so a series with exactly
    ``period + 2`` observations would otherwise report maximum seasonal strength regardless of
    its content, and the sampler, which ranks that magnitude, would preferentially pick it as
    its stratum's complexity extreme. That is precisely the short-series band the complexity
    axis exists to reach, so the artefact lands where it does the most damage.
    """
    import pandas as pd

    centered = valid["y"] - valid.groupby("ts_id", sort=True)["y"].transform("mean")
    lagged = centered.groupby(valid["ts_id"], sort=True).shift(period)
    pairs = pd.DataFrame({"ts_id": valid["ts_id"], "x": centered, "z": lagged}).dropna(
        subset=["z"]
    )
    pairs = pairs.assign(xx=pairs["x"] ** 2, zz=pairs["z"] ** 2, xz=pairs["x"] * pairs["z"])
    agg = pairs.groupby("ts_id", sort=True).agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sz=("z", "sum"),
        sxx=("xx", "sum"),
        szz=("zz", "sum"),
        sxz=("xz", "sum"),
    )

    cov = agg["n"] * agg["sxz"] - agg["sx"] * agg["sz"]
    var_x = agg["n"] * agg["sxx"] - agg["sx"] ** 2
    var_z = agg["n"] * agg["szz"] - agg["sz"] ** 2
    # >= 3, not >= 2: a correlation over two points is +/-1 by construction (see docstring).
    estimable = (agg["n"] >= 3) & (var_x > 0) & (var_z > 0)
    acf = (cov / (var_x * var_z).pow(0.5)).where(estimable, 0.0)
    return acf.clip(-1.0, 1.0)


# --- pure: which series to profile, and why ------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """One series chosen for the profiling pre-pass, with the reason it was chosen (pure).

    The reason is part of the contract, not a debug string. It is what makes a sizing
    decision reviewable after the fact ("this fleet was sized off the longest series and
    the two most intermittent ones"), and it is what a test asserts on instead of asserting
    on an opaque list of ids. ``stats`` is embedded rather than re-looked-up because the
    sample is snapshotted for audit, and a ``ts_id`` alone does not explain a sizing
    decision six weeks later.
    """

    ts_id: str
    n_obs: int
    stratum: int  # 0-based index into the deduped length strata, 0 == shortest band
    stratum_label: str  # "p50" | "p90" | "p99" | "max" — the quantile band it came from
    reason: str  # "longest" | "complexity_max" | "complexity_min" | "fill"
    complexity: float  # [0,1] equal-weighted mean of the rank-normalized _COMPLEXITY_AXES
    stats: SeriesStats

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the pick *and* the panel properties behind it.

        Without this the explanation stops at the telemetry boundary: a bare list of ids says
        which series were measured but not why, and "why" is the whole reason `reason` and
        `stats` are on the contract. Plain scalars only, same rule as
        `ComputeProfile.to_dict`.
        """
        return {
            "ts_id": self.ts_id,
            "n_obs": self.n_obs,
            "stratum": self.stratum,
            "stratum_label": self.stratum_label,
            "reason": self.reason,
            "complexity": self.complexity,
            "zero_fraction": self.stats.zero_fraction,
            "distinct_values": self.stats.distinct_values,
            "diff_cv": self.stats.diff_cv,
            "acf_at_season": self.stats.acf_at_season,
            "n_exog": self.stats.n_exog,
        }


def _length_strata(n_obs_values: list[int]) -> list[tuple[int, str]]:
    """The deduped ascending ``(edge, label)`` length bands for a candidate set (pure).

    Edges come from **nearest rank** on the sorted lengths — ``sorted[floor(q * (n - 1))]`` —
    not from an interpolating quantile: no numpy, no dtype-dependent midpoint, so the
    boundaries are exactly reproducible on any platform. The ``1.00`` edge is the longest
    series by construction, which is what makes the top band always representable.

    Duplicate edges collapse, keeping the **lowest** quantile that produced each one so its
    label survives. A panel whose series are all the same length therefore collapses to a
    single ``"p50"`` band instead of four identical ones.
    """
    ordered = sorted(n_obs_values)
    n = len(ordered)
    strata: list[tuple[int, str]] = []
    for quantile, label in zip(_LENGTH_QUANTILES, _LENGTH_LABELS, strict=True):
        edge = ordered[math.floor(quantile * (n - 1))]
        if not strata or edge != strata[-1][0]:
            strata.append((edge, label))
    return strata


def _rank_normalized(values: list[tuple[str, float]]) -> dict[str, float]:
    """Rank each ``(ts_id, value)`` ascending and scale the rank into ``[0, 1]`` (pure).

    Rank, not min-max: ranks are scale-free, so one outlier cannot flatten an axis for every
    other series (a legitimately huge ``diff_cv`` from a series whose level sits on the guard
    floor would otherwise collapse the axis to two useful values). Ties break on ``ts_id``, so
    every candidate gets a distinct rank and the composite is a total order. A single
    candidate ranks ``0.0`` rather than dividing by zero.
    """
    n = len(values)
    if n <= 1:
        return {ts_id: 0.0 for ts_id, _value in values}
    ordered = sorted(values, key=lambda pair: (pair[1], pair[0]))
    return {ts_id: rank / (n - 1) for rank, (ts_id, _value) in enumerate(ordered)}


def _complexity_scores(candidates: list[SeriesStats]) -> dict[str, float]:
    """Equal-weighted mean of the rank-normalized complexity axes, per ``ts_id`` (pure).

    The axes are volatility, intermittency and seasonal strength — the cost drivers that are
    *not* length. ``acf_at_season`` enters as a magnitude: a strong negative lag-m correlation
    is as much structure for a seasonal model to fit as a strong positive one.

    A non-finite value in a caller-supplied `SeriesStats` (a Spark aggregation over an all-null
    series returns ``NaN`` readily) is coerced to ``0.0`` rather than raised on, because a
    ``NaN`` sort key produces an order that varies with the pandas/numpy version — the exact
    class of nondeterminism this module cannot tolerate.

    Consequence worth stating: ``complexity`` is a property of the **panel**, not of the series
    alone. The same series in a different panel gets a different score.
    """
    ranked: list[dict[str, float]] = []
    for axis in _COMPLEXITY_AXES:
        raw: list[tuple[str, float]] = []
        for stats in candidates:
            value = float(getattr(stats, axis))
            value = value if math.isfinite(value) else 0.0
            raw.append((stats.ts_id, abs(value) if axis == "acf_at_season" else value))
        ranked.append(_rank_normalized(raw))
    return {
        stats.ts_id: sum(axis_ranks[stats.ts_id] for axis_ranks in ranked) / len(ranked)
        for stats in candidates
    }


def select_profile_sample(
    stats: Sequence[SeriesStats],
    *,
    samples: int = _DEFAULT_PROFILE_SAMPLES,
) -> list[SampleSpec]:
    """Choose which series to profile: stratified on length, spread on complexity (pure).

    Takes statistics, not a panel: `series_stats` is the pandas path, and a Spark aggregation
    can build the identical rows for a panel too big to collect. Takes no ``cfg`` either, which
    is what makes "same stats in, same sample out" trivially true — there is no clock, no RNG,
    and every intermediate collection is an explicitly sorted list, so the result is
    byte-identical for a shuffled input.

    The algorithm, in order:

    1. Candidates are the series with at least `_MIN_PROFILABLE_OBS` rows, sorted by ``ts_id``.
       A series too short for the *models* is deliberately kept — minimum history is model
       policy (``horizon``, ``backtest.min_train``), and duplicating that arithmetic here would
       put it in two places. Its fit fails, `measure_fit` records ``ok=False``, and
       `build_profile` skips it.
    2. Flat series (``distinct_values <= 1``) are demoted out of the complexity phase: they
       short-circuit most fits, so sizing a fleet on one under-provisions. They stay eligible
       for the longest pick and for backfill, and the demotion is skipped entirely when *every*
       candidate is flat — a real all-flat panel must still yield a sample.
    3. **The longest series is claimed first**, before any budget arithmetic. A memory bound
       derived from a sample that excludes the longest series is not a bound. Ties go to the
       smallest ``ts_id``.
    4. The remaining budget goes round-robin over the non-empty length strata, alternating the
       complexity **max** then **min** inside each — so a small budget spans the length range
       instead of exhausting itself in one bucket, and a short, violently intermittent series
       stays reachable.
    5. Anything left over backfills from the unclaimed candidates, longest first.

    ``samples`` below 1 raises `ValueError` — it is a caller-argument violation, and returning
    ``[]`` would present "no basis" as "nothing to profile". Above `_MAX_PROFILE_SAMPLES` it
    clamps silently, because that ceiling is a budget cap rather than bad input. A duplicate
    ``ts_id`` also raises: `series_stats` cannot produce one, so it means two stat sources were
    merged, and silently deduping would shrink the real budget and measure one series twice,
    biasing the median.

    ``result[0]`` is always the ``"longest"`` pick; the rest is ordered by
    ``(stratum, reason, ts_id)``.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")

    seen: set[str] = set()
    for stat in stats:
        if stat.ts_id in seen:
            raise ValueError(f"duplicate ts_id '{stat.ts_id}' in stats")
        seen.add(stat.ts_id)

    candidates = sorted(
        (stat for stat in stats if stat.n_obs >= _MIN_PROFILABLE_OBS), key=lambda s: s.ts_id
    )
    if not candidates:
        return []

    budget = min(samples, _MAX_PROFILE_SAMPLES)
    complexity = _complexity_scores(candidates)
    strata = _length_strata([stat.n_obs for stat in candidates])
    # First band whose edge covers the length; the last edge is the longest series, so this
    # always finds one.
    stratum_of = {
        stat.ts_id: next(i for i, (edge, _label) in enumerate(strata) if edge >= stat.n_obs)
        for stat in candidates
    }

    def spec(stat: SeriesStats, reason: str) -> SampleSpec:
        index = stratum_of[stat.ts_id]
        return SampleSpec(
            ts_id=stat.ts_id,
            n_obs=stat.n_obs,
            stratum=index,
            stratum_label=strata[index][1],
            reason=reason,
            complexity=complexity[stat.ts_id],
            stats=stat,
        )

    longest = min(candidates, key=lambda s: (-s.n_obs, s.ts_id))
    claimed = {longest.ts_id}
    picked = [spec(longest, "longest")]

    # Flat series are poor evidence, but "every series is flat" is a panel, not an error.
    eligible = [stat for stat in candidates if stat.distinct_values > 1] or candidates
    buckets: dict[int, list[SeriesStats]] = {}
    for stat in eligible:
        buckets.setdefault(stratum_of[stat.ts_id], []).append(stat)

    # One pick per stratum per pass, ascending; each stratum alternates max/min on its own turn
    # counter so a stratum that runs dry does not shift the alternation of the others.
    turns = dict.fromkeys(buckets, 0)
    while len(picked) < budget:
        progressed = False
        for index in sorted(buckets):
            if len(picked) >= budget:
                break
            pool = [stat for stat in buckets[index] if stat.ts_id not in claimed]
            if not pool:
                continue
            want_max = turns[index] % 2 == 0
            turns[index] += 1
            chosen = min(
                pool,
                key=(
                    (lambda s: (-complexity[s.ts_id], s.ts_id))
                    if want_max
                    else (lambda s: (complexity[s.ts_id], s.ts_id))
                ),
            )
            claimed.add(chosen.ts_id)
            picked.append(spec(chosen, "complexity_max" if want_max else "complexity_min"))
            progressed = True
        if not progressed:
            break

    # Backfill from whatever is left (demoted flat series included), longest first.
    if len(picked) < budget:
        rest = sorted(
            (stat for stat in candidates if stat.ts_id not in claimed),
            key=lambda s: (-s.n_obs, s.ts_id),
        )
        for stat in rest[: budget - len(picked)]:
            claimed.add(stat.ts_id)
            picked.append(spec(stat, "fill"))

    # The headline invariant is readable off index 0; the tail is sorted for a stable audit diff.
    reason_rank = {"complexity_max": 0, "complexity_min": 1, "fill": 2}
    tail = sorted(picked[1:], key=lambda s: (s.stratum, reason_rank[s.reason], s.ts_id))
    return [picked[0], *tail]


# --- I/O: one instrumented fit --------------------------------------------------


@dataclass(frozen=True)
class MeasuredFit:
    """Four measured axes from exactly one ``run_cell``, plus what it was measuring.

    ``family`` is carried on the measurement rather than re-derived in `build_profile`, so
    the aggregation is pure arithmetic over injected records: an offline test constructs
    `MeasuredFit` values directly and never touches the model registry, a GPU, or a
    cluster. Same injection seam as ``calibrate_gpu_fraction(measured_peaks_bytes=...)``.

    **Two memory numbers, because the obvious one does not survive contact with an allocator.**
    ``process_rss_bytes`` is what sizes a slot; ``peak_rss_bytes`` is a diagnostic. The reason
    is measured, not assumed — the same four models, fitted in three different orders:

    ==========  =====================================  =========================
    model       ``peak_rss_bytes`` (delta), by order    ``process_rss_bytes``
    ==========  =====================================  =========================
    theta       79.0 MB / 4.6 MB / 13.1 MB              646 / 672 / 649 MB
    sarimax     27.8 MB / 97.7 MB / 27.6 MB             676 / 666 / 676 MB
    ==========  =====================================  =========================

    The delta swings **17x on the order the models ran in**; the absolute high-water lands
    within 0.6% of 676 MB every time. Two effects cause the swing, and they push opposite ways:
    whichever model fits *first* is charged for lazily importing the shared model stack, while
    every model after it allocates inside a heap that is already warm, so the allocator serves
    the fit from resident pages and ``ru_maxrss`` never rises — a warmed-up run reports 0.00 MB
    for theta and holtwinters, which is not a claim that they are free.

    So **attribution is the wrong question.** A worker slot must hold the interpreter, the
    libraries, *and* the fit; the shared residency is not overhead to be factored out, it is
    part of what the slot must fit. ``process_rss_bytes`` measures exactly that and is stable,
    which is why `FamilyCost.slot_rss_bytes` is built from it. It over-states for a light family
    profiled in the same process as a heavy one — the deliberate direction, per the asymmetry
    `_DEFAULT_MEMORY_MARGIN` is chosen on: over-estimating memory costs money, under-estimating
    it kills the task. ``None`` means NOT MEASURED (no ``resource`` module), never zero.

    ``peak_rss_bytes`` is kept because the marginal cost is worth seeing even when it is not
    worth sizing on. Read it as a **lower bound where 0 means "no evidence"**, never as proof a
    fit was free; `build_profile` discards non-positive values rather than folding them into a
    max. It assumes fits are **sequential in this process** — concurrent `measure_fit` calls
    make each other's delta meaningless.

    ``rss_peak_reset`` records whether the high-water mark could be zeroed before the fit.
    Without the reset the mark is monotonic for the life of the interpreter, so the delta
    degrades from "order-dependent" to "zero for everything after the first heavy model".
    `measure_fit` resets it on Linux; ``False`` means the delta is worth even less than usual.
    The reset also rebases ``process_rss_bytes``, and helpfully so: the absolute reading becomes
    "this process's live footprint plus what this fit added" rather than "the largest transient
    any earlier fit ever reached and has since freed". The former is what a slot running this
    model needs; the latter is another model's spike wearing this model's name.

    ``peak_gpu_bytes is None`` means **NOT MEASURED** — never "measured zero". The profile
    runs on the driver at submit time, where there is usually no accelerator. If a missing
    GPU reading arrived as ``0``, a consumer would compute a minimum GPU fraction and pack
    ten tasks onto a device that fits two. ``None`` forces the fall-back to a nominal
    fraction and a refinement on-cluster, which is the existing two-phase behaviour.

    ``ok=False`` carries a failure the way `CellResult` does, rather than raising: a flaky
    probe widens sizing to nominal, it does not sink the run. Failed records are excluded
    from every aggregate but counted, so "we sized off 6 of 8 fits" stays visible.
    """

    ts_id: str
    model_type: str
    family: str  # "statistical" | "ml" | "deep_learning"; "unknown" if lookup failed
    n_obs: int  # rows fed to the fit; makes the measurement interpretable
    wall_s: float  # seconds, time.perf_counter delta — the throughput number
    cpu_s: float  # seconds, time.process_time delta — sums across threads
    peak_rss_bytes: int  # bytes, ru_maxrss delta x _RSS_UNIT_BYTES, floored at 0
    peak_gpu_bytes: int | None  # bytes, torch.cuda.max_memory_allocated; None == NOT MEASURED
    ok: bool  # False == run_cell returned status="error", or something raised
    error: str | None  # the failure message when ok is False, else None
    # --- how the measurement was taken; without these the numbers are uninterpretable ------
    intraop_threads: int | None = None  # native-pool cap in force; None == could not be pinned
    host_cpu_count: int | None = None  # os.cpu_count() of the measuring host
    rss_peak_reset: bool = False  # was the ru_maxrss high-water mark zeroed before the fit
    process_rss_bytes: int | None = None  # bytes, ABSOLUTE process peak; None == NOT MEASURED

    @property
    def effective_cores(self) -> float:
        """``cpu_s / wall_s``, floored at 1.0 — measured thread-parallelism of this fit.

        The direct answer to "can this model use more than one CPU", with no per-model
        declaration to maintain and no per-runtime variation. A ``wall_s`` below
        `_MIN_WALL_S` yields `_MIN_EFFECTIVE_CORES` rather than a division blow-up.

        **Only meaningful relative to ``intraop_threads``.** The ratio counts threads that
        actually ran, so an unpinned fit on an idle many-core driver reports that driver's core
        count for almost any model. Read as "parallelism this fit found under a cap of
        ``intraop_threads``"; ``intraop_threads=None`` means the cap could not be applied and
        the ratio is an upper bound contaminated by ``host_cpu_count``.
        """
        if not math.isfinite(self.wall_s) or self.wall_s < _MIN_WALL_S:
            return _MIN_EFFECTIVE_CORES
        return max(_MIN_EFFECTIVE_CORES, self.cpu_s / self.wall_s)


def _rss_bytes() -> int | None:  # pragma: no cover - platform probe, live-only
    """This process's peak-RSS high-water mark in bytes, or None where ``resource`` is absent.

    ``ru_maxrss`` is KiB on Linux and bytes on macOS (`_RSS_UNIT_BYTES`). Non-POSIX platforms
    have no ``resource`` module at all; ``None`` there becomes a ``0`` delta, which already
    means "no evidence" on this axis, so the platform gap needs no second representation.
    """
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RSS_UNIT_BYTES
    except Exception:  # noqa: BLE001 - a probe must never sink the run it is sizing
        return None


def _reset_rss_peak() -> bool:  # pragma: no cover - platform probe, live-only
    """Zero this process's peak-RSS high-water mark; True when it took, False otherwise.

    Linux exposes this as a write of ``"5"`` to ``/proc/self/clear_refs``
    (``CLEAR_REFS_MM_HIWATER_RSS``), which resets the mark to the current RSS. Without it
    ``ru_maxrss`` only ever rises, so in a sequential pre-pass every fit after the first heavy
    one measures as free and the profile becomes a function of the order the models ran in.
    Silently a no-op anywhere the file is absent or unwritable — the delta then keeps its
    documented lower-bound meaning, and `MeasuredFit.rss_peak_reset` records which happened.
    """
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except Exception:  # noqa: BLE001 - a probe must never sink the run it is sizing
        return False


@contextlib.contextmanager
def _pinned_intraop_threads(limit: int) -> Iterator[int | None]:
    """Cap every native thread pool to ``limit`` for the block; yield the cap actually applied.

    Yields ``None`` when the cap could not be put fully in force, so the caller records an
    honest ``intraop_threads`` instead of claiming a pin that did not happen.

    **Two mechanisms, because each one alone leaves half the threads running.** Measured on a
    32-core host, fitting the same four models:

    ==========================  ==============================================
    pin in force                measured ``effective_cores``
    ==========================  ==============================================
    neither                     theta 19.7, sarimax 11.5 — i.e. ``nproc``
    ``threadpoolctl`` only      theta 4.8, holtwinters 7.3 — still contaminated
    env vars only               unchanged; the loaded pool ignores them
    both                        1.00 on every model
    ==========================  ==============================================

    They are complementary, not alternatives, because the two pools have different lifetimes:

    * `threadpool_limits` re-sizes the pools of libraries **already loaded** — numpy's
      OpenBLAS, which was dlopened long before this module was imported and therefore read
      its environment long ago. Env vars cannot touch it.
    * The env vars cap the pool of a library loaded **later** — one statsmodels dlopens
      part-way through the fit itself, which is born at its default size (a second
      32-thread pool; ``/proc/self/status`` shows the thread count going 33 → 65 mid-fit)
      precisely because it reads the environment at *its* load time, which is after this
      context manager has run. `threadpoolctl` never saw it, so it cannot cap it.

    Setting the environment is therefore not the no-op an earlier reading of it suggested;
    it is the only handle on pools that do not exist yet. The variables are restored on exit
    — the profiling pre-pass runs inside the driver process, which goes on to do real work
    afterwards, and leaving the fleet's BLAS pinned to one thread would be a silent
    performance regression far larger than the pre-pass it came from.

    ``threadpoolctl`` arrives with scikit-learn but is imported defensively anyway, because a
    stripped environment must degrade to a recorded ``None`` rather than crash inside a sizing
    pre-pass. Env-only is *not* good enough to report a pin: it leaves the already-loaded pool
    at ``nproc``, which is the dominant contamination, so that path yields ``None`` too.

    The controller is entered by hand rather than with ``with threadpool_limits(...)`` so that
    an exception raised by the *measured fit* propagates cleanly instead of resuming this
    generator a second time.
    """
    previous = {name: os.environ.get(name) for name in _INTRAOP_ENV_VARS}
    for name in _INTRAOP_ENV_VARS:
        os.environ[name] = str(limit)

    def _restore_env() -> None:
        for name, was in previous.items():
            if was is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = was

    try:
        from threadpoolctl import threadpool_limits

        controller = threadpool_limits(limits=limit)
    except Exception:  # noqa: BLE001 - an un-pinnable environment is a state, not a failure
        try:
            yield None
        finally:
            _restore_env()
        return
    try:
        yield limit
    finally:
        try:
            controller.restore_original_limits()
        except Exception:  # noqa: BLE001 - restoring is best-effort; the fit already ran
            pass
        _restore_env()


def _peak_gpu_bytes(*, reset: bool = False) -> int | None:  # pragma: no cover - needs a GPU
    """Peak CUDA bytes allocated since the last reset, or None when NOT MEASURED.

    ``reset=True`` zeroes the allocator's high-water mark instead of reading it — the call that
    must happen *before* the fit — and returns ``None``, which is also what every no-GPU path
    returns. Keeping both halves in one function keeps the "torch might not be here" handling
    in one place: an absent torch, an absent CUDA build and an absent device all land on
    ``None`` rather than on a ``0`` that a consumer would size against.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - no accelerator is a normal state, not a failure
        return None


def measure_fit(
    series: pd.DataFrame,
    model_name: str,
    cfg: RunConfig,
    *,
    params: dict[str, Any] | None = None,
) -> MeasuredFit:
    """Run exactly one ``run_cell`` and report what it consumed (I/O; never raises).

    Mirrors ``run_cell(series, model_name, cfg, params)`` argument-for-argument, so the thing
    being measured is unmistakably the thing that runs in production. ``params`` is carried
    because HPO-tuned hyperparameters demonstrably change fit cost (``n_estimators``, epochs),
    and measuring an untuned fit for a tuned run would size the fleet for different work.

    Fixed measurement order, and it is part of the contract: reset the RSS high-water mark,
    reset the CUDA high-water mark, read ``ru_maxrss``, cap the native thread pools, start both
    clocks, run the cell, stop both clocks, release the cap, re-read ``ru_maxrss``, read the
    CUDA peak. There is no ``measure_gpu`` flag — the read is attempted and the absence of a
    device is expressed as ``peak_gpu_bytes=None``, so there is one path to that outcome.

    **The thread cap is part of the measurement, not an optimization.** Both resets and the cap
    exist because the two host axes are otherwise measurements of the machine rather than of
    the model: ``ru_maxrss`` is monotonic, so without a reset a fit's memory reads as whatever
    the fits before it left behind, and an uncapped fit takes every idle core, so its
    ``cpu_s / wall_s`` reads as the driver's ``nproc``. What each attempt achieved is recorded
    on the record (``rss_peak_reset``, ``intraop_threads``, ``host_cpu_count``) rather than
    assumed, because both are platform-dependent.

    **Writes nothing** — no registry, no GCS, no log line. The ``CellResult`` is deliberately
    discarded: this is a probe, not a run, and its forecasts must never reach the registry under
    the run's ``run_id``.

    **No timeout, deliberately.** ``signal.alarm`` is unsafe off the main thread (a Spark driver,
    an Airflow worker) and a thread-based kill cannot interrupt a C-extension fit, so any guard
    written here would be theatre. The real bound is the budget: ``samples x models`` sequential
    fits on the driver. A hung fit stalls submit visibly; it cannot corrupt sizing.

    **That budget is not the whole story under per-series HPO.** ``params=None`` defers to
    ``worker._resolve_params``, which runs a full Optuna study when ``hpo.enabled`` and
    ``hpo.granularity == "per_series"`` — so one call becomes ``n_trials`` fits (20 by default)
    times the backtest folds, and an 8-sample x 5-model pre-pass becomes hundreds of driver-side
    fits with, by the paragraph above, nothing to stop it. A caller profiling such a run should
    pass already-resolved ``params`` or shrink the sample.

    **Sequential use only.** ``peak_rss_bytes`` is a delta on a process-wide high-water mark, so
    two concurrent ``measure_fit`` calls make each other's RSS number meaningless.

    Best-effort throughout, exactly like the existing GPU calibration probe: a ``status="error"``
    cell keeps its measured numbers and records ``ok=False`` (never discard a fact at the
    measurement layer — `build_profile` decides usability), and anything that raises returns a
    zeroed ``ok=False`` record. There is no raising path.
    """
    ts_id = "unknown"
    family = "unknown"
    n_obs = 0
    try:  # pragma: no cover - live path: a real fit runs
        import time

        from .models import get_model
        from .worker import run_cell

        id_col = cfg.data.ts_id_col
        if id_col in series.columns and len(series):
            ts_id = str(series[id_col].iloc[0])
        n_obs = int(len(series))
        try:
            family = str(get_model(model_name).family)
        except Exception:  # noqa: BLE001 - an unknown model still yields a countable failure
            family = "unknown"

        rss_was_reset = _reset_rss_peak()
        _peak_gpu_bytes(reset=True)
        rss_before = _rss_bytes()

        with _pinned_intraop_threads(_PROBE_INTRAOP_THREADS) as pinned:
            wall_started = time.perf_counter()
            cpu_started = time.process_time()

            result = run_cell(series, model_name, cfg, params)

            wall_s = time.perf_counter() - wall_started
            cpu_s = time.process_time() - cpu_started

        rss_after = _rss_bytes()
        peak_gpu = _peak_gpu_bytes()

        rss_delta = 0
        if rss_before is not None and rss_after is not None:
            rss_delta = max(0, rss_after - rss_before)

        failed = result.status == "error"
        return MeasuredFit(
            ts_id=ts_id,
            model_type=model_name,
            family=family,
            n_obs=n_obs,
            wall_s=wall_s,
            cpu_s=cpu_s,
            peak_rss_bytes=rss_delta,
            peak_gpu_bytes=peak_gpu,
            ok=not failed,
            error=result.error if failed else None,
            intraop_threads=pinned,
            host_cpu_count=os.cpu_count(),
            rss_peak_reset=rss_was_reset,
            process_rss_bytes=rss_after,
        )
    except Exception as e:  # noqa: BLE001 - a flaky probe widens sizing, it never sinks a run
        return MeasuredFit(
            ts_id=ts_id,
            model_type=model_name,
            family=family,
            n_obs=n_obs,
            wall_s=0.0,
            cpu_s=0.0,
            peak_rss_bytes=0,
            peak_gpu_bytes=None,
            ok=False,
            error=repr(e),
            host_cpu_count=os.cpu_count(),
        )


# --- pure: measurements -> per-family cost model --------------------------------


@dataclass(frozen=True)
class ModelCost:
    """Aggregated cost of one model across the profiled sample — RAW, no margin (pure).

    Every axis is ``| None`` and independently derived: a measurement can be good evidence
    for wall time and no evidence at all for GPU memory, so usability is decided per axis,
    not per record. ``None`` means no usable measurement contributed to that axis.

    No margins are applied here. They ride on `FamilyCost` and `ComputeProfile`, which is
    what a consumer reads, so a margin can never be applied twice on the way through.
    """

    model_type: str
    family: str
    n_fits: int  # measurements supplied for this model
    n_ok: int  # of those, how many had ok=True
    max_n_obs: int  # length of the longest series actually measured, 0 if none
    max_peak_rss_bytes: int | None  # bytes, max over usable values — DIAGNOSTIC (marginal)
    max_peak_gpu_bytes: int | None  # bytes, max over usable values
    median_wall_s: float | None  # seconds, median over usable values — the throughput tail
    median_cpu_s: float | None  # seconds, median over usable values
    max_effective_cores: float | None  # cores, max over usable per-fit ratios
    max_process_rss_bytes: int | None = None  # bytes, max absolute footprint — SIZES THE SLOT


@dataclass(frozen=True)
class FamilyCost:
    """One family's slot shape and workload, rolled up from its models (pure).

    Roll-up rule, and the reason it differs per axis: **peaks take the max** across the
    family's models, because one slot must hold whichever model lands in it; **times take
    the median** for the per-cell question and the **sum** for the per-series question,
    because a family's models all run — they do not compete for one cell.

    The margins are carried on the record rather than applied by the caller so the sized
    numbers and the raw measurements travel together: an audit reads both "measured 3.1
    GiB" and "sized 4.1 GiB" off one object, and a consumer cannot accidentally apply the
    margin twice.

    **Invariant: every derived property is ``None`` if and only if its raw basis is
    ``None``.** There is no fabricated fallback here — a consumer that gets ``None`` has
    been told, unambiguously, that it must fall back to its own static default (for
    ``slot_cores`` that default is 1, today's hardcoded value).
    """

    family: str
    models: tuple[str, ...]  # sorted model_types that contributed a usable number
    # Counted over every measurement tagged with this family — including those from a model
    # that produced nothing usable and is therefore absent from ``models``. Scoping these to
    # the surviving models instead would report a family that was 2-of-4 as a clean 2-of-2,
    # which is not a missing number but a wrong one: a family whose heavyweight member OOM'd
    # on every fit would present as a fully successful measurement.
    n_fits: int  # measurements supplied for this family
    n_ok: int  # of those, how many had ok=True
    max_peak_rss_bytes: int | None  # bytes, max over models — RAW; DIAGNOSTIC (marginal)
    max_peak_gpu_bytes: int | None  # bytes, max over models — RAW; None == not measured
    max_effective_cores: float | None  # cores, max over models — RAW
    median_wall_s: float | None  # seconds, median over models' median_wall_s — RAW
    total_wall_s_per_series: float | None  # seconds, SUM over models' median_wall_s — RAW
    memory_margin: float  # the ratio the slot_* properties apply
    time_margin: float  # the ratio the planning_* properties apply
    max_process_rss_bytes: int | None = None  # bytes, max over models — RAW; SIZES THE SLOT

    # --- sized values: what a runtime translation actually consumes ---------------
    @property
    def slot_rss_bytes(self) -> int | None:
        """Host memory one cell of this family needs, margin applied (bytes, rounded up).

        Built on ``max_process_rss_bytes`` — the **absolute** footprint — not on the marginal
        ``max_peak_rss_bytes``. A slot holds an interpreter with the model stack imported, not
        just the incremental allocation of one fit, and the marginal number is in any case
        unstable: it swings 17x on the order the sample was measured in and reads 0.00 MB for a
        model whose fit is served entirely from already-resident pages. See `MeasuredFit`.

        ``None`` when nothing measured the absolute footprint (no ``resource`` module, or every
        fit failed), which tells the consumer to fall back to its own static default rather
        than to size against a number that was never taken.
        """
        if self.max_process_rss_bytes is None:
            return None
        return math.ceil(self.max_process_rss_bytes * self.memory_margin)

    @property
    def slot_gpu_bytes(self) -> int | None:
        """Device memory one cell needs, margin applied; None when the axis is unmeasured."""
        if self.max_peak_gpu_bytes is None:
            return None
        return math.ceil(self.max_peak_gpu_bytes * self.memory_margin)

    @property
    def slot_cores(self) -> int | None:
        """Whole cores one cell needs: ``ceil(max_effective_cores)``, at least 1.

        No margin — a core count is already discrete and already the max over the family — but
        a `_CORE_SNAP_TOLERANCE` snap before rounding up, because the input is a ratio of two
        clocks. A single-threaded fit lands on either side of 1.0 at random, and since this is
        the **max** over the family, the chance that at least one member lands a hair above
        grows with family size: five single-threaded models trip it ~97% of the time. Without
        the snap that reads as ``slot_cores=2`` and halves fleet density, with an audit record
        showing ``1.0000005`` that looks correct to two decimal places.
        """
        if self.max_effective_cores is None:
            return None
        return max(1, math.ceil(self.max_effective_cores - _CORE_SNAP_TOLERANCE))

    @property
    def planning_wall_s(self) -> float | None:
        """Expected seconds for the median cell of this family, time margin applied."""
        if self.median_wall_s is None:
            return None
        return self.median_wall_s * self.time_margin

    @property
    def planning_total_wall_s_per_series(self) -> float | None:
        """Expected seconds to run every model of this family for one series, margin applied."""
        if self.total_wall_s_per_series is None:
            return None
        return self.total_wall_s_per_series * self.time_margin


@dataclass(frozen=True)
class DataSignature:
    """What a set of measurements was taken *on* — the basis for "is this evidence yours?" (pure).

    Deliberately coarse. The question a signature has to answer is not "is this the same data"
    (nothing is) but "is this close enough that its costs transfer" — a fit's cost tracks the shape
    of the panel, not its values. Four fields cover that: a different **table** is different work, a
    10x change in **series count** changes nothing per-cell but everything about the fleet, a 10x
    change in **history length** changes the per-fit cost directly, and a different **frequency**
    changes the seasonality every model estimates.

    Every field is optional because the two sides are read from different places and neither is
    complete: a config knows its table and frequency but not how long its series are until it
    reads them, while a harvest knows the lengths it measured but not the table it read
    (``forecast_metadata`` records no source). `compare_signatures` compares what both sides have.
    """

    source_table: str | None = None
    n_series: int | None = None
    median_n_obs: int | None = None
    freq: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, for the telemetry stamp."""
        return {
            "source_table": self.source_table,
            "n_series": self.n_series,
            "median_n_obs": self.median_n_obs,
            "freq": self.freq,
        }


@dataclass(frozen=True)
class ProfileProvenance:
    """Where a profile's numbers came from, carried with the numbers (pure).

    A sizing decision an operator cannot attribute is one they cannot argue with. The load-bearing
    field is ``basis``:

    * ``measured`` — taken on this run's own data. The strongest claim.
    * ``reference`` — measured, genuinely, but **not on your data**: another run's harvest, or the
      baseline shipped with the product. Real evidence with a caveat, and without a name for that
      state an operator reading a resolved fleet shape cannot tell whose evidence produced it.
    * ``assumed`` — no measurement; the static arithmetic. Recorded so "we had nothing" is a stated
      outcome rather than an absent field.
    """

    basis: Literal["measured", "reference", "assumed"]
    source: str  # the `compute.profile.source` value that produced this — the audit key
    run_id: str | None = None  # the harvest's run, when the evidence came from one
    baseline_version: str | None = None  # the shipped baseline's version, when it came from that
    measured_at: str | None = None  # ISO-8601, when the evidence was recorded
    signature: DataSignature | None = None  # what it was measured on
    warnings: tuple[str, ...] = ()  # signature mismatches; see `compare_signatures`

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, for the telemetry stamp."""
        return {
            "basis": self.basis,
            "source": self.source,
            "run_id": self.run_id,
            "baseline_version": self.baseline_version,
            "measured_at": self.measured_at,
            "signature": self.signature.to_dict() if self.signature else None,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ComputeProfile:
    """The measured cost model for one run: per-model and per-family, both tails kept (pure).

    The whole object is stamped into run telemetry, so it stays JSON-representable and
    carries the margins it was built with — a sizing decision that cannot be re-derived
    from its own record is not auditable. Both mappings are built from sorted keys so the
    serialized form is byte-stable and an audit diff is meaningful.

    A family or model is **absent** when nothing usable was measured for it. Absence is the
    signal to fall back to static config; a zero-valued entry would be consumed as a real
    size. Absence is deliberately *not* silent: ``dropped_models`` and ``first_error_by_model``
    name what fell out and why, so "this family was sized off one of its two models, because
    the other one OOM'd" is readable straight off the record instead of being inferred from a
    gap in it.
    """

    families: dict[str, FamilyCost]  # keyed by family: statistical | ml | deep_learning
    models: dict[str, ModelCost]  # keyed by model_type, flat — no nesting to walk
    memory_margin: float
    time_margin: float
    n_measurements: int  # measurements supplied
    n_ok: int  # of those, how many had ok=True
    sample_ts_ids: tuple[str, ...]  # sorted unique ids that contributed usable evidence
    # Measured but contributed nothing: every fit failed, or every axis was unusable. These are
    # the models missing from ``models``/``FamilyCost.models``, named so the gap is legible.
    dropped_models: tuple[str, ...] = ()
    # model_type -> first error text seen for it. The one place a failure *reason* survives
    # aggregation; without it a profile can only say a fit produced nothing, never why.
    first_error_by_model: dict[str, str] | None = None
    # The pre-pass sample, when the caller passes it — which series were measured and why.
    sample: tuple[SampleSpec, ...] = ()
    # Whose evidence this is. ``None`` on a profile built straight from measurements by a caller
    # that has not decided yet — `resolve_profile_source` is what stamps it.
    provenance: ProfileProvenance | None = None

    @property
    def n_failed(self) -> int:
        """``n_measurements - n_ok`` — how much of the sample we could not use."""
        return self.n_measurements - self.n_ok

    @property
    def is_empty(self) -> bool:
        """True when nothing usable was measured — the caller must fall back to static config."""
        return not self.models

    def for_family(self, family: str) -> FamilyCost | None:
        """This family's cost, or None when it was never measured (fall back, don't guess)."""
        return self.families.get(family)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the whole profile — raw aggregates *and* the sized values.

        The sized values are serialized alongside the raw ones so telemetry answers both
        "what did we measure" and "what did we ask the platform for" without the reader
        re-deriving a margin. Plain ints/floats/strings/None only: ``json.dumps`` must
        succeed with no custom encoder.
        """
        return {
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "memory_margin": self.memory_margin,
            "time_margin": self.time_margin,
            "n_measurements": self.n_measurements,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "sample_ts_ids": list(self.sample_ts_ids),
            "dropped_models": list(self.dropped_models),
            "first_error_by_model": dict(self.first_error_by_model or {}),
            "sample": [spec.to_dict() for spec in self.sample],
            "models": {
                name: {
                    "model_type": cost.model_type,
                    "family": cost.family,
                    "n_fits": cost.n_fits,
                    "n_ok": cost.n_ok,
                    "max_n_obs": cost.max_n_obs,
                    "max_peak_rss_bytes": cost.max_peak_rss_bytes,
                    "max_process_rss_bytes": cost.max_process_rss_bytes,
                    "max_peak_gpu_bytes": cost.max_peak_gpu_bytes,
                    "median_wall_s": cost.median_wall_s,
                    "median_cpu_s": cost.median_cpu_s,
                    "max_effective_cores": cost.max_effective_cores,
                }
                for name, cost in self.models.items()
            },
            "families": {
                name: {
                    "family": cost.family,
                    "models": list(cost.models),
                    "n_fits": cost.n_fits,
                    "n_ok": cost.n_ok,
                    "max_peak_rss_bytes": cost.max_peak_rss_bytes,
                    "max_process_rss_bytes": cost.max_process_rss_bytes,
                    "max_peak_gpu_bytes": cost.max_peak_gpu_bytes,
                    "max_effective_cores": cost.max_effective_cores,
                    "median_wall_s": cost.median_wall_s,
                    "total_wall_s_per_series": cost.total_wall_s_per_series,
                    "memory_margin": cost.memory_margin,
                    "time_margin": cost.time_margin,
                    "slot_rss_bytes": cost.slot_rss_bytes,
                    "slot_gpu_bytes": cost.slot_gpu_bytes,
                    "slot_cores": cost.slot_cores,
                    "planning_wall_s": cost.planning_wall_s,
                    "planning_total_wall_s_per_series": cost.planning_total_wall_s_per_series,
                }
                for name, cost in self.families.items()
            },
        }


def _usable(value: float | int | None) -> bool:
    """True when a measured value is real evidence: present, finite, and strictly positive.

    Zero is not evidence on any axis here — an RSS delta of 0 is the high-water-mark artefact,
    a wall time of 0 is a fit too fast to time, and a GPU peak of 0 is a fit that allocated
    nothing. All three must collapse to ``None`` rather than be maxed or medianed into a size.
    """
    return value is not None and math.isfinite(value) and value > 0


def _safe_max(values: Sequence[float | int | None]) -> float | None:
    """Max over the usable values, or None when none of them is usable (pure)."""
    usable = [float(v) for v in values if _usable(v)]
    return max(usable) if usable else None


def _safe_median(values: Sequence[float | int | None]) -> float | None:
    """Median over the usable values, or None when none of them is usable (pure).

    ``statistics.median`` averages the two middle values on an even count, which is
    deterministic and does not depend on the input order.
    """
    usable = [float(v) for v in values if _usable(v)]
    return statistics.median(usable) if usable else None


def _model_cost(model_type: str, records: list[MeasuredFit]) -> ModelCost | None:
    """Roll one model's measurements up: max for peaks, median for times (pure).

    Returns ``None`` when no axis has usable evidence — the model is then absent from the
    profile entirely, because a `ModelCost` of zeros would be consumed as a real size while an
    absent one is unambiguously "fall back to static config". The measurements are still
    counted at the profile level, so the audit shows that fits happened and produced nothing.

    ``effective_cores`` has its own usability rule: a fit contributes iff it succeeded, was
    slow enough to time (`_MIN_WALL_S`), and reported a finite non-negative CPU time. The ratio
    is already floored at one core by `MeasuredFit.effective_cores`.
    """
    ok = [record for record in records if record.ok]
    cores = [
        record.effective_cores
        for record in ok
        if math.isfinite(record.wall_s)
        and record.wall_s >= _MIN_WALL_S
        and math.isfinite(record.cpu_s)
        and record.cpu_s >= 0
    ]

    max_rss = _safe_max([record.peak_rss_bytes for record in ok])
    max_process_rss = _safe_max([record.process_rss_bytes for record in ok])
    max_gpu = _safe_max([record.peak_gpu_bytes for record in ok])
    median_wall = _safe_median([record.wall_s for record in ok])
    median_cpu = _safe_median([record.cpu_s for record in ok])
    max_cores = _safe_max(cores)

    axes = (max_rss, max_process_rss, max_gpu, median_wall, median_cpu, max_cores)
    if all(axis is None for axis in axes):
        return None

    return ModelCost(
        model_type=model_type,
        family=min(record.family for record in ok),
        n_fits=len(records),
        n_ok=len(ok),
        max_n_obs=max((record.n_obs for record in ok), default=0),
        max_peak_rss_bytes=int(max_rss) if max_rss is not None else None,
        max_peak_gpu_bytes=int(max_gpu) if max_gpu is not None else None,
        median_wall_s=median_wall,
        median_cpu_s=median_cpu,
        max_effective_cores=max_cores,
        max_process_rss_bytes=int(max_process_rss) if max_process_rss is not None else None,
    )


def _family_cost(
    family: str,
    costs: list[ModelCost],
    memory_margin: float,
    time_margin: float,
    *,
    n_fits: int,
    n_ok: int,
) -> FamilyCost:
    """Roll a family's models up into one slot shape and one workload figure (pure).

    Peaks and cores take the **max** — any model in the family may land in the slot, so the
    slot must hold the widest of them. ``median_wall_s`` takes the median of the members'
    medians (what does a typical cell of this family cost?) while
    ``total_wall_s_per_series`` takes their **sum** (what does running the whole family for one
    series cost?). Those are different questions and a single field cannot answer both.

    ``n_fits`` / ``n_ok`` are passed in rather than summed from ``costs`` because a model that
    produced no usable axis has no `ModelCost` to sum — see the field comments.
    """
    member_walls: list[float | None] = [cost.median_wall_s for cost in costs]
    usable_walls = [wall for wall in member_walls if _usable(wall)]

    max_rss = _safe_max([cost.max_peak_rss_bytes for cost in costs])
    max_process_rss = _safe_max([cost.max_process_rss_bytes for cost in costs])
    max_gpu = _safe_max([cost.max_peak_gpu_bytes for cost in costs])

    return FamilyCost(
        family=family,
        models=tuple(sorted(cost.model_type for cost in costs)),
        n_fits=n_fits,
        n_ok=n_ok,
        max_peak_rss_bytes=int(max_rss) if max_rss is not None else None,
        max_peak_gpu_bytes=int(max_gpu) if max_gpu is not None else None,
        max_effective_cores=_safe_max([cost.max_effective_cores for cost in costs]),
        median_wall_s=_safe_median(member_walls),
        total_wall_s_per_series=sum(usable_walls) if usable_walls else None,
        memory_margin=memory_margin,
        time_margin=time_margin,
        max_process_rss_bytes=int(max_process_rss) if max_process_rss is not None else None,
    )


def build_profile(
    measurements: Sequence[MeasuredFit],
    *,
    sample: Sequence[SampleSpec] = (),
    memory_margin: float = _DEFAULT_MEMORY_MARGIN,
    time_margin: float = _DEFAULT_TIME_MARGIN,
) -> ComputeProfile:
    """Aggregate measurements into a per-family cost model: max for peaks, median for times (pure).

    The whole point of the split: ``max(peak) x memory_margin`` governs **safety** — how many
    tasks may share a device or an executor without an OOM — while ``median(time) x
    time_margin`` governs **throughput** — how many slots the load needs. Using the max for
    both systematically over-provisions every run; using the median for both OOM-kills it. Both
    tails are therefore kept and both are exposed.

    Usability is decided **per axis, not per record**: a measurement contributes to an axis iff
    it succeeded and that axis's value is finite and positive. One fit can be perfectly good
    evidence for wall time and no evidence at all for GPU memory, so filtering whole records
    would throw the first away and trusting whole records would fabricate the second. An axis
    with no usable value is ``None``, and a model or family with no usable axis is **absent**
    from the profile — absence is what tells a consumer to fall back to its static default.

    Margins are validated (below 1.0 asks for less headroom than the measurement, which is
    never valid), recorded on every record, and applied **only** in the ``slot_*`` /
    ``planning_*`` properties, so they can never be applied twice. Input is never mutated,
    grouping keys are iterated sorted, and the result is a pure function of the measurement
    *set* — a shuffled input yields an equal profile.

    ``sample`` is optional and carried through verbatim for the audit record: the profile then
    answers "what did this cost" and "which series was that measured on, and why those" from
    one object. It is not read by any arithmetic here.
    """
    for name, margin in (("memory_margin", memory_margin), ("time_margin", time_margin)):
        if not math.isfinite(margin) or margin < 1.0:
            raise ValueError(f"{name} must be a finite ratio >= 1.0, got {margin}")

    by_model: dict[str, list[MeasuredFit]] = {}
    for record in measurements:
        by_model.setdefault(record.model_type, []).append(record)

    models: dict[str, ModelCost] = {}
    for model_type in sorted(by_model):
        cost = _model_cost(model_type, by_model[model_type])
        if cost is not None:
            models[model_type] = cost

    by_family: dict[str, list[ModelCost]] = {}
    for cost in models.values():
        by_family.setdefault(cost.family, []).append(cost)

    # Counted off the measurements, not off the surviving ModelCosts, so a family that lost a
    # whole model still reports how many fits were really spent on it.
    family_fits: dict[str, int] = {}
    family_ok: dict[str, int] = {}
    for record in measurements:
        family_fits[record.family] = family_fits.get(record.family, 0) + 1
        family_ok[record.family] = family_ok.get(record.family, 0) + int(record.ok)

    families = {
        family: _family_cost(
            family,
            by_family[family],
            memory_margin,
            time_margin,
            n_fits=family_fits.get(family, 0),
            n_ok=family_ok.get(family, 0),
        )
        for family in sorted(by_family)
    }

    # What fell out, and the first reason given for it — the failure text stops here otherwise.
    dropped = tuple(sorted(name for name in by_model if name not in models))
    first_errors: dict[str, str] = {}
    for model_type in sorted(by_model):
        for record in by_model[model_type]:
            if not record.ok and record.error and model_type not in first_errors:
                first_errors[model_type] = record.error

    # The ids that actually backed a number, not merely the ids we tried — "we sized off these
    # six series" is the auditable claim.
    contributed = {
        record.ts_id
        for record in measurements
        if record.ok
        and (
            _usable(record.peak_rss_bytes)
            or _usable(record.peak_gpu_bytes)
            or _usable(record.wall_s)
            or _usable(record.cpu_s)
        )
    }

    return ComputeProfile(
        families=families,
        models=models,
        memory_margin=memory_margin,
        time_margin=time_margin,
        n_measurements=len(measurements),
        n_ok=sum(1 for record in measurements if record.ok),
        sample_ts_ids=tuple(sorted(contributed)),
        dropped_models=dropped,
        first_error_by_model=first_errors,
        sample=tuple(sample),
    )


def _harvest_family(model_type: str) -> str:
    """The compute family of ``model_type``, or ``"unknown"`` when it cannot be resolved.

    Resolved from the model registry rather than persisted per row: family is a property of the
    code, and a model that has since been re-homed should aggregate where it lives *now*. An
    unresolvable name (a model deleted since the run) lands in ``"unknown"``, which no translator
    consumes, so it is dropped from sizing without being hidden from the counts.
    """
    from .models import get_model

    try:
        return str(get_model(model_type).family)
    except Exception:  # noqa: BLE001 - a stale model name must not sink a profile read
        return "unknown"


def harvest_profile(
    rows: Iterable[Mapping[str, Any]],
    *,
    memory_margin: float = _DEFAULT_MEMORY_MARGIN,
    time_margin: float = _DEFAULT_TIME_MARGIN,
) -> ComputeProfile:
    """Aggregate persisted ``forecast_metadata`` rows into a `ComputeProfile` (pure).

    **The second producer, and the one that scales.** `build_profile` aggregates a pre-pass that
    deliberately fits a small sample; this aggregates the fits a completed run already performed.
    Both feed the identical aggregation, so a harvested profile and a measured one are the same
    object and every translator consumes them the same way. The difference is only in how the
    evidence was obtained — and this way it is obtained for free, from every cell rather than
    from eight, on the real hardware rather than on a driver.

    That is what makes "size this run like run X" a **query** instead of an artifact store: a
    completed ``run_id`` *is* a profile, with its config, data signature and lineage already
    recorded next to it in the registry. Nothing new to version, expire, or garbage-collect.

    ``rows`` are mappings with ``forecast_metadata`` column names — from a BigQuery read, a test
    fixture, or a committed baseline file; the function never touches BigQuery itself. Missing
    keys read as NULL, so rows written before the measurement columns existed degrade to
    wall-time-only evidence rather than raising.

    **Two filters, both structural.** Backtest fold rows (``fold_id`` set) are skipped because
    ``fit_seconds`` on the full-fit row already brackets the whole cell, folds included — counting
    folds too would double-count the same work. Ensemble rows (``ensemble_id`` set) are skipped
    because an ensemble is arithmetic over predictions, not a fit whose cost sizes a slot.

    **How success is inferred, since ``forecast_metadata`` carries no status column.** A cell that
    errored returns before its wall clock is recorded and lands with ``fit_seconds`` of zero, so a
    usable wall time is exactly the signal that a fit happened. Those rows still count toward
    ``n_measurements`` and ``n_failed``, so "we sized off 940 of 1000 cells" stays visible — the
    same honesty `build_profile` gives a pre-pass, at run scale.
    """
    measurements: list[MeasuredFit] = []
    family_of: dict[str, str] = {}
    for row in rows:
        if row.get("fold_id") is not None or row.get("ensemble_id") is not None:
            continue
        model_type = str(row.get("model_type") or "")
        if not model_type:
            continue
        if model_type not in family_of:
            family_of[model_type] = _harvest_family(model_type)
        wall_s = _as_number(row.get("fit_seconds"))
        measurements.append(
            MeasuredFit(
                ts_id=str(row.get("ts_id") or ""),
                model_type=model_type,
                family=family_of[model_type],
                n_obs=int(_as_number(row.get("n_obs"))),
                wall_s=wall_s,
                cpu_s=_as_number(row.get("cpu_seconds")),
                # The RSS *delta* axis is not harvested: it is order-dependent to the point of
                # uselessness (see `MeasuredFit`), and 0 already means "no evidence" there. The
                # absolute high-water — the number that actually sizes a slot — arrives on
                # ``process_rss_bytes`` below.
                peak_rss_bytes=0,
                peak_gpu_bytes=_as_optional_int(row.get("peak_gpu_bytes")),
                ok=_usable(wall_s),
                error=None,
                intraop_threads=_as_optional_int(row.get("intraop_threads")),
                host_cpu_count=None,
                rss_peak_reset=False,
                process_rss_bytes=_as_optional_int(row.get("process_rss_bytes")),
            )
        )
    return build_profile(
        measurements, memory_margin=memory_margin, time_margin=time_margin
    )


def signature_from_config(cfg: RunConfig) -> DataSignature:
    """What this run is about to read, as far as the config alone can say (pure).

    ``median_n_obs`` is always ``None`` here: history length is a property of the data, and at plan
    time nobody has read it. That asymmetry is deliberate rather than a gap — `compare_signatures`
    skips what one side cannot know, so a config-side signature checks the table, the series count
    and the frequency, and lets the length axis go unchecked rather than guessing at it.
    """
    return DataSignature(
        source_table=cfg.data.source_table,
        n_series=cfg.data.series_limit,
        median_n_obs=None,
        freq=cfg.data.freq,
    )


def signature_from_rows(
    rows: Iterable[Mapping[str, Any]], *, source_table: str | None = None
) -> DataSignature:
    """What a harvest was measured on, read off the rows themselves (pure).

    ``source_table`` is passed in because ``forecast_metadata`` does not record one — the run's
    header does. A caller that has the header supplies it; one that does not leaves it ``None``
    and the table axis simply goes unchecked.
    """
    ts_ids: set[str] = set()
    lengths: list[int] = []
    for row in rows:
        if row.get("fold_id") is not None or row.get("ensemble_id") is not None:
            continue
        ts_id = row.get("ts_id")
        if ts_id is not None:
            ts_ids.add(str(ts_id))
        n_obs = _as_optional_int(row.get("n_obs"))
        if n_obs:
            lengths.append(n_obs)
    return DataSignature(
        source_table=source_table,
        n_series=len(ts_ids) or None,
        median_n_obs=int(statistics.median(lengths)) if lengths else None,
        freq=None,
    )


# How far two signatures may drift on a *scale* axis before the evidence stops transferring. An
# order of magnitude, because that is the point at which per-fit cost and fleet width stop being
# the same problem — and because a tighter band would warn on every ordinary week-to-week change
# in a panel and train operators to ignore the warning, which is worse than not emitting one.
_SIGNATURE_DRIFT_FACTOR = 10.0


def compare_signatures(want: DataSignature, have: DataSignature) -> tuple[str, ...]:
    """Human-readable reasons ``have``'s evidence may not describe ``want``'s run (pure).

    Warnings, never errors. A profile is a *hint* about hardware; sizing off drifted evidence is
    still better than sizing off nothing, and a hard failure here would turn a convenience into a
    thing that blocks runs. But it is never silent either — §3.10's rule is that a pinned profile
    from months ago on different data is worse than no profile precisely *because it looks
    authoritative*, and an unnamed mismatch is how it gets to look that way.

    Axes either side leaves ``None`` are skipped: an unchecked axis is honest, an axis compared
    against a placeholder is not.
    """
    out: list[str] = []
    if want.source_table and have.source_table and want.source_table != have.source_table:
        out.append(
            f"measured on a different table ({have.source_table} vs {want.source_table})"
        )
    if want.freq and have.freq and want.freq != have.freq:
        out.append(f"measured at a different frequency ({have.freq} vs {want.freq})")
    for label, mine, theirs in (
        ("series count", want.n_series, have.n_series),
        ("history length", want.median_n_obs, have.median_n_obs),
    ):
        if not mine or not theirs:
            continue
        ratio = max(mine, theirs) / min(mine, theirs)
        if ratio >= _SIGNATURE_DRIFT_FACTOR:
            out.append(f"{label} differs by {ratio:.0f}x ({theirs} measured vs {mine} planned)")
    return tuple(out)


def _as_number(value: Any) -> float:
    """A row cell as a finite float, with NULL / non-numeric / non-finite all reading as ``0.0``.

    Zero is already this module's "no evidence" value on every axis `_usable` guards, so a missing
    reading needs no second representation and no branch at every call site.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _as_optional_int(value: Any) -> int | None:
    """A row cell as an int, preserving ``None`` — the axes where NULL means NOT MEASURED.

    ``peak_gpu_bytes`` and ``process_rss_bytes`` must not collapse a missing reading to ``0``: a
    consumer reading zero device bytes would pack ten tasks onto a device that fits two.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --- I/O: the driver-side pre-pass ----------------------------------------------


def should_profile(cfg: RunConfig, n_cells: int) -> bool:
    """Is a measurement pre-pass worth running for a fan-out of ``n_cells``? (pure)

    The three ``compute.profile.mode`` values, and the reasoning behind the middle one:

    * ``off`` — never. The escape hatch: static config, exactly as before the profiler existed.
    * ``always`` — unconditionally. What the smokes use, so the path is exercised on runs small
      enough that exercising it is cheap.
    * ``auto`` (default) — only when the fan-out is big enough to repay the pre-pass. The
      pre-pass costs ``samples x models`` real fits on the driver; below ``min_cells`` that is a
      measurable fraction of the whole run, spent to size a fleet that barely needs sizing. Above
      it, the same fixed cost is rounding error against the work it is optimising.

    The comparison is on **cells**, not series, because a 100-series run over 6 models is the
    same amount of work as a 600-series run over one, and it is the work — not the panel — that
    the fleet is sized for.
    """
    mode = cfg.compute.profile.mode
    if mode == "off":
        return False
    if mode == "always":
        return True
    return n_cells >= cfg.compute.profile.min_cells


# --- the consumer: which evidence this run sizes from ---------------------------

# A loader for one run's harvest. Returns the ``forecast_metadata`` rows and the source table the
# run read (which those rows do not carry — the run header does), or ``None`` when the run has no
# harvest to give. The tuple is the seam's whole vocabulary: rows say what the fits cost, the table
# says what they cost it *on*, and `compare_signatures` needs both to say anything useful.
RunHarvestLoader = Callable[[str], "tuple[Sequence[Mapping[str, Any]], str | None] | None"]
# Finds the newest run whose harvest matches a signature, or ``None``. Separate from the loader
# because "which run" is a search and "what did it cost" is a read, and only `auto` does the search.
RunDiscoverer = Callable[[DataSignature], "str | None"]
# The shipped, versioned baseline (W13), or ``None`` before one exists.
BaselineLoader = Callable[[], "ComputeProfile | None"]


def resolve_profile_source(
    cfg: RunConfig,
    *,
    load_run: RunHarvestLoader | None = None,
    load_baseline: BaselineLoader | None = None,
    discover: RunDiscoverer | None = None,
) -> ComputeProfile | None:
    """The profile this run should size from, per ``compute.profile.source`` (pure + injected I/O).

    Returns ``None`` for "no evidence" — the static-config case every consumer already handles
    (`resources.resource_slot` takes ``profile=None`` as its identity case). ``None`` is a
    *decision*, not a failure: sizing from declared config is the behaviour this product shipped
    with, and it stays the floor under every path here.

    The precedence, resolved outside-in:

    1. ``mode == "off"`` or ``source == "none"`` → ``None``. Nothing consulted, nothing loaded.
    2. ``source == "<run_id>"`` → that run's harvest. An operator naming a run has made a
       decision; it is honoured even if the signature has drifted, and the drift comes back as
       warnings on the provenance rather than as a substitution they did not ask for.
    3. ``source == "auto"`` → the newest run whose harvest matches this run's signature, if
       ``discover`` finds one.
    4. the shipped baseline, if one is loadable.
    5. ``None``.

    Every loader is injected. That keeps this function pure enough to test the whole precedence
    chain offline with no BigQuery and no baseline file — the same seam ``resolve_profile`` uses
    for ``measure``. A loader that raises is treated as a loader that found nothing: sizing
    evidence is an optimisation, and a registry hiccup must not sink a run that would otherwise
    size itself from config.
    """
    profile_cfg = cfg.compute.profile
    if not profile_cfg.consumes_evidence:
        return None

    want = signature_from_config(cfg)
    source = profile_cfg.source

    run_id = source if source != "baseline" else None
    if source == "auto":
        run_id = _try(lambda: discover(want)) if discover else None

    if run_id and load_run:
        loaded = _try(lambda: load_run(run_id))
        if loaded:
            rows, source_table = loaded
            profile = harvest_profile(
                rows,
                memory_margin=profile_cfg.memory_margin,
                time_margin=profile_cfg.time_margin,
            )
            have = signature_from_rows(rows, source_table=source_table)
            warnings = compare_signatures(want, have)
            return _with_provenance(
                profile,
                ProfileProvenance(
                    # No drift on any axis both sides can see is as close to "measured on your
                    # data" as harvested evidence gets; anything else is honest about being
                    # someone else's measurement.
                    basis="measured" if not warnings else "reference",
                    source=source,
                    run_id=run_id,
                    measured_at=_measured_at(rows),
                    signature=have,
                    warnings=warnings,
                ),
            )

    if load_baseline:
        baseline = _try(load_baseline)
        if baseline is not None:
            existing = baseline.provenance
            return _with_provenance(
                baseline,
                ProfileProvenance(
                    basis="reference",  # measured, but never on your data — that is what it is for
                    source=source,
                    baseline_version=existing.baseline_version if existing else None,
                    measured_at=existing.measured_at if existing else None,
                    signature=existing.signature if existing else None,
                    warnings=(
                        "sized from the shipped baseline, not from a measurement of your data",
                    ),
                ),
            )
    return None


def _try(load: Callable[[], Any]) -> Any:
    """Run a loader, swallowing its failure (see `resolve_profile_source`)."""
    try:
        return load()
    except Exception as e:  # noqa: BLE001 - evidence is an optimisation; never fatal
        _log.warning("profile source lookup failed, falling back: %r", e)
        return None


def _with_provenance(profile: ComputeProfile, provenance: ProfileProvenance) -> ComputeProfile:
    """``profile`` with its provenance stamped on (frozen dataclass → a copy)."""
    return replace(profile, provenance=provenance)


def _measured_at(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """The newest ``created_at`` in a harvest, as a string, or None when the rows carry none."""
    stamps = [str(row["created_at"]) for row in rows if row.get("created_at") is not None]
    return max(stamps) if stamps else None


def _profilable_models(models: Sequence[str]) -> list[str]:
    """The subset of ``models`` that runs as a Python fit, in the given order (I/O: imports).

    BigQuery-native models (``runtime == "bigquery"``) execute as SQL inside BigQuery: there is no
    process whose cores, RSS or device bytes we could measure, and no slot to size. Measuring one
    would mean issuing a real ``CREATE MODEL`` from a sizing pre-pass, which is both expensive and
    a write. An unresolvable name is dropped rather than raised on — the router has already
    validated the model list, so a failure here is a probe problem, and a probe must not sink a run.
    """
    from .models import get_model

    keep: list[str] = []
    for name in models:
        try:
            if get_model(name).runtime != "bigquery":
                keep.append(name)
        except Exception:  # noqa: BLE001 - a name we cannot resolve is simply not profilable
            continue
    return keep


def resolve_profile(
    panel: pd.DataFrame,
    cfg: RunConfig,
    models: Sequence[str],
    *,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    measure: Callable[..., MeasuredFit] | None = None,
) -> ComputeProfile | None:
    """Driver-side measurement pre-pass: sample the panel, fit, aggregate → `ComputeProfile`.

    The structural twin of the fleetwide-HPO pre-pass (``resolve_fleetwide_hpo``) — same seam,
    same place, same "resolve once on the driver before fanning out" shape — and it runs *after*
    that one, so tuned hyperparameters can be measured rather than defaults.

    Returns ``None`` when no measurement was taken: profiling is off, the fan-out is below
    ``min_cells``, nothing profilable is in the model list, or the panel yields no usable
    statistics. ``None`` is the signal to size from static config, and every consumer already
    treats it that way (`resources.resource_slot` takes ``profile=None`` as its
    identity case). A profile that *was* taken but measured nothing usable comes back as an empty
    ``ComputeProfile`` rather than ``None`` — the distinction is "we did not look" versus "we
    looked and found nothing", and only the second is worth an audit record.

    **The sample loop is the outer one, deliberately.** Absolute process RSS only grows within a
    process, so a model measured exactly once is charged whatever happened to be imported by the
    time its turn came — the first model measured looks artificially small because the later
    models' libraries were not loaded yet. Cycling every model across every sampled series means
    each model is measured at least once against a fully warm heap, and the ``max`` aggregation
    picks that measurement up. (At ``samples=1`` there is no second pass to warm into, so a
    one-sample budget under-states early models. It also has no length spread and no complexity
    spread; one sample is a smoke-test setting, not a sizing one.)

    **Hyperparameters.** ``params_by_model`` (the fleetwide pre-pass's output) is passed straight
    through, so the measured fit is the fit that will run. Under **per-series** HPO the pre-pass
    instead passes ``{}``: leaving it ``None`` would make each probe call ``worker._resolve_params``
    and run a full Optuna study, turning an 8-sample pre-pass into hundreds of driver-side fits
    with nothing to bound it (see `measure_fit`). Profiling an untuned fit under per-series
    tuning is a known under-statement; running the tuner ``samples x models`` times to avoid it is
    worse.

    ``measure`` injects the measurement function for the offline gate — the default is
    `measure_fit`, and the tests pass a deterministic stand-in so the whole pre-pass, gate
    included, is testable with no fit, no accelerator, and no cloud.
    """
    measure_one = measure or measure_fit
    profilable = _profilable_models(models if models is not None else cfg.models)
    id_col = cfg.data.ts_id_col
    n_series = int(panel[id_col].nunique()) if len(panel) and id_col in panel.columns else 0
    if not profilable or not should_profile(cfg, n_series * len(profilable)):
        return None

    try:
        stats = series_stats(panel, cfg)
    except DataError:
        # A panel we cannot even describe is a panel we cannot sample. Fall back to static
        # config rather than to an arbitrary subset — the run itself will report the real error.
        return None
    sample = select_profile_sample(stats, samples=cfg.compute.profile.samples)
    if not sample:
        return None

    # One frame per sampled id, taken once: slicing the panel inside the loop would rescan it
    # samples x models times for no benefit.
    wanted = {spec.ts_id for spec in sample}
    frames = {
        str(key): frame.reset_index(drop=True)
        for key, frame in panel.groupby(panel[id_col].astype(str), sort=False)
        if str(key) in wanted
    }

    tuned = dict(params_by_model or {})
    untuned: dict[str, Any] | None = (
        {} if (cfg.hpo.enabled and cfg.hpo.granularity == "per_series") else None
    )

    measurements: list[MeasuredFit] = []
    for spec in sample:
        series = frames.get(spec.ts_id)
        if series is None:
            continue
        for model_name in profilable:
            measurements.append(
                measure_one(series, model_name, cfg, params=tuned.get(model_name, untuned))
            )

    profile = build_profile(
        measurements,
        sample=sample,
        memory_margin=cfg.compute.profile.memory_margin,
        time_margin=cfg.compute.profile.time_margin,
    )
    # The one path that measures this run's own data, on this run's own hardware, in-run: the
    # only place `basis="measured"` is unconditional.
    return _with_provenance(
        profile,
        ProfileProvenance(
            basis="measured", source="in-run", signature=signature_from_config(cfg)
        ),
    )
