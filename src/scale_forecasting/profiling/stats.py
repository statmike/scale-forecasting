"""Cheap per-series statistics — one pass over the panel, no fits.

The inputs to sampling. Everything here is a pure function of a value column: length,
volatility, intermittency, and seasonal strength. No model is fit and no accelerator is
touched, so the whole module is unit-testable with injected numbers.

Feeds `sampling.select_profile_sample`, which turns these statistics into the decision of
*which* series are worth the cost of a real instrumented fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors import DataError

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig

# Denominator guard for ``diff_cv``. A series whose mean level is 0 (all zeros) must yield
# 0.0 volatility; an inf/NaN here would propagate into a sort key and destroy determinism.
_MIN_LEVEL = 1e-12  # target units


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
    from ..seasonality import seasonal_period

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
    pairs = pd.DataFrame({"ts_id": valid["ts_id"], "x": centered, "z": lagged}).dropna(subset=["z"])
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
