"""Pure example-data generator — no I/O.

Deterministic panel math: each series is a sum of readable components —
``trend + short-cycle & yearly seasonality + holiday bumps + AR(1) noise`` — then shaped by
an **archetype** (smooth-seasonal, intermittent, trending, promo-spiky, noisy) that applies
intermittency, level shifts, and promo spikes. Seasonality is measured in *steps* (via the
shared `seasonality` maps), so the panel is coherent at any ``freq`` — daily, weekly,
monthly, hourly — and daily output is byte-for-byte the same as day-count math. Every series
draws its parameters from an rng **seeded by its own index**, so:

* series ``i`` is byte-for-byte identical no matter which partition produced it — hence
  ``generate_panel(n)`` equals the union of any partitioning of ``range(n)`` (the property
  the Spark seed job relies on), and
* one master ``seed`` reproduces the whole 100k dataset on every deployment.

The five archetypes deliberately favor different models — that's what makes the ensemble and
straggler contrasts visible downstream. The golden test fixture is a tiny call into this same
code, so tests and shipped data share one path.

Public surface: `GenConfig`, `generate_partition`, `generate_panel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..seasonality import periods_per_year, seasonal_period

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# --- configuration -------------------------------------------------------------


@dataclass(frozen=True)
class GenConfig:
    """How to *shape* each series — not how many. The count is the caller's ``id_range`` /
    ``n`` (a partition of series ids), which keeps the partition-union invariant orthogonal
    to the per-series knobs. Pure parameters — no source/sink here (that's ``seed_spark.py``).
    """

    history: int = 1460  # number of periods of history (at freq); ~4 years daily
    freq: str = "D"
    start: str = "2021-01-01"
    holidays: tuple[str, ...] = ("US",)
    with_exog: bool = False  # emit covariate drivers + hierarchy attributes
    with_hierarchy: bool = False  # emit static hierarchy attributes (region, category)


# Static hierarchy dimensions assigned deterministically from series index `i`.
# 4 regions × 3 categories = 12 bottom-level cells before series-id disaggregation.
REGIONS: tuple[str, ...] = ("NA", "EMEA", "APAC", "LATAM")
CATEGORIES: tuple[str, ...] = ("enterprise", "SMB", "consumer")


# Deterministic static covariate signal maps (applied when `with_exog=True`).
# Give `region` and `category` real cross-series level, trend, seasonal, and promo-elasticity
# effects so global models consuming `static_covariates` learn genuine structural differences.
_REGION_LEVEL_OFFSET: dict[str, float] = {
    "NA": 0.15,
    "EMEA": 0.00,
    "APAC": 0.08,
    "LATAM": -0.12,
}
_REGION_SEASONAL_MOD: dict[str, float] = {
    "NA": 0.06,
    "EMEA": 0.02,
    "APAC": -0.04,
    "LATAM": -0.05,
}
_CATEGORY_TREND_BONUS: dict[str, float] = {
    "enterprise": 0.12,
    "SMB": 0.04,
    "consumer": -0.03,
}
_CATEGORY_PROMO_LIFT: dict[str, float] = {
    "enterprise": 0.04,
    "SMB": 0.09,
    "consumer": 0.16,
}


# --- archetypes ----------------------------------------------------------------


@dataclass(frozen=True)
class Archetype:
    """One series "personality". Each field is a ``(low, high)`` range the per-series rng
    samples uniformly, or a scalar probability. Distinct profiles make distinct models win.
    """

    name: str
    base: tuple[float, float]  # baseline level
    trend_frac: tuple[float, float]  # total drift over history, as a fraction of base
    weekly_amp: tuple[float, float]  # weekly seasonal amplitude, fraction of base
    yearly_amp: tuple[float, float]  # yearly seasonal amplitude, fraction of base
    noise_frac: tuple[float, float]  # AR(1) innovation sigma, fraction of base
    ar1_rho: tuple[float, float]  # AR(1) persistence
    holiday_frac: tuple[float, float]  # holiday bump, fraction of base
    zero_inflation: float  # P(a given day is dropped to zero) — intermittency
    spike_rate: float  # P(a given day gets a promo spike)
    spike_mult: tuple[float, float]  # promo spike multiplier
    level_shift_prob: float  # P(the series has one abrupt level shift)


# ~5 buckets. Order fixed so ``i % len`` assigns them reproducibly.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        name="smooth_seasonal",
        base=(80.0, 200.0),
        trend_frac=(-0.1, 0.3),
        weekly_amp=(0.15, 0.35),
        yearly_amp=(0.2, 0.5),
        noise_frac=(0.02, 0.06),
        ar1_rho=(0.1, 0.4),
        holiday_frac=(0.1, 0.3),
        zero_inflation=0.0,
        spike_rate=0.0,
        spike_mult=(1.0, 1.0),
        level_shift_prob=0.05,
    ),
    Archetype(
        name="intermittent",
        base=(2.0, 12.0),
        trend_frac=(-0.05, 0.1),
        weekly_amp=(0.0, 0.1),
        yearly_amp=(0.0, 0.15),
        noise_frac=(0.1, 0.3),
        ar1_rho=(0.0, 0.2),
        holiday_frac=(0.0, 0.1),
        zero_inflation=0.6,
        spike_rate=0.01,
        spike_mult=(2.0, 5.0),
        level_shift_prob=0.05,
    ),
    Archetype(
        name="trending",
        base=(40.0, 120.0),
        trend_frac=(0.6, 1.8),
        weekly_amp=(0.05, 0.2),
        yearly_amp=(0.1, 0.3),
        noise_frac=(0.04, 0.1),
        ar1_rho=(0.2, 0.6),
        holiday_frac=(0.05, 0.2),
        zero_inflation=0.0,
        spike_rate=0.0,
        spike_mult=(1.0, 1.0),
        level_shift_prob=0.15,
    ),
    Archetype(
        name="promo_spiky",
        base=(50.0, 150.0),
        trend_frac=(-0.1, 0.4),
        weekly_amp=(0.1, 0.25),
        yearly_amp=(0.1, 0.3),
        noise_frac=(0.04, 0.1),
        ar1_rho=(0.1, 0.4),
        holiday_frac=(0.1, 0.3),
        zero_inflation=0.0,
        spike_rate=0.03,
        spike_mult=(2.5, 6.0),
        level_shift_prob=0.1,
    ),
    Archetype(
        name="noisy",
        base=(30.0, 100.0),
        trend_frac=(-0.3, 0.3),
        weekly_amp=(0.03, 0.12),
        yearly_amp=(0.03, 0.15),
        noise_frac=(0.2, 0.45),
        ar1_rho=(0.5, 0.85),
        holiday_frac=(0.0, 0.1),
        zero_inflation=0.0,
        spike_rate=0.005,
        spike_mult=(2.0, 4.0),
        level_shift_prob=0.1,
    ),
)


# --- panel time axis + holiday calendar (shared across series) -----------------


def _time_axis(cfg: GenConfig) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The shared date index and its integer step positions (0, 1, 2, … at ``freq``).

    Seasonality is built off *step position*, not calendar days, so a cycle spans the
    right number of steps at any frequency (a "yearly" cycle is 52 weekly steps or 12
    monthly steps, not 365 days). For daily data step == day-offset, so daily panels are
    byte-for-byte unchanged.
    """
    index = pd.date_range(cfg.start, periods=cfg.history, freq=cfg.freq).as_unit("ns")
    pos = np.arange(len(index), dtype=float)
    return index, pos


def _holiday_mask(index: pd.DatetimeIndex, codes: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``index`` marking holidays for the given country codes.

    Uses the same ``holidays`` package the models/features use, so generated holiday bumps
    line up with the holiday features the models see (parity).
    """
    if not codes:
        return np.zeros(len(index), dtype=bool)
    import holidays as holidays_pkg

    years = range(index[0].year, index[-1].year + 1)
    days: set[pd.Timestamp] = set()
    for code in codes:
        cal = holidays_pkg.country_holidays(code, years=years)
        days.update(pd.Timestamp(d) for d in cal)
    holiday_days = pd.DatetimeIndex(sorted(days)).as_unit("ns").to_numpy()
    return np.isin(index.normalize().to_numpy(), holiday_days)


def is_holiday_flags(ds: Sequence[object] | pd.Series, holidays: Sequence[str]) -> np.ndarray:
    """Boolean holiday flags for a sequence of dates — the public view of the panel's calendar.

    The generator applies holidays as a numeric *bump* on ``y`` (it emits no holiday column),
    but the ``source_series`` table carries an ``is_holiday`` column the models read. This
    derives that flag from the **same** ``holidays`` calendar the bump uses, so the shipped
    ``is_holiday`` agrees with the effect baked into ``y`` (parity). Pure — used by
    the Spark seed transform (`data_gen.seed_spark`).

    ``ds`` is any date-like sequence (a column of ``generate_partition`` output, a list of
    ``date``/``Timestamp``); ``holidays`` is the country codes (e.g. ``("US",)``). Returns a
    ``bool`` array aligned to ``ds``.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(ds))).as_unit("ns")
    return _holiday_mask(index, holidays)


# --- one series ----------------------------------------------------------------


def _uniform(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def _series_seed(master: int, i: int) -> np.random.Generator:
    """A generator seeded from ``(master, i)`` — independent and reproducible per series."""
    return np.random.default_rng([master, i])


def _one_series(
    i: int,
    cfg: GenConfig,
    seed: int,
    pos: np.ndarray,
    holiday_mask: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Generate the components of a single series ``i`` as plain arrays.

    ``pos`` is the integer step position (0, 1, 2, …) at the run frequency. All cycles are
    measured in steps via the shared seasonality maps, so a series is coherent at any freq:
    the "weekly" amplitude drives the dominant sub-annual cycle (7 steps daily, 24 hourly)
    and the "yearly" amplitude drives the annual cycle (365.25 steps daily, 52 weekly, 12
    monthly). For daily data these are 7 and 365.25 — byte-for-byte the original.
    """
    arch = ARCHETYPES[i % len(ARCHETYPES)]
    rng = _series_seed(seed, i)
    short_period = float(seasonal_period(cfg.freq))
    year_period = periods_per_year(cfg.freq)
    n = pos.size
    t = pos / max(pos[-1], 1.0)  # normalized time in [0, 1]

    base = _uniform(rng, arch.base)
    trend = base * _uniform(rng, arch.trend_frac) * t
    weekly = base * _uniform(rng, arch.weekly_amp) * np.sin(2 * np.pi * pos / short_period)
    yearly = base * _uniform(rng, arch.yearly_amp) * np.sin(2 * np.pi * pos / year_period)
    holiday = base * _uniform(rng, arch.holiday_frac) * holiday_mask

    # AR(1) colored noise via a one-pole filter on white innovations.
    rho = _uniform(rng, arch.ar1_rho)
    sigma = base * _uniform(rng, arch.noise_frac)
    innovations = rng.normal(0.0, sigma, size=n)
    noise = lfilter([1.0], [1.0, -rho], innovations)

    # Optional three-tier covariate drivers (`with_exog=True`):
    # 1. `static_covariates` (`region`, `category`): cross-series level, seasonal, trend, and
    #    promo-elasticity differences (`consumer` responds 4x stronger to promos than `enterprise`).
    # 2. `future_covariates` (`price_index`, `promo_flag` + `is_holiday`): known-in-advance drivers
    #    (`price_index` drawn from primary `rng` preserving legacy draw order).
    # 3. `past_covariates` (`temperature`): observed weather driver (annual cycle + AR(1) noise from
    #    a sub-seed) with contemporaneous + lag-1 + lag-season carry-over so `exog_lags` carries
    #    genuine predictive signal into the forecast horizon.
    exog: np.ndarray | None = None
    promo_flag: np.ndarray | None = None
    temperature: np.ndarray | None = None
    exog_effect = np.zeros(n)
    region = REGIONS[i % len(REGIONS)]
    category = CATEGORIES[(i // len(REGIONS)) % len(CATEGORIES)]
    if cfg.with_exog:
        driver_period = year_period / 4.0
        exog = 100.0 + 20.0 * np.sin(2 * np.pi * pos / driver_period + _uniform(rng, (0.0, 6.28)))
        price_effect = base * _uniform(rng, (0.05, 0.2)) * (exog - 100.0) / 20.0

        static_effect = (
            base * _REGION_LEVEL_OFFSET[region]
            + _REGION_SEASONAL_MOD[region] * yearly
            + base * _CATEGORY_TREND_BONUS[category] * t
        )

        promo_cycle = max(int(round(short_period * 4)), 14)
        promo_window = max(1, min(3, promo_cycle // 10))
        promo_flag = (((pos.astype(int) + i * 7) % promo_cycle) < promo_window).astype(np.int64)
        promo_lift = base * _CATEGORY_PROMO_LIFT[category] * promo_flag.astype(float)

        cov_rng = np.random.default_rng([seed, i, 1])
        temp_phase = _uniform(cov_rng, (0.0, 6.28))
        temp_clean = 18.0 + 12.0 * np.sin(
            2 * np.pi * pos / year_period - np.pi / 2 + temp_phase * 0.1
        )
        temp_noise = lfilter([1.0], [1.0, -0.7], cov_rng.normal(0.0, 2.0, size=n))
        temperature = temp_clean + temp_noise
        temp_norm = (temperature - 18.0) / 12.0
        temp_lag1 = np.roll(temp_norm, 1)
        temp_lag1[0] = temp_norm[0]
        lag_s = max(1, min(int(round(short_period)), max(n - 1, 1)))
        temp_lags = np.roll(temp_norm, lag_s)
        temp_lags[:lag_s] = temp_norm[:lag_s]
        temp_effect = base * (0.03 * temp_norm + 0.04 * temp_lag1 + 0.025 * temp_lags)
        exog_effect = static_effect + price_effect + promo_lift + temp_effect

    y = base + trend + weekly + yearly + holiday + noise + exog_effect

    # --- archetype shaping ---
    # One abrupt level shift (a changepoint) partway through the history.
    if rng.random() < arch.level_shift_prob:
        cut = int(rng.uniform(0.3, 0.7) * n)
        y[cut:] += base * _uniform(rng, (-0.5, 0.5))

    # Promo spikes: a few steps multiplied up (outliers/promos).
    if arch.spike_rate > 0:
        spike_days = rng.random(n) < arch.spike_rate
        y[spike_days] *= _uniform(rng, arch.spike_mult)

    # Intermittency: zero-inflate slow movers so preprocessing/robust models matter.
    if arch.zero_inflation > 0:
        y[rng.random(n) < arch.zero_inflation] = 0.0

    y = np.clip(y, 0.0, None).round(3)

    out: dict[str, np.ndarray | str] = {"archetype": arch.name, "y": y}
    if cfg.with_hierarchy or cfg.with_exog:
        out["region"] = region
        out["category"] = category
    if promo_flag is not None:
        out["promo_flag"] = promo_flag
    if exog is not None:
        out["price_index"] = exog.round(3)
    if temperature is not None:
        out["temperature"] = temperature.round(3)
    return out


# --- public API ----------------------------------------------------------------


def generate_partition(id_range: Iterable[int], cfg: GenConfig, seed: int) -> pd.DataFrame:
    """Generate the long-format panel for a set of series ids.

    ``id_range`` is any iterable of integer series indices (a partition of the full range in
    the Spark seed job). Returns a frame with columns ``ts_id, archetype, ds, y`` (plus
    ``region, category`` when ``cfg.with_hierarchy`` or ``cfg.with_exog``, and
    ``promo_flag, price_index, temperature`` when ``cfg.with_exog``), sorted by ``ts_id``
    then ``ds``. Pure and deterministic: identical output for identical ``(id, cfg, seed)``.
    """
    index, pos = _time_axis(cfg)
    holiday_mask = _holiday_mask(index, cfg.holidays)
    ds = index.to_numpy()

    frames: list[pd.DataFrame] = []
    for i in id_range:
        comp = _one_series(i, cfg, seed, pos, holiday_mask)
        cols: dict[str, object] = {
            "ts_id": f"s_{i:06d}",
            "archetype": comp["archetype"],
        }
        if "region" in comp:
            cols["region"] = comp["region"]
            cols["category"] = comp["category"]
        cols["ds"] = ds
        cols["y"] = comp["y"]
        for cov in ("promo_flag", "price_index", "temperature"):
            if cov in comp:
                cols[cov] = comp[cov]
        frames.append(pd.DataFrame(cols))

    if not frames:
        base_cols = ["ts_id", "archetype"]
        if cfg.with_hierarchy or cfg.with_exog:
            base_cols.extend(["region", "category"])
        base_cols.extend(["ds", "y"])
        if cfg.with_exog:
            base_cols.extend(["promo_flag", "price_index", "temperature"])
        return pd.DataFrame({c: pd.Series(dtype="object") for c in base_cols})
    return pd.concat(frames, ignore_index=True)


def generate_panel(n: int, cfg: GenConfig, seed: int) -> pd.DataFrame:
    """Generate the full panel of the first ``n`` series — ``generate_partition(range(n))``.

    Because each series is seeded by its own index, this is exactly the union of any
    partitioning of ``range(n)`` (the invariant the distributed seed job depends on), and
    subsetting to the first ``k`` series (``data.series_limit``) is just a prefix of this.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return generate_partition(range(n), cfg, seed)
