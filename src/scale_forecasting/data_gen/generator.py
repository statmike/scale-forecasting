"""Pure example-data generator — no I/O (CONTRACTS §6, DESIGN §13.1).

Deterministic panel math: each series is a sum of readable components —
``trend + weekly & yearly seasonality + holiday bumps + AR(1) noise`` — then shaped by an
**archetype** (smooth-seasonal, intermittent, trending, promo-spiky, noisy) that applies
intermittency, level shifts, and promo spikes. Every series draws its parameters from an rng
**seeded by its own index**, so:

* series ``i`` is byte-for-byte identical no matter which partition produced it — hence
  ``generate_panel(n)`` equals the union of any partitioning of ``range(n)`` (the property
  the Spark seed job in Arc B relies on), and
* one master ``seed`` reproduces the whole 100k dataset on every deployment.

The five archetypes deliberately favor different models — that's what makes the ensemble and
straggler contrasts visible downstream. The golden test fixture is a tiny call into this same
code, so tests and shipped data share one path.

Public surface: :class:`GenConfig`, :func:`generate_partition`, :func:`generate_panel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import lfilter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# --- configuration -------------------------------------------------------------


@dataclass(frozen=True)
class GenConfig:
    """What to generate. Pure parameters — no source/sink here (that's ``seed_spark.py``)."""

    n_series: int = 100_000
    history_days: int = 1460  # ~4 years of daily history
    freq: str = "D"
    start: str = "2021-01-01"
    holidays: tuple[str, ...] = ("US",)
    with_exog: bool = False  # emit a driver column + let y respond to it (xreg paths)


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


# ~5 buckets (DESIGN §13.1). Order fixed so ``i % len`` assigns them reproducibly.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        name="smooth_seasonal",
        base=(80.0, 200.0), trend_frac=(-0.1, 0.3),
        weekly_amp=(0.15, 0.35), yearly_amp=(0.2, 0.5),
        noise_frac=(0.02, 0.06), ar1_rho=(0.1, 0.4), holiday_frac=(0.1, 0.3),
        zero_inflation=0.0, spike_rate=0.0, spike_mult=(1.0, 1.0), level_shift_prob=0.05,
    ),
    Archetype(
        name="intermittent",
        base=(2.0, 12.0), trend_frac=(-0.05, 0.1),
        weekly_amp=(0.0, 0.1), yearly_amp=(0.0, 0.15),
        noise_frac=(0.1, 0.3), ar1_rho=(0.0, 0.2), holiday_frac=(0.0, 0.1),
        zero_inflation=0.6, spike_rate=0.01, spike_mult=(2.0, 5.0), level_shift_prob=0.05,
    ),
    Archetype(
        name="trending",
        base=(40.0, 120.0), trend_frac=(0.6, 1.8),
        weekly_amp=(0.05, 0.2), yearly_amp=(0.1, 0.3),
        noise_frac=(0.04, 0.1), ar1_rho=(0.2, 0.6), holiday_frac=(0.05, 0.2),
        zero_inflation=0.0, spike_rate=0.0, spike_mult=(1.0, 1.0), level_shift_prob=0.15,
    ),
    Archetype(
        name="promo_spiky",
        base=(50.0, 150.0), trend_frac=(-0.1, 0.4),
        weekly_amp=(0.1, 0.25), yearly_amp=(0.1, 0.3),
        noise_frac=(0.04, 0.1), ar1_rho=(0.1, 0.4), holiday_frac=(0.1, 0.3),
        zero_inflation=0.0, spike_rate=0.03, spike_mult=(2.5, 6.0), level_shift_prob=0.1,
    ),
    Archetype(
        name="noisy",
        base=(30.0, 100.0), trend_frac=(-0.3, 0.3),
        weekly_amp=(0.03, 0.12), yearly_amp=(0.03, 0.15),
        noise_frac=(0.2, 0.45), ar1_rho=(0.5, 0.85), holiday_frac=(0.0, 0.1),
        zero_inflation=0.0, spike_rate=0.005, spike_mult=(2.0, 4.0), level_shift_prob=0.1,
    ),
)


# --- panel time axis + holiday calendar (shared across series) -----------------


def _time_axis(cfg: GenConfig) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The shared date index and its day-offsets from the start (for seasonality)."""
    index = pd.date_range(cfg.start, periods=cfg.history_days, freq=cfg.freq).as_unit("ns")
    day = (index - index[0]).days.to_numpy().astype(float)
    return index, day


def _holiday_mask(index: pd.DatetimeIndex, codes: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``index`` marking holidays for the given country codes.

    Uses the same ``holidays`` package the models/features use, so generated holiday bumps
    line up with the holiday features the models see (parity, DESIGN §4).
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
    day: np.ndarray,
    holiday_mask: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Generate the components of a single series ``i`` as plain arrays."""
    arch = ARCHETYPES[i % len(ARCHETYPES)]
    rng = _series_seed(seed, i)
    t = day / max(day[-1], 1.0)  # normalized time in [0, 1]

    base = _uniform(rng, arch.base)
    trend = base * _uniform(rng, arch.trend_frac) * t
    weekly = base * _uniform(rng, arch.weekly_amp) * np.sin(2 * np.pi * day / 7.0)
    yearly = base * _uniform(rng, arch.yearly_amp) * np.sin(2 * np.pi * day / 365.25)
    holiday = base * _uniform(rng, arch.holiday_frac) * holiday_mask

    # AR(1) colored noise via a one-pole filter on white innovations.
    rho = _uniform(rng, arch.ar1_rho)
    sigma = base * _uniform(rng, arch.noise_frac)
    innovations = rng.normal(0.0, sigma, size=day.size)
    noise = lfilter([1.0], [1.0, -rho], innovations)

    # Optional exogenous driver: a smooth index the target partially follows (xreg paths).
    exog: np.ndarray | None = None
    exog_effect = np.zeros(day.size)
    if cfg.with_exog:
        exog = 100.0 + 20.0 * np.sin(2 * np.pi * day / 90.0 + _uniform(rng, (0.0, 6.28)))
        exog_effect = base * _uniform(rng, (0.05, 0.2)) * (exog - 100.0) / 20.0

    y = base + trend + weekly + yearly + holiday + noise + exog_effect

    # --- archetype shaping ---
    # One abrupt level shift (a changepoint) partway through the history.
    if rng.random() < arch.level_shift_prob:
        cut = int(rng.uniform(0.3, 0.7) * day.size)
        y[cut:] += base * _uniform(rng, (-0.5, 0.5))

    # Promo spikes: a few days multiplied up (outliers/promos).
    if arch.spike_rate > 0:
        spike_days = rng.random(day.size) < arch.spike_rate
        y[spike_days] *= _uniform(rng, arch.spike_mult)

    # Intermittency: zero-inflate slow movers so preprocessing/robust models matter.
    if arch.zero_inflation > 0:
        y[rng.random(day.size) < arch.zero_inflation] = 0.0

    y = np.clip(y, 0.0, None).round(3)

    out: dict[str, np.ndarray | str] = {"archetype": arch.name, "y": y}
    if exog is not None:
        out["price_index"] = exog.round(3)
    return out


# --- public API ----------------------------------------------------------------


def generate_partition(
    id_range: Iterable[int], cfg: GenConfig, seed: int
) -> pd.DataFrame:
    """Generate the long-format panel for a set of series ids (CONTRACTS §6).

    ``id_range`` is any iterable of integer series indices (a partition of the full range in
    the Spark seed job). Returns a frame with columns ``ts_id, archetype, ds, y`` (plus
    ``price_index`` when ``cfg.with_exog``), sorted by ``ts_id`` then ``ds``. Pure and
    deterministic: identical output for identical ``(id, cfg, seed)``.
    """
    index, day = _time_axis(cfg)
    holiday_mask = _holiday_mask(index, cfg.holidays)
    ds = index.to_numpy()

    frames: list[pd.DataFrame] = []
    for i in id_range:
        comp = _one_series(i, cfg, seed, day, holiday_mask)
        cols: dict[str, object] = {
            "ts_id": f"s_{i:06d}",
            "archetype": comp["archetype"],
            "ds": ds,
            "y": comp["y"],
        }
        if "price_index" in comp:
            cols["price_index"] = comp["price_index"]
        frames.append(pd.DataFrame(cols))

    if not frames:
        base_cols = ["ts_id", "archetype", "ds", "y"] + (["price_index"] if cfg.with_exog else [])
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
