"""Asymmetric pre-holiday build-up and post-holiday slump window covariates.

Generates deterministic distance-to-holiday and distance-from-holiday ramp
features (``pre_holiday_window``, ``post_holiday_window``,
``days_to_next_holiday``, ``days_since_prev_holiday``, ``holiday_offset_days``)
around country/custom holiday calendars so both linear and tree-based models
capture multi-day pre-holiday stockpiling and post-holiday demand slumps
rather than only single-day ``is_holiday`` spikes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

HOLIDAY_WINDOW_COLUMNS: tuple[str, ...] = (
    "pre_holiday_window",
    "post_holiday_window",
    "days_to_next_holiday",
    "days_since_prev_holiday",
    "holiday_offset_days",
)


class HolidayWindowConfig(BaseModel):
    """Configuration for asymmetric pre- and post-holiday window covariates.

    Attributes:
        enabled: Whether to emit holiday-window lead/lag covariates when
            ``features.holidays`` is non-empty.
        pre_holiday_days: Number of calendar days prior to each holiday date
            (``1..pre_holiday_days``) over which the pre-holiday build-up
            ramp is active. Also accepted as ``pre_days``.
        post_holiday_days: Number of calendar days following each holiday date
            (``1..post_holiday_days``) over which the post-holiday slump
            recovery ramp is active. Also accepted as ``post_days``.
        kernel: Shape of the proximity weight curve across the window:
            ``"linear"`` (default triangular ramp peaking at 1.0 adjacent to the
            holiday), ``"indicator"`` (uniform 1.0 across the window), or
            ``"exponential"`` (exponential decay away from the holiday). Also
            accepted as ``decay`` (where ``"none"`` maps to ``"indicator"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    pre_holiday_days: int = Field(default=7, ge=0, le=30)
    post_holiday_days: int = Field(default=3, ge=0, le=30)
    kernel: Literal["indicator", "linear", "exponential"] = "linear"

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, bool):
            return (
                {"enabled": True}
                if data
                else {"enabled": False, "pre_holiday_days": 0, "post_holiday_days": 0}
            )
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return {
                "enabled": True,
                "pre_holiday_days": int(data[0]),
                "post_holiday_days": int(data[1]),
            }
        if isinstance(data, Mapping):
            d = dict(data)
            if "pre_days" in d:
                val_pre = d.pop("pre_days")
                d.setdefault("pre_holiday_days", val_pre)
            if "post_days" in d:
                val_post = d.pop("post_days")
                d.setdefault("post_holiday_days", val_post)
            if "decay" in d:
                val_decay = d.pop("decay")
                norm_kernel = "indicator" if str(val_decay).lower() == "none" else val_decay
                d.setdefault("kernel", norm_kernel)
            return d
        return data

    @property
    def pre_days(self) -> int:
        """Alias for ``pre_holiday_days``."""
        return self.pre_holiday_days

    @property
    def post_days(self) -> int:
        """Alias for ``post_holiday_days``."""
        return self.post_holiday_days

    @property
    def decay(self) -> str:
        """Alias for ``kernel`` (``'none'`` when ``kernel == 'indicator'``)."""
        return "none" if self.kernel == "indicator" else self.kernel

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-ready dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_value(cls, val: Any) -> HolidayWindowConfig | None:
        """Coerce shorthand representations into a :class:`HolidayWindowConfig` or ``None``."""
        return coerce_holiday_window_config(val)


def coerce_holiday_window_config(val: Any) -> HolidayWindowConfig | None:
    """Coerce shorthand config representations into a :class:`HolidayWindowConfig`.

    Supports ``None``, ``bool``, ``(pre_days, post_days)`` sequences, mappings,
    and :class:`HolidayWindowConfig` instances.
    """
    if val is None or val is False:
        return None
    if val is True:
        return HolidayWindowConfig()
    if isinstance(val, HolidayWindowConfig):
        return val
    if isinstance(val, (tuple, list)):
        if len(val) != 2:
            raise ValueError(
                "features.holiday_window tuple/list shorthand must have length 2 "
                f"(pre_holiday_days, post_holiday_days), got length {len(val)}."
            )
        return HolidayWindowConfig(
            pre_holiday_days=int(val[0]),
            post_holiday_days=int(val[1]),
        )
    if isinstance(val, Mapping):
        return HolidayWindowConfig(**dict(val))
    raise ValueError(
        "features.holiday_window must be a HolidayWindowConfig, dict, "
        f"(pre_days, post_days) tuple, bool, or None; got {type(val).__name__}."
    )


def holiday_window_columns(
    window_cfg: HolidayWindowConfig | Mapping[str, Any] | bool | Sequence[int] | None,
) -> list[str]:
    """Return the deterministic list of holiday-window feature names when active."""
    cfg = coerce_holiday_window_config(window_cfg)
    if cfg is None or not cfg.enabled:
        return []
    if cfg.pre_holiday_days == 0 and cfg.post_holiday_days == 0:
        return []
    return list(HOLIDAY_WINDOW_COLUMNS)


def extract_holiday_day_ordinals(
    holiday_source: pd.DataFrame | Sequence[Any] | pd.DatetimeIndex,
    *,
    ref_dates: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """Extract sorted unique integer day ordinals from a holiday frame, date sequence,
    or ISO country codes.
    """
    import numpy as np
    import pandas as pd

    raw_dates: pd.DatetimeIndex
    if isinstance(holiday_source, pd.DataFrame):
        if holiday_source.empty:
            return np.zeros(0, dtype=np.int64)
        if "ds" in holiday_source.columns:
            raw_dates = pd.DatetimeIndex(pd.to_datetime(holiday_source["ds"]))
        else:
            raw_dates = pd.DatetimeIndex(pd.to_datetime(holiday_source.index))
    elif isinstance(holiday_source, (list, tuple)):
        if not holiday_source:
            return np.zeros(0, dtype=np.int64)
        country_codes = [
            str(x).strip().upper()
            for x in holiday_source
            if isinstance(x, str) and 2 <= len(x.strip()) <= 3 and x.strip().isalpha()
        ]
        explicit_items = [
            x
            for x in holiday_source
            if not (isinstance(x, str) and 2 <= len(x.strip()) <= 3 and x.strip().isalpha())
        ]
        h_dates: list[pd.Timestamp] = []
        if country_codes:
            import holidays as holidays_pkg

            y_min = (
                int(ref_dates.min().year) - 1
                if ref_dates is not None and len(ref_dates) > 0
                else 2020
            )
            y_max = (
                int(ref_dates.max().year) + 1
                if ref_dates is not None and len(ref_dates) > 0
                else 2030
            )
            years = range(y_min, y_max + 1)
            for code in country_codes:
                cal = holidays_pkg.country_holidays(code, years=years)
                for day in cal:
                    h_dates.append(pd.Timestamp(day))
        if explicit_items:
            for ts in pd.to_datetime(pd.Index(explicit_items)):
                h_dates.append(pd.Timestamp(ts))
        if not h_dates:
            return np.zeros(0, dtype=np.int64)
        raw_dates = pd.DatetimeIndex(h_dates)
    else:
        raw_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(holiday_source)))

    if len(raw_dates) == 0:
        return np.zeros(0, dtype=np.int64)

    dt_idx = pd.DatetimeIndex(raw_dates)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_localize(None)
    norm = dt_idx.normalize()
    epoch = pd.Timestamp("1970-01-01")
    ordinals = ((norm - epoch) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)
    return np.unique(ordinals)


_extract_holiday_day_ordinals = extract_holiday_day_ordinals


def _compute_kernel_weights(
    distances: np.ndarray,
    active_mask: np.ndarray,
    window_days: int,
    kernel: Literal["indicator", "linear", "exponential"],
) -> np.ndarray:
    """Compute proximity weights in ``(0, 1]`` for active window observations."""
    import numpy as np

    weights = np.zeros(len(distances), dtype=float)
    if window_days <= 0 or not np.any(active_mask):
        return weights

    active_dist = distances[active_mask].astype(float)
    if kernel == "indicator":
        weights[active_mask] = 1.0
    elif kernel == "exponential":
        scale = max(float(window_days) / 2.0, 1.0)
        weights[active_mask] = np.exp(-(active_dist - 1.0) / scale)
    else:
        weights[active_mask] = (float(window_days) - active_dist + 1.0) / float(window_days)
    return weights


def build_holiday_window_features(
    dates: pd.DatetimeIndex | Sequence[Any],
    holiday_df: pd.DataFrame | Sequence[Any],
    window_cfg: HolidayWindowConfig | Mapping[str, Any] | bool | Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Build deterministic pre-holiday build-up and post-holiday slump features.

    For each timestamp in ``dates``, computes the calendar-day distance to the
    next Strictly Upcoming holiday (``d_to >= 1``) and from the most recent
    Strictly Past holiday (``d_since >= 1``). Outside the configured
    ``[1, pre_holiday_days]`` and ``[1, post_holiday_days]`` windows, all
    returned features are ``0.0`` so non-holiday baselines remain unperturbed.

    Args:
        dates: Observation timestamps (``pd.DatetimeIndex`` or sequence).
        holiday_df: DataFrame with a ``ds`` column (as returned by
            :func:`scale_forecasting.features.fetch_holidays`), sequence of
            holiday dates, or sequence of ISO country codes (e.g. ``["US"]``).
        window_cfg: :class:`HolidayWindowConfig` or compatible shorthand.

    Returns:
        Dictionary mapping each column in :data:`HOLIDAY_WINDOW_COLUMNS` to a
        1-D ``float64`` NumPy array of length ``len(dates)``, or ``{}`` when
        ``window_cfg`` is disabled.
    """
    import numpy as np
    import pandas as pd

    cfg = coerce_holiday_window_config(window_cfg)
    if cfg is None or not cfg.enabled:
        return {}
    if cfg.pre_holiday_days == 0 and cfg.post_holiday_days == 0:
        return {}

    dt_idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    n_obs = len(dt_idx)
    if n_obs == 0:
        return {col: np.zeros(0, dtype=float) for col in HOLIDAY_WINDOW_COLUMNS}

    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_localize(None)
    norm_dates = dt_idx.normalize()
    epoch = pd.Timestamp("1970-01-01")
    d_ord = ((norm_dates - epoch) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)

    h_ord = extract_holiday_day_ordinals(holiday_df, ref_dates=norm_dates)
    if len(h_ord) == 0:
        return {col: np.zeros(n_obs, dtype=float) for col in HOLIDAY_WINDOW_COLUMNS}

    # Strictly upcoming holiday: h_ord > d_ord (side="right")
    idx_next = np.searchsorted(h_ord, d_ord, side="right")
    has_next = idx_next < len(h_ord)
    safe_next = np.minimum(idx_next, len(h_ord) - 1)
    dist_to_next = np.where(has_next, h_ord[safe_next] - d_ord, np.iinfo(np.int64).max)

    # Strictly preceding holiday: h_ord < d_ord (side="left" - 1)
    idx_prev = np.searchsorted(h_ord, d_ord, side="left") - 1
    has_prev = idx_prev >= 0
    safe_prev = np.maximum(idx_prev, 0)
    dist_since_prev = np.where(has_prev, d_ord - h_ord[safe_prev], np.iinfo(np.int64).max)

    pre_days = int(cfg.pre_holiday_days)
    post_days = int(cfg.post_holiday_days)

    pre_mask = (
        (dist_to_next >= 1) & (dist_to_next <= pre_days)
        if pre_days > 0
        else np.zeros(n_obs, dtype=bool)
    )
    post_mask = (
        (dist_since_prev >= 1) & (dist_since_prev <= post_days)
        if post_days > 0
        else np.zeros(n_obs, dtype=bool)
    )

    pre_window = _compute_kernel_weights(dist_to_next, pre_mask, pre_days, cfg.kernel)
    post_window = _compute_kernel_weights(dist_since_prev, post_mask, post_days, cfg.kernel)

    days_to_next = np.where(pre_mask, dist_to_next.astype(float), 0.0)
    days_since_prev = np.where(post_mask, dist_since_prev.astype(float), 0.0)

    # Signed offset within active windows: negative leading into holiday, positive after
    offset = np.zeros(n_obs, dtype=float)
    offset = np.where(post_mask, dist_since_prev.astype(float), offset)
    offset = np.where(
        pre_mask & (~post_mask | (dist_to_next <= dist_since_prev)),
        -dist_to_next.astype(float),
        offset,
    )

    return {
        "pre_holiday_window": pre_window,
        "post_holiday_window": post_window,
        "days_to_next_holiday": days_to_next,
        "days_since_prev_holiday": days_since_prev,
        "holiday_offset_days": offset,
    }


__all__ = [
    "HOLIDAY_WINDOW_COLUMNS",
    "HolidayWindowConfig",
    "build_holiday_window_features",
    "coerce_holiday_window_config",
    "extract_holiday_day_ordinals",
    "holiday_window_columns",
]
