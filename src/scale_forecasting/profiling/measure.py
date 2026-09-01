"""One instrumented fit — what a single cell of a model family actually consumed.

Static per-model resource guesses fail in both directions: too generous and every run
over-provisions, too tight and the task OOMs. `measure_fit` replaces the guess by
bracketing exactly one ``run_cell`` and reporting five axes — wall time, CPU time, absolute
process footprint, marginal host RSS, peak device bytes — from which
``effective_cores = cpu_s / wall_s`` answers "does this model use more than one core" by
observation instead of by assertion.

**Two of those axes are traps, and both traps were found by measuring rather than by
reasoning.** They are stated up front because in each case the obvious implementation
produces a plausible number that is wrong by more than an order of magnitude — the failure
mode a sizing pre-pass can least afford, since nothing downstream can tell a bad
measurement from a good one.

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
fits are served from an already-warm heap and report 0.00 MB. The absolute high-water lands
within 0.6% regardless of order, and it is also the number a slot actually needs — a slot
holds the interpreter and the libraries too. See `MeasuredFit` for the measurements.

A real fit runs here, so this module is the I/O side of the package's pure/I-O seam: it is
exercised live, while `cost` — the arithmetic that turns these records into a sizing
decision — stays testable with no accelerator, no cluster and no cloud.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig

# Floor on ``cpu_s / wall_s``. A fit cannot be scheduled on less than one core, and a fit
# too fast to time would otherwise report 0.
_MIN_EFFECTIVE_CORES = 1.0  # cores

# Below this a wall-clock reading is under the resolution of ``perf_counter`` on any
# platform we run, so the ratio is meaningless rather than large.
_MIN_WALL_S = 1e-6  # seconds

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

# ``getrusage.ru_maxrss`` is KiB on Linux and bytes on macOS. This is not pedantry: getting
# it wrong is a silent 1024x error in the axis that sizes host memory — one direction on a
# dev laptop, the other on a cluster.
_RSS_UNIT_BYTES = 1 if sys.platform == "darwin" else 1024  # bytes per ru_maxrss unit


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

        from ..models import get_model
        from ..worker import run_cell

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
