"""Layer 4 of the GPU contract: the evidence a cell records, and the verdict drawn from it.

Layer 3 makes a cell *state* its device. This is what checks the statement came true — and the
checking has to be able to fail, which is the whole point of the negative case below. The failure
being guarded against is the one the ledger already contains: a job attaches a T4, every cell fits
on the CPU, every forecast is correct, and nothing anywhere says so.

Two rules are asserted more than once because both were violated by the code this replaces:

* **Recorded, never inferred.** ``peak_gpu_bytes`` cannot stand in for "a device was used" — its
  ``None`` is overloaded across four unrelated causes (no torch, no CUDA build, no device,
  ``measure="off"``), and a *non*-``None`` value proves only that somebody once allocated in this
  process, not that this family's cells ran on a card.
* **Idle is not missing.** ``ENGAGED_IDLE`` is the expected verdict for every GPU run shipped
  today; it is a cost finding, and it warns rather than failing a correct run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from scale_forecasting import device_audit, hardware
from scale_forecasting.device_audit import (
    ENGAGED_IDLE,
    ENGAGED_UTILISED,
    MISSING_DEVICE,
    audit_device_use,
    device_verdict,
)
from scale_forecasting.engines.ray_io import device_memory_bytes
from scale_forecasting.models.base_model import BaseModel

_T4 = device_memory_bytes("T4")
_FLOOR = _T4 // 100  # 1% of the card — the line between engaged-idle and engaged-utilised


# --- T-D: the verdict, per row of the table ------------------------------------------------


@pytest.mark.parametrize(
    ("hw", "gpu_type", "on_device", "peak", "expected"),
    [
        # A family that never asked for a device has no claim to check.
        ("cpu", None, 0, None, None),
        ("cpu", None, 40, 999_999_999, None),
        (None, None, 0, None, None),
        # Asked, and nothing reports having run on one. After Layer 3 this should be unreachable;
        # it is kept as the regression detector for the whole contract.
        ("gpu", "T4", 0, None, MISSING_DEVICE),
        # Ran on the device and barely touched it — the Phase 0 shape, 75 KB on a 17 GB card.
        ("gpu", "T4", 200, 75_000, ENGAGED_IDLE),
        # Exactly at the floor is engaged: the comparison is >=, so the boundary is inclusive and
        # a device sized to exactly 1% does not flip verdict on a rounding difference.
        ("gpu", "T4", 200, _FLOOR, ENGAGED_UTILISED),
        ("gpu", "T4", 200, _FLOOR - 1, ENGAGED_IDLE),
        ("gpu", "T4", 1, 8_000_000_000, ENGAGED_UTILISED),
        # The floor is a share of *this* card, not a constant: 1% of an L4 is above 1% of a T4, so
        # the same peak can be utilised on the smaller device and idle on the larger one.
        ("gpu", "L4", 200, _FLOOR, ENGAGED_IDLE),
        ("gpu", "L4", 200, device_memory_bytes("L4") // 100, ENGAGED_UTILISED),
    ],
)
def test_verdict_table(
    hw: str | None, gpu_type: str | None, on_device: int, peak: int | None, expected: str | None
) -> None:
    assert (
        device_verdict(
            hardware=hw, gpu_type=gpu_type, cells_on_device=on_device, max_peak_gpu_bytes=peak
        )
        == expected
    )


def test_a_device_that_was_used_but_unmeasured_reads_idle_not_missing() -> None:
    """The cells say ``cuda``; the allocator high-water is unreadable. That is a gap in the
    measurement, not evidence the card was absent — and calling it MISSING_DEVICE would fire the
    contract's regression alarm every time ``measure="off"``."""
    for peak in (None, 0):
        assert (
            device_verdict(
                hardware="gpu", gpu_type="T4", cells_on_device=500, max_peak_gpu_bytes=peak
            )
            == ENGAGED_IDLE
        )


# --- T-E: the negative case — the audit has to be able to fail -----------------------------


def test_gpu_requested_cpu_used_is_missing_even_with_bytes_on_the_clock() -> None:
    """The case the whole layer exists for, and the one an inferred verdict gets wrong.

    A family provisioned onto GPU hardware whose cells all report ``device_used="cpu"``, while
    ``peak_gpu_bytes`` reads 75,000 — because *something* in the process touched CUDA (the Ray
    fraction calibration probe does exactly this on the head node). Inferring the verdict from the
    bytes would call this a working GPU run. It is not one, and it must come back MISSING_DEVICE.
    """
    verdict = device_verdict(
        hardware="gpu", gpu_type="T4", cells_on_device=0, max_peak_gpu_bytes=75_000
    )
    assert verdict == MISSING_DEVICE

    blob = audit_device_use(
        "run-abc123def456",
        "deep_learning",
        ["neuralprophet"],
        hardware="gpu",
        gpu_type="T4",
        read=_reader({"cells": 200, "cells_on_device": 0, "max_peak_gpu_bytes": 75_000}),
    )
    assert blob is not None
    assert blob["verdict"] == MISSING_DEVICE
    assert blob["cells_on_device"] == 0


# --- the blob that gets filed ---------------------------------------------------------------


def _reader(row: dict[str, Any]) -> Any:
    """Stand in for the BigQuery aggregate, the way `profiling.source` injects a measurement."""

    def read(run_id: str, models: list[str], *, settings: Any = None) -> dict[str, Any]:
        read.seen = (run_id, models)  # type: ignore[attr-defined]
        return row

    return read


def test_a_cpu_family_never_runs_the_query() -> None:
    """The audit costs a CPU family one comparison, not a BigQuery job. Most families are CPU."""

    def explode(*a: Any, **k: Any) -> dict[str, Any]:
        raise AssertionError("a cpu family must short-circuit before the aggregate")

    assert (
        audit_device_use("r", "statistical", ["theta"], hardware="cpu", gpu_type=None, read=explode)
        is None
    )


def test_an_empty_aggregate_files_nothing() -> None:
    """A failed or empty read returns ``{}``; there is no verdict to record, and inventing one
    (``MISSING_DEVICE`` from zero rows) would report a contract breach that never happened."""
    assert (
        audit_device_use(
            "r", "dl", ["neuralprophet"], hardware="gpu", gpu_type="T4", read=_reader({})
        )
        is None
    )


def test_the_filed_blob_carries_the_counts_the_verdict_was_drawn_from() -> None:
    """A verdict with no evidence beside it cannot be re-checked six months later."""
    blob = audit_device_use(
        "run-abc123def456",
        "deep_learning",
        ["neuralprophet"],
        hardware="gpu",
        gpu_type="T4",
        read=_reader(
            {
                "cells": 200,
                "cells_on_device": 198,
                "cells_no_device": 2,
                "max_peak_gpu_bytes": 78_000,
                "device_name": "Tesla T4",
            }
        ),
    )
    assert blob == {
        "verdict": ENGAGED_IDLE,
        "gpu_type": "T4",
        "cells": 200,
        "cells_on_device": 198,
        "cells_no_device": 2,
        "max_peak_gpu_bytes": 78_000,
        "device_name": "Tesla T4",
    }


def test_idle_and_missing_warn_and_neither_raises() -> None:
    """``ENGAGED_IDLE`` is the expected verdict on every GPU run shipped today. It is a cost
    finding on a correct run, so it says so in the log and returns — it never fails the job."""
    lines: list[str] = []

    class _Spy:
        def warning(self, fmt: str, *args: Any) -> None:
            lines.append(fmt % args)

    original = device_audit._log
    device_audit._log = _Spy()  # type: ignore[assignment]
    try:
        for on_device in (0, 200):
            audit_device_use(
                "r",
                "deep_learning",
                ["neuralprophet"],
                hardware="gpu",
                gpu_type="T4",
                read=_reader(
                    {"cells": 200, "cells_on_device": on_device, "max_peak_gpu_bytes": 75_000}
                ),
            )
    finally:
        device_audit._log = original
    assert len(lines) == 2
    assert "deep_learning" in lines[0] and "deep_learning" in lines[1]


def test_engaged_utilised_is_not_worth_a_warning() -> None:
    """The good outcome is silent. A log line on success trains people to ignore the log."""
    lines: list[str] = []

    class _Spy:
        def warning(self, fmt: str, *args: Any) -> None:
            lines.append(fmt % args)

    original = device_audit._log
    device_audit._log = _Spy()  # type: ignore[assignment]
    try:
        blob = audit_device_use(
            "r",
            "deep_learning",
            ["neuralprophet"],
            hardware="gpu",
            gpu_type="T4",
            read=_reader({"cells": 200, "cells_on_device": 200, "max_peak_gpu_bytes": _T4 // 2}),
        )
    finally:
        device_audit._log = original
    assert blob is not None and blob["verdict"] == ENGAGED_UTILISED
    assert lines == []


def test_every_verdict_has_two_words_for_a_chart_and_nothing_else_does() -> None:
    """The display vocabulary lives beside the verdicts so the two cannot drift apart.

    A word nobody defined is dropped rather than printed raw — the end of a progress bar is not
    where a reader should first meet a verdict string.
    """
    assert device_audit.verdict_label(MISSING_DEVICE) == "no gpu"
    assert device_audit.verdict_label(ENGAGED_IDLE) == "gpu idle"
    assert device_audit.verdict_label(ENGAGED_UTILISED) == "gpu used"
    assert device_audit.verdict_label(None) is None
    assert device_audit.verdict_label("SOMETHING_NEW") is None


# --- the evidence half: what a worker can see, and where the weights landed -----------------


def test_visible_device_answers_in_three_words_and_is_memoized() -> None:
    """``"cuda"`` / ``"cpu"`` / ``"unknown"``, and the last one is load-bearing: it means torch
    could not be imported, so *nobody asked* — which is a different fact from "there is no card"
    and must not be filed as one.

    Memoized because a device does not appear part-way through a process, and the probe runs once
    per cell in a loop that may run tens of thousands of times.
    """
    hardware._visible = None
    try:
        availability, name = hardware.visible_device()
        assert availability in ("cuda", "cpu", "unknown")
        assert name is None or isinstance(name, str)
        assert hardware.visible_device() is hardware.visible_device()
        hardware._visible = ("cuda", "Tesla T4")
        assert hardware.visible_device() == ("cuda", "Tesla T4")
    finally:
        hardware._visible = None


def test_a_broken_probe_answers_unknown_rather_than_raising(monkeypatch: Any) -> None:
    """A cell must not die because the evidence collector did. ``unknown`` is a legal answer."""
    import builtins

    real_import = builtins.__import__

    def boom(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise RuntimeError("this install is broken in an interesting way")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert hardware._probe_visible_device() == ("unknown", None)


def test_the_base_model_device_hook_is_none_and_never_raises() -> None:
    """``None`` means "this model has no device concept" — the honest answer for the twelve models
    with no tensor library under them. Returning ``"cpu"`` instead would be a claim about hardware
    that a statsmodels fit is in no position to make, and would show up as evidence."""

    class _Plain(BaseModel):
        name = "_plain_for_device_hook"
        runtime = "python"
        family = "statistical"

        def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
            return None

        def predict(self, horizon: int, X: Any = None, quantiles: Any = ()) -> pd.DataFrame:
            return pd.DataFrame()

    assert _Plain.device_used(_Plain.__new__(_Plain)) is None


def test_every_model_class_can_answer_the_device_question() -> None:
    """The hook is on the base, so a model that never heard of it still answers — the audit reads
    ``device_used()`` on whatever the factory built, and an AttributeError there would sink a
    perfectly good cell."""
    from scale_forecasting.models import get_model, list_models

    names = list_models()
    assert names, "no models registered — the assertion below would be vacuous"
    for name in names:
        assert callable(get_model(name).device_used)
