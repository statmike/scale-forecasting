"""Pre-break snapshots: what the digests are today, and what the numbers are today.

The refinement stage takes one deliberate `run_id` break — a batch of new `RunConfig` fields,
added together so the identity moves exactly once instead of six times. That break makes two
claims, and neither is checkable without artefacts captured *before* it lands:

* **Every id moved.** A snapshot written in the break commit pins post-break ids and proves
  nothing at all.
* **No output changed.** The fields arrive with defaults chosen to preserve today's behaviour, so
  the forecasts must be numerically identical on both sides of it.

So this module holds both snapshots and one switch. `_BREAK_LANDED` is `True` as of the break
commit: before it, the digest test asserted the ids still *matched*; now it asserts every one
*differs*. The golden panel test does not have a switch, because it must stay green throughout —
that is the claim.

The switch matters more than it looks. The obvious alternative is to leave a test that is known to
fail from the break onward, and the reason not to is that a permanently-red test gets muted, and a
muted test is worse than no test. A one-line flip is something a reviewer can see and argue with.

**`run_ids.json` is the one that keeps working.** "Every id differs from the pre-break set" is
satisfied forever once the break lands, so on its own it stops detecting anything the moment it
goes green. The post-break snapshot is a plain equality pin against *today's* digests, and it is
what actually fails when an unplanned field is added next. Six ledger rows recorded a `run_id` their
config no longer produced because nothing was watching this; something is watching now.

**Regenerating.** ``uv run python tests/unit/test_prebreak_snapshots.py --write``. Both snapshots
are built by the same functions the tests read them with, so there is no second code path that can
drift. Do not regenerate to make a failure go away — the failures are the deliverable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting.backtest import make_folds
from scale_forecasting.config import RunConfig
from scale_forecasting.models import get_model, list_models
from scale_forecasting.registry.ids import make_run_id

# Flipped by the commit that landed the digest break (plan P3), and by no other commit.
_BREAK_LANDED = True

_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS = Path(__file__).parent / "snapshots"
_RUN_IDS = _SNAPSHOTS / "run_ids_prebreak.json"
_RUN_IDS_NOW = _SNAPSHOTS / "run_ids.json"
_PANEL = _SNAPSHOTS / "golden_panel_prebreak.json"

# Not a run config: a zone/region failover map with its own schema and no `run_name`.
_NON_RUNCONFIG = {"compute_fallback.json"}

# The golden panel's fixture. A fixed seed lives inside `playground.sample_data`, so the only
# knobs are shape; 400 observations is long enough for the three-fold backtest every model runs
# here and short enough that the whole panel builds in about twelve seconds.
_HISTORY = 400
_HORIZON = 28

# Observations the fold geometry is computed against — the shipped seed's daily history, so the
# snapshot describes folds the shipped data can actually produce.
_FOLD_OBS = 1460

# Absolute vs relative tolerance for the numbers. Loose enough to survive a BLAS or libm
# difference between machines, tight enough that any change with a *cause* moves further than
# this. A behavioural regression does not land at 1e-9.
_RTOL = 1e-9
_ATOL = 1e-9

# `neuralprophet` is excluded from the numeric panel for two independent reasons, and both would
# have to stop being true to include it: it costs ~51 s for this one series (85% of the panel's
# whole runtime, for one of sixteen models), and it is an optional extra that a plain `uv sync`
# does not install, so the snapshot would be unbuildable on a normal checkout. Its fold geometry
# is still covered — that half is pure arithmetic and does not depend on the model.
_TOO_SLOW = {"neuralprophet"}


def _shipped_configs() -> list[tuple[str, RunConfig]]:
    """Every config we ship, keyed by repo-relative path, in a stable order."""
    paths = sorted(
        [*(_ROOT / "configs").glob("*.json"), *(_ROOT / "configs" / "smokes").glob("*.json")],
        key=lambda p: p.relative_to(_ROOT).as_posix(),
    )
    return [
        (p.relative_to(_ROOT).as_posix(), RunConfig(**json.loads(p.read_text())))
        for p in paths
        if p.name not in _NON_RUNCONFIG
    ]


def build_run_ids() -> dict[str, str]:
    return {name: make_run_id(cfg) for name, cfg in _shipped_configs()}


def _panel_models() -> list[str]:
    """The Python models the numeric panel runs, sorted."""
    return sorted(
        n for n in list_models() if get_model(n).runtime == "python" and n not in _TOO_SLOW
    )


def build_golden_panel() -> dict[str, Any]:
    """Fold geometry for every backtesting config, plus real cell output for every fast model.

    The two halves guard different things. **Folds** are pure arithmetic over the backtest block,
    and they are what the new `min_folds` / `gap` / `window` / `min_train_floor` fields could most
    easily move by accident — an instant check on the exact surface the break touches. **Cells**
    run `worker.run_cell`, the same function every engine calls, so they catch a change that
    reaches a model's parameters rather than the fold grid.
    """
    from scale_forecasting import playground

    data = playground.sample_data(n_series=1, history=_HISTORY)
    cells: dict[str, Any] = {}
    for model in _panel_models():
        run = playground.run_model(model, data=data, horizon=_HORIZON, backtest=True)
        result = run.result
        preds = result.predictions
        cells[model] = {
            "status": result.status,
            # Sorted so the snapshot does not encode a dict iteration order, and NaN written as
            # null so the file stays valid JSON — `_close` treats null and NaN as the same value.
            "metrics": {k: _jsonable(result.metrics[k]) for k in sorted(result.metrics)},
            "columns": sorted(preds.columns),
            "yhat": [_jsonable(v) for v in preds["yhat"].tolist()],
            "n_oof_rows": 0 if result.oof is None else int(len(result.oof)),
        }
    return {"folds": build_folds(), "cells": cells}


def build_folds() -> dict[str, list[list[int]]]:
    """Fold grid each backtesting config produces at `_FOLD_OBS` observations.

    Split out from `build_golden_panel` because it is instant while the cells cost ~12 s, and the
    fold test has no use for the cells.
    """
    return {
        name: [
            [f.fold_id, f.train_start, f.train_end, f.val_start, f.val_end]
            for f in make_folds(_FOLD_OBS, cfg)
        ]
        for name, cfg in _shipped_configs()
        if cfg.backtest.enabled
    }


def _jsonable(v: Any) -> Any:
    """NaN/inf are not JSON, and NaN is a real metric value here (a full-fit run scores nothing)."""
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else f


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=_RTOL, abs_tol=_ATOL)


@pytest.fixture(scope="module")
def snapshot_run_ids() -> dict[str, str]:
    return json.loads(_RUN_IDS.read_text())


@pytest.fixture(scope="module")
def snapshot_run_ids_now() -> dict[str, str]:
    return json.loads(_RUN_IDS_NOW.read_text())


@pytest.fixture(scope="module")
def snapshot_panel() -> dict[str, Any]:
    return json.loads(_PANEL.read_text())


# --- the digests -----------------------------------------------------------------------


def test_every_shipped_config_is_in_the_digest_snapshot(snapshot_run_ids: dict[str, str]) -> None:
    """A config added after the snapshot has no pre-break id, so the break is unprovable for it."""
    assert sorted(snapshot_run_ids) == sorted(dict(_shipped_configs())), (
        "the shipped configs and the pre-break digest snapshot disagree. If a config was added, "
        "regenerate — but note that a config created after the break has no pre-break identity "
        "and the digest claim below is vacuous for it."
    )


def test_the_digest_break_has_or_has_not_happened(snapshot_run_ids: dict[str, str]) -> None:
    """Before the break every id matches the snapshot; after it, every single one must differ.

    "Every single one" is the strict reading on purpose. The break adds fields to `RunConfig` with
    defaults, and `_canonical_config` hashes the **dumped values**, so a new field with a default
    lands in every config's payload whether or not that config mentions it. A config whose id did
    *not* move therefore means the field is missing from the dump, not that the config was
    unaffected — which is exactly the bug this catches.
    """
    current = build_run_ids()
    if not _BREAK_LANDED:
        assert current == snapshot_run_ids, (
            "run_ids moved without the digest break being declared. Either an unplanned config "
            "field was added — in which case revert it and fold it into the break batch — or the "
            "break just landed, in which case set _BREAK_LANDED = True in this file."
        )
        return

    unmoved = [name for name, rid in current.items() if rid == snapshot_run_ids.get(name)]
    assert not unmoved, (
        f"the digest break landed but {len(unmoved)} config(s) kept their pre-break run_id: "
        f"{sorted(unmoved)}. A new field with a default must appear in every config's dumped "
        f"payload; one that does not is excluded from the digest somewhere it should not be."
    )


def test_current_digests_match_the_pinned_snapshot(snapshot_run_ids_now: dict[str, str]) -> None:
    """The tripwire that survives the break, and the one that would have caught 2026-09-01.

    The test above stops discriminating the moment the break lands — "differs from the pre-break
    set" is true forever afterwards. This one is a plain equality pin against today's digests, so
    the next unplanned field fails in the commit that adds it rather than being noticed months
    later by someone regenerating a snapshot for an unrelated reason.
    """
    assert build_run_ids() == snapshot_run_ids_now, (
        "a run_id moved. If a config field was added, the config surface is frozen — fold it into "
        "the next planned break instead. If the break is the intent, regenerate with --write and "
        "re-grade the affected rows in docs/validation.md, whose recorded ids are now pointers "
        "into the registry that no config reproduces."
    )


# --- the numbers -----------------------------------------------------------------------


def test_fold_geometry_is_unchanged(snapshot_panel: dict[str, Any]) -> None:
    """Pure arithmetic, and the surface the break's five new backtest fields touch directly."""
    assert build_folds() == snapshot_panel["folds"], (
        "fold geometry moved. The new backtest fields are supposed to be inert at their "
        "defaults; if this is intentional it is a behaviour change and needs its own decision."
    )


def test_golden_cell_output_is_unchanged(snapshot_panel: dict[str, Any]) -> None:
    """The claim the break rests on: the forecasts are numerically the same on both sides of it.

    Slow by the standards of this suite (~12 s) because it runs the real cell for fifteen models
    rather than a stub. That is the cost of the claim being about output rather than about
    plumbing, and it is paid once per gate run.
    """
    current = build_golden_panel()["cells"]
    expected = snapshot_panel["cells"]
    assert sorted(current) == sorted(expected), (
        "the model registry and the golden panel disagree. A new model needs a regenerated "
        "snapshot; a removed one needs a deliberate edit."
    )

    for model in sorted(current):
        got, want = current[model], expected[model]
        assert got["status"] == want["status"], f"{model}: cell status changed"
        assert got["columns"] == want["columns"], f"{model}: prediction columns changed"
        assert got["n_oof_rows"] == want["n_oof_rows"], f"{model}: out-of-fold row count changed"
        assert sorted(got["metrics"]) == sorted(want["metrics"]), f"{model}: metric set changed"
        for key in sorted(want["metrics"]):
            assert _close(got["metrics"][key], want["metrics"][key]), (
                f"{model}: metric {key} moved {want['metrics'][key]!r} -> {got['metrics'][key]!r}"
            )
        assert len(got["yhat"]) == len(want["yhat"]), f"{model}: forecast length changed"
        drifted = [
            (i, w, g)
            for i, (w, g) in enumerate(zip(want["yhat"], got["yhat"], strict=True))
            if not _close(g, w)
        ]
        assert not drifted, (
            f"{model}: {len(drifted)} of {len(want['yhat'])} forecast values moved, first at "
            f"step {drifted[0][0]}: {drifted[0][1]!r} -> {drifted[0][2]!r}"
        )


def _write() -> None:
    """Regenerate the post-break pin and the golden panel. Never rewrites the pre-break digests.

    `run_ids_prebreak.json` is a historical record — it is what the ids were before the break, and
    rewriting it would erase the only evidence that the break moved anything. Restore it from git
    if it is ever lost.
    """
    _RUN_IDS_NOW.write_text(json.dumps(build_run_ids(), indent=2, sort_keys=True) + "\n")
    _PANEL.write_text(json.dumps(build_golden_panel(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {_RUN_IDS_NOW.relative_to(_ROOT)} and {_PANEL.relative_to(_ROOT)}")


if __name__ == "__main__":  # pragma: no cover - regeneration entrypoint
    _write()
