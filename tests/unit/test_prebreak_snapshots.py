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

**Each pre-break snapshot has a same-shape "today" pin beside it**, and the pair does two different
jobs. The pre-break file answers "did the thing we promised not to change, change?" — a question
that stops discriminating once its claim is settled. The today file is a plain pin on current
output, and it is what fails in the commit that moves something nobody meant to move. Six ledger
rows recorded a `run_id` their config no longer produced because nothing was watching the digests;
something is watching both now.

**A deliberate change to the numbers gets named, not absorbed.** `_SCORED_AT_2_3` is the only
exemption in the pre-break comparison, and it is written as narrowly as the change it describes:
four named metrics, allowed to move in one direction only (from "not computed" to a number). A
regression back to NaN still fails, and a metric that already had a value still fails. Widening
this set, rather than adding a new one beside it, is how an exemption quietly becomes a hole.

**Regenerating.** ``uv run python tests/unit/test_prebreak_snapshots.py --write``. Every snapshot is
built by the same functions the tests read it with, so there is no second code path that can drift.
`--write` never touches the two pre-break files: they are a historical record, and rewriting one
erases the only evidence the break moved anything. Do not regenerate to make a failure go away —
the failures are the deliverable.
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
_PANEL_NOW = _SNAPSHOTS / "golden_panel.json"

# The one declared movement in the numbers since the pre-break panel was captured (plan item 2.3).
# Every model in the suite returns `yhat_lower` / `yhat_upper` on every fold, and the backtest loop
# used to drop them, so these four interval metrics were NaN on every Python cell in every run while
# the BigQuery-native path scored them. Scoring the folds on bounds that were already there turned
# fifteen nulls into fifteen numbers, per metric, and moved nothing else — not a forecast value, not
# a fold, not one of the other eleven metrics.
_SCORED_AT_2_3 = frozenset({"coverage", "pinball", "interval_score", "interval_width"})

# Not a run config: a zone/region failover map with its own schema and no `run_name`.
_NON_RUNCONFIG = {"compute_fallback.json"}

# Configs written *after* the break. They have no pre-break identity — there was nothing to hash —
# so they are named here rather than given an invented row in `run_ids_prebreak.json`, which is a
# historical record and must stay one. Naming them keeps the vacancy visible: the "every id moved"
# claim below simply does not cover these files, and a reader can see exactly which ones.
_POST_BREAK = {
    "configs/smokes/17_gpu_absent_serverless.json",
    "configs/smokes/18_gpu_absent_cluster.json",
    "configs/smokes/19_gpu_absent_ray.json",
}

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


@pytest.fixture(scope="module")
def snapshot_panel_now() -> dict[str, Any]:
    return json.loads(_PANEL_NOW.read_text())


@pytest.fixture(scope="module")
def current_panel() -> dict[str, Any]:
    """Built once for the whole module — it runs fifteen real models and costs ~12 s."""
    return build_golden_panel()


# --- the digests -----------------------------------------------------------------------


def test_every_shipped_config_is_in_the_digest_snapshot(snapshot_run_ids: dict[str, str]) -> None:
    """A config added after the snapshot has no pre-break id, so the break is unprovable for it."""
    shipped = [name for name in dict(_shipped_configs()) if name not in _POST_BREAK]
    assert sorted(snapshot_run_ids) == sorted(shipped), (
        "the shipped configs and the pre-break digest snapshot disagree. A config written after "
        "the break belongs in `_POST_BREAK`, which says so out loud; one written before it should "
        "already be in the snapshot, and its absence means the snapshot was regenerated at the "
        "wrong time."
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


def _cell_complaints(
    current: dict[str, Any],
    expected: dict[str, Any],
    newly_scored: frozenset[str] = frozenset(),
) -> list[str]:
    """Every way the cells in `current` differ from `expected`, as sentences. Empty means same.

    Shared by both panel tests so that "unchanged" means one thing rather than two. Numbers are
    compared with `_close` on both sides: a BLAS or libm difference between two machines moves the
    last bit or two of a float, and a pin that fails on that is a pin someone switches off.

    `newly_scored` names metrics allowed to have gone from "not computed" to a number since the
    snapshot was taken. It is one-directional on purpose — the reverse move, a metric that used to
    have a value and now reads NaN, is a regression and still reported.
    """
    if sorted(current) != sorted(expected):
        return [
            "the model registry and the panel disagree: "
            f"{sorted(set(current) - set(expected))} added, "
            f"{sorted(set(expected) - set(current))} missing. A new model needs a regenerated "
            "snapshot; a removed one needs a deliberate edit."
        ]

    out: list[str] = []
    for model in sorted(current):
        got, want = current[model], expected[model]
        if got["status"] != want["status"]:
            out.append(f"{model}: cell status {want['status']!r} -> {got['status']!r}")
        if got["columns"] != want["columns"]:
            out.append(f"{model}: prediction columns {want['columns']} -> {got['columns']}")
        if got["n_oof_rows"] != want["n_oof_rows"]:
            out.append(f"{model}: out-of-fold rows {want['n_oof_rows']} -> {got['n_oof_rows']}")
        if sorted(got["metrics"]) != sorted(want["metrics"]):
            out.append(f"{model}: metric set changed")
            continue
        for key in sorted(want["metrics"]):
            w, g = want["metrics"][key], got["metrics"][key]
            if key in newly_scored and w is None and g is not None:
                continue  # the movement 2.3 declared, in the only direction it declared it
            if not _close(g, w):
                out.append(f"{model}: metric {key} moved {w!r} -> {g!r}")
        if len(got["yhat"]) != len(want["yhat"]):
            out.append(f"{model}: forecast length {len(want['yhat'])} -> {len(got['yhat'])}")
            continue
        drifted = [
            (i, w, g)
            for i, (w, g) in enumerate(zip(want["yhat"], got["yhat"], strict=True))
            if not _close(g, w)
        ]
        if drifted:
            out.append(
                f"{model}: {len(drifted)} of {len(want['yhat'])} forecast values moved, first at "
                f"step {drifted[0][0]}: {drifted[0][1]!r} -> {drifted[0][2]!r}"
            )
    return out


def test_golden_cell_output_is_unchanged(
    current_panel: dict[str, Any], snapshot_panel: dict[str, Any]
) -> None:
    """The claim the break rests on: the forecasts are numerically the same on both sides of it.

    Slow by the standards of this suite (~12 s) because it runs the real cell for fifteen models
    rather than a stub. That is the cost of the claim being about output rather than about
    plumbing, and it is paid once per gate run.

    The four metrics in `_SCORED_AT_2_3` are exempt, in one direction, for the reason recorded
    there. The exemption is deliberately not "the interval metrics may differ" — it is "these four
    may go from unmeasured to measured", which is a thing that can only happen once.
    """
    complaints = _cell_complaints(
        current_panel["cells"], snapshot_panel["cells"], newly_scored=_SCORED_AT_2_3
    )
    assert not complaints, "output moved across the digest break:\n" + "\n".join(complaints)


def test_current_cell_output_matches_the_pinned_panel(
    current_panel: dict[str, Any], snapshot_panel_now: dict[str, Any]
) -> None:
    """The panel's counterpart to `test_current_digests_match_the_pinned_snapshot`.

    The pre-break comparison above answers a question that is settled — it carries an exemption now
    and will carry more as more deliberate changes land, and each one narrows what it can still
    catch. This one has no exemptions and never will: it pins today's numbers exactly, so the next
    unintended movement fails in the commit that causes it rather than being discovered later by
    someone regenerating a snapshot for an unrelated reason.

    When a change here *is* intended, `--write` and say so in the commit body. That is the whole
    ceremony, and it is worth having: it makes moving a number a thing somebody decided.
    """
    assert build_folds() == snapshot_panel_now["folds"], (
        "fold geometry moved from the pinned panel. If this is intentional it is a behaviour "
        "change and needs its own decision; regenerate with --write once it has one."
    )
    complaints = _cell_complaints(current_panel["cells"], snapshot_panel_now["cells"])
    assert not complaints, "output moved from the pinned panel:\n" + "\n".join(complaints)


def _write() -> None:
    """Regenerate the two today-pins. Never rewrites either pre-break file.

    `run_ids_prebreak.json` and `golden_panel_prebreak.json` are a historical record — they are what
    the ids and the numbers were before the break, and rewriting one would erase the only evidence
    that the break moved anything. Restore them from git if they are ever lost.
    """
    _RUN_IDS_NOW.write_text(json.dumps(build_run_ids(), indent=2, sort_keys=True) + "\n")
    _PANEL_NOW.write_text(json.dumps(build_golden_panel(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {_RUN_IDS_NOW.relative_to(_ROOT)} and {_PANEL_NOW.relative_to(_ROOT)}")


if __name__ == "__main__":  # pragma: no cover - regeneration entrypoint
    _write()
