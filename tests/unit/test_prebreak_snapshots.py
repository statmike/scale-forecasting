"""Pre-break snapshots: what the digests are today, and what the numbers are today.

The refinement stage takes one deliberate `run_id` break — a batch of new `RunConfig` fields,
added together so the identity moves exactly once instead of six times. That break makes two
claims, and neither is checkable without artefacts captured *before* it lands:

* **Every id moved.** A snapshot written in the break commit pins post-break ids and proves
  nothing at all.
* **No output changed.** The fields arrive with defaults chosen to preserve today's behaviour, so
  the forecasts must be numerically identical on both sides of it.

So this module holds both snapshots and two switches. `_BREAK_LANDED` is `True` as of the break
commit: before it, the digest test asserted the ids still *matched*; now it asserts every one
*differs*. `_MOVED_AT_2_5` is `True` as of the commit that changed what a point forecast means:
before it, the golden panel test compared the numbers, which was the second claim above; now it
compares everything about the cells except the numbers, and says in one place why.

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

# The second declared movement, and the first one that reaches the forecast itself (plan item 2.5,
# 2026-09-08). `yhat` used to be the model's own prediction plus the median of its *in-sample*
# residuals — an accident of `residual_intervals` returning a 0.5 quantile that `_assemble_frame`
# then read as the point forecast. Measured on a backtest, that shift cost 5.7% of WAPE, and the
# band it came from achieved 0.601 coverage against a nominal 0.8. Item 2.5 replaced both: when a
# backtest ran, the correction and the band are estimated per horizon step from out-of-fold
# residuals, and the model's own output is kept beside them as `yhat_raw`.
#
# So every forecast value in the panel moves, and no config reproduces the old behaviour to let the
# pre-break comparison make its numeric claim any more. Rather than mute the test or widen
# `_SCORED_AT_2_3` until it means nothing, the claim is narrowed once and in writing: the pre-break
# panel now checks *structure* — same models, same cell status, same out-of-fold row counts, same
# metric set, same forecast length, and columns differing from the pre-break set by exactly the two
# names below. That is the half that still catches a plumbing regression, and it is the half 2.5
# did not touch.
#
# The numeric pin is not lost, it moved: `golden_panel.json` carries it, with no exemptions and at
# an exact tolerance for every model that can hold one (see `_UNSTABLE_FIT` for the five that
# cannot). `test_current_cell_output_matches_the_pinned_panel` is now the only test in this module
# that reads a forecast value. That test is the one to keep sharp.
_MOVED_AT_2_5 = True
_COLUMNS_ADDED_AT_2_5 = frozenset({"yhat_raw", "yhat_adjusted"})

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
    "configs/smokes/20_gpu_intent_cpu_family.json",
    # The GPU-vs-CPU A/B arms, written 2026-09-10. Nothing to hash before the break either. Their
    # own invariants — that the two arms stay identical apart from the accelerator — live in
    # `test_ab_preregistration.py`, and that is the stronger pin: it constrains the two configs
    # against *each other*, which is what the experiment depends on, rather than against a
    # recorded digest.
    "configs/neuralprophet_ab_gpu.json",
    "configs/neuralprophet_ab_cpu.json",
    # The two repair-ladder rungs, written 2026-09-11 — also nothing to hash before the break. Both
    # are deliberately ordinary two-family Spark runs, because everything they prove happens *after*
    # submission and a pinned digest would say nothing about it. They differ only in how much of a
    # family had landed when it died, which is the whole distinction the ladder turns on:
    # `repair_demo` loses a family mid-write and `--retry` refuses it (a partly-landed model cannot
    # be re-asked at v1's model grain without duplicating what survived), while `repair_retry_demo`
    # loses one during provisioning, lands nothing, and is repaired end to end.
    "configs/repair_demo.json",
    "configs/repair_retry_demo.json",
    # The Dataproc-cluster half of the same A/B, written 2026-09-13 — nothing to hash before the
    # break, and the argument given for the Ray arms above applies here with more force. What these
    # two files have to guarantee is that they describe one fleet and differ only in the card, and a
    # digest cannot say that about a *pair*; `test_ab_preregistration.py` compares them to each
    # other instead, down to the Spark properties each arm would submit.
    "configs/neuralprophet_ab_cluster_gpu.json",
    "configs/neuralprophet_ab_cluster_cpu.json",
    # The CPU twin of the per-family runtime demo, written 2026-09-15 once both A/B pairs had shown
    # the accelerator losing on wall clock as well as on cost. It exists so the recommended shape is
    # a file to copy rather than a paragraph to apply, and it is the GPU demo with one word changed
    # — so what guards it is `per_family_runtimes_demo.json` sitting beside it, not a pre-break
    # digest it never had.
    "configs/per_family_runtimes_cpu_demo.json",
    # The catalogue sweep, written 2026-09-23 — also nothing to hash before the break. It is the
    # first config to name every registered model at once, and it exists because the twenty smokes
    # before it named only seven of the eighteen between them, so eleven had never executed anywhere
    # but a unit test. A pre-break digest could not have covered it and would say nothing about what
    # it proves: what guards this file is the live row it owes in `docs/validation.md`, not an
    # identity recorded before the models were ever run.
    "configs/smokes/21_full_catalogue.json",
    # The backtest-semantics quartet, written 2026-09-23 — post-break like the rest, and grouped
    # here because they are one sweep rather than four runtime combinations. Each over-asks the fold
    # grid on purpose so that a `short_series` policy actually *branches*: naming a policy in a
    # config that the data comfortably satisfies exercises nothing, and a coverage tool reading
    # configs cannot tell the difference.
    "configs/smokes/22_backtest_sliding_overlap.json",
    "configs/smokes/23_backtest_frozen_shrink.json",
    "configs/smokes/24_backtest_stale.json",
    "configs/smokes/25_backtest_skip.json",
}

# The golden panel's fixture. A fixed seed lives inside `playground.sample_data`, so the only
# knobs are shape; 400 observations is long enough for the three-fold backtest every model runs
# here and short enough that the whole panel builds in about twelve seconds.
_HISTORY = 400
_HORIZON = 28

# Observations the fold geometry is computed against — the shipped seed's daily history, so the
# snapshot describes folds the shipped data can actually produce.
_FOLD_OBS = 1460

# Absolute vs relative tolerance for the numbers: an exact pin in everything but name, and what
# most of the panel is held to. Six of the fifteen models reproduce bit for bit on every machine
# the fleet survey below reached, and four more move by less than this band: `prophet` and
# `naive_moving_average` by a single ULP, `regression_lags` by 4.5e-13, `stl_bagging` by 4.3e-10.
# That last one has only about twice the room it needs, so it is the name to look at first if this
# band ever fails on a runner and passes at a desk.
_RTOL = 1e-9
_ATOL = 1e-9

# The other five, and the one place this module admits a number it cannot pin exactly.
#
# These five fit by iterative numerical optimisation inside `statsmodels`, and the optimiser does
# not halt at the same point on every machine: the BLAS kernel it dispatches to depends on the
# instruction set found at load time, the reduction order changes with the kernel, and the search
# stops an iteration sooner or later. The evidence that it is the machine and not the code: exactly
# these five move on CI and the other ten stay inside the exact band, while locally the whole panel
# is reproducible to the bit under any thread count. Note that `stl_bagging` also fits an ARIMA and
# stayed inside the exact band — so this set is what was observed, not a category anyone reasoned
# their way to. A sixth name belongs here only with the same kind of evidence behind it.
#
# **The bands are measured, and here is the measurement.** The first pair of numbers came off a
# single runner, which is a sample of one drawn from a fleet that is heterogeneous by CPU make and
# generation — so the band was right for that machine and too tight for the fleet, and CI went red
# on a commit that had changed nothing numeric. A 48-sample survey replaced the guess: 24 runners,
# two panel builds each, six CPU families (AMD EPYC 7763 / 9V45 / 9V74, Intel Xeon Platinum 8370C /
# 8573C, Xeon 6973P-C). Three findings, all of which shape the bands below.
#
#   * **The movement is discrete, not noisy.** Forecast values landed on exactly three results:
#     0, 2.7682023e-05 and 8.9757297e-05 relative, and nothing in between. Both builds on a machine
#     agreed every time, 24 out of 24. That is kernel dispatch picking one of a few code paths,
#     deterministic once picked — not an optimiser wandering.
#   * **The CPU model does not predict which path you get.** The EPYC 9V74 produced 0 on four
#     runners and 2.768e-05 on a fifth. So there is nothing to key a per-machine expectation on,
#     and the band has to cover the worst path rather than the likely one.
#   * **The five are not one population.** `autoets` and `holtwinters` move 9.0e-05, `ucm` 1.1e-06,
#     `sarimax` 1.8e-09, `theta` 2.7e-11 — seven orders of spread. One flat band is therefore sized
#     by the worst two and is very loose for the other three, and that is a deliberate trade: a
#     band fitted tightly to `theta` would go red the first time an unsampled runner dispatches
#     `theta` down a different path, and a permanently-red test gets muted.
#
# `_UNSTABLE_FIT_RTOL_FORECAST` is five times the worst movement observed, the same headroom the
# original band used. At 5e-4 it is still two orders below any change that reaches a model's
# parameters, which moves forecasts by percent rather than by parts per million.
#
# **`bias` carries its own band and every other metric keeps the tight one.** Across the same
# survey `bias` moved 7.0e-03 while no other metric moved more than 1.4e-04 — fifty times less.
# `bias` averages *signed* errors, so they cancel and what is left is small: the pinned value is
# -1.97 on a series whose level is ~270, and a 0.0137 absolute shift is a large fraction of itself
# while being invisible inside `mae`. One band wide enough for `bias` would have to be ~4e-2, and
# that is what the split buys, because of the next paragraph.
#
# What the metric band still catches: item 2.5, the last deliberate movement in this panel, moved
# WAPE by 5.7e-2. Against the band WAPE is actually held to that is 5.7x, and against the largest
# movement WAPE made anywhere in the survey (1.2e-04) it is nearly five hundred times. Absorbing
# `bias` into the shared band instead would have left that margin at 1.4x. A change with a cause is
# not subtle, and the point of splitting `bias` out is to keep it that way.
_UNSTABLE_FIT = frozenset({"autoets", "holtwinters", "sarimax", "theta", "ucm"})
_UNSTABLE_FIT_RTOL_FORECAST = 5e-4
_UNSTABLE_FIT_RTOL_METRIC = 1e-2
_UNSTABLE_FIT_RTOL_BIAS = 4e-2

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


def _relative(want: Any, got: Any) -> float:
    """How far `got` moved from `want`, as a fraction of `want`. A value gained or lost is inf."""
    if want is None or got is None:
        return math.inf
    return abs(float(got) - float(want)) / max(abs(float(want)), 1e-12)


def _close(a: Any, b: Any, rtol: float = _RTOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=_ATOL)


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
    """Pure arithmetic, and the surface the break's five new backtest fields touch directly.

    Post-break configs are dropped before the comparison for the same reason they are dropped from
    the pre-break id list: there is no pre-break geometry for them to have kept. This did not come
    up until 2026-09-10 because every post-break config until then had backtesting off, and
    `build_folds` only visits configs that backtest — so the exclusion was latent rather than
    absent. The filter is here and not in `build_folds`, which stays complete: the *current*
    snapshot should record the new configs' folds, and only the historical comparison should skip
    them.
    """
    current = {k: v for k, v in build_folds().items() if k not in _POST_BREAK}
    assert current == snapshot_panel["folds"], (
        "fold geometry moved. The new backtest fields are supposed to be inert at their "
        "defaults; if this is intentional it is a behaviour change and needs its own decision."
    )


def _cell_complaints(
    current: dict[str, Any],
    expected: dict[str, Any],
    newly_scored: frozenset[str] = frozenset(),
    columns_added: frozenset[str] = frozenset(),
    compare_values: bool = True,
) -> list[str]:
    """Every way the cells in `current` differ from `expected`, as sentences. Empty means same.

    Shared by both panel tests so that "unchanged" means one thing rather than two. Numbers are
    compared with `_close`, at the exact tolerance for most models and at the wider cross-machine
    band for the five named in `_UNSTABLE_FIT` — with `bias` wider still, being the one metric whose
    signed errors cancel. See the comment by those constants for why those five, why `bias` is on
    its own, and where every width was measured. A pin that fails on a difference between two CPUs
    is a pin someone switches off.

    `newly_scored` names metrics allowed to have gone from "not computed" to a number since the
    snapshot was taken. It is one-directional on purpose — the reverse move, a metric that used to
    have a value and now reads NaN, is a regression and still reported.

    `columns_added` names prediction columns a declared change introduced: the current column set
    must be the expected one plus exactly those, so an *extra* new column, or a lost old one, still
    fails. `compare_values=False` drops the forecast and metric *values* from the comparison while
    keeping every structural check — the state the pre-break panel is in once a change deliberately
    moves the numbers. Both are here rather than at the call site so the two panel tests keep
    disagreeing in exactly one place, which is the whole point of sharing this function.
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
        unstable = model in _UNSTABLE_FIT
        metric_rtol = _UNSTABLE_FIT_RTOL_METRIC if unstable else _RTOL
        forecast_rtol = _UNSTABLE_FIT_RTOL_FORECAST if unstable else _RTOL
        if got["status"] != want["status"]:
            out.append(f"{model}: cell status {want['status']!r} -> {got['status']!r}")
        if set(got["columns"]) != set(want["columns"]) | columns_added:
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
            # `bias` is the one metric whose signed errors cancel, so the same absolute wobble is
            # a fifty-times larger fraction of it than of anything else. It gets its own band so
            # the shared one can stay tight enough to still catch a deliberate change.
            rtol = _UNSTABLE_FIT_RTOL_BIAS if unstable and key == "bias" else metric_rtol
            if compare_values and not _close(g, w, rtol):
                out.append(f"{model}: metric {key} moved {w!r} -> {g!r}")
        if len(got["yhat"]) != len(want["yhat"]):
            out.append(f"{model}: forecast length {len(want['yhat'])} -> {len(got['yhat'])}")
            continue
        if not compare_values:
            continue
        drifted = [
            (i, w, g)
            for i, (w, g) in enumerate(zip(want["yhat"], got["yhat"], strict=True))
            if not _close(g, w, forecast_rtol)
        ]
        if drifted:
            # The worst movement, not just the first: it is the number anyone re-sizing a
            # tolerance needs, and reading it off one failure beats another round trip to find it.
            i, w, g = max(drifted, key=lambda d: _relative(d[1], d[2]))
            out.append(
                f"{model}: {len(drifted)} of {len(want['yhat'])} forecast values moved, worst at "
                f"step {i}: {w!r} -> {g!r} ({_relative(w, g):.2e} relative)"
            )
    return out


def test_golden_cell_output_is_unchanged(
    current_panel: dict[str, Any], snapshot_panel: dict[str, Any]
) -> None:
    """The claim the break rests on: the forecasts are the same on both sides of it.

    Slow by the standards of this suite (~12 s) because it runs the real cell for fifteen models
    rather than a stub. That is the cost of the claim being about output rather than about
    plumbing, and it is paid once per gate run.

    "The same" meant *numerically* the same until item 2.5 deliberately moved every forecast value
    (`_MOVED_AT_2_5`). It now means structurally the same, which is a real claim and a smaller one:
    the same fifteen models still run, still succeed, still produce the same number of out-of-fold
    rows over the same fold grid, still score the same metric set, and still return a horizon-length
    frame whose columns are the pre-break set plus exactly `_COLUMNS_ADDED_AT_2_5`. What it can no
    longer catch, `test_current_cell_output_matches_the_pinned_panel` catches instead.

    The four metrics in `_SCORED_AT_2_3` are exempt, in one direction, for the reason recorded
    there. The exemption is deliberately not "the interval metrics may differ" — it is "these four
    may go from unmeasured to measured", which is a thing that can only happen once.
    """
    complaints = _cell_complaints(
        current_panel["cells"],
        snapshot_panel["cells"],
        newly_scored=_SCORED_AT_2_3,
        columns_added=_COLUMNS_ADDED_AT_2_5 if _MOVED_AT_2_5 else frozenset(),
        compare_values=not _MOVED_AT_2_5,
    )
    assert not complaints, "output moved across the digest break:\n" + "\n".join(complaints)


def test_current_cell_output_matches_the_pinned_panel(
    current_panel: dict[str, Any], snapshot_panel_now: dict[str, Any]
) -> None:
    """The panel's counterpart to `test_current_digests_match_the_pinned_snapshot`.

    The pre-break comparison above answers a question that is settled — it carries an exemption now
    and will carry more as more deliberate changes land, and each one narrows what it can still
    catch. This one pins today's numbers, so the next unintended movement fails in the commit that
    causes it rather than being discovered later by someone regenerating a snapshot for an
    unrelated reason.

    It carries no *exemption* — nothing here is allowed to change direction or appear from nowhere
    — but it does carry a tolerance, and for the five models in `_UNSTABLE_FIT` that tolerance is
    wide enough to absorb the difference between two CPUs. That is a limit of what a numeric pin
    can promise across machines, written down where it applies rather than left for whoever next
    sees the gate go red on a runner and green at their desk.

    When a change here *is* intended, `--write` and say so in the commit body. That is the whole
    ceremony, and it is worth having: it makes moving a number a thing somebody decided.
    """
    assert build_folds() == snapshot_panel_now["folds"], (
        "fold geometry moved from the pinned panel. If this is intentional it is a behaviour "
        "change and needs its own decision; regenerate with --write once it has one."
    )
    complaints = _cell_complaints(current_panel["cells"], snapshot_panel_now["cells"])
    assert not complaints, "output moved from the pinned panel:\n" + "\n".join(complaints)


def test_the_cross_machine_band_names_only_models_the_panel_still_runs() -> None:
    """A widened tolerance for a model that no longer exists is a dead line nobody can see is dead.

    The band in `_UNSTABLE_FIT` is the one place this module gives up exactness, so it is the one
    place worth checking stays honest: a renamed or retired model must take its exemption with it
    rather than leave a name behind that reads like a live claim about today's suite.
    """
    stale = sorted(_UNSTABLE_FIT - set(_panel_models()))
    assert not stale, (
        f"{stale} carry the cross-machine tolerance band but are not in the numeric panel any "
        "more. Remove them from `_UNSTABLE_FIT` — every model left in it must be one whose "
        "movement across machines was actually observed."
    )


def test_the_cross_machine_band_is_wide_for_five_models_and_for_no_others() -> None:
    """The band is the module's one loose thread, so it is worth testing and not just writing.

    Two claims in one: the same 1e-6 movement that a named model is allowed to make is a failure
    for a model that is not named, and the width really is per-model rather than global.
    """
    base = {
        "status": "ok",
        "metrics": {"wape": 0.1},
        "columns": ["yhat"],
        "yhat": [100.0],
        "n_oof_rows": 3,
    }
    nudged = {**base, "metrics": {"wape": 0.1 * (1 + 1e-6)}, "yhat": [100.0 * (1 + 1e-6)]}

    assert not _cell_complaints({"autoets": nudged}, {"autoets": base})
    assert _cell_complaints({"croston": nudged}, {"croston": base})


def test_bias_gets_its_own_band_and_takes_nothing_else_with_it() -> None:
    """The split is the reason the shared metric band could stay tight, so test that it held.

    A movement between the two widths must be tolerated in `bias` and reported in every other
    metric of the same cell — otherwise the split has quietly become a general loosening, which is
    exactly what it was chosen over.
    """
    base = {
        "status": "ok",
        "metrics": {"bias": -1.97, "wape": 0.0577},
        "columns": ["yhat"],
        "yhat": [100.0],
        "n_oof_rows": 3,
    }
    between = (_UNSTABLE_FIT_RTOL_METRIC + _UNSTABLE_FIT_RTOL_BIAS) / 2
    assert _UNSTABLE_FIT_RTOL_METRIC < between < _UNSTABLE_FIT_RTOL_BIAS

    moved_bias = {**base, "metrics": {"bias": -1.97 * (1 + between), "wape": 0.0577}}
    moved_wape = {**base, "metrics": {"bias": -1.97, "wape": 0.0577 * (1 + between)}}

    assert not _cell_complaints({"autoets": moved_bias}, {"autoets": base})
    assert _cell_complaints({"autoets": moved_wape}, {"autoets": base})
    # And the wide `bias` band is the unstable five's alone — an exactly-pinned model keeps the
    # exact tolerance for every metric it has, `bias` included.
    assert _cell_complaints({"croston": moved_bias}, {"croston": base})


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
