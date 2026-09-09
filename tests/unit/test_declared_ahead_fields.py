"""The fields that landed on the config surface ahead of their code, and the claim that outlives it.

Six things arrived in one commit — five `backtest` fields, the `expanding_frozen` scheme, and
`model_params` — before anything read them. That was deliberate: `run_id` is a digest of the whole
config, so every added field moves every identity ever recorded, and landing them together cost one
break instead of six.

**All six are now honoured.** `model_params` is read at all three places a model's params resolve
(`worker._resolve_params` and both halves of `hpo.tune_model`), validated against the registry at
plan time (`dag.check_model_params`), and consumed by `NeuralProphetModel`. `expanding_frozen` and
`expanding_stale` are carried by `backtest._walk_folds`. `gap` and `window` are wired through
`backtest.make_folds`, `backtest.training_width` and the native fold SQL. `short_series`,
`min_folds` and `min_train_floor` are `backtest.resolve_geometry`'s whole subject. There is no
longer an inert half to pin, and the structural test that used to catch the day one of them was
wired up has done its job and gone.

What remains is the half that is true for the life of a field, not just while it is unread:
**setting any of them moves the `run_id`.** If one did not, it would be missing from the dumped
payload, and the identity break would have to be paid a second time to add it. Leaving the digest
is not something implementing a field is allowed to do quietly, so this module keeps watching for
it long after the behaviour tests moved to `test_backtest.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id

# field -> a value that differs from the default. `min_train_floor` is paired with the one policy
# that reads it, because `BacktestConfig` rejects it anywhere else rather than accept a knob it
# would then ignore — so "does it reach the digest" can only be asked of the pair.
_DECLARED_AHEAD: dict[str, dict[str, Any]] = {
    "short_series": {"short_series": "skip"},
    "min_folds": {"min_folds": 3},
    "min_train_floor": {"short_series": "shrink_train", "min_train_floor": 90},
    "gap": {"gap": 14},
    "window": {"window": 200},
}

_BASE: dict[str, Any] = {
    "run_name": "declared_ahead",
    "data": {"source_table": "series"},
    "models": ["theta"],
    "backtest": {"enabled": True, "n_folds": 3, "horizon": 28, "step": 28, "min_train": 180},
}


def _cfg(**overrides: Any) -> RunConfig:
    backtest = {**_BASE["backtest"], **overrides.pop("backtest", {})}
    return RunConfig(**{**_BASE, **overrides, "backtest": backtest})


# --- accepted -----------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "override"), sorted(_DECLARED_AHEAD.items()))
def test_the_declared_ahead_backtest_fields_are_accepted(field: str, override: dict) -> None:
    bt = _cfg(backtest=override).backtest
    assert all(getattr(bt, k) == v for k, v in override.items())


@pytest.mark.parametrize("scheme", ["expanding_frozen", "expanding_stale"])
def test_the_frozen_schemes_are_accepted(scheme: str) -> None:
    assert _cfg(backtest={"scheme": scheme}).backtest.scheme == scheme


def test_model_params_accepts_scalars_and_flat_lists() -> None:
    params = {"neuralprophet": {"n_lags": 28, "lr": 0.01, "freeze": True, "order": [1, 1, 1]}}
    assert _cfg(model_params=params).model_params == params


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_model_params_rejects_values_that_are_not_json(bad: float) -> None:
    """`json.dumps` emits bare NaN/Infinity, so the digest stops being reproducible elsewhere."""
    with pytest.raises(ValueError, match="not JSON"):
        _cfg(model_params={"m": {"k": bad}})
    with pytest.raises(ValueError, match="not JSON"):
        _cfg(model_params={"m": {"k": [1.0, bad]}})


# --- the one geometry invariant that was never a "not yet" ----------------------------------


@pytest.mark.parametrize("scheme", ["expanding_frozen", "expanding_stale"])
def test_a_frozen_scheme_lays_out_folds_exactly_as_expanding(scheme: str) -> None:
    """A standing invariant, and the reason the comparison is fair.

    The frozen schemes change how the model is *carried* between origins, never where a fold
    starts or what it is scored on. If the grids diverged, a frozen leaderboard and a refit
    leaderboard would be scored on different windows and the staleness gap would be measuring the
    geometry. `make_folds` writes this as a membership test rather than ``== "expanding"``, which
    is what keeps a newly-added scheme from silently inheriting sliding's fixed-width window.
    """
    from scale_forecasting.backtest import make_folds

    def folds(cfg: RunConfig) -> list[tuple[int, int, int, int, int]]:
        return [
            (f.fold_id, f.train_start, f.train_end, f.val_start, f.val_end)
            for f in make_folds(1460, cfg)
        ]

    assert folds(_cfg(backtest={"scheme": scheme})) == folds(_cfg(backtest={"scheme": "expanding"}))


# --- in the digest ------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "override"), sorted(_DECLARED_AHEAD.items()))
def test_every_declared_ahead_field_reaches_the_run_id(field: str, override: dict) -> None:
    """The claim that outlives implementation — that is what made landing them early worth it."""
    assert make_run_id(_cfg(backtest=override)) != make_run_id(_cfg())


def test_model_params_reaches_the_run_id() -> None:
    assert make_run_id(_cfg(model_params={"theta": {"deseasonalize": False}})) != make_run_id(
        _cfg()
    )


def test_the_declared_ahead_fields_serialize_to_json_the_digest_can_reproduce() -> None:
    """`_canonical_config` dumps then `json.dumps`; a value surviving neither breaks identity."""
    cfg = _cfg(
        backtest={"gap": 7, "window": 120, "short_series": "skip", "min_folds": 2},
        model_params={"neuralprophet": {"n_lags": 28, "order": [1, 1, 1]}},
    )
    payload = cfg.model_dump(mode="json")
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
