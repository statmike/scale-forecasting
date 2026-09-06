"""The fields that are accepted but not yet honoured, and the proof that they are both.

Six things landed on the config surface in one commit — five `backtest` fields, the
`expanding_frozen` scheme, and `model_params` — ahead of the code that reads them. That is
deliberate: `run_id` is a digest of the whole config, so every added field moves every identity
ever recorded, and landing them together costs one break instead of six.

**`model_params` is now honoured** and has left the inert list: it is read at all three places a
model's params resolve (`worker._resolve_params` and both halves of `hpo.tune_model`), validated
against the registry at plan time (`dag.check_model_params`), and consumed by
`NeuralProphetModel`. Its digest test stays. Five backtest fields and the `expanding_frozen`
scheme remain inert.

It also creates a gap between what the schema says and what the code does, and a gap nobody is
watching becomes a lie. So this module pins both halves:

* **Inert** — setting any of them changes no fold and no forecast, and no module outside `config.py`
  so much as mentions them. The structural half is the one that matters: it fails on the day
  somebody wires a field up, which is exactly when the "not yet honoured" wording in the config
  reference and the `BacktestConfig` docstring stops being true and has to be deleted.
* **In the digest** — setting any of them moves the `run_id`. If one did not, it would be missing
  from the dumped payload, and the break would have to be paid for a second time to add it.

When you implement one of these: delete its entry from `_INERT_BACKTEST_FIELDS` (or
`_UNREAD_IN_SOURCE`) in the same commit, and update `docs/configuration_reference.md`. The digest
half stays — it is true for the life of the field.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting.backtest import make_folds
from scale_forecasting.config import RunConfig
from scale_forecasting.registry.ids import make_run_id

_SRC = Path(__file__).resolve().parents[2] / "src" / "scale_forecasting"

# field -> a value that differs from the default and would visibly change fold geometry if read.
_INERT_BACKTEST_FIELDS: dict[str, Any] = {
    "short_series": "error",
    "min_folds": 3,
    "min_train_floor": 90,
    "gap": 14,
    "window": 200,
}

# Names no module outside `config.py` may mention while the field is unread. `scheme` is absent
# because it *is* read — only its new `expanding_frozen` value is unhonoured, covered separately.
# `model_params` left this tuple when it was wired up; see the module docstring.
_UNREAD_IN_SOURCE = tuple(_INERT_BACKTEST_FIELDS)

_BASE: dict[str, Any] = {
    "run_name": "inert_fields",
    "data": {"source_table": "series"},
    "models": ["theta"],
    "backtest": {"enabled": True, "n_folds": 3, "horizon": 28, "step": 28, "min_train": 180},
}

# Long enough for the base fold grid with room to spare, so a field that *were* honoured would
# reshape the folds rather than raise.
_OBS = 1460


def _cfg(**overrides: Any) -> RunConfig:
    backtest = {**_BASE["backtest"], **overrides.pop("backtest", {})}
    return RunConfig(**{**_BASE, **overrides, "backtest": backtest})


def _folds(cfg: RunConfig) -> list[tuple[int, int, int, int, int]]:
    return [
        (f.fold_id, f.train_start, f.train_end, f.val_start, f.val_end)
        for f in make_folds(_OBS, cfg)
    ]


# --- accepted -----------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), sorted(_INERT_BACKTEST_FIELDS.items()))
def test_the_new_backtest_fields_are_accepted(field: str, value: Any) -> None:
    assert getattr(_cfg(backtest={field: value}).backtest, field) == value


def test_expanding_frozen_is_accepted_as_a_scheme() -> None:
    assert _cfg(backtest={"scheme": "expanding_frozen"}).backtest.scheme == "expanding_frozen"


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


# --- not yet honoured ---------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), sorted(_INERT_BACKTEST_FIELDS.items()))
def test_the_new_backtest_fields_change_no_fold(field: str, value: Any) -> None:
    assert _folds(_cfg(backtest={field: value})) == _folds(_cfg())


def test_expanding_frozen_lays_out_folds_exactly_as_expanding() -> None:
    """It names an intent — freeze the fitted model and re-condition — not a different grid."""
    assert _folds(_cfg(backtest={"scheme": "expanding_frozen"})) == _folds(
        _cfg(backtest={"scheme": "expanding"})
    )


def _reads_of_unhonoured_fields(tree: ast.AST) -> list[str]:
    """Attribute reads of an unhonoured field, as `<something backtest-ish>.<field>`.

    An AST walk rather than a text search, and the reason is that `gap` and `window` are ordinary
    English: a substring scan matches "validation window" and "no gaps" in thirty files and reports
    the whole package as an offender. Matching the attribute *access* also excludes docstrings and
    comments for free, which is right — prose describing a field is not code reading it.

    `backtest`/`bt` is required on the left, because `window` is a legitimate attribute name
    elsewhere (a rolling window, a buffer window).
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in _INERT_BACKTEST_FIELDS:
            owner = node.value
            name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
            if name in {"backtest", "bt"}:
                found.add(node.attr)
    return sorted(found)


def test_no_module_outside_config_reads_the_unhonoured_fields() -> None:
    """The half that catches wiring. A behavioural test cannot: an unread field looks inert whether
    it is unwired or wired-but-broken."""
    offenders = {
        str(path.relative_to(_SRC)): reads
        for path in sorted(_SRC.rglob("*.py"))
        if path.name != "config.py"
        and (reads := _reads_of_unhonoured_fields(ast.parse(path.read_text())))
    }
    assert not offenders, (
        f"a field documented as 'accepted, not yet honoured' is now read in {offenders}. "
        f"If it was implemented, remove it from this module's lists and correct the wording in "
        f"BacktestConfig's docstring and docs/configuration_reference.md, which both still say "
        f"nothing reads it."
    )


# --- in the digest ------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), sorted(_INERT_BACKTEST_FIELDS.items()))
def test_every_new_backtest_field_reaches_the_run_id(field: str, value: Any) -> None:
    """Inert in behaviour, not in identity — that is what makes landing them early worth it."""
    assert make_run_id(_cfg(backtest={field: value})) != make_run_id(_cfg())


def test_model_params_reaches_the_run_id() -> None:
    assert make_run_id(_cfg(model_params={"theta": {"deseasonalize": False}})) != make_run_id(
        _cfg()
    )


def test_the_new_fields_serialize_to_json_the_digest_can_reproduce() -> None:
    """`_canonical_config` dumps then `json.dumps`; a value surviving neither breaks identity."""
    cfg = _cfg(
        backtest={"gap": 7, "window": 120, "short_series": "skip"},
        model_params={"neuralprophet": {"n_lags": 28, "order": [1, 1, 1]}},
    )
    payload = cfg.model_dump(mode="json")
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
