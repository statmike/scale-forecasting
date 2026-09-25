"""Tripwire: every documented config value has evidence, or a written reason it has none.

The configuration reference documents the config surface, and a documented option is a promise. A
reader who sets `backtest.scheme: "expanding_stale"` believes somebody has run it. Nothing else in
this repository can check that belief: `mkdocs --strict` cannot, the offline gate cannot — a
`Literal` member is valid the moment it is declared — and `docs/validation.md` records *runs*, not
the options those runs happened to exercise. So the gap between "declared" and "proven" is invisible
at exactly the moment it matters most, which is when somebody sits down to document the surface.

This file closes that by measuring the gap on every push and refusing to let it grow silently.

**What it joins.** Three sources:

1. **The declared surface** — every enumerable value reachable from `RunConfig`, read off the
   Pydantic models rather than a hand-kept list: each `Literal`'s members, each `bool`'s two states,
   and the two registry-backed catalogues (`models`, `backtest.decision_metric`) whose members no
   annotation can see.
2. **The authored surface** — what the shipped configs produce. Every file under `configs/` is
   loaded *through* `RunConfig`, so the **effective** value is counted, not the authored one. That
   distinction is the whole point: no config sets `backtest.scheme`, but every backtest run still
   ran `expanding`, and a reader of the raw JSON would call the best-tested value on the surface
   untested.
3. **The proven surface** — `docs/validation.md`'s per-config status. A value counts as proven only
   if a config producing it has a `CURRENT` row, because `CURRENT` is the ledger's own word for
   "this result still describes today's architecture".

Anything left over is listed in `UNPROVEN` below, and **the assertion is set equality**. A new
`Literal` member, or a config deleted, fails this file until somebody writes down which of the four
things it is. That is the entire mechanism, and it is the same one
`test_registry_column_parity.py` uses for reserved columns — a set that must be maintained by hand
is exactly as good as the diff it shows up in, which is what makes it better than a count.

**The four kinds of entry, and why the grammar is machine-checked.** A map of free prose would rot
into a list of claims nobody re-reads. So three of the four forms are verified:

* ``"tests/unit/test_x.py::test_name"`` — an offline test drives this value. Verified: the file
  exists, the test exists, and **that test's own source span** names both the field and the value.
  The span matters. A pointer verified against the whole file would pass on any module that happens
  to mention the word somewhere.
* ``"tests/unit/test_x.py::test_name (over CONSTANT)"`` — an offline test drives this value by
  iterating a catalogue rather than by naming it. Verified: the test's span references `CONSTANT`,
  and `CONSTANT`, imported from the product, contains the value. This form exists because a loop
  over `METRIC_NAMES` is the *strongest* kind of coverage and the least literal-matchable, and a
  grammar that could not express it would push ten honest entries into prose.
* ``"reason: …"`` — not work. The value is unreachable (a default validation resolves away before
  any object holds it) or is exercised somewhere this join structurally cannot see (a knob with an
  environment-variable twin). Unverifiable by construction; visible in a diff.
* ``"gap: …"`` — **real unproven surface.** Nothing drives it, offline or live. These are the honest
  ones, and their count is the campaign's actual metric.

**What this file does not claim.** An offline pointer proves a value was *exercised*, not that it
was exercised *well* — `features.fourier=true` reaching `build_features` proves the pipeline
tolerates the flag, not that the Fourier terms are right. And a pointer to a test that *rejects* a
value is not coverage of that value at all; those are `gap:` entries here, deliberately, because a
`pytest.raises` proving a combination is refused says nothing about the combination working.

**Two things the join deliberately does not measure**, both inherited from the tool this grew out
of. Scalar fields — `series_limit`, `horizon`, `n_folds`, the capacity backoff knobs — are left out
entirely: their surface is a number line, and "`horizon=28` is proven" is not a statement about
coverage. And `compute.families` is a map keyed by family name, so its values collapse:
`compute.families.hardware=gpu` means *some* family ran on a GPU, not that every family did.
"""

from __future__ import annotations

import ast
import json
import types
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

# Borrowed from the tripwire that already owns them: the ledger's table layout and the one config
# that is not a run config are facts about those two artefacts, written down once. A second copy
# here would be a second thing to drift.
from test_validation_ledger import _NOT_A_RUN_CONFIG, _table_rows  # noqa: PLC2701

from scale_forecasting.config import RunConfig
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.models import list_models

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO_ROOT / "configs"
_LEDGER = _REPO_ROOT / "docs" / "validation.md"
_TESTS = _REPO_ROOT / "tests"

# Catalogues a `(over CONSTANT)` pointer may name. Kept to an explicit map rather than resolved by
# `getattr` on an arbitrary module, so a pointer cannot quietly start meaning something else.
_CATALOGUES: dict[str, tuple[str, ...]] = {
    "METRIC_NAMES": tuple(METRIC_NAMES),
    "MODEL_NAMES": tuple(sorted(list_models())),
}


# Every declared value with no `CURRENT` ledger row, and what stands behind it. See the module
# docstring for the four forms. Keep it sorted; the failure message diffs against it.
UNPROVEN: dict[str, str] = {
    # --- the decision metric -----------------------------------------------------------------
    # One test constructs `BacktestConfig(decision_metric=name)` for every name in the panel, which
    # is the config value itself being exercised rather than a proxy for it. No run has ever
    # *ranked* on any of these but `wape`, which is the part that stays unproven.
    "backtest.decision_metric=bias": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=coverage": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=interval_score": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=interval_width": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=mape": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=mase": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=mase_seasonal": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=mse": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=pinball": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    "backtest.decision_metric=rmsse": (
        "tests/unit/test_metrics.py::test_config_accepts_every_panel_metric_and_no_other "
        "(over METRIC_NAMES)"
    ),
    # --- backtest ----------------------------------------------------------------------------
    "backtest.short_series=error": (
        "tests/unit/test_backtest.py::"
        "test_the_panel_gate_refuses_only_under_error_and_only_when_a_series_is_short"
    ),
    # --- compute -------------------------------------------------------------------------------
    "compute.capacity.enabled=false": (
        "tests/unit/test_capacity.py::test_disabling_capacity_retry_beats_an_authored_pass_count"
    ),
    "compute.capacity.preflight=false": (
        "tests/unit/test_dataproc_cluster.py::"
        "test_preflight_off_asks_for_the_planned_fleet_and_never_reads_the_quota"
    ),
    # --- the ensemble node's compute, which is declared and not wired ----------------------------
    # `EnsembleCompute.runtime`, `.spark_mode` and `.spark_cluster_name` are read by nothing. The
    # ensemble node is hard-wired to the driver: `dag.build_dag_nodes` stamps it `runtime=bigquery`
    # with `spark_mode=None`, and `job_launch.run_ensemble` blends in driver pandas and says so. The
    # two live fields on that model — `mode` and `microbatch_interval_s` — are covered elsewhere.
    #
    # These entries are `gap:` rather than `reason:` on purpose. A declared value nothing reads is
    # not exempt work; it is worse than untested, because the reference promises a choice the
    # product does not offer. It closes by wiring the fields up or by deleting them, and deleting
    # them re-keys every run_id (`_canonical_config` dumps defaults too), so it is an owner call.
    "compute.ensemble.runtime=ray": (
        "gap: nothing reads `cfg.compute.ensemble.runtime`. A test asserts the field accepts the "
        "value and rejects Spark-only siblings beside it, which is validation, not behaviour — the "
        "ensemble runs on the driver either way."
    ),
    "compute.ensemble.spark_mode=cluster": (
        "gap: nothing reads `cfg.compute.ensemble.spark_mode`. The only test naming this value "
        "asserts it is *rejected* alongside `runtime=ray`; no accepting path exists to cover."
    ),
    "compute.ensemble.spark_mode=serverless": (
        "gap: nothing reads `cfg.compute.ensemble.spark_mode`. Setting it is documented, accepted, "
        "and inert."
    ),
    "compute.families.runtime=spark": (
        "tests/unit/test_airflow_emit.py::test_two_cluster_spark_families_emit_a_shared_dataproc_bracket"
    ),
    "compute.families.spark_mode=serverless": (
        "tests/unit/test_config_families.py::"
        "test_spark_mode_serverless_written_out_resolves_the_same_but_is_a_different_run"
    ),
    "compute.machine_family=c2": (
        "tests/unit/test_dataproc_cluster.py::test_machine_family_selects_the_cpu_worker_and_master_family"
    ),
    "compute.machine_family=e2": (
        "tests/unit/test_dataproc_cluster.py::test_machine_family_selects_the_cpu_worker_and_master_family"
    ),
    "compute.machine_family=n1": (
        "tests/unit/test_dataproc_cluster.py::"
        "test_machine_family_is_ignored_on_gpu_because_the_accelerator_dictates_the_machine"
    ),
    "compute.machine_family=n2": (
        "tests/unit/test_dataproc_cluster.py::test_machine_family_selects_the_cpu_worker_and_master_family"
    ),
    "compute.machine_family=n2d": (
        "tests/unit/test_dataproc_cluster.py::test_machine_family_selects_the_cpu_worker_and_master_family"
    ),
    "compute.profile.measure=controlled": (
        "tests/unit/test_resources.py::"
        "test_a_controlled_measurement_run_can_unpin_them_and_says_so_on_the_plan"
    ),
    "compute.profile.measure=off": (
        "tests/unit/test_worker.py::test_measurement_off_leaves_every_axis_null_rather_than_zero"
    ),
    "compute.profile.mode=always": "tests/unit/test_config.py::test_profile_is_part_of_the_run_id",
    "compute.profile.mode=off": (
        "tests/unit/test_config.py::test_profile_defaults_are_off_the_shelf_and_conservative"
    ),
    "compute.ray_autoscale=false": (
        "tests/unit/test_ray_io.py::test_plan_autoscale_false_restores_fixed_plan"
    ),
    "compute.ray_read_mode=ray_data": (
        "tests/unit/test_ray_engine.py::test_read_source_series_ray_data_mode_dispatches_and_limits"
    ),
    "compute.spark_deps=container": (
        "reason: this knob has an environment-variable twin, `SF_SERVERLESS_DEPS`, which lives on "
        "the infra object rather than the config precisely so it cannot move the `run_id`. Every "
        "container-image deployment therefore exercises it without any config naming it, and this "
        "join — which reads configs — is structurally blind to that. Read the ledger's prose, not "
        "this entry, before concluding the path is untested."
    ),
    # --- features ------------------------------------------------------------------------------
    "features.fourier=true": "tests/unit/test_features.py::test_build_features_fourier_terms",
    "features.level_shift=true": (
        "tests/unit/test_features.py::test_build_features_level_shift_column_is_opt_in"
    ),
    "features.transform=boxcox": (
        "tests/unit/test_features.py::test_boxcox_roundtrips_with_fitted_lambda"
    ),
    # --- hpo -------------------------------------------------------------------------------------
    "hpo.enabled=true": (
        "tests/unit/test_worker.py::test_per_series_hpo_tunes_and_records_best_params"
    ),
    "hpo.engine=optuna": (
        "gap: the only member of its Literal and the default, so every HPO test runs it without "
        "naming it — and no shipped config enables HPO at all, which is why the whole block reads "
        "untouched. It closes the day an HPO config is written, not before."
    ),
    "hpo.granularity=fleetwide": (
        "tests/unit/test_config.py::test_hpo_defaults_are_off_and_fleetwide"
    ),
    "hpo.granularity=per_series": (
        "tests/unit/test_worker.py::test_per_series_hpo_tunes_and_records_best_params"
    ),
    # --- output ------------------------------------------------------------------------------
    "output.point_forecast=(unset)": (
        "reason: unreachable rather than unproven. The field is declared `… | None = None`, but "
        "validation resolves the `None` into a concrete arm before any object holds it, so no "
        "loaded config can leave it unset. There is no work behind this line and never will be."
    ),
    "output.point_forecast=mean": (
        "tests/unit/test_config.py::test_auto_is_a_different_run_from_every_fixed_arm"
    ),
    "output.point_forecast=raw": (
        "tests/unit/test_config.py::test_raw_against_a_squared_error_metric_does_not_warn"
    ),
}


# --- the join ------------------------------------------------------------------------------


def _token(value: object) -> str:
    """One spelling for a value, used on both sides of the join.

    The declared side reads `True` off a type and the authored side reads `True` off an instance,
    and `str()` renders both as ``"True"`` — but a `Literal["auto"]` renders as ``"auto"``, so the
    two sides have to agree on booleans explicitly or every `bool` field reports as unproven with
    its two states sitting unmatched in the declared column. Lowercasing is also how JSON spells
    them, which is what the configs being measured are written in.
    """
    if value is None:
        return "(unset)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _enumerable(annotation: object) -> tuple[str, ...] | None:
    """The values this annotation can take, if it is enumerable; else None.

    Unwraps optionals and lists on the way down, because `Runtime | None` and `list[Strategy]` are
    both enumerable surfaces even though neither is a bare `Literal`.
    """
    origin = typing.get_origin(annotation)
    if annotation is bool:
        return ("true", "false")
    if origin is typing.Literal:
        return tuple(_token(a) for a in typing.get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        values: list[str] = []
        optional = False
        for arg in typing.get_args(annotation):
            if arg is type(None):
                optional = True
                continue
            nested = _enumerable(arg)
            if nested:
                values.extend(nested)
        # `(unset)` is only a surface *value* when the field has other values to be unset instead
        # of. `series_limit: int | None` is not enumerable, and "(unset) is proven" says nothing
        # about it — including it would bulk the map out with two dozen optional ints whose real
        # surface is a number line.
        if not values:
            return None
        if optional:
            values.append("(unset)")
        return tuple(dict.fromkeys(values))
    if origin in (list, set, tuple):
        args = typing.get_args(annotation)
        return _enumerable(args[0]) if args else None
    return None


def _model_types(annotation: object) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        found.append(annotation)
    for arg in typing.get_args(annotation):
        found.extend(_model_types(arg))
    return found


def declared_surface() -> dict[str, tuple[str, ...]]:
    """``{field path: possible values}`` for every enumerable field reachable from `RunConfig`.

    Read off the models rather than maintained by hand, so a new `Literal` member joins the map the
    day it is declared — which is the point of the file. The two registry-backed catalogues are
    grafted on afterwards: `models` and `decision_metric` validate against runtime registries rather
    than against a type, so the annotation alone cannot see their members.
    """
    surface: dict[str, tuple[str, ...]] = {}

    def walk(model: type[BaseModel], prefix: str, seen: frozenset[type]) -> None:
        if model in seen:  # pragma: no cover - defensive against a recursive config model
            return
        seen = seen | {model}
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            values = _enumerable(field.annotation)
            if values:
                surface[path] = values
            for candidate in _model_types(field.annotation):
                walk(candidate, f"{path}.", seen)

    walk(RunConfig, "", frozenset())
    surface["models"] = _CATALOGUES["MODEL_NAMES"]
    surface["backtest.decision_metric"] = _CATALOGUES["METRIC_NAMES"]
    return surface


def authored_surface(surface: dict[str, tuple[str, ...]]) -> dict[str, set[str]]:
    """``{"field=value": {configs producing it}}``, using each config's **effective** value."""
    produced: dict[str, set[str]] = {}

    def record(path: str, value: object, config: str) -> None:
        if path not in surface:
            return
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            produced.setdefault(f"{path}={_token(item)}", set()).add(config)

    def walk(obj: BaseModel, prefix: str, config: str) -> None:
        fields = type(obj).model_fields
        # A default inside a switched-off block was never exercised, whatever the object says it
        # holds. Every config leaves `hpo.enabled` false, so `hpo.granularity` is `fleetwide` in all
        # of them — and counting that as coverage would report the HPO surface as proven when no run
        # has ever tuned anything. When the gate is off, only the gate itself is recorded.
        gated_off = "enabled" in fields and not getattr(obj, "enabled", True)
        for name in fields:
            if gated_off and name != "enabled":
                continue
            value = getattr(obj, name, None)
            path = f"{prefix}{name}"
            if isinstance(value, BaseModel):
                walk(value, f"{path}.", config)
                continue
            if isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, BaseModel):
                        walk(nested, f"{path}.", config)
                continue
            record(path, value, config)

    for path in sorted(_CONFIGS.rglob("*.json")):
        if path.name in _NOT_A_RUN_CONFIG:
            continue
        cfg = RunConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
        walk(cfg, "", path.name)
    return produced


def config_status() -> dict[str, str]:
    """``{config filename: ledger status}`` for every config the ledger has a row for."""
    ledger = _LEDGER.read_text(encoding="utf-8")
    status: dict[str, str] = {}
    for row in _table_rows(ledger, "# "):
        status[row[1].strip("`")] = row[3]
    for row in _table_rows(ledger, "Config "):
        status[row[0].strip("`")] = row[2]
    return status


def unproven_values() -> set[str]:
    """Every declared ``field=value`` with no config carrying it into a `CURRENT` ledger row."""
    surface = declared_surface()
    produced = authored_surface(surface)
    status = config_status()
    unproven = set()
    for path, values in surface.items():
        for value in values:
            key = f"{path}={value}"
            configs = produced.get(key, set())
            if not any(status.get(c) == "CURRENT" for c in configs):
                unproven.add(key)
    return unproven


# --- the pointer grammar -------------------------------------------------------------------


def _test_span(pointer: str) -> str:
    """The source of the single test function a pointer names.

    The *span*, not the file. A pointer checked against a whole module passes on any file that
    happens to mention the word somewhere, which is the failure mode that makes a map of pointers
    worth less than no map at all.

    Decorators are part of the span. `ast` puts a decorated function's `lineno` on the `def`, so
    `get_source_segment` would cut `@pytest.mark.parametrize("measure", ["off", …])` off the top —
    and a parametrize list is one of the two ways a test honestly drives several values at once.
    Losing it would push true entries into prose for a reason that is an artefact of the parser.
    """
    rel, _, name = pointer.partition("::")
    path = _REPO_ROOT / rel
    assert path.is_file(), f"{pointer}: no such file"
    lines = path.read_text(encoding="utf-8").splitlines()
    for node in ast.walk(ast.parse("\n".join(lines))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            return "\n".join(lines[start - 1 : node.end_lineno])
    pytest.fail(f"{pointer}: no test function by that name")


def _pointers() -> dict[str, tuple[str, str | None]]:
    """``{value: (pointer, catalogue or None)}`` for the entries that claim an offline exercise."""
    out: dict[str, tuple[str, str | None]] = {}
    for key, entry in UNPROVEN.items():
        if entry.startswith(("reason:", "gap:")):
            continue
        pointer, sep, tail = entry.partition(" (over ")
        out[key] = (pointer, tail.rstrip(")") if sep else None)
    return out


# --- the tripwires -------------------------------------------------------------------------


def test_the_unproven_set_is_exactly_what_is_written_down() -> None:
    """The assertion the file exists for. Equality, not containment, in both directions.

    A value that becomes unproven — a new `Literal` member, a config deleted, a ledger row going
    `STALE` — fails here until somebody classifies it. A value that becomes proven fails here too,
    which is the half a one-directional check would miss: the entry would sit in the map forever,
    still claiming the gap, long after the run that closed it.
    """
    actual = unproven_values()
    declared = set(UNPROVEN)
    assert actual == declared, (
        f"newly unproven, add to UNPROVEN: {sorted(actual - declared)}\n"
        f"now proven, delete from UNPROVEN: {sorted(declared - actual)}"
    )


@pytest.mark.parametrize("value", sorted(_pointers()))
def test_every_offline_pointer_names_a_test_that_drives_the_value(value: str) -> None:
    """A pointer is a claim, and an unverified claim is what this map exists to replace."""
    pointer, catalogue = _pointers()[value]
    span = _test_span(pointer)
    path, _, token = value.rpartition("=")
    leaf = path.rsplit(".", 1)[-1]
    assert leaf in span, f"{value}: {pointer} never mentions `{leaf}`"

    if catalogue is not None:
        assert catalogue in _CATALOGUES, f"{value}: unknown catalogue `{catalogue}`"
        assert catalogue in span, f"{value}: {pointer} does not iterate `{catalogue}`"
        assert token in _CATALOGUES[catalogue], f"{value}: `{catalogue}` does not contain `{token}`"
        return

    # A bool is spelled `True`/`False` in a test and `true`/`false` in the map, because the map is
    # keyed the way JSON spells it. Everything else is compared as written.
    spelled = {"true": "True", "false": "False"}.get(token, token)
    assert spelled in span, f"{value}: {pointer} never names `{spelled}`"


def test_no_entry_is_prose_pretending_to_be_a_pointer() -> None:
    """The grammar is the whole defence. An entry that is neither a resolvable pointer nor one of
    the two declared prose forms would be a claim nothing checks, which is the state this file was
    written to end."""
    for key, entry in sorted(UNPROVEN.items()):
        if entry.startswith(("reason:", "gap:")):
            # Long enough to be an argument rather than a shrug. The shortest honest one in the map
            # runs to two sentences; a one-word "gap: todo" is the failure this catches.
            assert len(entry) > 60, f"{key}: {entry!r} does not say enough to be a reason"
            continue
        assert "::" in entry, f"{key}: {entry!r} is neither a pointer nor a reason"


def test_every_config_the_join_skips_is_one_the_ledger_also_excludes() -> None:
    """A join that silently drops a file it could not parse reports coverage it did not measure.

    The tool this grew out of printed `skipped compute_fallback.json: ValidationError` to stderr and
    carried on — correct for that file, which is a zone/region failover map rather than a run
    config, and wrong as a policy: a config that genuinely broke would print the same line and be
    counted the same way. Here the exclusion list is shared with the ledger tripwire, and anything
    not on it must load.
    """
    for path in sorted(_CONFIGS.rglob("*.json")):
        if path.name in _NOT_A_RUN_CONFIG:
            continue
        RunConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_the_gaps_are_the_number_worth_watching() -> None:
    """Not a threshold — a statement of where the campaign is, in a place that cannot go stale.

    Splitting the map by kind is the whole reason it has four forms. `reason:` entries are not work
    and never will be; pointers are work already done offline; `gap:` is what is genuinely unproven,
    and it is the only one of the three that should be shrinking.
    """
    gaps = {k for k, v in UNPROVEN.items() if v.startswith("gap:")}
    reasons = {k for k, v in UNPROVEN.items() if v.startswith("reason:")}
    offline = set(_pointers())
    assert gaps | reasons | offline == set(UNPROVEN)
    # The three do not overlap, so the split is a partition and the counts add up.
    assert len(gaps) + len(reasons) + len(offline) == len(UNPROVEN)
