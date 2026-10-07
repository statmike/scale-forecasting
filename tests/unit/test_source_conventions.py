"""Repo-wide static checks on the source tree that ruff does not make.

Four checks. The first two exist because of a bug class that is genuinely invisible until it
runs on a launch point that is not a developer's laptop; the last two keep a deliberate
duplication and a naming convention honest (see their own docstrings at the bottom).

The first is a relative import at the **wrong level**. Ruff resolves undefined names and unused
imports, not whether ``from .models import get_model`` names a module that exists. When that
import sits at the top of a file, any test that imports the file catches it immediately. When
it sits inside a function body — which this codebase does deliberately, to keep heavy model,
Spark and BigQuery imports off the submit path — nothing catches it until that branch runs,
and the branch that runs it is often on a cluster, mid-job, an hour in.

Moving a module into a package is exactly when this breaks: every ``from .x import y`` in the
moved code silently starts meaning something else. The check is a few lines of ``ast`` and it
covers the whole tree, so it costs nothing to keep and pays for itself on the first move.

The second is a module whose name **shadows a stdlib module**. Inside a package this is normally
harmless — ``scale_forecasting.profiling.numbers`` and stdlib ``numbers`` coexist fine, because
only the package root is ever on ``sys.path``. It stops being harmless the moment something puts
an inner directory on ``sys.path``, and a Composer plugins delivery does exactly that: a live
Airflow run died on ``cannot import name 'Number' from 'numbers'`` because a *third-party*
library's ``import numbers`` resolved to our file. Nothing local reproduces it — the same code
imports cleanly from a repo checkout — so a static name check is the only cheap guard.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "scale_forecasting"


def _relative_imports() -> list[tuple[Path, int, str, int]]:
    """Every relative import in the source tree as ``(file, lineno, module, level)``."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                found.append((path, node.lineno, node.module or "", node.level))
    return found


def _resolves(path: Path, module: str, level: int) -> bool:
    """Does ``from <'.' * level><module> import ...`` inside ``path`` name something on disk?"""
    # `level` 1 means "this file's own package", 2 means its parent, and so on.
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    return (target / "__init__.py").is_file() or target.with_suffix(".py").is_file()


def test_every_relative_import_resolves_to_a_module_that_exists() -> None:
    """A wrong-level relative import inside a lazy function body must fail here, not in a job.

    The failure message names file, line and the exact import, because the fix is always to
    add or drop a dot and the only hard part is finding which one.
    """
    broken = [
        f"{path.relative_to(SRC)}:{lineno}: from {'.' * level}{module} import ..."
        for path, lineno, module, level in _relative_imports()
        if not _resolves(path, module, level)
    ]
    assert not broken, "relative imports that name no module:\n  " + "\n  ".join(broken)


def test_the_check_would_notice_a_wrong_level_import() -> None:
    """Guard the guard: a resolver that always returns True would pass the test above silently."""
    package_module = SRC / "profiling" / "cost.py"
    assert _resolves(package_module, "models", 2), "`..models` from inside profiling/ is real"
    assert not _resolves(package_module, "models", 1), "`.models` from inside profiling/ is not"


def test_the_source_tree_actually_has_relative_imports_to_check() -> None:
    """Guard the guard: an rglob that matched nothing would also report zero breakage."""
    assert len(_relative_imports()) > 50


def test_no_module_name_shadows_a_stdlib_module() -> None:
    # See the module docstring: a Composer plugins delivery puts inner directories on sys.path, so
    # `profiling/numbers.py` made an unrelated library's `import numbers` resolve to our file and
    # killed a live Ray submit with `cannot import name 'Number' from 'numbers'`. A repo checkout
    # never reproduces it, so nothing but this check stands between us and the next one.
    offenders = [
        f"{path.relative_to(SRC.parent)} shadows stdlib '{path.stem}'"
        for path in sorted(SRC.rglob("*.py"))
        if path.stem in sys.stdlib_module_names
    ]
    assert not offenders, "rename these — stdlib-shadowing module names:\n" + "\n".join(offenders)


def test_the_shadowing_check_knows_what_a_stdlib_name_looks_like() -> None:
    # Guard the guard: if `sys.stdlib_module_names` ever came back empty the check above would pass
    # vacuously and we would learn about it on a cluster instead of here.
    assert {"numbers", "json", "types"} <= sys.stdlib_module_names


# --- the deliberately-duplicated terminal-status set ---------------------------


def test_every_copy_of_the_terminal_status_set_agrees() -> None:
    """Four modules each keep their own copy of "a run/job has stopped changing". They must match.

    The duplication is on purpose and stays: `probes` is imported low and `sdk` high, so a shared
    constant would mean importing the SDK from the probe layer (and the registry layer from the
    probes package) purely to name one frozenset. Each site says so in its own comment.

    What duplication costs is drift, and drift here is not cosmetic — the four copies answer
    "should I keep waiting?", "should I probe the runtime?", "may I close this header?", and "did
    this family finish?". A status added to one and not the others makes `Forecaster.wait` hang on
    a run the registry considers settled, or lets close-runs write a verdict on a live job. This
    test is the cheap thing that makes the deliberate copy safe; it is not a hint to merge them.
    """
    from scale_forecasting.airflow_tasks import _TERMINAL_STATUSES as airflow_set
    from scale_forecasting.probes.vocabulary import _TERMINAL as probe_set
    from scale_forecasting.registry.ops import _TERMINAL_JOB_STATUSES as ops_set
    from scale_forecasting.sdk import _TERMINAL_STATUSES as sdk_set

    assert sdk_set == airflow_set == probe_set == ops_set
    # Named outright rather than compared to each other alone: four copies of the *wrong* set would
    # also be equal, and CANCELLED is the member most likely to be dropped by someone reasoning that
    # a stopped run "never finished".
    assert sdk_set == frozenset({"COMPLETED", "FAILED", "PARTIAL", "CANCELLED"})


# --- cross-module imports of underscore-prefixed names (a ratchet) ---------------------------

# Every `from .somewhere import _name` in the tree today, as `<file> <relative module> <name>`.
# The test below fails on any entry that is NOT here (so the list can only shrink) and on any
# entry here that no longer exists (so it is never stale). To clear a failure, promote the helper
# — drop the underscore, give it a docstring — or move it to a shared module; do not add here.
_PRIVATE_CROSS_IMPORTS = """
cluster_deps.py .batch_infra _ENV_VENV_ARCHIVE
cluster_submit.py .cluster_deps _VENV_JOB_PROPERTIES
cluster_submit.py .cluster_deps _resolve_cluster_deps
cluster_submit.py .cluster_deps _stage_cluster_init
cluster_submit.py .cluster_telemetry _job_client
cluster_submit.py .cluster_telemetry _stamp_cluster_telemetry
cluster_submit.py .dataproc_cluster _DEFAULT_WORKER_COUNT
cluster_submit.py .dataproc_cluster _cluster_client
cluster_submit.py .dataproc_cluster _create_cluster_across_candidates
cluster_submit.py .dataproc_cluster _delete_cluster
config.py .metrics.base_metric _REGISTRY
dataproc_cluster.py .cluster_deps _VENV_ARCHIVE_METADATA_KEY
dataproc_cluster.py .cluster_deps _VENV_DIR
dataproc_cluster.py .cluster_deps _VENV_DIR_METADATA_KEY
dataproc_cluster.py .cluster_deps _resolve_cluster_deps
dataproc_cluster.py .cluster_deps _stage_cluster_init
engines/automl_engine.py ..worker _backtest_outcome
engines/automl_engine.py ..worker _worker_id
engines/automl_engine.py .ray_engine _assert_source_supports_folds
engines/automl_engine.py .ray_engine _read_source_series
engines/bigquery_engine.py ..registry.rows _as_date
engines/bigquery_engine.py ..registry.rows _as_float
engines/bigquery_engine.py ..registry.write_api _META_SPEC
engines/bigquery_engine.py ..registry.write_api _OOF_SPEC
engines/bigquery_engine.py ..registry.write_api _append_via_write_api
engines/bigquery_engine.py ..registry.write_api _encode_rows
engines/bigquery_engine.py ..registry.write_api _proto_for
engines/ray_engine.py .spark_io _MODEL_COL
engines/ray_engine.py .spark_io _needed_columns
engines/ray_engine.py .spark_io _resolve_source_table
engines/ray_engine.py .spark_io _snapshot_millis
engines/ray_io.py .spark_io _MODEL_COL
engines/vertex_engine.py .ray_engine _assert_source_supports_folds
engines/vertex_engine.py .ray_engine _chunk_count
engines/vertex_engine.py .ray_engine _create_read_session
engines/vertex_engine.py .ray_engine _failed_chunk_status
engines/vertex_engine.py .ray_engine _limit_series
engines/vertex_engine.py .ray_engine _needed_columns
engines/vertex_engine.py .ray_engine _read_source_series
engines/vertex_engine.py .ray_engine _read_streams
engines/vertex_engine.py .ray_engine _resolve_fleetwide_hpo
engines/vertex_engine.py .ray_engine _stamp_executed_sizing
ensemble_run.py .registry.rows _as_float
ensemble_run.py .registry.write_api _META_SPEC
ensemble_run.py .registry.write_api _OOF_SPEC
ensemble_run.py .registry.write_api _PRED_SPEC
hpo.py .worker _model_context
job_launch.py .registry.lifecycle _STICKY_STATUSES
launch_plan.py .job_launch _entry_handle
launch_plan.py .job_launch _system_job_id
metrics/__init__.py .base_metric _REGISTRY
models/__init__.py .base_model _REGISTRY
playground.py .ensemble_run _apply_weights
probes/cancel.py .vocabulary _AWAITING_CAPACITY
probes/cancel.py .vocabulary _CANCELLED
probes/cancel.py .vocabulary _TERMINAL
probes/cancel.py .vocabulary _parse_ts
probes/reconcile.py ..review _assemble_progress
probes/reconcile.py .vocabulary _AWAITING_CAPACITY
probes/reconcile.py .vocabulary _EMITTED
probes/reconcile.py .vocabulary _REGISTRY_RUNNING
probes/reconcile.py .vocabulary _TERMINAL
probes/runtimes.py ..automl_submit _pipeline_client
probes/runtimes.py ..batch_telemetry _batch_client
probes/runtimes.py ..gke_submit _ephemeral_cluster_name
probes/runtimes.py ..ray_cluster _get_cluster
probes/runtimes.py ..ray_cluster _init_vertex
probes/runtimes.py ..ray_jobs _connect_job_client
probes/runtimes.py ..ray_jobs _is_job_absent_error
probes/runtimes.py ..vertex_submit _job_client
probes/runtimes.py .vocabulary _parse_ts
probes/settle.py .vocabulary _EMITTED
probes/settle.py .vocabulary _TERMINAL
profiling/cost.py .measure _MIN_WALL_S
quota.py .dataproc_cluster _DEFAULT_WORKER_COUNT
registry/cells.py .write_api _CELL_TABLES
registry/cells.py .write_api _META_SPEC
registry/cells.py .write_api _OOF_SPEC
registry/cells.py .write_api _PRED_SPEC
registry/cells.py .write_api _append_via_write_api
registry/cells.py .write_api _encode_rows
registry/cells.py .write_api _proto_for
registry/header.py .params _HEADER_PARAM_TYPES
registry/header.py .params _status_guard_param
registry/jobs.py .params _JOB_PARAM_TYPES
registry/jobs.py .params _job_param
registry/jobs.py .params _status_guard_param
resources/cluster.py .catalog _DEFAULT_TARGET_CELLS_PER_SLOT
resources/cluster.py .catalog _MIB
resources/cluster.py .catalog _SPARK_JVM_MB_PER_CORE
resources/fleet.py .catalog _DEFAULT_TARGET_CELLS_PER_SLOT
resources/fleet.py .catalog _MAX_SLOT_MEMORY_FRACTION
resources/fleet.py .catalog _MIN_GPU_FRACTION
resources/fleet.py .catalog _RESERVED_CORES_PER_UNIT
resources/fleet.py .catalog _SCHEDULABLE_MEMORY_FRACTION
resources/serverless.py .catalog _DEFAULT_TARGET_CELLS_PER_SLOT
resources/serverless.py .catalog _MIB
resources/serverless.py .catalog _MIN_GPU_FRACTION
resources/serverless.py .catalog _SPARK_JVM_MB_PER_CORE
resources/slot.py .catalog _DEFAULT_SLOT_CORES
resources/slot.py .catalog _MIN_GPU_FRACTION
resources/slot.py .catalog _NOMINAL_GPU_FRACTION
sdk.py .ensemble_run _override_ensemble
submit.py .batch_infra _DEFAULT_TTL_SECONDS
submit.py .batch_telemetry _batch_client
submit.py .batch_telemetry _stamp_job_telemetry
vertex_submit.py .engines.ray_io _ACCELERATOR_TYPES
vertex_submit.py .ray_cluster _resolve_regions
worker.py .profiling.measure _peak_gpu_bytes
worker.py .profiling.measure _rss_bytes
worker.py .resources.catalog _INTRAOP_ENV_VARS
"""


def _parse_private_cross_imports(block: str) -> set[tuple[str, str, str]]:
    """The allowlist block as ``{(file, relative_module, name)}``; blank lines ignored."""
    found: set[tuple[str, str, str]] = set()
    for line in block.splitlines():
        if line.strip():
            file, module, name = line.split()
            found.add((file, module, name))
    return found


def _private_cross_imports() -> set[tuple[str, str, str]]:
    """Every relative import of an underscore-prefixed *name* in the tree.

    Importing an underscore-prefixed *module* (``from . import _lag_forecaster``) is not counted:
    that is how a package keeps a shared implementation file out of its public surface, and the
    module's own leading underscore is the convention working as intended.
    """
    found: set[tuple[str, str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level):
                continue
            module = node.module or ""
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            target = base.joinpath(*module.split(".")) if module else base
            for alias in node.names:
                if not alias.name.startswith("_"):
                    continue
                if target.is_dir() and (target / f"{alias.name}.py").is_file():
                    continue  # a private module, not a private name
                found.add((str(path.relative_to(SRC)), "." * node.level + module, alias.name))
    return found


def test_cross_module_private_imports_only_ratchet_down() -> None:
    """A leading underscore means "not for use outside this module"; importing it elsewhere says
    otherwise. Every such import is a small lie about the module's surface, and the honest fix is
    cheap: drop the underscore (the helper is shared, so say so) or move it to a module whose job
    is to be shared. The list above is the debt as of this test's introduction; it may only shrink.
    """
    live = _private_cross_imports()
    allowed = _parse_private_cross_imports(_PRIVATE_CROSS_IMPORTS)

    added = sorted(live - allowed)
    assert not added, (
        "new cross-module imports of underscore-prefixed names:\n  "
        + "\n  ".join(" ".join(entry) for entry in added)
        + "\n\nPromote the helper (drop the underscore, add a docstring) or move it to a shared"
        " module. Do not add it to _PRIVATE_CROSS_IMPORTS; that list only ratchets down."
    )

    removed = sorted(allowed - live)
    assert not removed, (
        "allowlist entries that no longer exist — delete them from _PRIVATE_CROSS_IMPORTS so the"
        " list stays an exact picture of the remaining debt:\n  "
        + "\n  ".join(" ".join(entry) for entry in removed)
    )


def test_the_ratchet_sees_private_names_but_not_private_modules() -> None:
    """Guard the guard: the walker must count a known private-name import and skip the private
    module imports (``from . import _lag_forecaster``) that the lag-based models rely on."""
    live = _private_cross_imports()
    assert ("worker.py", ".resources.catalog", "_INTRAOP_ENV_VARS") in live
    assert not any(name == "_lag_forecaster" for _, _, name in live)
    assert all(len(entry) == 3 for entry in _parse_private_cross_imports(_PRIVATE_CROSS_IMPORTS))
