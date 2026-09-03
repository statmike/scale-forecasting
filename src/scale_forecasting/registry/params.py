"""Registry column types — how a Python value is bound into a parameterized query.

One table per registry table (``run_registry``, ``run_jobs``) mapping column name to its
BigQuery type, plus the binder that turns a name/value pair into a query parameter. The two
binders live together because they are the same idea applied twice; they live apart from
`registry.ddl` because that renders the schema while this consumes it.

The two SQL fragments here — the status guard and the ``job_telemetry`` merge — are for the same
reason: both tables' writers need them, so neither writer can own them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# run_registry columns that may be set by write_header / update_header, with their BQ types.
_HEADER_PARAM_TYPES: dict[str, str] = {
    "run_id": "STRING",
    "created_at": "TIMESTAMP",
    "snapshot_millis": "INT64",
    "user_id": "STRING",
    "git_sha": "STRING",
    "python_runtime": "STRING",
    "bq_models": "ARRAY<STRING>",
    "backtest_on": "BOOL",
    "decision_metric": "STRING",
    "ensemble_strategies": "ARRAY<STRING>",
    "raw_config": "JSON",
    "status": "STRING",
    "n_series": "INT64",
    "n_models": "INT64",
    "runtime_seconds": "FLOAT64",
    "job_telemetry": "JSON",
}


def _header_param(name: str, value: Any) -> Any:
    """Build a scalar or array query parameter for a run_registry column."""
    from google.cloud import bigquery

    bq_type = _HEADER_PARAM_TYPES[name]
    if bq_type.startswith("ARRAY<"):
        element_type = bq_type[len("ARRAY<") : -1]
        return bigquery.ArrayQueryParameter(name, element_type, list(value or []))
    return bigquery.ScalarQueryParameter(name, bq_type, value)


# run_jobs columns that may be set by write_job / update_job, with their BQ types.
_JOB_PARAM_TYPES: dict[str, str] = {
    "job_id": "STRING",
    "run_id": "STRING",
    "family": "STRING",
    "attempt": "INT64",
    "runtime": "STRING",
    "spark_mode": "STRING",
    "hardware": "STRING",
    "gpu_type": "STRING",
    "system_job_id": "STRING",
    "status": "STRING",
    "created_at": "TIMESTAMP",
    "started_at": "TIMESTAMP",
    "ended_at": "TIMESTAMP",
    "runtime_seconds": "FLOAT64",
    # Why a FAILED row failed, as a short machine-readable token (`capacity.CAPACITY_EXHAUSTED` is
    # the first). A column rather than another JSON path because this is the field an operator
    # filters a whole registry on — "show me every job that ran out of regions" has to be a WHERE
    # clause, not something you need to know a JSON path to find. NULL for every other failure and
    # for every row written before it existed.
    "failure_reason": "STRING",
    "job_telemetry": "JSON",
}


def _job_param(name: str, value: Any) -> Any:
    """Build a scalar query parameter for a run_jobs column."""
    from google.cloud import bigquery

    return bigquery.ScalarQueryParameter(name, _JOB_PARAM_TYPES[name], value)


# The parameter name both status-guarded UPDATEs bind their protected-status list to.
_STATUS_GUARD_PARAM = "unless_status_in"


def render_status_guard(unless_status_in: Sequence[str]) -> str:
    """The ``AND status …`` tail that makes an UPDATE skip rows already in a protected state (pure).

    Empty sequence → empty string, so an unguarded call renders exactly the SQL it always did. The
    ``status IS NULL`` arm is deliberate: SQL three-valued logic makes ``NULL NOT IN (…)`` unknown,
    which would silently drop the row from the update, and a row with no status is precisely one
    that has nothing worth protecting.
    """
    if not unless_status_in:
        return ""
    return f" AND (status IS NULL OR status NOT IN UNNEST(@{_STATUS_GUARD_PARAM}))"


def _status_guard_param(unless_status_in: Sequence[str]) -> Any:
    """Bind the protected-status list for `render_status_guard`'s tail."""
    from google.cloud import bigquery

    return bigquery.ArrayQueryParameter(_STATUS_GUARD_PARAM, "STRING", list(unless_status_in))


# A `job_telemetry` merge path: dot-separated lower-snake segments, rendered as ``$.a.b``. The
# charset is enforced rather than escaped because every caller is our own code writing a known
# key — a path that needs quoting is a bug in the caller, not an input to accommodate.
_TELEMETRY_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")


def render_telemetry_merge(paths: Sequence[str]) -> str:
    """The ``job_telemetry = JSON_SET(…)`` assignment that merges ``paths`` into place (pure).

    ``JSON_SET`` writes each path independently and leaves the rest of the document alone, which is
    the whole point: both telemetry columns are written by more than one author, so a whole-column
    write means whichever finishes last is the only one that leaves a trace. On the header that is
    several jobs of one run stamping their sizing; on a job row it is the capacity ledger, the probe
    handle and the cancel/settle audit accreting on the same row from different code paths at
    different times. ``IFNULL(…, JSON '{}')`` covers the first writer, whose column is still NULL;
    nested paths create their parent objects.

    Returned as the bare SET assignment (no table, no WHERE) so the two tables' writers can each
    wrap it in their own statement. Parameters are named ``@t0…@tN`` positionally against ``paths``;
    the caller binds them in the same order.
    """
    sets = ", ".join(f"'$.{path}', @t{i}" for i, path in enumerate(paths))
    return f"job_telemetry = JSON_SET(IFNULL(job_telemetry, JSON '{{}}'), {sets})"


def telemetry_merge_params(patch: Mapping[str, Any], *, caller: str) -> list[Any]:
    """Bind a ``{path: value}`` telemetry patch to ``@t0…@tN``, validating the paths (pure-ish).

    Values are bound as ``JSON`` parameters, so a dict lands as an object rather than as a string.
    An illegal path raises `errors.RegistryError` naming ``caller`` rather than being escaped into
    SQL. Order matches ``list(patch)``, which is the order `render_telemetry_merge` numbers.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    bad = [path for path in patch if not _TELEMETRY_PATH_RE.match(path)]
    if bad:
        raise RegistryError(f"{caller}: illegal telemetry path(s): {sorted(bad)}")
    return [
        bigquery.ScalarQueryParameter(f"t{i}", "JSON", patch[path]) for i, path in enumerate(patch)
    ]
