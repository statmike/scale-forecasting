"""Strict BigQuery Registry & Source Scope Discovery + `geminidataanalytics` Context Builder.

Builds the complete ``GoogleCloudGeminidataanalyticsV1Context`` payload for a registered BigQuery
Conversational Analytics Agent (`DataAgent`), enforcing strict object-level scoping:

- **Included:** Only the 5 canonical `REGISTRY_TABLE_NAMES` and 11 canonical `VIEW_NAMES` in the
  target registry dataset (plus any explicitly federated ``additional_registries``) and the
  verified ``source_table``s discovered from ``run_registry.raw_config`` (`$.data.source_table`).
- **Excluded:** Unrelated user tables, staging tables, or ephemeral AutoML tables
  (``_sf_automl_train_*``, ``_sf_automl_infer_*``) that happen to share the same BigQuery dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..errors import ConfigError
from ..registry.ddl import REGISTRY_TABLE_NAMES
from ..registry.tables import bigquery_client
from ..registry.views import VIEW_NAMES
from .glossary import build_context_glossary_terms
from .golden_queries import build_golden_queries, format_example_queries_for_context

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

IncludeSourcesMode = Literal["auto", "none", "explicit"]


@dataclass(frozen=True)
class ScopedTableRef:
    """One verified BigQuery table or view included in the `DataAgent` scope."""

    project_id: str
    dataset_id: str
    table_id: str
    role: Literal["registry_table", "registry_view", "source_table"]

    @property
    def fqn(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    def to_api_ref(self) -> dict[str, str]:
        return {
            "projectId": self.project_id,
            "datasetId": self.dataset_id,
            "tableId": self.table_id,
        }


@dataclass(frozen=True)
class BQAgentScope:
    """Resolved strict scope for a BigQuery Registry `DataAgent`."""

    project_id: str
    dataset_id: str
    registry_tables: tuple[ScopedTableRef, ...]
    registry_views: tuple[ScopedTableRef, ...]
    source_tables: tuple[ScopedTableRef, ...]

    @property
    def all_objects(self) -> tuple[ScopedTableRef, ...]:
        return (*self.registry_tables, *self.registry_views, *self.source_tables)

    @property
    def physical_tables(self) -> tuple[ScopedTableRef, ...]:
        """Physical/Iceberg tables eligible for Knowledge Catalog DataProfile scans (no views)."""
        return (*self.registry_tables, *self.source_tables)


def parse_table_fqn(
    raw_ref: str,
    *,
    default_project_id: str,
    default_dataset_id: str,
) -> tuple[str, str, str] | None:
    """Parse ``table``, ``dataset.table``, or ``project.dataset.table`` into a 3-tuple."""
    cleaned = raw_ref.strip().strip("`")
    if not cleaned or cleaned.startswith("<") or "/" in cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(".") if p.strip()]
    if len(parts) == 1:
        return (default_project_id, default_dataset_id, parts[0])
    if len(parts) == 2:
        return (default_project_id, parts[0], parts[1])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    return None


def _table_exists(client: Any, fqn: str) -> bool:
    """Return True if ``fqn`` exists in BigQuery and is accessible to ``client``."""
    from google.api_core import exceptions as gcp_exceptions

    try:
        client.get_table(fqn)
        return True
    except gcp_exceptions.NotFound:
        return False
    except gcp_exceptions.Forbidden:
        logger.warning("Skipping inaccessible table %s during BQ agent scope discovery", fqn)
        return False


def discover_registry_scope(
    project_id: str,
    dataset_id: str,
    *,
    include_sources: IncludeSourcesMode = "auto",
    source_tables: Sequence[str] | None = None,
    default_source_dataset_id: str | None = None,
    additional_registries: Sequence[str] = (),
    client: Any = None,
) -> BQAgentScope:
    """Discover the exact allowlisted registry tables, views, and active source tables.

    Never calls ``list_tables`` across a dataset, guaranteeing that ephemeral staging tables or
    unrelated tables in the same dataset are strictly excluded.
    """
    if include_sources not in ("auto", "none", "explicit"):
        raise ConfigError(
            f"include_sources must be 'auto', 'none', or 'explicit'; got {include_sources!r}"
        )

    bq = client if client is not None else bigquery_client(project_id)
    src_default_ds = default_source_dataset_id or dataset_id

    registry_targets: list[tuple[str, str]] = [(project_id, dataset_id)]
    for extra in additional_registries:
        cleaned = extra.strip().strip("`")
        if not cleaned:
            continue
        parts = cleaned.split(".")
        if len(parts) == 1:
            pair = (project_id, parts[0])
        elif len(parts) == 2:
            pair = (parts[0], parts[1])
        else:
            raise ConfigError(
                f"additional_registries entry must be 'dataset' or 'project.dataset'; got {extra!r}"
            )
        if pair not in registry_targets:
            registry_targets.append(pair)

    reg_tables: list[ScopedTableRef] = []
    reg_views: list[ScopedTableRef] = []
    seen_fqns: set[str] = set()

    for reg_proj, reg_ds in registry_targets:
        for t_name in REGISTRY_TABLE_NAMES:
            fqn = f"{reg_proj}.{reg_ds}.{t_name}"
            if fqn not in seen_fqns and _table_exists(bq, fqn):
                reg_tables.append(ScopedTableRef(reg_proj, reg_ds, t_name, "registry_table"))
                seen_fqns.add(fqn)
        for v_name in VIEW_NAMES:
            fqn = f"{reg_proj}.{reg_ds}.{v_name}"
            if fqn not in seen_fqns and _table_exists(bq, fqn):
                reg_views.append(ScopedTableRef(reg_proj, reg_ds, v_name, "registry_view"))
                seen_fqns.add(fqn)

    candidate_sources: list[tuple[str, str, str]] = []
    if include_sources == "explicit":
        for raw_src in source_tables or ():
            parsed = parse_table_fqn(
                raw_src,
                default_project_id=project_id,
                default_dataset_id=src_default_ds,
            )
            if parsed is not None:
                candidate_sources.append(parsed)
    elif include_sources == "auto":
        for reg_proj, reg_ds in registry_targets:
            rr_fqn = f"{reg_proj}.{reg_ds}.run_registry"
            if rr_fqn not in seen_fqns:
                continue
            sql = (
                "SELECT DISTINCT JSON_VALUE(raw_config, '$.data.source_table') AS src "
                f"FROM `{rr_fqn}` "
                "WHERE raw_config IS NOT NULL"
            )
            try:
                rows = list(bq.query(sql).result())
            except Exception as exc:  # noqa: BLE001 - tolerate empty or restricted table
                logger.warning("Source discovery query on %s skipped: %s", rr_fqn, exc)
                continue
            for row in rows:
                row_src = row.get("src") if isinstance(row, dict) else row["src"]
                if not row_src:
                    continue
                parsed = parse_table_fqn(
                    str(row_src),
                    default_project_id=reg_proj,
                    default_dataset_id=src_default_ds,
                )
                if parsed is not None:
                    candidate_sources.append(parsed)

    resolved_sources: list[ScopedTableRef] = []
    for s_proj, s_ds, s_tbl in candidate_sources:
        if s_tbl.startswith("_sf_automl_"):
            continue
        fqn = f"{s_proj}.{s_ds}.{s_tbl}"
        if fqn in seen_fqns:
            continue
        if _table_exists(bq, fqn):
            resolved_sources.append(ScopedTableRef(s_proj, s_ds, s_tbl, "source_table"))
            seen_fqns.add(fqn)

    return BQAgentScope(
        project_id=project_id,
        dataset_id=dataset_id,
        registry_tables=tuple(reg_tables),
        registry_views=tuple(reg_views),
        source_tables=tuple(resolved_sources),
    )


def build_schema_relationships(scope: BQAgentScope) -> list[dict[str, Any]]:
    """Build explicit `SchemaRelationship` join edges across scoped registry & source objects."""
    existing = {
        ref.table_id: ref.fqn for ref in scope.all_objects if ref.dataset_id == scope.dataset_id
    }
    relationships: list[dict[str, Any]] = []

    def _add_edge(left_table: str, right_table: str, cols: list[str]) -> None:
        left_fqn = existing.get(left_table)
        right_fqn = existing.get(right_table)
        if not left_fqn or not right_fqn:
            return
        relationships.append(
            {
                "leftSchemaPaths": {"tableFqn": left_fqn, "paths": cols},
                "rightSchemaPaths": {"tableFqn": right_fqn, "paths": cols},
                "sources": ["BIGQUERY_TABLE_CONSTRAINTS"],
                "confidenceScore": 1.0,
            }
        )

    for child in (
        "run_jobs",
        "forecast_metadata",
        "forecast_predictions",
        "backtest_oof",
        "v_run_summary",
        "v_run_jobs",
        "v_model_leaderboard",
        "v_model_leaderboard_comparable",
        "v_fva_waterfall",
        "v_horizon_decay",
        "v_bias_and_exceptions",
        "v_anomalies",
        "v_feature_attributions",
        "v_executive_sop_summary",
    ):
        if child != "run_registry":
            _add_edge("run_registry", child, ["run_id"])

    _add_edge("forecast_metadata", "forecast_predictions", ["run_id", "ts_id", "model_type"])
    _add_edge("forecast_metadata", "backtest_oof", ["run_id", "ts_id", "model_type"])
    _add_edge("v_executive_sop_summary", "v_bias_and_exceptions", ["run_id"])
    _add_edge("v_executive_sop_summary", "forecast_predictions", ["run_id"])

    for src_ref in scope.source_tables:
        for target_tbl in ("forecast_predictions", "backtest_oof", "v_bias_and_exceptions"):
            target_fqn = existing.get(target_tbl)
            if target_fqn:
                relationships.append(
                    {
                        "leftSchemaPaths": {"tableFqn": target_fqn, "paths": ["ts_id"]},
                        "rightSchemaPaths": {"tableFqn": src_ref.fqn, "paths": ["ts_id"]},
                        "sources": ["BIGQUERY_TABLE_CONSTRAINTS"],
                        "confidenceScore": 1.0,
                    }
                )
    return relationships


def build_system_instruction(scope: BQAgentScope) -> str:
    """Build the domain system instruction teaching the BigQuery `DataAgent` how to query."""
    ds = f"{scope.project_id}.{scope.dataset_id}"
    src_list = (
        ", ".join(f"`{r.fqn}`" for r in scope.source_tables)
        if scope.source_tables
        else "none attached (`include_sources='none'` or no runs recorded yet)"
    )
    return (
        f"You are the scale-forecasting Conversational Analytics Agent for BigQuery registry "
        f"`{ds}`.\n\n"
        "FOLLOW THESE FIVE NON-NEGOTIABLE RULES ON EVERY QUESTION:\n"
        "1. DEFAULT TO THE LATEST COMPLETED RUN: Whenever the user does not specify an explicit "
        "`run_id` (or `[Context: run_id=...]`) and is not asking for cross-run history, scope "
        f"your SQL to the newest completed run in `{ds}.v_run_summary` (`ORDER BY IF(status = "
        "'COMPLETED', 0, 1), created_at DESC LIMIT 1`), and always name that `run_id` and its "
        "`created_at` timestamp in your explanation.\n"
        "2. QUERY CURATED `v_*` VIEWS FIRST: Always prefer the 11 pre-deduplicated analytical "
        "views over raw append-only tables:\n"
        f"   - `{ds}.v_executive_sop_summary`: one-row executive scorecard per run (winning "
        "champion_model, champion_pooled_wape, fva_relative_lift_pct, total_forecast_units, "
        "exception_rate, n_gpu_waste_jobs).\n"
        f"   - `{ds}.v_model_leaderboard_comparable`: apples-to-apples holdout-fold pooled WAPE "
        "and pooled MAE ranking across base models, ensembles, and reconciled variants.\n"
        f"   - `{ds}.v_fva_waterfall`: Forecast Value Added across 6 pipeline complexity stages.\n"
        f"   - `{ds}.v_bias_and_exceptions`: planner triage queue with Trigg's `tracking_signal`, "
        "`sbc_class`, `is_exception`, and `exception_reason`.\n"
        f"   - `{ds}.v_horizon_decay`: accuracy and coverage by lead-time `horizon_step`.\n"
        f"   - `{ds}.v_anomalies`: historical demand spikes/dips (`|residual_zscore| >= 2.5`).\n"
        f"   - `{ds}.v_feature_attributions`: global and horizon-local feature importances.\n"
        f"   - `{ds}.v_run_summary`, `{ds}.v_run_jobs`, `{ds}.v_model_leaderboard`, "
        f"`{ds}.v_backtest_coverage`: run overhead, cloud compute telemetry, and fold coverage.\n"
        "3. DEDUPLICATE APPEND-ONLY TABLES WHEN QUERIED DIRECTLY: If you query "
        "`forecast_predictions`, `forecast_metadata`, `backtest_oof`, `run_registry`, or "
        "`run_jobs` directly, ALWAYS deduplicate retries with `QUALIFY ROW_NUMBER() OVER "
        "(... ORDER BY created_at DESC NULLS LAST) = 1`.\n"
        "4. SNAPSHOT & TEMPORAL LINEAGE (`snapshot_millis`): Use `backtest_oof.y_true` for "
        "historical backtest accuracy (it is snapshot-locked as of run launch). Only join "
        f"`forecast_predictions` to discovered source tables ({src_list}) when evaluating "
        "post-run realized accuracy on fresh actuals or rolling up by source business attributes.\n"
        "5. NATIVE JSON PATHS: Quote decimal keys in `quantiles` as "
        '`SAFE_CAST(JSON_VALUE(quantiles, \'$."0.1"\') AS FLOAT64)` (and `"0.5"`, `"0.9"`). '
        "Read run configs from `JSON_VALUE(raw_config, '$.data.source_table')`."
    )


def build_published_context(scope: BQAgentScope) -> dict[str, Any]:
    """Assemble the full `GoogleCloudGeminidataanalyticsV1Context` JSON payload for ``scope``."""
    default_src = scope.source_tables[0].fqn if scope.source_tables else None
    golden = build_golden_queries(
        scope.project_id,
        scope.dataset_id,
        default_source_table=default_src,
    )
    return {
        "systemInstruction": build_system_instruction(scope),
        "datasourceReferences": {
            "bq": {
                "tableReferences": [ref.to_api_ref() for ref in scope.all_objects],
            }
        },
        "exampleQueries": format_example_queries_for_context(golden),
        "glossaryTerms": build_context_glossary_terms(),
        "schemaRelationships": build_schema_relationships(scope),
    }
