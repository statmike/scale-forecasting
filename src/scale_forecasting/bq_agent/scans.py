"""Knowledge Catalog DataScan (`DATA_PROFILE` & `DATA_DOCUMENTATION`) provisioner.

Creates and optionally triggers Knowledge Catalog `DataScan` jobs (`dataplex.googleapis.com/v1`)
with ``catalogPublishingEnabled = True`` over the physical/Iceberg tables in a `BQAgentScope`
(registry tables + discovered source tables). Logical SQL views are automatically skipped because
Knowledge Catalog DataProfile scans operate on storage-backed tables.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

from .rest import KNOWLEDGE_CATALOG_BASE_URL, build_authorized_session, request_json, wait_for_lro

if TYPE_CHECKING:
    from .context import BQAgentScope

logger = logging.getLogger(__name__)


def _scan_slug(raw: str) -> str:
    """Normalize a scan ID to Knowledge Catalog's ``[a-z][a-z0-9-]{0,62}`` grammar."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"sf-{cleaned}" if cleaned else "sf-scan"
    if len(cleaned) <= 62:
        return cleaned.rstrip("-")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned[:55].rstrip('-')}-{digest}"


def default_profile_scan_id(dataset_id: str, table_id: str) -> str:
    """Return the deterministic Knowledge Catalog DataProfile scan ID for a table."""
    return _scan_slug(f"sf-prof-{dataset_id}-{table_id}")


def ensure_knowledge_catalog_scans(
    scope: BQAgentScope,
    *,
    region: str = "us-central1",
    trigger_run: bool = True,
    include_documentation_scan: bool = False,
    max_source_tables: int = 10,
    session: Any = None,
) -> dict[str, Any]:
    """Idempotently create (and optionally trigger) Knowledge Catalog DataProfile scans.

    Publishes column cardinality, null rates, min/max, and quantile distributions into Knowledge
    Catalog (``catalogPublishingEnabled=True``) so BigQuery Conversational Analytics Agents ground
    value predicates and distributions against real table statistics. Wraps each table individually
    so unsupported external/BigLake tables or cross-region sources skip cleanly without aborting
    native BigQuery registry table scans.
    """
    project_id = scope.project_id
    sess = session or build_authorized_session(project_id)
    parent = f"projects/{project_id}/locations/{region}"
    scans_collection_url = f"{KNOWLEDGE_CATALOG_BASE_URL}/{parent}/dataScans"

    ensured_scans: list[str] = []
    triggered_jobs: list[str] = []
    skipped_tables: list[str] = []

    target_tables = (*scope.registry_tables, *scope.source_tables[: max(0, max_source_tables)])

    for ref in target_tables:
        scan_id = default_profile_scan_id(ref.dataset_id, ref.table_id)
        scan_name = f"{parent}/dataScans/{scan_id}"
        scan_url = f"{KNOWLEDGE_CATALOG_BASE_URL}/{scan_name}"
        resource_uri = (
            f"//bigquery.googleapis.com/projects/{ref.project_id}"
            f"/datasets/{ref.dataset_id}/tables/{ref.table_id}"
        )
        scan_body: dict[str, Any] = {
            "displayName": f"scale-forecasting Profile ({ref.dataset_id}.{ref.table_id})",
            "description": (
                f"Automated Knowledge Catalog data profile scan for {ref.fqn} "
                "(published to Knowledge Catalog for BigQuery Data Agents)."
            ),
            "data": {"resource": resource_uri},
            "dataProfileSpec": {
                "samplingPercent": 100.0,
                "catalogPublishingEnabled": True,
            },
            "executionSpec": {
                "trigger": {"onDemand": {}},
            },
        }
        try:
            existing = request_json(
                sess,
                "GET",
                scan_url,
                allow_not_found=True,
                service_label="Knowledge Catalog DataScan",
            )
            if existing is None:
                op = request_json(
                    sess,
                    "POST",
                    scans_collection_url,
                    params={"dataScanId": scan_id},
                    json_body=scan_body,
                    service_label="Knowledge Catalog DataScan",
                )
                wait_for_lro(
                    sess,
                    op,
                    base_url=KNOWLEDGE_CATALOG_BASE_URL,
                    service_label="Knowledge Catalog DataScan",
                )
            ensured_scans.append(scan_name)
        except Exception as exc:  # noqa: BLE001 - external/Iceberg or cross-region tables may reject DataScan
            logger.debug("DataScan creation skipped for %s: %s", ref.fqn, exc)
            skipped_tables.append(ref.fqn)
            continue

        if trigger_run:
            try:
                run_resp = request_json(
                    sess,
                    "POST",
                    f"{scan_url}:run",
                    json_body={},
                    service_label="Knowledge Catalog DataScan",
                )
                job_name = str(run_resp.get("job", {}).get("name", scan_name))
                triggered_jobs.append(job_name)
            except Exception as exc:  # noqa: BLE001 - concurrent scan run already active is safe
                logger.debug(
                    "DataScan trigger skipped for %s (may already be running): %s", scan_id, exc
                )

        if include_documentation_scan:
            doc_scan_id = _scan_slug(f"sf-doc-{ref.dataset_id}-{ref.table_id}")
            doc_scan_url = f"{KNOWLEDGE_CATALOG_BASE_URL}/{parent}/dataScans/{doc_scan_id}"
            try:
                existing_doc = request_json(
                    sess,
                    "GET",
                    doc_scan_url,
                    allow_not_found=True,
                    service_label="Knowledge Catalog DataScan",
                )
                if existing_doc is None:
                    doc_body: dict[str, Any] = {
                        "displayName": f"scale-forecasting Docs ({ref.dataset_id}.{ref.table_id})",
                        "data": {"resource": resource_uri},
                        "dataDocumentationSpec": {"catalogPublishingEnabled": True},
                        "executionSpec": {"trigger": {"onDemand": {}}},
                    }
                    op = request_json(
                        sess,
                        "POST",
                        scans_collection_url,
                        params={"dataScanId": doc_scan_id},
                        json_body=doc_body,
                        service_label="Knowledge Catalog DataScan",
                    )
                    wait_for_lro(
                        sess,
                        op,
                        base_url=KNOWLEDGE_CATALOG_BASE_URL,
                        service_label="Knowledge Catalog DataScan",
                    )
                ensured_scans.append(f"{parent}/dataScans/{doc_scan_id}")
            except Exception as exc:  # noqa: BLE001 - optional preview feature in some regions
                logger.debug("Optional DataDocumentation scan skipped on %s: %s", ref.fqn, exc)

    return {
        "region": region,
        "scans_ensured": ensured_scans,
        "jobs_triggered": triggered_jobs,
        "skipped_tables": skipped_tables,
    }
