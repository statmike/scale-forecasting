"""Pure Google Cloud Console deep-link builders for compute jobs, BigQuery views & Agents.

Generates clickable Google Cloud Console URLs across all 7 ``scale-forecasting`` compute runtimes
(``spark`` Serverless/Cluster, ``vertex``, ``ray``, ``gce``, ``gke``, ``vertex_automl``, and
``bigquery``), the BigQuery Registry Dataset & analytical SQL views, Dataplex **Knowledge Catalog**
Business Glossaries, **BigQuery Studio -> Agents**, and GCS artifact roots.

Zero GCP client imports at module load time: when ``project_id`` is known at runtime (from ``SF_*``
or ADC), URLs include ``?project=<project_id>`` so clicking opens directly in the caller's project;
when ``project_id=None`` (such as offline or public documentation), URLs remain clean, valid,
project-neutral Google Cloud Console links.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from .glossary import default_glossary_id

if TYPE_CHECKING:
    import pandas as pd

CONSOLE_BASE_URL = "https://console.cloud.google.com"

_CUSTOM_JOB_RE = re.compile(r"^projects/[^/]+/locations/([^/]+)/customJobs/([^/]+)$")
_PERSISTENT_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/([^/]+)/persistentResources/([^/]+)$"
)

_REGISTRY_VIEW_SPECS: tuple[tuple[str, str], ...] = (
    (
        "v_executive_sop_summary",
        "One-row S&OP scorecard: winning model, FVA lift %, exceptions, forward volume",
    ),
    (
        "v_model_leaderboard_comparable",
        "Apples-to-apples pooled holdout WAPE & MASE model leaderboard",
    ),
    (
        "v_fva_waterfall",
        "Forecast Value Added (FVA) waterfall from naive baseline to ensemble",
    ),
    (
        "v_bias_and_exceptions",
        "Per-series Trigg tracking-signal alerts & planner exception triage",
    ),
    (
        "v_horizon_decay",
        "Lead-time horizon accuracy decay & empirical interval coverage",
    ),
    (
        "v_feature_attributions",
        "Global & per-step promo, price, and holiday covariate attributions",
    ),
    (
        "v_run_summary",
        "Cross-run execution ledger (live runs alongside 100k-series benchmarks)",
    ),
    (
        "v_run_jobs",
        "Per-family cloud job trace, runtimes, and 0.00s prior-run cell borrows",
    ),
)


def _slug_agent_id(dataset_id: str) -> str:
    """Compute the deterministic `DataAgent` ID without importing `client.py`."""
    slug = re.sub(r"[^a-z0-9-]+", "-", dataset_id.lower()).strip("-")
    base = f"sf-registry-{slug}" if slug else "sf-registry-agent"
    if not base[0].isalpha():
        base = f"sf-{base}"
    return base[:62].rstrip("-")


def _append_query(base_url: str, params: Mapping[str, str | None]) -> str:
    """Append non-empty query parameters to ``base_url`` in deterministic order."""
    filtered = [(key, val) for key, val in params.items() if val is not None and val != ""]
    if not filtered:
        return base_url
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode(filtered)}"


def _parse_probe_handle(raw_handle: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a serialized or mapping ``probe_handle`` into a plain dictionary."""
    if raw_handle is None:
        return {}
    if isinstance(raw_handle, Mapping):
        return dict(raw_handle)
    if isinstance(raw_handle, str) and raw_handle.strip():
        try:
            decoded = json.loads(raw_handle)
            if isinstance(decoded, dict):
                return decoded
        except ValueError:
            return {}
    return {}


def build_job_console_url(
    runtime: str,
    *,
    system_job_id: str | None = None,
    project_id: str | None = None,
    region: str | None = None,
    spark_mode: str | None = None,
    probe_handle: str | Mapping[str, Any] | None = None,
) -> str:
    """Build a direct Google Cloud Console URL for a single family job row.

    Supports all 7 ``scale-forecasting`` compute runtimes plus ``borrowed:<run_id>`` zero-second
    cell-reuse entries recorded when ``compute.reuse_prior_runs=True``.
    """
    handle = _parse_probe_handle(probe_handle)
    eff_project = (project_id or "").strip() or None
    handle_region = str(handle.get("region") or "").strip()
    eff_region = (region or handle_region or "us-central1").strip()
    handle_native_id = str(handle.get("native_id") or "").strip()
    job_id = (system_job_id or handle_native_id or "").strip()

    if job_id.startswith("borrowed:"):
        return _append_query(f"{CONSOLE_BASE_URL}/bigquery", {"project": eff_project})

    runtime_key = (runtime or "bigquery").strip().lower()

    if runtime_key == "spark":
        handle_mode = str(handle.get("spark_mode") or "").strip().lower()
        eff_mode = (spark_mode or handle_mode or "serverless").strip().lower()
        if eff_mode == "cluster":
            if job_id:
                return _append_query(
                    f"{CONSOLE_BASE_URL}/dataproc/jobs/{job_id}",
                    {"region": eff_region, "project": eff_project},
                )
            return _append_query(f"{CONSOLE_BASE_URL}/dataproc/jobs", {"project": eff_project})
        if job_id:
            return _append_query(
                f"{CONSOLE_BASE_URL}/dataproc/batches/{eff_region}/{job_id}/monitoring",
                {"project": eff_project},
            )
        return _append_query(f"{CONSOLE_BASE_URL}/dataproc/batches", {"project": eff_project})

    if runtime_key == "vertex":
        match = _CUSTOM_JOB_RE.match(job_id)
        if match:
            location, custom_job_num = match.group(1), match.group(2)
            return _append_query(
                f"{CONSOLE_BASE_URL}/vertex-ai/locations/{location}/training/{custom_job_num}/cpu",
                {"project": eff_project},
            )
        return _append_query(
            f"{CONSOLE_BASE_URL}/vertex-ai/training/custom-jobs",
            {"project": eff_project},
        )

    if runtime_key == "ray":
        resource_name = str(handle.get("resource_name") or job_id).strip()
        match = _PERSISTENT_RESOURCE_RE.match(resource_name)
        if match:
            location, cluster_id = match.group(1), match.group(2)
            return _append_query(
                f"{CONSOLE_BASE_URL}/vertex-ai/locations/{location}/ray/{cluster_id}",
                {"project": eff_project},
            )
        return _append_query(f"{CONSOLE_BASE_URL}/vertex-ai/ray", {"project": eff_project})

    if runtime_key == "gce":
        return _append_query(f"{CONSOLE_BASE_URL}/compute/instances", {"project": eff_project})

    if runtime_key == "gke":
        return _append_query(
            f"{CONSOLE_BASE_URL}/kubernetes/workload/overview",
            {"project": eff_project},
        )

    if runtime_key == "vertex_automl":
        return _append_query(
            f"{CONSOLE_BASE_URL}/vertex-ai/pipelines/runs",
            {"project": eff_project},
        )

    return _append_query(f"{CONSOLE_BASE_URL}/bigquery", {"project": eff_project})


@dataclass(frozen=True)
class ConsoleLink:
    """One labeled Google Cloud Console deep link with presenter navigation guidance."""

    category: str
    label: str
    url: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize this link to a plain dictionary."""
        return {
            "category": self.category,
            "label": self.label,
            "url": self.url,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class ConsoleLinkSet:
    """Curated collection of Google Cloud Console links for a forecast registry or run."""

    project_id: str | None
    dataset_id: str
    region: str
    run_id: str | None
    agent_id: str
    links: tuple[ConsoleLink, ...]
    project_number: str | None = None

    @property
    def bq_agents_url(self) -> str:
        """Console URL for BigQuery Agents Hub."""
        for link in self.links:
            if link.category == "agent":
                return link.url
        return _append_query(
            f"{CONSOLE_BASE_URL}/bigquery/agents_hub", {"project": self.project_id}
        )

    @property
    def dataset_url(self) -> str:
        """Console URL for the BigQuery Registry Dataset."""
        for link in self.links:
            if link.category == "registry":
                return link.url
        return _append_query(f"{CONSOLE_BASE_URL}/bigquery", {"project": self.project_id})

    @property
    def glossary_url(self) -> str:
        """Console URL for Dataplex Knowledge Catalog Business Glossaries."""
        for link in self.links:
            if link.category == "catalog":
                return link.url
        return _append_query(
            f"{CONSOLE_BASE_URL}/dataplex/dp-glossaries",
            {"project": self.project_id},
        )

    def by_category(self, category: str) -> tuple[ConsoleLink, ...]:
        """Filter links matching ``category``."""
        return tuple(link for link in self.links if link.category == category)

    def to_dict(self) -> dict[str, Any]:
        """Serialize link set to a JSON-compatible dictionary."""
        return {
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "region": self.region,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "bq_agents_url": self.bq_agents_url,
            "dataset_url": self.dataset_url,
            "glossary_url": self.glossary_url,
            "links": [link.to_dict() for link in self.links],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return links as a tidy ``DataFrame`` for notebook inspection."""
        import pandas as pd

        columns = ["category", "label", "url", "hint"]
        if not self.links:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame.from_records(
            [link.to_dict() for link in self.links],
            columns=columns,
        )

    def format_text(self, *, include_views: bool = True) -> str:
        """Format Console deep links as a numbered terminal / presenter checklist."""
        proj_label = self.project_id or "<select-project-in-console>"
        header = f"Google Cloud Console Links (project={proj_label}, dataset={self.dataset_id}"
        if self.run_id:
            header += f", run_id={self.run_id}"
        header += "):"
        lines = [header]
        visible = [link for link in self.links if include_views or link.category != "view"]
        for idx, link in enumerate(visible, start=1):
            lines.append(f"  [{idx}] {link.label}")
            lines.append(f"      {link.url}")
            if link.hint:
                lines.append(f"      -> {link.hint}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Render clickable Markdown links in Jupyter / Colab notebooks."""
        proj_label = f"`{self.project_id}`" if self.project_id else "*(project-neutral)*"
        title = f"#### Google Cloud Console Deep Links ({proj_label} · `{self.dataset_id}`)"
        if self.run_id:
            title += f" — Run `{self.run_id}`"
        rows = [
            title,
            "",
            "| Surface | Console Link | Navigation Note |",
            "| :--- | :--- | :--- |",
        ]
        for link in self.links:
            note = link.hint.replace("|", "\\|")
            label = link.label.replace("|", "\\|")
            rows.append(f"| **{label}** | [Open in Console]({link.url}) | {note} |")
        return "\n".join(rows)


def build_registry_console_urls(
    *,
    project_id: str | None = None,
    project_number: str | None = None,
    dataset_id: str = "scale_forecasting",
    region: str = "us-central1",
    run_id: str | None = None,
    agent_id: str | None = None,
    warehouse_uri: str | None = None,
    job_rows: Sequence[Mapping[str, Any]] = (),
    include_service_hubs: bool = True,
) -> ConsoleLinkSet:
    """Construct a complete :class:`ConsoleLinkSet` for a forecast registry and optional ``run_id``.

    Includes direct links for **BigQuery Agents Hub**, the **Registry Dataset**, key analytical SQL
    views, **Knowledge Catalog** Business Glossaries, executed per-family compute jobs (or runtime
    hubs), and GCS warehouse artifacts.
    """
    eff_project = (project_id or "").strip() or None
    raw_proj_num = (project_number or "").strip()
    eff_proj_num = raw_proj_num if raw_proj_num.isdigit() else None
    eff_dataset = (dataset_id or "scale_forecasting").strip()
    eff_region = (region or "us-central1").strip()
    eff_agent_id = (agent_id or _slug_agent_id(eff_dataset)).strip()
    glossary_id = default_glossary_id(eff_dataset)

    if eff_proj_num:
        glossary_path = (
            f"{CONSOLE_BASE_URL}/dataplex/dp-glossaries/projects/{eff_proj_num}"
            f"/locations/global/glossaries/{glossary_id}"
        )
    else:
        glossary_path = f"{CONSOLE_BASE_URL}/dataplex/dp-glossaries"

    links: list[ConsoleLink] = [
        ConsoleLink(
            category="agent",
            label=f"BigQuery Studio — Agents Hub (Registry Agent: {eff_agent_id})",
            url=_append_query(f"{CONSOLE_BASE_URL}/bigquery/agents_hub", {"project": eff_project}),
            hint=(
                f"Open scale-forecasting Registry Agent ({eff_dataset}) "
                f"({eff_agent_id}) for zero-code chat"
            ),
        ),
        ConsoleLink(
            category="registry",
            label=f"BigQuery Registry Dataset ({eff_dataset})",
            url=(
                _append_query(
                    f"{CONSOLE_BASE_URL}/bigquery",
                    {
                        "project": eff_project,
                        "p": eff_project,
                        "d": eff_dataset,
                        "page": "dataset",
                    },
                )
                if eff_project
                else f"{CONSOLE_BASE_URL}/bigquery"
            ),
            hint="5 append-only registry tables + 11 self-documenting analytical SQL views",
        ),
        ConsoleLink(
            category="catalog",
            label=f"Knowledge Catalog Business Glossary ({glossary_id})",
            url=_append_query(glossary_path, {"project": eff_project}),
            hint="44 governed forecasting & S&OP definitions linked to registry tables and views",
        ),
    ]

    for row in job_rows:
        family = str(row.get("family") or "job")
        rt = str(row.get("runtime") or "bigquery")
        sys_id = str(row.get("system_job_id") or "") or None
        status = str(row.get("status") or "")
        job_url = build_job_console_url(
            rt,
            system_job_id=sys_id,
            project_id=eff_project,
            region=eff_region,
            spark_mode=str(row.get("spark_mode") or "") or None,
            probe_handle=row.get("probe_handle"),
        )
        if sys_id and sys_id.startswith("borrowed:"):
            hint_text = f"0.00s cell reuse ({sys_id}); inspect in v_run_jobs"
        else:
            hint_parts = [p for p in (f"status={status}" if status else "", sys_id or "") if p]
            hint_text = " · ".join(hint_parts)
        links.append(
            ConsoleLink(
                category="compute",
                label=f"Job [{family}] on {rt}",
                url=job_url,
                hint=hint_text,
            )
        )

    if include_service_hubs and not job_rows:
        links.extend(
            [
                ConsoleLink(
                    category="compute",
                    label="Dataproc Serverless Spark Batches",
                    url=_append_query(
                        f"{CONSOLE_BASE_URL}/dataproc/batches",
                        {"project": eff_project},
                    ),
                    hint="Zero-idle serverless Spark model family batches",
                ),
                ConsoleLink(
                    category="compute",
                    label="Vertex AI Custom Training Jobs",
                    url=_append_query(
                        f"{CONSOLE_BASE_URL}/vertex-ai/training/custom-jobs",
                        {"project": eff_project},
                    ),
                    hint="Ephemeral CPU & GPU (T4 / L4 / A100) family workers",
                ),
                ConsoleLink(
                    category="compute",
                    label="Vertex AI Ray Clusters",
                    url=_append_query(
                        f"{CONSOLE_BASE_URL}/vertex-ai/ray",
                        {"project": eff_project},
                    ),
                    hint="Autoscaling Ray clusters for distributed Python families",
                ),
            ]
        )

    if warehouse_uri and warehouse_uri.startswith("gs://"):
        bucket_and_prefix = warehouse_uri[len("gs://") :].strip("/")
        if bucket_and_prefix:
            if run_id and eff_project:
                storage_path = f"{bucket_and_prefix}/artifacts/{eff_project}/{eff_dataset}/{run_id}"
            else:
                storage_path = bucket_and_prefix
            links.append(
                ConsoleLink(
                    category="storage",
                    label="Cloud Storage Warehouse & Run Artifacts",
                    url=_append_query(
                        f"{CONSOLE_BASE_URL}/storage/browser/{storage_path}",
                        {"project": eff_project},
                    ),
                    hint=warehouse_uri,
                )
            )

    for view_name, view_desc in _REGISTRY_VIEW_SPECS:
        view_url = (
            _append_query(
                f"{CONSOLE_BASE_URL}/bigquery",
                {
                    "project": eff_project,
                    "p": eff_project,
                    "d": eff_dataset,
                    "t": view_name,
                    "page": "table",
                },
            )
            if eff_project
            else f"{CONSOLE_BASE_URL}/bigquery"
        )
        links.append(
            ConsoleLink(
                category="view",
                label=f"View `{view_name}`",
                url=view_url,
                hint=view_desc,
            )
        )

    return ConsoleLinkSet(
        project_id=eff_project,
        dataset_id=eff_dataset,
        region=eff_region,
        run_id=run_id,
        agent_id=eff_agent_id,
        links=tuple(links),
        project_number=eff_proj_num,
    )
