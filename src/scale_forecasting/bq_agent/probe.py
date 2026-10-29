"""Non-raising pre-flight readiness probe for Knowledge Catalog & BigQuery Conversational Agent.

Allows notebooks, CLI callers, and SDK workflows to safely check whether Google Cloud
Conversational Analytics (`geminidataanalytics.googleapis.com`) and **Knowledge Catalog**
(`dataplex.googleapis.com`) are enabled and authorized before calling `ensure_registry_agent`
or `ask_registry` ("check-and-run" pattern).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors import is_importable
from .client import default_agent_id, resolve_target_coordinates
from .glossary import default_glossary_id
from .rest import GEMINI_ANALYTICS_BASE_URL, KNOWLEDGE_CATALOG_BASE_URL, build_authorized_session

if TYPE_CHECKING:
    from ..settings import Settings


@dataclass(frozen=True)
class BQAgentProbeResult:
    """Structured, non-raising pre-flight status for `probe_bq_agent_access`."""

    ready: bool
    conversational_api_ready: bool
    knowledge_catalog_ready: bool
    agent_registered: bool
    project_id: str | None = None
    dataset_id: str | None = None
    location: str = "global"
    agent_id: str | None = None
    agent_name: str | None = None
    missing_apis: tuple[str, ...] = ()
    reason: str = ""
    setup_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize probe status to a JSON-compatible dictionary."""
        return {
            "ready": self.ready,
            "conversational_api_ready": self.conversational_api_ready,
            "knowledge_catalog_ready": self.knowledge_catalog_ready,
            "agent_registered": self.agent_registered,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "location": self.location,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "missing_apis": list(self.missing_apis),
            "reason": self.reason,
            "setup_hint": self.setup_hint,
        }


def _build_enablement_hint(project_id: str, missing_apis: list[str], *, iam_denied: bool) -> str:
    """Render an actionable CLI / Terraform setup hint when an API or IAM role is missing."""
    parts: list[str] = []
    if missing_apis:
        apis_str = " ".join(missing_apis)
        parts.append(
            f"Enable required Google Cloud APIs: `gcloud services enable {apis_str} "
            f"--project={project_id}` (or apply `terraform/main` with "
            "`enable_bq_conversational_agent = true`)."
        )
    if iam_denied:
        parts.append(
            "Grant Conversational Analytics & Knowledge Catalog roles (`roles/geminidataanalytics"
            ".dataAgentOwner`, `roles/cloudaicompanion.user`, `roles/dataplex.catalogEditor`) to "
            "the active caller or service account."
        )
    return " ".join(parts)


def probe_bq_agent_access(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    agent_id: str | None = None,
    location: str = "global",
    region: str | None = None,
    settings: Settings | None = None,
    session: Any = None,
    timeout: float = 15.0,
) -> BQAgentProbeResult:
    """Check BigQuery `DataAgent` & **Knowledge Catalog** API readiness without raising.

    Returns a :class:`BQAgentProbeResult` where ``ready=True`` indicates that
    `ensure_registry_agent` and `ask_registry` can execute against ``project_id``. When an API is
    disabled, credentials are absent, or IAM permissions are missing, ``ready`` is ``False`` and
    ``setup_hint`` provides the exact remediation command so notebooks and pipelines can skip the
    optional agent step cleanly without failing.
    """
    if session is None and not (
        is_importable("google.auth")
        and is_importable("google.auth.transport.requests")
        and is_importable("requests")
    ):
        return BQAgentProbeResult(
            ready=False,
            conversational_api_ready=False,
            knowledge_catalog_ready=False,
            agent_registered=False,
            location=location,
            reason="Optional [gcp] extra is not installed.",
            setup_hint='Install Google Cloud dependencies: pip install "scale-forecasting[gcp]"',
        )

    eff_proj: str | None = None
    eff_ds: str | None = None
    try:
        eff_proj, eff_ds, _, _ = resolve_target_coordinates(
            project_id=project_id,
            dataset_id=dataset_id,
            region=region,
            settings=settings,
        )
    except Exception:  # noqa: BLE001 - fall back to bare env inspection when SF_* is partial
        eff_proj = (
            project_id
            or os.environ.get("SF_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or ""
        ).strip() or None
        eff_ds = (
            dataset_id
            or os.environ.get("SF_REGISTRY_DATASET_ID")
            or os.environ.get("SF_DATASET_ID")
            or "scale_forecasting"
        )

    if not eff_proj:
        return BQAgentProbeResult(
            ready=False,
            conversational_api_ready=False,
            knowledge_catalog_ready=False,
            agent_registered=False,
            dataset_id=eff_ds,
            location=location,
            reason="No Google Cloud project_id configured in Settings or environment.",
            setup_hint=(
                "Set SF_PROJECT_ID (or pass project_id='your-gcp-project-id') and run "
                "`gcloud auth application-default login`."
            ),
        )

    assert eff_ds is not None
    eff_agent_id = agent_id or default_agent_id(eff_ds)
    expected_name = f"projects/{eff_proj}/locations/{location}/dataAgents/{eff_agent_id}"

    try:
        sess = session if session is not None else build_authorized_session(eff_proj)
    except Exception as exc:  # noqa: BLE001 - non-raising ADC check
        return BQAgentProbeResult(
            ready=False,
            conversational_api_ready=False,
            knowledge_catalog_ready=False,
            agent_registered=False,
            project_id=eff_proj,
            dataset_id=eff_ds,
            location=location,
            agent_id=eff_agent_id,
            reason=f"Application Default Credentials unavailable: {exc}",
            setup_hint="Authenticate locally via `gcloud auth application-default login`.",
        )

    missing_apis: list[str] = []
    iam_denied = False
    conv_ready = False
    agent_registered = False
    resolved_agent_name: str | None = None
    conv_error_detail = ""

    agent_url = f"{GEMINI_ANALYTICS_BASE_URL}/{expected_name}"
    try:
        resp = sess.request(method="GET", url=agent_url, timeout=timeout)
        code = int(resp.status_code)
        if code == 200:
            conv_ready = True
            agent_registered = True
            body = resp.json() if resp.text.strip() else {}
            resolved_agent_name = str(body.get("name") or expected_name)
        elif code == 404:
            conv_ready = True
            agent_registered = False
        else:
            conv_error_detail = resp.text.strip()
            if "SERVICE_DISABLED" in conv_error_detail or "has not been used" in conv_error_detail:
                missing_apis.extend(
                    ["geminidataanalytics.googleapis.com", "cloudaicompanion.googleapis.com"]
                )
            elif code in (401, 403):
                iam_denied = True
    except Exception as exc:  # noqa: BLE001 - network / transport probe guard
        conv_error_detail = str(exc)

    kc_ready = False
    glossary_id = default_glossary_id(eff_ds)
    glossary_url = (
        f"{KNOWLEDGE_CATALOG_BASE_URL}/projects/{eff_proj}/locations/global/"
        f"glossaries/{glossary_id}"
    )
    try:
        kc_resp = sess.request(method="GET", url=glossary_url, timeout=timeout)
        kc_code = int(kc_resp.status_code)
        if kc_code in (200, 404):
            kc_ready = True
        else:
            kc_text = kc_resp.text.strip()
            if "SERVICE_DISABLED" in kc_text or "has not been used" in kc_text:
                missing_apis.append("dataplex.googleapis.com")
    except Exception:  # noqa: BLE001 - optional catalog probe guard
        kc_ready = False

    if conv_ready:
        status_note = "registered" if agent_registered else "not yet created (ready to provision)"
        kc_note = (
            "Knowledge Catalog API ready"
            if kc_ready
            else "Knowledge Catalog API unavailable (glossary sync will be skipped)"
        )
        return BQAgentProbeResult(
            ready=True,
            conversational_api_ready=True,
            knowledge_catalog_ready=kc_ready,
            agent_registered=agent_registered,
            project_id=eff_proj,
            dataset_id=eff_ds,
            location=location,
            agent_id=eff_agent_id,
            agent_name=resolved_agent_name,
            missing_apis=tuple(missing_apis),
            reason=f"BigQuery Conversational Analytics API ready ({status_note}); {kc_note}.",
            setup_hint=(
                _build_enablement_hint(eff_proj, missing_apis, iam_denied=False)
                if missing_apis
                else ""
            ),
        )

    hint = _build_enablement_hint(eff_proj, missing_apis, iam_denied=iam_denied)
    return BQAgentProbeResult(
        ready=False,
        conversational_api_ready=False,
        knowledge_catalog_ready=kc_ready,
        agent_registered=False,
        project_id=eff_proj,
        dataset_id=eff_ds,
        location=location,
        agent_id=eff_agent_id,
        missing_apis=tuple(missing_apis),
        reason=(
            f"BigQuery Conversational Analytics API is not ready on `{eff_proj}`: "
            f"{conv_error_detail[:240] or 'unreachable'}"
        ),
        setup_hint=hint,
    )
