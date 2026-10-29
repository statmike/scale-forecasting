"""Authenticated Google Cloud REST transport for Knowledge Catalog & Conversational Analytics API.

Uses ``google.auth.transport.requests.AuthorizedSession`` (already included in the ``[gcp]`` extra
via ``google-auth`` and ``requests``) so ``scale_forecasting.bq_agent`` talks directly to the GA
``v1`` endpoints of ``geminidataanalytics.googleapis.com`` and ``dataplex.googleapis.com`` without
forcing extra generated gRPC client distributions onto workers or notebooks.
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import RegistryError, require_modules

GEMINI_ANALYTICS_BASE_URL = "https://geminidataanalytics.googleapis.com/v1"
KNOWLEDGE_CATALOG_BASE_URL = "https://dataplex.googleapis.com/v1"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_DEFAULT_TIMEOUT_SECONDS = 120.0
_LRO_POLL_INTERVAL_SECONDS = 1.5
_LRO_MAX_WAIT_SECONDS = 90.0


def build_authorized_session(project_id: str) -> Any:  # pragma: no cover - GCP auth boundary
    """Build an ADC ``AuthorizedSession`` billing API quota against ``project_id``."""
    require_modules(
        ("google.auth", "google.auth.transport.requests", "requests"),
        extra="gcp",
        purpose="Calling BigQuery Conversational Analytics & Knowledge Catalog APIs",
    )
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    session = AuthorizedSession(credentials)
    session.headers.update(
        {
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }
    )
    return session


def request_json(
    session: Any,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    allow_not_found: bool = False,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    service_label: str = "Google Cloud API",
) -> Any:
    """Execute an authenticated JSON HTTP request and raise `RegistryError` on error."""
    response = session.request(
        method=method,
        url=url,
        params=params,
        json=json_body,
        timeout=timeout,
    )
    if allow_not_found and response.status_code == 404:
        return None
    if response.status_code >= 400:
        detail = response.text.strip()
        hint = ""
        if response.status_code == 403 and "SERVICE_DISABLED" in detail:
            hint = (
                " Ensure geminidataanalytics.googleapis.com, cloudaicompanion.googleapis.com, "
                "and dataplex.googleapis.com are enabled on the target project (or apply the "
                "updated terraform/main module)."
            )
        raise RegistryError(
            f"{service_label} {method} {url} failed (HTTP {response.status_code}): {detail}{hint}"
        )
    if not response.text.strip():
        return {}
    return response.json()


def wait_for_lro(
    session: Any,
    operation: dict[str, Any],
    *,
    base_url: str,
    service_label: str,
    max_wait_seconds: float = _LRO_MAX_WAIT_SECONDS,
) -> dict[str, Any]:
    """Poll a Google Cloud Long-Running Operation (LRO) until ``done=True``."""
    raw_name = str(operation.get("name", ""))
    if operation.get("done") or "/operations/" not in raw_name:
        if "error" in operation and isinstance(operation["error"], dict):
            raise RegistryError(f"{service_label} operation failed: {operation['error']}")
        return dict(operation.get("response", operation))

    op_name = str(operation["name"]).lstrip("/")
    op_url = f"{base_url}/{op_name}"
    deadline = time.monotonic() + max_wait_seconds
    current = operation
    while not current.get("done"):
        if time.monotonic() >= deadline:
            raise RegistryError(
                f"{service_label} operation {op_name} timed out after {max_wait_seconds:.0f}s"
            )
        time.sleep(_LRO_POLL_INTERVAL_SECONDS)
        current = request_json(session, "GET", op_url, service_label=service_label)
    if "error" in current:
        raise RegistryError(f"{service_label} operation {op_name} failed: {current['error']}")
    return dict(current.get("response", current))
