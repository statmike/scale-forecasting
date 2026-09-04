"""Offline tests for the principal-resolution seam (`scale_forecasting.identity`).

Only the pure parts are unit-tested — the ADC lookup and the userinfo network call in
`resolve_principal` / `_userinfo_email` are best-effort I/O (marked ``pragma: no cover``).
`_sa_email` is what decides "runner SA → its email for free" vs. "user cred → fall through to
userinfo", and `_without_quota_project` is what keeps that fall-through from 403-ing, so both carry
branch logic worth pinning.
"""

from __future__ import annotations

import types

from scale_forecasting.identity import _sa_email, _without_quota_project


def test_sa_email_reads_service_account_email() -> None:
    creds = types.SimpleNamespace(service_account_email="runner@proj.iam.gserviceaccount.com")
    assert _sa_email(creds) == "runner@proj.iam.gserviceaccount.com"


def test_sa_email_falls_back_to_signer_email() -> None:
    # Some SA credential types expose the identity as signer_email, not service_account_email.
    creds = types.SimpleNamespace(service_account_email=None, signer_email="s@proj.iam")
    assert _sa_email(creds) == "s@proj.iam"


def test_sa_email_default_placeholder_is_unresolved() -> None:
    # A compute-metadata credential reads "default" before its first refresh — not a real email.
    creds = types.SimpleNamespace(service_account_email="default")
    assert _sa_email(creds) is None


def test_sa_email_user_credential_has_no_email() -> None:
    # A user credential (laptop ADC) exposes neither attribute → None, so resolve_principal falls
    # through to the userinfo lookup.
    assert _sa_email(types.SimpleNamespace()) is None


def test_sa_email_empty_string_is_unresolved() -> None:
    assert _sa_email(types.SimpleNamespace(service_account_email="", signer_email="")) is None


class _QuotaCreds:
    """A credential that records the quota project it was asked to drop."""

    def __init__(self, quota_project_id: str | None = "some-unrelated-project") -> None:
        self.quota_project_id = quota_project_id
        self.asked: list[str | None] = []

    def with_quota_project(self, project: str | None) -> _QuotaCreds:
        self.asked.append(project)
        return _QuotaCreds(quota_project_id=project)


def test_without_quota_project_strips_the_header_source() -> None:
    # The userinfo 403: AuthorizedSession sends x-goog-user-project from the credential's quota
    # project, so a caller lacking serviceusage.services.use on *that* project loses the audit line
    # even though userinfo itself needs no project at all.
    creds = _QuotaCreds()
    stripped = _without_quota_project(creds)
    assert creds.asked == [None]
    assert stripped.quota_project_id is None


def test_without_quota_project_passes_through_credentials_that_cannot_strip() -> None:
    # Not every credential type implements with_quota_project; those go out unchanged rather than
    # failing, because attribution is advisory and must never block the operation it annotates.
    creds = types.SimpleNamespace(service_account_email="runner@proj.iam.gserviceaccount.com")
    assert _without_quota_project(creds) is creds


def test_without_quota_project_falls_back_when_stripping_raises() -> None:
    class _Raises:
        def with_quota_project(self, project: str | None) -> object:
            raise RuntimeError("credential refuses to be copied")

    creds = _Raises()
    assert _without_quota_project(creds) is creds
