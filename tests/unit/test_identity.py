"""Offline tests for the principal-resolution seam (`scale_forecasting.identity`).

Only the pure credential→email selection (`_sa_email`) is unit-tested — the ADC lookup and the
userinfo network call in `resolve_principal` / `_userinfo_email` are best-effort I/O (marked
``pragma: no cover``). `_sa_email` is what decides "runner SA → its email for free" vs. "user cred →
fall through to userinfo", so it carries the branch logic worth pinning.
"""

from __future__ import annotations

import types

from scale_forecasting.identity import _sa_email


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
