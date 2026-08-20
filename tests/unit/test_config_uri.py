"""Offline tests for ``load_config_uri`` — the local-path-or-``gs://`` config source.

A staged ``gs://`` config is the portable handle an emitted launch command references, so both
launchers accept it. These tests cover the URI parse, the local-path delegation, and the single
``ConfigError`` funnel. The one network call (GCS fetch) is faked.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scale_forecasting.config import RunConfig, load_config_uri
from scale_forecasting.errors import ConfigError


def _cfg_dict() -> dict[str, Any]:
    return {
        "run_name": "uri test",
        "data": {"source_table": "source_series_native", "horizon": 28},
        "models": ["theta", "holtwinters"],
    }


def test_local_path_delegates_to_load_config(tmp_path) -> None:
    p = tmp_path / "run.json"
    p.write_text(json.dumps(_cfg_dict()))
    cfg = load_config_uri(str(p))
    assert isinstance(cfg, RunConfig)
    assert cfg.run_name == "uri test"


def test_malformed_uri_is_a_config_error_before_any_fetch() -> None:
    # bucket-only, no blob — must fail on parse (never touches the network).
    with pytest.raises(ConfigError, match="malformed config URI"):
        load_config_uri("gs://bucket-only")


def test_gs_uri_fetches_parses_and_validates(monkeypatch) -> None:
    payload = json.dumps(_cfg_dict())
    seen: dict[str, str] = {}

    class _FakeBlob:
        def __init__(self, name: str) -> None:
            seen["blob"] = name

        def download_as_text(self) -> str:
            return payload

    class _FakeBucket:
        def __init__(self, name: str) -> None:
            seen["bucket"] = name

        def blob(self, name: str) -> _FakeBlob:
            return _FakeBlob(name)

    class _FakeClient:
        def bucket(self, name: str) -> _FakeBucket:
            return _FakeBucket(name)

    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", lambda: _FakeClient())

    cfg = load_config_uri("gs://code-bkt/runs/run-abc.json")
    assert isinstance(cfg, RunConfig)
    assert cfg.run_name == "uri test"
    assert seen == {"bucket": "code-bkt", "blob": "runs/run-abc.json"}


def test_gs_uri_bad_json_surfaces_as_config_error(monkeypatch) -> None:
    class _FakeClient:
        def bucket(self, name: str):  # noqa: ANN202 - test double
            class _B:
                def blob(self, _n: str):  # noqa: ANN202
                    class _Bl:
                        def download_as_text(self) -> str:
                            return "{not json"

                    return _Bl()

            return _B()

    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", lambda: _FakeClient())

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config_uri("gs://code-bkt/runs/bad.json")


def test_gs_uri_fetch_failure_surfaces_as_config_error(monkeypatch) -> None:
    class _FakeClient:
        def bucket(self, name: str):  # noqa: ANN202 - test double
            raise RuntimeError("boom")

    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", lambda: _FakeClient())

    with pytest.raises(ConfigError, match="cannot read config URI"):
        load_config_uri("gs://code-bkt/runs/x.json")
