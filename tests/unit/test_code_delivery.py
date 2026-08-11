"""The code-from-outside-container guarantee, made un-regressable (PLANS_PROD §C1).

Q1's architecture ships the ``scale_forecasting`` package **at runtime** on every compute path —
never baked into the container image — so a data scientist edits ``src/`` and re-runs against the
*existing* slow-moving image, with no rebuild and no stale code hiding in the container. These tests
lock that property at every seam:

* the **image** carries dependencies only (no ``COPY src``, no ``pip install`` of the package);
* the **Cloud Build** step and the **Dockerfile** together never add the package to the image;
* every **submit path** delivers the package externally — ``python_file_uris`` (Spark batch / seed /
  smoke) or ``runtime_env.working_dir`` (Ray on Vertex).

A regression in any of these — someone "helpfully" adding ``COPY . /app`` + ``pip install .`` to the
Dockerfile, or dropping the package zip from a submit — would re-bake code into the image and break
the edit→re-run loop silently. These are cheap file/spec assertions that fail loudly instead.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile"
_CLOUDBUILD = _REPO_ROOT / "docker" / "cloudbuild.yaml"
_SRC = _REPO_ROOT / "src"


# --- the image bakes dependencies only, never the package ----------------------


def _dockerfile_directives() -> list[str]:
    """The Dockerfile's instructions, one per logical directive: ``\\``-continuations joined and
    comments/blanks stripped (comments describe the contract in prose — ``# NOT the package`` — and
    must not trip the source-code checks). Joining continuations keeps a multi-line ``RUN pip
    install ... \\ -r requirements.txt`` a single directive so it reads as one install."""
    joined = re.sub(r"\\\s*\n", " ", _DOCKERFILE.read_text())
    lines = joined.splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def test_dockerfile_never_copies_source() -> None:
    # No ``COPY src`` / ``COPY . ...`` / ``ADD src`` — the package must not enter the build context.
    for directive in _dockerfile_directives():
        upper = directive.upper()
        if upper.startswith(("COPY ", "ADD ")):
            # The only COPY we permit is requirements.txt (the locked dep set), nothing under src/.
            assert "requirements.txt" in directive, f"image bakes source via: {directive}"
            assert " src" not in f" {directive}", f"image bakes src/ via: {directive}"


def test_dockerfile_never_installs_the_package() -> None:
    # No ``pip install .`` / ``-e .`` / ``scale-forecasting`` / a built wheel — deps only. (The
    # ``via scale-forecasting`` provenance comments live in requirements.txt, not the Dockerfile.)
    text = "\n".join(_dockerfile_directives())
    bakes = re.search(r"pip install[^\n]*(\s\.(\s|$)|\s-e\b|scale[_-]forecasting|\.whl)", text)
    assert not bakes, (
        "Dockerfile installs the package into the image — code must ship at runtime, not baked in"
    )


def test_dockerfile_only_installs_locked_requirements() -> None:
    # The one install path is ``-r requirements.txt`` (the exported lock, package excluded).
    installs = [d for d in _dockerfile_directives() if "pip install" in d and "-r" in d]
    assert installs, "expected a `pip install -r requirements.txt` layer"
    assert all("requirements.txt" in d for d in installs)


def test_requirements_lock_does_not_install_the_package() -> None:
    # ``# via scale-forecasting`` lines are pip-compile provenance comments (deps pulled in *by* the
    # package), not an install directive. No *uncommented* line may name the package as a dep.
    reqs = _REPO_ROOT / "docker" / "requirements.txt"
    for ln in reqs.read_text().splitlines():
        code = ln.split("#", 1)[0].strip()
        assert "scale-forecasting" not in code and "scale_forecasting" not in code, (
            f"requirements.txt lists the package as a dependency: {ln!r}"
        )


def test_cloudbuild_builds_only_the_dockerfile() -> None:
    # The build step points at docker/Dockerfile; there is no separate "pip install the package" or
    # source-copy step in the pipeline. Guards against the image gaining code via cloudbuild.
    text = _CLOUDBUILD.read_text()
    assert "docker/Dockerfile" in text
    assert not re.search(r"pip install[^\n]*scale[_-]forecasting", text)


# --- every submit path delivers the package externally -------------------------


def test_spark_batch_ships_the_package_via_python_file_uris() -> None:
    import pytest

    pytest.importorskip("google.cloud.dataproc_v1")  # the [spark] extra; parity with test_submit
    from scale_forecasting.settings import Settings
    from scale_forecasting.submit import BatchInfra, build_batch

    infra = BatchInfra(
        code_bucket="code-bkt",
        container_image="us-docker.pkg.dev/p/repo/runtime:latest",
        compute_sa="compute@p.iam.gserviceaccount.com",
        subnetwork_uri="projects/p/regions/us-central1/subnetworks/sf",
    )
    settings = Settings(
        project_id="p",
        connection="p.us-central1.conn",
        warehouse_uri="gs://b/w",
        dataset_id="ds",
        region="us-central1",
    )
    batch = build_batch(
        infra=infra,
        settings=settings,
        engine="explode",
        package_uri="gs://code-bkt/runs/pkg-1234.zip",
        launcher_uri="gs://code-bkt/runs/spark_main.py",
        config_uri="gs://code-bkt/runs/run-abc.json",
    )
    # The package rides python_file_uris (runtime delivery), never the container_image.
    assert list(batch.pyspark_batch.python_file_uris) == ["gs://code-bkt/runs/pkg-1234.zip"]
    assert "scale" not in batch.runtime_config.container_image.rsplit("/", 1)[-1]


def test_ray_runtime_env_ships_the_package_via_working_dir() -> None:
    from scale_forecasting.ray_submit import build_runtime_env

    env = build_runtime_env()
    # working_dir is the real on-disk src/ (uploaded by Ray at submit) — code ships with the job.
    assert env["working_dir"].endswith("/src")
    assert Path(env["working_dir"]).is_dir()


# --- the Terraform submit paths zip src/ and deliver it at runtime -------------


def _tf(module: str) -> str:
    return (_REPO_ROOT / "terraform" / "main" / "modules" / module / "main.tf").read_text()


def test_seed_module_zips_src_and_ships_via_python_file_uris() -> None:
    text = _tf("seed")
    assert 'data "archive_file" "package"' in text  # zips src/ at apply time
    assert "/src" in text  # the archived source dir
    assert "python_file_uris" in text  # delivered to the batch at runtime


def test_smoke_module_zips_src_and_ships_via_py_files() -> None:
    text = _tf("smoke")
    assert 'data "archive_file" "package"' in text
    assert "/src" in text
    assert "--py-files=" in text  # the gcloud batches submit form of python_file_uris


def test_no_terraform_module_bakes_source_into_the_image() -> None:
    # The container module builds the deps-only image; no module may COPY src into an image build.
    modules = _REPO_ROOT / "terraform" / "main" / "modules"
    for tf in modules.rglob("*.tf"):
        assert "COPY src" not in tf.read_text(), f"{tf} bakes src/ into an image"


# --- the shared package-zip builder (batch + interactive Spark Connect both use it) --------------


def test_build_package_zip_is_root_importable() -> None:
    import io
    import zipfile

    from scale_forecasting import code_delivery

    data, code_hash = code_delivery.build_package_zip()
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    # The package sits at the zip ROOT (no src/ prefix) so it imports the moment it's on sys.path —
    # the same layout python_file_uris (batch) and addArtifacts(pyfile=True) (Connect) both need.
    assert "scale_forecasting/__init__.py" in names
    assert not any(n.startswith("src/") for n in names)
    assert all(n.endswith(".py") for n in names)
    assert len(code_hash) == 8


def test_build_package_zip_is_deterministic() -> None:
    from scale_forecasting import code_delivery

    d1, h1 = code_delivery.build_package_zip()
    d2, h2 = code_delivery.build_package_zip()
    # Same source → same hash → a stable artifact name (no in-place-overwrite races).
    assert h1 == h2 and d1 == d2


def test_write_package_zip_round_trips(tmp_path: Path) -> None:
    from scale_forecasting import code_delivery

    out = code_delivery.write_package_zip(tmp_path)
    assert out.exists() and out.parent == tmp_path
    _, code_hash = code_delivery.build_package_zip()
    # Filename carries the hash so re-adding after an edit is a distinct artifact.
    assert code_hash in out.name
