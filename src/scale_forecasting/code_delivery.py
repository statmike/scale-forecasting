"""Deliver the ``scale_forecasting`` source to remote Spark/Ray workers at run time.

Every remote compute path needs the *package* on its workers — but the package deliberately is NOT
baked into the runtime container image (``docker/Dockerfile``: deps ship in the image, code ships at
runtime, so a code edit never needs an image rebuild and no stale code can hide). There are three
mechanisms, and they all read the same ``src/`` from this one place, so worker code cannot drift
between them:

* **Dataproc batch** (`submit`) uploads the zip to GCS and passes it on ``python_file_uris``.
* **Interactive Spark Connect** (notebook 01) adds the same zip to the session with
  ``spark.addArtifacts(path, pyfile=True)`` — Connect only accepts *local* files, so the notebook
  writes it to a temp path first.
* **Ray on Vertex** (`ray_submit`) ships ``src/`` as the job's ``runtime_env`` ``working_dir``
  instead of a zip, because that is the handle the Ray Jobs API takes.

The zip contains only ``scale_forecasting/`` at its root (no ``src/`` prefix), so it is importable
the moment it lands on ``sys.path`` — the same layout the Terraform seed module relies on.

The Ray mechanism carries the *dependencies* too (`build_runtime_env`), which the other two do not:
a Spark surface gets its locked environment from the container image or a venv archive, resolved
before anything is submitted, while Ray's ``runtime_env`` is a single dict describing both halves
and Vertex's prebuilt Ray image has none of our packages. Splitting that dict across two modules
would mean neither one could be read as "what the job receives", so the deps ride along here.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any

# The package root that gets zipped. This file is ``…/src/scale_forecasting/code_delivery.py``, so
# parents[1] is the ``src/`` dir, whose child ``scale_forecasting/`` is the importable package.
# ``_SRC_DIR`` is also what Ray ships as ``working_dir`` (it contains ``scale_forecasting/``, so
# ``python -m scale_forecasting.ray_entry`` resolves on the cluster). The locked cluster deps live
# at docker/requirements.txt — the same file the container image pins, reused for the on-cluster uv.
_SRC_DIR = Path(__file__).resolve().parent.parent
_PKG_ROOT = _SRC_DIR / "scale_forecasting"
_REPO_ROOT = _SRC_DIR.parent

# Where those locked deps can be, in priority order. A launch point has one of two shapes: a full
# repo checkout (local dev, CI, a notebook clone), where ``docker/`` sits beside ``src/``; or a
# **src-only delivery**, where something rsynced just the code somewhere and there is no repo root
# above it. Composer is the second shape — ``make composer-sync`` puts ``src/`` in the environment's
# plugins prefix, so it copies ``docker/`` in beside it and the second candidate is the one that
# resolves. Searching both keeps ONE file the source of truth instead of shipping a second copy
# inside the package for the delivery mechanisms to keep in sync.
_REQUIREMENTS_CANDIDATES = (
    _REPO_ROOT / "docker" / "requirements.txt",
    _SRC_DIR / "docker" / "requirements.txt",
)

# torch's x86_64/linux pin is the CUDA-12.6 local build (``torch==2.13.0+cu126``) for the Vertex T4
# driver — that ``+cuXXX`` local version exists ONLY on the PyTorch index, never on PyPI. The
# on-cluster runtime_env pip install must therefore add the SAME ``--extra-index-url`` the image
# build uses (docker/Dockerfile), or the pin 404s ("No matching distribution for torch==…+cu126")
# and the whole Ray job fails at env setup. ``--extra-index-url`` (not ``--index-url``) keeps PyPI
# primary for every other package; pip honors the option line when it appears in the requirements
# list Ray materializes. Source of truth for the URL is docker/Dockerfile — keep them in lockstep.
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def build_package_zip() -> tuple[bytes, str]:
    """Build the ``scale_forecasting`` package zip in memory.

    Returns ``(data, code_hash)`` where ``code_hash`` is an 8-char md5 of the bytes — a non-crypto
    object-name tag so a code change is a new artifact (no in-place-overwrite races). The walk is
    sorted for a deterministic archive (same source → same hash).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(_PKG_ROOT.rglob("*.py")):
            zf.write(path, arcname=str(path.relative_to(_SRC_DIR)))
    data = buf.getvalue()
    code_hash = hashlib.md5(data).hexdigest()[:8]  # noqa: S324 - non-crypto object-name tag
    return data, code_hash


# Packages the cluster already provides — never reinstall these via runtime_env or a pip-installed
# version would fight the one baked into the Vertex Ray image (the cluster's Ray is pinned at create
# via ``ray_version``, and requirements.txt may pin a newer Ray than Vertex supports, so swapping it
# out from under the running head/workers breaks the job). Matched on the PEP-508 project name.
_CLUSTER_PROVIDED = frozenset({"ray"})


def _requirements_path() -> Path:
    """The first `_REQUIREMENTS_CANDIDATES` entry that exists, or a `FileNotFoundError` naming both.

    A bare ``FileNotFoundError`` on the repo-shaped path is a genuinely confusing failure on a
    launch point that has no repo — it names a directory the operator never chose and cannot
    create. Say what was searched and what fixes it.
    """
    for candidate in _REQUIREMENTS_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(p) for p in _REQUIREMENTS_CANDIDATES)
    raise FileNotFoundError(
        f"locked cluster requirements not found (searched: {searched}). On a launch point that "
        "holds only src/ — a Composer environment, say — deliver them with `make composer-sync`, "
        "which copies docker/ in beside src/."
    )


def _requirements_packages() -> list[str]:
    """Parse ``docker/requirements.txt`` into a package-spec list, dropping cluster-provided deps.

    The uv-exported file is ``name==version [; marker]`` lines interleaved with ``# via`` comment
    blocks; we keep only the requirement lines and skip anything whose project name is in
    `_CLUSTER_PROVIDED` (see its note — Ray must come from the image, not pip).
    """
    packages: list[str] = []
    for raw in _requirements_path().read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Project name = everything before the first version/extra/marker delimiter.
        name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower()
        if name in _CLUSTER_PROVIDED:
            continue
        packages.append(line)
    return packages


def build_runtime_env() -> dict[str, Any]:
    """The Ray ``runtime_env``: current ``src/`` + on-cluster deps installed by uv (pure).

    Delivers code at RUNTIME the way the Spark path uploads a ``src/`` zip: ``working_dir`` is the
    package root, so ``python -m scale_forecasting.ray_entry`` imports the code that was just
    submitted, not anything baked into the image — the "same code local and in the cloud" seam.

    Ray always runs on Vertex's prebuilt image (a custom node image fails Vertex Ray GPU-node
    provisioning), which lacks our deps — so **uv** installs the requirements package **list minus
    Ray** into the per-job virtualenv (`_requirements_packages` — Ray stays the image's pinned
    version rather than being swapped by a conflicting pin). uv resolves from the same pinned
    requirements export the container is built from, so the on-cluster env is byte-aligned with
    every other surface, and it installs markedly faster than pip; Ray 2.47's runtime_env uv plugin
    self-bootstraps uv into the prebuilt image if absent, so nothing has to preinstall it.
    ``--extra-index-url`` adds the PyTorch CUDA wheels (`_TORCH_CUDA_INDEX`, mirroring
    docker/Dockerfile): the x86_64/linux torch pin is a ``+cu126`` local build that only resolves
    from that index, and ``--index-strategy unsafe-best-match`` lets uv pick it from the extra index
    even though the same name exists on PyPI (uv's default first-index strategy would stop at PyPI
    and never find the ``+cu126`` build). PyPI stays the primary index.
    """
    return {
        "working_dir": str(_SRC_DIR),
        "uv": {
            "packages": _requirements_packages(),
            # Passed through to ``uv pip install`` — this REPLACES the plugin default
            # ``["--no-cache"]``, so re-list it. See the docstring for why the extra index +
            # unsafe-best-match are needed for the ``+cu126`` torch build.
            "uv_pip_install_options": [
                "--no-cache",
                "--extra-index-url",
                _TORCH_CUDA_INDEX,
                "--index-strategy",
                "unsafe-best-match",
            ],
            # Run ``uv pip check`` after install so dependency drift fails loudly at env setup
            # rather than as a confusing runtime import error — the byte-alignment guarantee.
            "uv_check": True,
        },
    }


def write_package_zip(dest_dir: str | Path | None = None) -> Path:
    """Write the package zip to a local file and return its path.

    For the Spark Connect path, which needs a local file to ``addArtifacts``. The filename carries
    the code hash so re-adding after an edit is a distinct artifact. ``dest_dir`` defaults to a temp
    directory (the caller need not manage cleanup for a short-lived notebook run).
    """
    import tempfile

    data, code_hash = build_package_zip()
    base = Path(dest_dir) if dest_dir is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"scale_forecasting-{code_hash}.zip"
    out.write_bytes(data)
    return out
