"""Package the ``scale_forecasting`` source for delivery to remote Spark/Ray workers.

Every remote compute path needs the *package* on its workers — but the package deliberately is NOT
baked into the runtime container image (``docker/Dockerfile``: deps ship in the image, code ships at
runtime, so a code edit never needs an image rebuild and no stale code can hide). The two delivery
mechanisms both build the SAME zip from this one place, so worker code can never drift between them:

* **Dataproc batch** (:mod:`.submit`) uploads the zip to GCS and passes it on ``python_file_uris``.
* **Interactive Spark Connect** (notebook 01) adds it to the session with
  ``spark.addArtifacts(path, pyfile=True)`` — Connect only accepts *local* files, so the notebook
  writes the zip to a temp path first.

The zip contains only ``scale_forecasting/`` at its root (no ``src/`` prefix), so it is importable
the moment it lands on ``sys.path`` — the same layout the Terraform seed module relies on.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

# The package root that gets zipped. This file is ``…/src/scale_forecasting/code_delivery.py``, so
# parents[1] is the ``src/`` dir, whose child ``scale_forecasting/`` is the importable package.
_SRC_DIR = Path(__file__).resolve().parent.parent
_PKG_ROOT = _SRC_DIR / "scale_forecasting"


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
