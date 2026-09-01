"""How the locked Python environment reaches a Dataproc **cluster**.

A cluster cannot use the Serverless custom container, so the one viable mechanism is a
self-contained venv archive unpacked to an absolute path on every node. That answer is spread
across two surfaces and so has to live in neither of them: the *cluster* carries the archive URI as
metadata and runs the init action that unpacks it (`dataproc_cluster.build_cluster`), while the
*job* points Spark's interpreter at the result (`cluster_submit.build_job`). Split those apart and
the two halves drift; a job whose ``spark.pyspark.python`` disagrees with where the init action
landed the venv is a run that fits nothing, with no error until the first import.

So this module is the whole delivery contract in one place: where the venv lands, the metadata keys
the init action reads, the script itself, the check that the deployment actually configured an
archive (`_resolve_cluster_deps`), and the upload of the script (`_stage_cluster_init`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .batch_infra import _ENV_VENV_ARCHIVE
from .errors import ConfigError

if TYPE_CHECKING:
    from .batch_infra import BatchInfra
    from .config import RunConfig



# Packed-venv delivery: a Dataproc cluster can't use the Serverless custom container, so the locked
# dependency env is shipped as a self-contained venv archive and unpacked to a fixed absolute path
# on every node by a cluster **init action** (below). Job ``archive_uris`` are localized only to the
# *executors'* working dirs — never the client-mode *driver's* CWD (the driver runs on the master) —
# so a relative ``./env/bin/python`` fails for the driver with ``error=2, No such file or dir``.
# The init action sidesteps that: it lands the venv at ``/opt/sf-venv`` on master + workers alike,
# so both driver and executors point at the same absolute interpreter and run the exact same locked
# env as the container path — the model libraries (statsmodels, xgboost, …) live inside it.
_VENV_DIR = "/opt/sf-venv"
_VENV_PYTHON = f"{_VENV_DIR}/bin/python"
_VENV_JOB_PROPERTIES = {
    "spark.pyspark.python": _VENV_PYTHON,
    "spark.pyspark.driver.python": _VENV_PYTHON,
}

# Cluster metadata keys the init action reads to know what to fetch + where to unpack it. Metadata
# rides on the cluster (not the job), so it's available to the init action at create time on every
# node; the archive URI is the same ``venv_archive_uri`` a forecast job would otherwise attach.
_VENV_ARCHIVE_METADATA_KEY = "sf-venv-archive-uri"
_VENV_DIR_METADATA_KEY = "sf-venv-dir"

# The init-action script: on every node at cluster create, download the venv archive named in
# cluster metadata and unpack it to the absolute venv dir. The archive is a plain tar of the venv's
# *contents* (packed with ``tar -C /opt/venv .``), so it extracts straight into the target dir; the
# bundled interpreter + relative ``bin/python`` symlink make it runnable at any absolute path. Fails
# the node (``set -e``) if the metadata is missing or the fetch/unpack errors, so a broken env
# surfaces at create rather than as a silent bare-Python job later.
_CLUSTER_INIT_SCRIPT = f"""#!/bin/bash
set -euo pipefail
ARCHIVE_URI="$(/usr/share/google/get_metadata_value attributes/{_VENV_ARCHIVE_METADATA_KEY})"
VENV_DIR="$(/usr/share/google/get_metadata_value attributes/{_VENV_DIR_METADATA_KEY})"
if [[ -z "${{ARCHIVE_URI}}" || -z "${{VENV_DIR}}" ]]; then
  echo "sf venv init: missing venv cluster metadata" >&2
  exit 1
fi
mkdir -p "${{VENV_DIR}}"
gsutil -q cp "${{ARCHIVE_URI}}" /tmp/sf-venv.tar.gz
tar xzf /tmp/sf-venv.tar.gz -C "${{VENV_DIR}}"
rm -f /tmp/sf-venv.tar.gz
"""


def _resolve_cluster_deps(cfg: RunConfig, infra: BatchInfra) -> str:
    """The packed-venv archive URI a cluster job must attach, per ``compute.spark_deps`` (pure).

    A Dataproc cluster can't use the Serverless custom container, so ``packed_venv`` (the default)
    is the only viable dependency mechanism on a cluster — it requires ``infra.venv_archive_uri``
    (the ``SF_VENV_ARCHIVE`` env / terraform ``venv_archive_uri`` output). ``container`` is a
    Serverless-only mechanism, so requesting it for a cluster is a config error rather than a
    silent run with no model libraries. Raises `ConfigError` on ``container`` or a missing URI.
    """
    spark_deps = cfg.compute.spark_deps
    if spark_deps == "container":
        raise ConfigError(
            "compute.spark_deps='container' is a Dataproc Serverless mechanism, not available on a "
            "Dataproc cluster; use spark_deps='packed_venv' for cluster families"
        )
    if not infra.venv_archive_uri:
        raise ConfigError(
            "a Dataproc cluster forecast job needs the packed-venv archive but none is configured; "
            f"set {_ENV_VENV_ARCHIVE} (or the terraform 'venv_archive_uri' output). Without it the "
            "cluster runs bare Python with no model libraries and every fit fails"
        )
    return infra.venv_archive_uri


def _stage_cluster_init(infra: BatchInfra) -> str:
    """Upload the venv init-action script to the code bucket; return its ``gs://`` URI.

    Mirrors `staging.stage_code`'s pattern. The object name carries the script's md5 so an
    edit to the script is a new object (no in-place-overwrite races) and an unchanged script re-uses
    the same URI across runs. `dataproc_cluster.build_cluster` points a `NodeInitializationAction`
    at the returned
    URI.
    """
    import hashlib

    from google.cloud import storage

    data = _CLUSTER_INIT_SCRIPT.encode("utf-8")
    digest = hashlib.md5(data, usedforsecurity=False).hexdigest()[:8]
    name = f"init/sf-venv-init-{digest}.sh"
    client = storage.Client()
    bucket = client.bucket(infra.code_bucket)
    bucket.blob(name).upload_from_string(data, content_type="text/x-shellscript")
    return f"gs://{infra.code_bucket}/{name}"
