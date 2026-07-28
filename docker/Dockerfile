# Custom container for Dataproc Serverless Spark (Arc B) — the shared runtime image.
#
# This one image is the dependency-delivery mechanism for every Spark path: the seed job (B0.4)
# and, later, the `explode`/`multi` forecast engines (B2). It carries the `scale_forecasting`
# package + its CORE deps, so the identical code runs local == Managed Spark == (eventually) Ray.
#
# Dataproc Serverless custom-container rules (https://cloud.google.com/dataproc-serverless/docs/
# guides/custom-containers) baked in below:
#   * Base a slim Debian; do NOT install Spark, PySpark, Java, or Python-the-runtime for Spark —
#     the Serverless runtime mounts its own Spark + JDK at launch. We only provide the *user*
#     Python env + libraries.
#   * The image MUST contain `procps` (Spark shells out to `ps`), and it's strongly recommended to
#     ship `tini` (PID-1 reaping) and `libjemalloc2` (allocator Spark tunes for).
#   * A `spark` user with UID **1099** and GID **1099** must exist and own the runtime dirs.
#   * Build for **linux/amd64** and push to Artifact Registry (see docker/cloudbuild.yaml).
#
# The spark-bigquery connector is BUILT-IN on Serverless (do not add it here); this image just
# needs to be able to import `pyspark` at runtime — which the mounted runtime provides — and our
# pure code. `pyspark` is therefore intentionally NOT pip-installed (it's an optional extra used
# only for local UDF testing); installing it here would shadow the runtime's version.

FROM debian:12-slim

# --- OS packages required/recommended by Dataproc Serverless -------------------------------------
# procps: `ps` for Spark. tini: init/reaper. libjemalloc2: allocator. python3.11 is Debian 12's
# system Python (matches requires-python >=3.11,<3.12); pip + venv to build an isolated user env.
RUN apt-get update && apt-get install --no-install-recommends -y \
        procps \
        tini \
        libjemalloc2 \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# jemalloc: Spark on Serverless expects it available for preload.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# --- user Python environment ---------------------------------------------------------------------
# An isolated venv keeps our deps off the system Python. Serverless launches the driver/executors
# with PYSPARK_PYTHON, so point it at this interpreter.
ENV VENV=/opt/venv
RUN python3.11 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
ENV PYSPARK_PYTHON=${VENV}/bin/python

# Install the package + its CORE deps only (numpy/pandas/scipy/holidays/google-cloud-*, protobuf).
# The [spark]/[ray]/[models] extras are deliberately excluded: pyspark is runtime-mounted, and the
# heavy model extras are added when B2/B4 need them (keeps the base image lean). Copy pyproject +
# sources, then editable-install so the layer caches on dependency changes, not every source edit.
WORKDIR /opt/scale-forecasting
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# --- the mandatory spark user (UID/GID 1099) -----------------------------------------------------
RUN groupadd --gid 1099 spark \
    && useradd --uid 1099 --gid 1099 --create-home --shell /bin/bash spark
USER spark

# tini as PID 1 (Serverless overrides the command; ENTRYPOINT is the reaper).
ENTRYPOINT ["/usr/bin/tini", "--"]
