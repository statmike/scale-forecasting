# Custom container for Dataproc Serverless Spark (Arc B) — the shared runtime image.
#
# This one image is the DEPENDENCY-delivery mechanism for every compute path: the seed job (B0.4),
# the `explode`/`multi` forecast engines (B2), and Ray on Vertex (B4). It carries the full locked
# dependency set (core + [models] + [ray]) from docker/requirements.txt — but NOT the
# `scale_forecasting` package itself.
#
# CODE ships at RUNTIME, not in the image: the Dataproc batch passes src/ as a zip via
# python_file_uris (Terraform rebuilds+uploads it every apply — see modules/seed). So the image is
# slow-moving (rebuild only when deps change), and a code edit never requires an image rebuild —
# no stale code can hide in the container.
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
# needs to be able to import `pyspark` at runtime — which the mounted runtime provides — and the
# code that ships via python_file_uris. `pyspark` is therefore intentionally NOT in
# requirements.txt (the export omits the [spark] extra); installing it here would shadow the
# runtime's version.

FROM debian:12-slim

# --- OS packages required/recommended by Dataproc Serverless -------------------------------------
# procps: `ps` for Spark. tini: init/reaper. libjemalloc2: allocator. python3.11 is Debian 12's
# system Python (matches requires-python >=3.11,<3.12); pip + venv to build an isolated user env.
# libgomp1: OpenMP runtime that lightgbm hard-links against (xgboost bundles its own); without it
# lightgbm import fails and every lightgbm cell degrades to an error at runtime.
RUN apt-get update && apt-get install --no-install-recommends -y \
        procps \
        tini \
        libjemalloc2 \
        libgomp1 \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# jemalloc: Spark on Serverless expects it available for preload.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# --- NVIDIA driver visibility (Ray-on-Vertex GPU pool) -------------------------------------------
# On a T4 worker, Ray counts GPUs via NVML (pynvml.nvmlInit() → nvmlDeviceGetCount()), which loads
# libnvidia-ml.so.1 from the HOST driver. Ray-on-Vertex runs on GKE, which injects the host driver
# into the container at the GKE convention path /usr/local/nvidia (nvidia-smi in .../bin, the driver
# libs incl. libnvidia-ml.so.1 + libcuda.so in .../lib64). Two things are needed for our bare-Debian
# image to SEE that injected driver:
#   1. NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES so the injection happens at all.
#   2. /usr/local/nvidia/{bin,lib64} on PATH / LD_LIBRARY_PATH so nvidia-smi + NVML can be loaded —
#      without this, NVML fails ("libnvidia-ml.so.1 not found"), Ray sees 0 GPUs, and every
#      fractional-GPU task (NeuralProphet: {'CPU':1,'GPU':<frac>}) is unschedulable forever.
# We do NOT install the CUDA toolkit here — torch ships its own CUDA-12.6 userspace wheels (see
# docker/requirements.txt) and only needs the injected DRIVER at runtime. All of this is inert on
# CPU-only nodes (Dataproc Serverless, Ray CPU pool): the env vars do nothing without a GPU, and the
# /usr/local/nvidia path simply doesn't exist there (a missing dir on PATH/LD_LIBRARY_PATH is a
# no-op). So this stays one shared image across every compute path (G1).
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/usr/local/nvidia/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

# --- user Python environment ---------------------------------------------------------------------
# An isolated venv keeps our deps off the system Python. Serverless launches the driver/executors
# with PYSPARK_PYTHON, so point it at this interpreter.
ENV VENV=/opt/venv
RUN python3.11 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
ENV PYSPARK_PYTHON=${VENV}/bin/python

# Install the LOCKED dependency set only (core + [models] + [ray]; NOT [spark], NOT the package).
# See docker/requirements.txt for the export command + why pyspark and the package are excluded.
# Copy only requirements.txt so this layer caches on dependency changes — never on a source edit
# (the package isn't here; it ships at runtime via python_file_uris).
WORKDIR /opt/scale-forecasting
COPY docker/requirements.txt ./requirements.txt
# torch is pinned to the CUDA-12.6 local build (torch==2.13.0+cu126) for the Vertex T4 driver — that
# local version only exists on the PyTorch index, so add it alongside PyPI. --extra-index-url (not
# --index-url) keeps PyPI primary for every other package; the pin's exact +cu126 version is only
# satisfiable from the PyTorch index, so there's no ambiguity about where torch comes from.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r requirements.txt

# --- the mandatory spark user (UID/GID 1099) -----------------------------------------------------
RUN groupadd --gid 1099 spark \
    && useradd --uid 1099 --gid 1099 --create-home --shell /bin/bash spark
USER spark

# tini as PID 1 (Serverless overrides the command; ENTRYPOINT is the reaper).
ENTRYPOINT ["/usr/bin/tini", "--"]
