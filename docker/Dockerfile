# Custom container for Dataproc Serverless Spark — the shared runtime image.
#
# This one image is the DEPENDENCY-delivery mechanism for every compute path: the seed job,
# the `explode`/`multi` forecast engines, and Ray on Vertex. It carries the full locked
# dependency set (core + [models] + [ray]) resolved straight from uv.lock — but NOT the
# `scale_forecasting` package itself.
#
# ONE TOOL, WHOLE STACK: uv manages BOTH the interpreter and the dependencies here, from the exact
# same uv.lock + .python-version that pin local dev, CI, and the notebooks. So the interpreter is
# patch-identical (3.11.15) and the dependency set is byte-identical everywhere — no second resolve,
# no drift. docker/requirements.txt is a derived, human-readable export of the same lock (regenerate
# with `make lock`); the image builds from the lock directly.
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
# code that ships via python_file_uris. `pyspark` is therefore intentionally NOT in the installed
# set (the [spark] extra is omitted); installing it here would shadow the runtime's version.

FROM debian:12-slim

# --- OS packages required/recommended by Dataproc Serverless -------------------------------------
# procps: `ps` for Spark. tini: init/reaper. libjemalloc2: allocator. libgomp1: OpenMP runtime that
# lightgbm hard-links against (xgboost bundles its own); without it lightgbm import fails and every
# lightgbm cell degrades to an error at runtime. ca-certificates: TLS for uv's downloads. NOTE we no
# longer install a system Python — uv brings the exact interpreter (see below), so debian's python
# is irrelevant and never used.
RUN apt-get update && apt-get install --no-install-recommends -y \
        procps \
        tini \
        libjemalloc2 \
        libgomp1 \
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
# We do NOT install the CUDA toolkit here — torch ships its own CUDA-12.6 userspace wheels (routed
# from the PyTorch index by [tool.uv.sources]) and only needs the injected DRIVER at runtime. All of
# this is inert on CPU-only nodes (Dataproc Serverless, Ray CPU pool): the env vars do nothing
# without a GPU, and the /usr/local/nvidia path simply doesn't exist there (a missing dir on
# PATH/LD_LIBRARY_PATH is a no-op). So this stays one shared image across every compute path.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/usr/local/nvidia/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

# --- uv: the single tool that manages BOTH the interpreter and the dependencies ------------------
# Pin uv itself so the build tool is reproducible (matches the version dev + CI use).
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

ENV VENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=${VENV} \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Copy ONLY the lockfiles first so this heavy layer caches on dependency changes only, never on a
# source edit (the package isn't installed here; it ships at runtime via python_file_uris).
WORKDIR /opt/scale-forecasting
COPY .python-version pyproject.toml uv.lock ./
# Install the exact interpreter named by .python-version (3.11.15), then the LOCKED dependency set
# (core + [models] + [ray]; NOT [spark], NOT the project). --frozen uses uv.lock verbatim (no
# re-resolve); --no-install-project keeps the package out of the image; [tool.uv.sources] routes the
# torch cu126 build from the PyTorch index automatically (no manual --extra-index-url needed).
RUN uv python install "$(cat .python-version)" \
    && uv sync --frozen --no-dev --no-install-project --extra models --extra ray

# --- make /opt/venv fully SELF-CONTAINED (for the Dataproc-cluster packed-venv archive) ----------
# A Dataproc *cluster* can't take a custom container, so the cluster path ships this /opt/venv as a
# tar archive attached to the job. A venv normally only *symlinks* to its base interpreter, so a
# packed venv would dangle on a bare cluster node (which has no uv-installed Python). uv's Python is
# a python-build-standalone build (relative RPATHs, its own libpython + stdlib), so we copy it
# in-tree and repoint bin/python to a RELATIVE in-tree symlink: the result runs from any unpack path
# with NO external Python dependency — and still runs in place here for Serverless/Ray. One layout,
# both consumers. (Proven: unpacks + imports on a node with no uv present.)
RUN set -eux; \
    PYHOME="$(${VENV}/bin/python -c 'import sys; print(sys.base_prefix)')"; \
    cp -a "$PYHOME" "${VENV}/_python"; \
    rm -f "${VENV}/bin/python" "${VENV}/bin/python3" "${VENV}/bin/python3.11"; \
    ln -s ../_python/bin/python3.11 "${VENV}/bin/python"; \
    ln -s python "${VENV}/bin/python3"; \
    ln -s python "${VENV}/bin/python3.11"; \
    sed -i "s|^home = .*|home = ${VENV}/_python/bin|" "${VENV}/pyvenv.cfg"; \
    rm -rf /opt/uv-python; \
    "${VENV}/bin/python" -c "import sys; print('self-contained venv:', sys.executable, sys.version.split()[0])"

# --- make the interpreter discoverable on the DEFAULT system PATH ---------------------------------
# Serverless points PYSPARK_PYTHON at an absolute path and every in-container shell inherits the ENV
# PATH below, so those paths always find /opt/venv/bin. A managed head bootstrap (Ray on Vertex),
# however, does NOT inherit this image's ENV PATH — it invokes `python`/`ray` by bare name on the
# base system PATH, where — now that the self-contained venv is the ONLY interpreter — nothing
# resolves. These wrappers restore that discoverability the way the removed apt python used to. They
# must be WRAPPER SCRIPTS, not symlinks: a symlink into the venv resolves through to the in-tree base
# binary (/opt/venv/_python/bin/python3.11), which does not activate the venv (its base_prefix is
# in-tree) and so cannot import the installed packages; exec'ing /opt/venv/bin/python keeps sys.prefix
# on the venv. Placed in both /usr/local/bin (default PATH front) and /usr/bin (hard-coded callers).
# Purely additive — the in-container ENV-PATH and PYSPARK_PYTHON callers are unchanged, and the
# packed-venv archive (which ships only /opt/venv) is untouched.
RUN set -eux; \
    for d in /usr/local/bin /usr/bin; do \
        for n in python python3 python3.11; do \
            printf '#!/bin/sh\nexec %s/bin/python "$@"\n' "${VENV}" > "$d/$n"; \
            chmod 0755 "$d/$n"; \
        done; \
        printf '#!/bin/sh\nexec %s/bin/ray "$@"\n' "${VENV}" > "$d/ray"; \
        chmod 0755 "$d/ray"; \
    done; \
    env -i PATH=/usr/local/bin:/usr/bin:/bin python -c "import ray, sys; print('bare-PATH import ok:', sys.executable, ray.__version__)"

# Serverless launches the driver/executors with PYSPARK_PYTHON; point it at this interpreter.
ENV PATH="${VENV}/bin:${PATH}"
ENV PYSPARK_PYTHON=${VENV}/bin/python

# --- the mandatory spark user (UID/GID 1099) -----------------------------------------------------
RUN groupadd --gid 1099 spark \
    && useradd --uid 1099 --gid 1099 --create-home --shell /bin/bash spark
USER spark

# tini as PID 1 (Serverless overrides the command; ENTRYPOINT is the reaper).
ENTRYPOINT ["/usr/bin/tini", "--"]
