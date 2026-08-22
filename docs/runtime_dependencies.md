# Runtime dependencies — how each compute surface gets its packages

The same forecasting code runs on four very different compute surfaces — Colab Enterprise runtimes,
Dataproc Serverless batches, Dataproc **clusters**, and Ray on Vertex — and **each has its own native
way of installing software**. This page is the single source of truth for *how the dependencies reach
each surface*, so that a fit running on any of them imports the identical `scale_forecasting` package
and the identical third-party stack (statsmodels, lightgbm, torch, …).

This is about **dependencies** (the third-party libraries + the interpreter). *Your* source code ships
separately at runtime — see [editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md).
For the Python **version** each surface runs, see [version_matrix.md](./version_matrix.md).

## The one source of truth: `docker/requirements.txt`

Every mechanism below resolves back to a single locked dependency set:

- **`docker/requirements.txt`** — the locked, hashed dependency list (core + models + Ray extras).
- **`docker/Dockerfile`** — installs that list into an isolated `python3.11` venv at **`/opt/venv`**
  on `debian:12-slim`, plus the handful of **system** libraries the wheels link against (e.g.
  `libgomp1` for lightgbm). The result is the **shared runtime image**.

Everything else is a delivery detail. The image *is* the dependency set; the packed-venv archive is
built *from* that same image; the notebook bootstrap installs the *same* project extras. There is one
list to bump when a dependency changes, and it is content-addressed — the image and the archive both
rebuild only when `requirements.txt` (or the `Dockerfile`) changes, never on a source edit.

## The matrix

| Surface | Mechanism | What carries the deps | Set by |
|---------|-----------|-----------------------|--------|
| **Dataproc Serverless** (`explode` / `multi` batches) | **Custom container** | the shared runtime image, attached on every submit | `submit.py` (`runtime_config.container_image`) |
| **Spark Connect** (interactive, nb01) | **Custom container** + artifacts | image pinned on the session; code via `addArtifacts` | notebook session cell |
| **Ray on Vertex** (nb04) | **Custom container** | the same image as the Ray cluster's worker image | `ray_submit.py` |
| **Dataproc cluster** (`spark_mode="cluster"`) | **Packed-venv archive** | a `venv-pack` tarball attached to the job (`#env`) | `dataproc_cluster.py` + `compute.spark_deps` |
| **Colab Enterprise** (all notebooks) | **Project install** | `pip install -e .[extra]` in the runtime (bootstrap or post-startup) | notebook bootstrap cell / template |
| **Local dev** | **`uv` project env** | `uv sync` resolves the same `pyproject.toml` | `pyproject.toml` |

The rest of this page walks each row: *why* that mechanism, and what to know operationally.

## Dataproc Serverless — custom container

Serverless batches accept a **custom container image**, so the cleanest path is to hand them the
shared runtime image directly. `submit.py` attaches it on **every** batch via
`runtime_config.container_image`, which overrides the base runtime's interpreter and libraries for
both driver and executors. Nothing is installed at launch — the image is already the environment — so
startup is fast and the executed environment is byte-for-byte the one that was built and tested.

This is the default Spark path (`compute.spark_deps = "container"` is the Serverless behaviour) and
needs no per-run configuration beyond `SF_CONTAINER_IMAGE` (populated from `terraform output`).

## Spark Connect (interactive notebook) — container + shipped artifacts

The interactive Connect session (nb01) reuses the **same container image** — pinned on the `Session`
so its executors carry the deps — and additionally ships the **code** with
`spark.addArtifacts(zip, pyfile=True)`, because an interactive session runs a closure pickled on the
notebook kernel. The dep story is identical to Serverless; only the code-delivery step differs. Full
detail lives in [notebook_runtimes.md](./notebook_runtimes.md#per-notebook-mapping).

## Ray on Vertex — custom container

A Vertex Ray cluster takes a **worker container image**, and it is the **same shared runtime image**.
`ray_submit.py` points the cluster's workers at it, so a Ray task imports the identical stack a Spark
task does. The image also carries the CUDA-12.6 torch build the GPU path needs (see
[the size note](#known-limits-and-gotchas) below). Because client↔cluster Ray parity is strict, the
image's Python (3.11) and the pinned Ray version are load-bearing — see
[version_matrix.md](./version_matrix.md#why-311-everywhere--the-four-parity-boundaries).

## Dataproc cluster — packed-venv archive

**A Dataproc cluster cannot use a custom container** (that is a Serverless-only feature). So the
cluster path delivers the *same* locked environment a different way: a **relocatable venv archive**.

How it is produced and consumed:

1. **Pack (build time).** After the shared image is built, the build packs its `/opt/venv` with
   [`venv-pack`](https://jcristharif.com/venv-pack/) into `env.tar.gz` and uploads it to the code
   bucket at **`envs/<requirements-hash>.tar.gz`**. Building the archive *from the image* guarantees
   it is the exact same dependency set — no second resolve, no drift. (See `docker/cloudbuild.yaml`;
   Terraform runs this automatically on apply and exposes the URI as the `venv_archive_uri` output.)
2. **Attach (submit time).** `dataproc_cluster.py` attaches the archive to the PySpark job with
   `--archives=<uri>#env` and points Spark at the unpacked interpreter via
   `spark.pyspark.python` / `spark.pyspark.driver.python = ./env/bin/python`. Dataproc unpacks the
   tarball into `./env` on every node before the job starts; `venv-pack` has already rewritten the
   venv's internal paths so `./env/bin/python` works from wherever it lands.

Config + identity:

- **`compute.spark_deps`** selects the mechanism. On a cluster it must be **`"packed_venv"`** (the
  default); `"container"` is rejected on a cluster because clusters can't take a container.
- The archive URI is resolved from **`SF_VENV_ARCHIVE`** (or `BatchInfra.venv_archive_uri` /
  `terraform output venv_archive_uri`). If `spark_deps="packed_venv"` and no archive URI is
  available, submission fails fast with a clear error rather than running against the bare cluster.

> **Why not a cluster init-action `pip install`?** It would re-resolve deps on every cluster create
> (slow, and not guaranteed identical to the tested image), and it would need network egress to PyPI
> from the cluster. The packed archive is a single content-addressed artifact that is provably the
> same environment as the container — one build, reused everywhere.

## Colab Enterprise — project install

A Colab runtime is a general VM; it gets the package the ordinary Python way — **`pip install -e
.[extra]`** against the checked-out project. Two timings:

- **Bootstrap cell (default).** Each notebook's first cell does `git clone` + `pip install -e
  .[extra]`, installing on first run. Simple and self-contained; a little slower on a cold runtime.
- **Post-startup script (opt-in).** Set `install_via_post_startup = true` to pre-install at
  runtime-creation time for a faster cold start. Off by default — the field is deprecated and some
  orgs block it.

The runtime's Python is pinned to 3.11 to match every other surface. Full Colab detail (templates,
baked `SF_*` env, headless acceptance) is in [notebook_runtimes.md](./notebook_runtimes.md).

## Local dev — the `uv` project env

Locally, `uv sync` resolves `pyproject.toml` (the same dependency intent `requirements.txt` locks for
the clusters) into a project venv. This is what the notebooks and the offline test gate run on. See
[running_and_reviewing.md](./running_and_reviewing.md).

## Known limits and gotchas

- **System libraries are not in the venv archive.** `venv-pack` captures the Python packages, not the
  OS packages the container installs via `apt` (e.g. `libgomp1`, which **lightgbm** links against).
  The custom-container paths (Serverless, Ray) have these baked in; a Dataproc **cluster** relies on
  its base image providing them. If a cluster fit fails with a missing `.so` (e.g. `libgomp.so.1`),
  the fix is a cluster **init action** that `apt-get install`s the library — the venv archive can't
  carry it.
- **The archive is large.** The venv includes the CUDA-12.6 **torch** build (the GPU deep-learning
  path), which is on the order of a gigabyte. That is fine for a cluster job (staged once to nodes)
  but is the reason the archive is content-addressed and cached in the code bucket rather than rebuilt
  per run.
- **The packer runs as root.** `/opt/venv` is root-owned while the image's default user is the
  `spark` user (UID 1099), so the build packs the venv as root (`docker run --user root …`). This is
  a build-time detail; nothing about the running job changes.
- **One version knob for all surfaces.** Because every mechanism traces back to `requirements.txt`,
  bump it once and re-apply (or rebuild): the image, the packed archive, and — after a `uv sync` /
  Colab reinstall — the notebook and local envs all move together. Don't pin a dependency in only one
  place; that reintroduces the drift this design exists to prevent.

---

Sources:
[venv-pack](https://jcristharif.com/venv-pack/) ·
[Dataproc Serverless custom containers](https://cloud.google.com/dataproc-serverless/docs/guides/custom-containers) ·
[Dataproc: Python environment on clusters](https://cloud.google.com/dataproc/docs/tutorials/python-configuration)

See also: [architecture.md](./architecture.md) (the compute tracks) ·
[version_matrix.md](./version_matrix.md) (the Python version per surface) ·
[editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md) (how *code* ships at runtime) ·
[deploying_on_gcp.md](./deploying_on_gcp.md) (the Terraform that builds the image + archive).
