# Version matrix — one Python everywhere, and why

This page is the single source of truth for **which Python, Spark, and Ray versions run where**, and
**why the whole system is pinned to Python 3.11**. If you're debugging a `PYTHON_VERSION_MISMATCH`, a
Ray `JobSubmissionClient` handshake hang, or a runtime-version choice, start here.

## TL;DR

**Python 3.11 is the single supported version across every surface** — the `uv` project kernel, the
custom container image, every Dataproc batch, every Ray cluster + client, the Spark Connect session,
and every Colab template. This is not a coincidence to preserve casually: it's the one constraint that
keeps four independent driver↔worker boundaries in parity. Deviating on any one of them breaks that
surface.

## The matrix

| Surface | Where it runs | Runtime version | Base runtime Python | **Effective Python** | Spark / Ray | How Python is set |
|---------|---------------|-----------------|---------------------|----------------------|-------------|-------------------|
| **Project / kernel** | local `uv`, Colab `sf-main` | — | — | **3.11** | — | `pyproject.toml` `requires-python = ">=3.11,<3.12"`; Colab template PATCHed to `py311` |
| **Custom container** | attached to batch + Spark Connect | — | — | **3.11** | Spark 3.5.x (from base) | `docker/Dockerfile` — `python3.11` venv on `debian:12-slim` |
| **Dataproc batch** (`explode` / `multi`) | Serverless | **2.2** (default) | 3.12 | **3.11** ← *container wins* | Spark 3.5.3 | container image attached on **every** submit (`submit.py`), overriding base |
| **Spark Connect** (nb01, interactive) | Serverless session | **2.3** | 3.11 | **3.11** | Spark 3.5.3 | runtime 2.3 base is already 3.11 **and** container attached |
| **Ray on Vertex** (nb04) | Vertex Ray cluster | Ray **2.47** | 3.11 | **3.11** | Ray 2.47.1 | `ray_submit.py` pins `python_version="3.11"`, `ray_version="2.47"` |
| **BigQuery-native** (nb02: ARIMA_PLUS, TimesFM) | BigQuery engine | — | — | n/a (SQL) | — | no client Python on the compute path |

> **"Effective Python" is what your code actually executes on.** For batch it's the *container's*
> 3.11, not the base runtime's 3.12 — the attached image replaces the runtime's interpreter for both
> driver and executors. That's why batch can sit on runtime 2.2 (base 3.12) and still run 3.11 cleanly.

## Why 3.11 everywhere — the four parity boundaries

Each of these is a place where two processes exchange **pickled Python objects** and therefore must
share a Python **minor** version (3.11 ≠ 3.12 for pickle/`applyInPandas`/cloudpickle purposes):

1. **Dataproc batch: container ↔ itself.** The container sets both driver and executor Python, so this
   is self-consistent by construction — the base runtime (2.2 = 3.12) never touches your code because
   the image overrides it. This is why the batch default runtime version is unremarkable: the
   container, not the runtime, decides.

2. **Spark Connect: Colab kernel ↔ session executors.** The `applyInPandas` fan-out pickles the
   group-runner closure **on the notebook kernel** (`sf-main`, py3.11) and unpickles it on the session
   executors. So the executors must be 3.11. Runtime **2.3**'s workers are Python 3.11; runtime **3.0**'s
   are Python 3.12 → `PYTHON_VERSION_MISMATCH`. **This is the whole reason nb01 uses 2.3, not 3.0.**

3. **Ray on Vertex: client ↔ cluster.** The `JobSubmissionClient` handshake (`GET /api/version`)
   requires the **client Ray version to equal the cluster's**. Vertex offers Ray only for a fixed set
   (2.9.3 / 2.33.0 / 2.42.0 / 2.47.1), and **on Python 3.11 only 2.42 or 2.47 are available**. A
   version-skewed client doesn't error cleanly — the dashboard proxy **hangs** (→ HTTP 524). So the
   `[ray]` extra is capped and `ray_submit.py` defaults `ray_version=2.47`, `python_version=3.11`.

4. **Same code locally and under Composer.** The `uv` project itself is `>=3.11,<3.12`, so a developer's
   local kernel, the CI kernel, and the Composer runner all resolve the same interpreter — the code
   that runs locally is byte-identical to what runs in the cloud.

## The Spark Connect 2.x vs 3.x decision (the "Connect 2 / Connect 3" question)

Both Dataproc runtime **2.3** and **3.0** support Spark Connect. We use **2.3**. The trade:

| | Runtime **2.3** ✅ (chosen) | Runtime **3.0** ❌ |
|---|---|---|
| Spark | 3.5.3 | 4.0.x |
| Base worker Python | **3.11** | 3.12 |
| Parity with `sf-main` kernel (py3.11) | ✅ holds | ❌ `PYTHON_VERSION_MISMATCH` on `applyInPandas` |
| Parity with custom container (py3.11) | ✅ | ❌ |
| Needs a new py3.12 container + py3.12 kernel? | no | **yes** — a whole second toolchain |

Runtime 3.0 (Spark 4) is the newer line and drops Jupyter in favor of Connect-only sessions — but
adopting it would force the **entire stack to Python 3.12**, which is blocked anyway: **Vertex Ray
caps at Python 3.11** (no 3.12 cluster image), and the client must match the cluster. So going to 3.0
would *split* the toolchain (3.12 for Spark/batch, 3.11 for Ray) — reintroducing the two-container,
two-template mess that the single `sf-main` template deliberately removed. **3.11-everything is the
only configuration that unifies all tracks under one container and one template — 2.3 is a
consequence of that, not a compromise.**

Revisit only if Vertex Ray ships a Python 3.12 cluster image; until then 3.11 is load-bearing.

## Where these numbers live in the code

| Value | File |
|-------|------|
| `requires-python = ">=3.11,<3.12"` | `pyproject.toml` |
| `python3.11` venv, `debian:12-slim` | `docker/Dockerfile` |
| Batch default runtime `"2.2"`; container attached on submit | `src/scale_forecasting/submit.py` |
| Spark Connect session `version = "2.3"` + container | `notebooks/01_spark_via_connect.ipynb` (session cell) |
| Ray `python_version="3.11"`, `ray_version="2.47"` | `src/scale_forecasting/ray_submit.py` |
| Colab template pinned to `py311` (REST PATCH) | `terraform/main/modules/colab/main.tf` |

## Sources

- [Dataproc Serverless runtime 2.2](https://docs.cloud.google.com/dataproc-serverless/docs/concepts/versions/spark-runtime-2.2) — Spark 3.5.3, Python 3.12
- [Dataproc Serverless runtime 2.3](https://docs.cloud.google.com/dataproc-serverless/docs/concepts/versions/spark-runtime-2.3) — Spark 3.5.3, **Python 3.11**
- [Dataproc Serverless runtime 3.0](https://docs.cloud.google.com/dataproc-serverless/docs/concepts/versions/spark-runtime-3.0) — Spark 4.0.x, Python 3.12
- [Vertex AI: Ray on Vertex setup](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/set-up) — supported Ray versions / Python 3.11 cap
- [Colab Enterprise runtimes](https://cloud.google.com/colab/docs/runtimes) — template Python versions

---

See also: [notebook_runtimes.md](./notebook_runtimes.md) (per-notebook Python + template mapping) ·
[architecture.md](./architecture.md) (the compute tracks) ·
[deploying_on_gcp.md](./deploying_on_gcp.md) (the Terraform).
