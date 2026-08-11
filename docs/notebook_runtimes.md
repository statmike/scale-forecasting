# Notebook runtimes — Python versions, locally and on Colab

The notebooks are first-class run drivers: two run fully local, the rest **orchestrate** the
Dataproc / Ray / BigQuery work over ADC. This page maps **which Python version each notebook needs
and how it behaves under 3.11 vs 3.12** — both locally and on **Colab Enterprise** — and documents
the two runtime templates Terraform ships.

## The one load-bearing fact: the project targets Python 3.11

`pyproject.toml` sets `requires-python = ">=3.11,<3.13"`, but **3.11 is the tested, supported
default everywhere**. This is deliberate:

- **Vertex Ray parity** — the client-side Ray must match the cluster's Ray version (2.47); on Python
  3.11 the supported versions are 2.42 / 2.47. A mismatched client floats to an unsupported Ray and
  the `JobSubmissionClient` handshake hangs (HTTP 524).
- **Dataproc packed-venv** — the packed virtual-env shipped to Serverless is built against 3.11.

So **3.11 is the default everywhere** — the `uv` project kernel, the runtime image, every cluster.
There is exactly **one** exception, notebook 01's *interactive* Spark Connect path (below): its
Colab template is Python **3.12** (Dataproc 3.0 Connect workers are 3.12 and refuse mismatched
minors), and the notebook's bootstrap `pip install -e .` must be allowed to run there. That single
requirement is why the ceiling is `<3.13` rather than `<3.12` — the core is pure-Python and
3.12-safe, and NB01 pulls only the `[spark]` extra (not `[models]`). Nothing else is tested or
supported on 3.12.

## Running locally

Every notebook except `model_playground` needs **ADC** and the `SF_*` identity. The `model_playground`
needs neither — it runs the model suite on in-memory sample data with no cloud calls.

```bash
gcloud auth application-default login          # ADC — required for all but model_playground
uv sync                                        # core deps incl. ipykernel + matplotlib
uv run python -m ipykernel install --user --name scale-forecasting --display-name "scale-forecasting (uv)"
```

Then select the **scale-forecasting (uv)** kernel. Full kernel guidance (and the interactive-Connect
3.12 note) lives in [running_and_reviewing.md](./running_and_reviewing.md#notebooks-and-kernels) —
this page is the per-notebook mapping.

### The `SF_*` identity (read by `Settings.resolve()`)

Source of truth: [`src/scale_forecasting/settings.py`](../src/scale_forecasting/settings.py). Every
value comes straight from `terraform output` (see
[running_and_reviewing.md](./running_and_reviewing.md#prerequisites)).

| Variable | Required | Default |
|----------|----------|---------|
| `SF_PROJECT_ID` | yes | — |
| `SF_CONNECTION` | yes | — |
| `SF_WAREHOUSE_URI` | yes | — |
| `SF_DATASET_ID` | no | `scale_forecasting` |
| `SF_REGION` | no | `us-central1` |

## Per-notebook mapping

| Notebook | Cloud compute | Python | 3.11 vs 3.12 | Extra | Colab template |
|----------|---------------|--------|--------------|-------|----------------|
| `model_playground` | none (fully local) | 3.11 | either — pure local, no cloud | none | `sf-main` |
| `02_bigquery_native` | BigQuery | 3.11 | either — orchestration only | none | `sf-main` |
| `03_combo_and_ensemble` | Dataproc (submit) + BigQuery | 3.11 | either — submits a Spark batch ∥ BQ, runs on-cluster | none | `sf-main` |
| `05_spark_naive` | Dataproc (submit) | 3.11 | either — submits a batch, runs on-cluster | none | `sf-main` |
| `07_scale_review` | BigQuery (read-only) | 3.11 | either — reads the registry views | none | `sf-main` |
| `06_spark_multi` | Dataproc (submit) | 3.11 | either — client submits, cluster runs | `[spark]` | `sf-main` |
| `04_ray_on_vertex` | Ray on Vertex | **3.11** | **3.11 required** — client↔cluster Ray parity (2.47); a 3.12 client floats to an unsupported Ray → HTTP 524 | `[ray]` | `sf-main` |
| `01_spark_via_connect` | Dataproc Spark Connect | **3.12** (interactive) | **3.12 for interactive Connect**; from 3.11 it uses the remote-batch fallback | `[spark]` | `sf-spark-connect` |

**How to read the "3.11 vs 3.12" column.** Most notebooks are *orchestration* — they submit work to
Dataproc / Ray / BigQuery, which runs on-cluster Python, so the local/kernel minor doesn't change the
result. The two that genuinely care:

- **`04_ray_on_vertex` needs 3.11.** Not the Python minor per se — the *Ray package* version. On 3.11
  the client resolves to a Vertex-supported Ray (2.47) that matches the cluster. Run it on `sf-main`.
- **`01_spark_via_connect` needs 3.12 for the interactive path.** Dataproc 3.0 Spark Connect workers
  run Python 3.12 and Connect refuses mismatched minors (`PYTHON_VERSION_MISMATCH`). From a **3.11**
  kernel the notebook falls back to the **remote-batch** path (`main.run(cfg)` with no injected
  session) — the *identical* engine on-cluster, same `run_id`, same results — so 01 still works on
  `sf-main`, just non-interactively. For live interactive Connect, use `sf-spark-connect` (3.12).

Note NB01's own bootstrap cell installs the package *plain* (not `[spark]`), so the `sf-spark-connect`
template is what carries `dataproc-spark-connect` — see below.

## On Colab Enterprise

Terraform ships **two runtime templates** (a template is a blueprint for the VM a runtime runs on).
They are **free at rest** — a template costs nothing until you start a runtime from it, and runtimes
idle-shutdown — so both are created **on by default** (`create_colab_templates = true`).

| Template | Python | Extra | Use it for |
|----------|--------|-------|------------|
| `sf-main` | **3.11** | `.[ray]` | notebooks 02–07 + `model_playground` (everything but interactive 01) |
| `sf-spark-connect` | **3.12** | `.[spark]` | **only** notebook 01's interactive Spark Connect |

After `terraform apply`, the template resource names are surfaced as outputs
(`colab_main_runtime_template_id`, `colab_spark_runtime_template_id`).

### One-click open + run (no environment cell)

Each notebook carries a header with a **Run in Colab Enterprise** badge. The click-path is:

1. Click the badge — it imports the notebook straight into Colab Enterprise.
2. Create/pick a runtime from the matching template (`sf-main` for everything but interactive 01;
   `sf-spark-connect` for notebook 01's interactive Spark Connect).
3. **Run all.**

There is **no `SF_*` environment cell to fill in**. Terraform bakes the full run identity
(`SF_PROJECT_ID`, `SF_CONNECTION`, `SF_WAREHOUSE_URI`, `SF_DATASET_ID`, `SF_REGION`, plus the batch /
Ray infra vars) into each template's `software_config.env`, so a freshly-started runtime already
resolves `Settings.resolve()` — the same identity a headless execution gets (below). Running locally
still uses ADC + the `SF_*` vars from `terraform output` as [above](#running-locally); only the Colab
path is env-baked.

### Headless acceptance (verify every notebook runs green)

`scale_forecasting.notebook_acceptance` runs the notebooks **headless** on their templates via the
Vertex AI `NotebookExecutionJob` API (serviceAccount mode — no browser consent), downloads each
executed notebook from GCS, and asserts no cell errored. It's both the acceptance test for a fresh
deploy and the regression guard when a notebook changes. Because the templates carry the same baked
env, a headless run validates exactly what a human gets on open (G1).

Tiers escalate cost (each runs its tier plus the cheaper ones), gated like the other billed smokes:

| Tier | Adds | Gate |
|------|------|------|
| `smoke` (default) | `02`, `07`, `model_playground` (BQ-only / local) | `@gcp` (`SF_PROJECT_ID` + ADC) |
| `batch` | `01`, `03`, `05`, `06` (submit a Dataproc batch) | `SF_ENABLE_NB_BATCH` |
| `full` | `04_ray_on_vertex` (live Ray cluster) | `SF_ENABLE_NB_FULL` |

```bash
# pytest wrapper (reads template ids + runner SA from `terraform output`):
uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py            # smoke
SF_ENABLE_NB_BATCH=1 uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py
SF_ENABLE_NB_FULL=1  uv run --active pytest -m gcp tests/integration/test_notebook_acceptance.py

# or the CLI directly:
uv run --active python -m scale_forecasting.notebook_acceptance --tier smoke \
  --project "$(terraform -chdir=terraform/main output -raw project_id)" \
  --main-template  "$(terraform -chdir=terraform/main output -raw colab_main_runtime_template_id)" \
  --spark-template "$(terraform -chdir=terraform/main output -raw colab_spark_runtime_template_id)" \
  --service-account "$(terraform -chdir=terraform/main output -raw runner_sa)" \
  --gcs-output "gs://$(terraform -chdir=terraform/main output -raw code_bucket)"
```

**Packages.** By default each notebook's **bootstrap cell** (`git clone` + `pip install -e .[extra]`)
installs the package on first cell-run — correct, just a little slower on a cold runtime. To
pre-install at runtime-creation time instead (faster cold start), set `install_via_post_startup =
true` on the module (off by default: the post-startup-script field is deprecated and some orgs block
it on new templates).

**Network.** Templates attach the project VPC by default (`colab_attach_network = true`): the
greenfield stack builds a *custom* VPC and has no `default` network, so a public runtime would 404
looking for one. Attached runtimes get egress with **no external IP** via the Cloud NAT + Private
Google Access from `modules/network` (also compatible with a `compute.vmExternalIpAccess = DENY` org
policy). Set `colab_attach_network = false` only in a brownfield project that has a usable `default`
network and permits external IPs.

### Maintenance note — the py311 pin

The Terraform provider can't set a template's Python version (issue
[hashicorp/terraform-provider-google#25217](https://github.com/hashicorp/terraform-provider-google/issues/25217)),
so **both** templates are pinned via a tolerant REST PATCH inside the module — `sf-main` to `py311`
and `sf-spark-connect` to `py312`. `sf-spark-connect` is pinned *explicitly* even though `py312` is
Colab's current Latest: if it were left on Latest it would silently drift to `py313` when Colab
advances Latest, re-breaking NB01 interactive Connect with `PYTHON_VERSION_MISMATCH` (Dataproc 3.0
Connect workers run 3.12). When a Python version reaches **end-of-availability**, Colab auto-upgrades
templates to Latest; bump `colab_main_release_name` / `colab_spark_release_name` and re-apply
**before** that date to keep each pin. Track the supported versions in the
[Colab Enterprise runtime docs](https://cloud.google.com/colab/docs/runtimes).

---

See also: [running_and_reviewing.md](./running_and_reviewing.md) (the operator loop + kernel setup) ·
[configuration_reference.md](./configuration_reference.md) (every config field) ·
[deploying_on_gcp.md](./deploying_on_gcp.md) (the Terraform).
