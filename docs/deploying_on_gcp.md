# Deploying on GCP — a reviewer's guide to the Terraform

This is the deep-dive companion to the [README quickstart](../README.md#deploy-on-gcp). It exists so
you can **read the Terraform before you run it**: what gets created, which GCP services it uses and
how, why each permission is granted and who uses it, and how to fit the deployment into an existing
locked-down project instead of a fresh one.

The Terraform itself is heavily commented — every module header explains its own *what* and *why*.
This document is the map that ties them together. If a claim here and a module comment ever disagree,
the module comment (next to the resource) wins.

- **Terraform lives in** [`terraform/`](../terraform/); the operator runbook is
  [`terraform/README.md`](../terraform/README.md).
- **One rule to keep in mind:** Terraform owns the *containers* (project, dataset, buckets,
  connection, network, service accounts, roles). The **application** owns the *tables* — the six
  registry/data tables are defined once in `src/scale_forecasting/registry/ddl.py` and created by
  `registry.bq.ensure_tables()` at run time, so there is a single source of truth for the DDL and no
  HCL↔Python drift.

---

## The shape: two stages

```
terraform/
├── bootstrap/   # Stage 1 — run once. Creates the project (optional) + the state bucket.
└── main/        # Stage 2 — everything else. State lives in the bucket stage 1 made.
    ├── main.tf         # wires the modules together (read this top-to-bottom)
    ├── variables.tf    # every input + its default (the toggles live here)
    ├── outputs.tf      # the handful of values the app + operator consume after apply
    └── modules/        # one capability per module (mirrors the Python one-file-one-capability rule)
```

**Why two stages?** Terraform needs somewhere to store its state. We want that somewhere to be a GCS
bucket (durable, shareable), but a bucket can't hold the state that describes its own creation — a
chicken-and-egg. Stage 1 (**bootstrap**) runs with *local* state and creates just two things: the
project (optionally) and the state bucket. Stage 2 (**main**) then stores its state in that bucket
and builds the rest. You run bootstrap once; after that you only ever touch `main`.

Bootstrap is deliberately tiny and free at rest — an empty versioned bucket costs nothing, and it
never mutates your `gcloud`/ADC default project (the create-project call is org/folder-scoped by
design).

---

## What one `terraform apply` builds

Read `terraform/main/main.tf` top-to-bottom — it's ordered by dependency, and Terraform infers the
rest of the order from which module's output feeds the next. `apis` comes first (nothing works until
the services are on); everything else depends on it.

| Module | Creates | Notes |
|--------|---------|-------|
| `apis` | Enables the ~15 Google APIs the platform uses | `disable_on_destroy = false` so a teardown never yanks APIs other work relies on |
| `iam` | The two service accounts + their least-privilege roles + three custom roles | The heart of "why permissions" — see [Permissions](#permissions-why-each-is-granted-and-who-uses-it) |
| `storage` | Three GCS buckets: `warehouse`, `artifacts`, `code` | Separate buckets, not one with prefixes — see [Storage](#storage-three-buckets-on-purpose) |
| `bigquery` | The dataset + the BigLake (Cloud Resource) connection | Tables are **not** here — the app creates them. Connection exists for the Iceberg source variant |
| `budget` | A monthly cost budget with 50/90/100% alert thresholds | A budget *alerts*, it does not *cap*. Safety net before any spend |
| `container` | The Artifact Registry Docker repo **and the Cloud Build run that fills it** | Owns the *repo* and builds + pushes the shared runtime image on apply (`build_image`, content-addressed on `docker/`) — see [The runtime image](#the-runtime-image) |
| `network` | VPC + subnet + firewall + PSA peering + Cloud NAT + PSC-I attachment | Serverless compute needs a private-access subnet; Ray needs the private path — see [Networking](#networking-what-each-piece-is-for) |
| `composer` | *(gated, off by default)* Composer 3 (Airflow) environment | The only real at-rest cost (~$300–400/mo). Start/stop with one variable |
| `seed` | *(on by default)* The Dataproc Serverless batch that materializes the example dataset | Runs once on the first apply; `terraform apply` submits **and waits** for the batch. See [The example dataset](#the-example-dataset) |

The **infrastructure** — everything except the image build, the seed batch, and `composer` — is
**effectively free at rest**: empty buckets, an empty dataset, service accounts, a connection, and
network plumbing all cost nothing until compute runs. Two on-by-default costs run on the first apply:
the **runtime-image build** and the **example-data seed** (both below), each a one-time step.

### The runtime image

`build_image` defaults to **true**, so the first `terraform apply` doesn't just create the empty
Artifact Registry repo — it **fills it**. The `container` module runs `gcloud builds submit` against
the repo's own [`docker/cloudbuild.yaml`](../docker/cloudbuild.yaml), building the shared Spark/Ray
runtime image from [`docker/Dockerfile`](../docker/Dockerfile) and pushing it to the repo. This
matters because *everything downstream pulls that image* — the seed batch below, and every forecast
engine. Without it, a fresh deploy with `run_seed = true` would fail: the seed batch would try to
pull an image that doesn't exist yet. One apply now does the whole chain: repo → image → seed.

- **It's content-addressed on `docker/`.** The build is triggered by an md5 of the `Dockerfile` and
  `requirements.txt` (the two files that define the image), so it runs on the **first** apply and
  re-runs **only** when those change. A source-code edit does **not** rebuild it — application code
  ships to jobs at runtime via `python_file_uris` (see the seed module), never baked into the image.
- **Cost/mechanics.** A Cloud Build run is a few minutes and cents of build time. The image is
  `linux/amd64` (Dataproc Serverless requires amd64; Cloud Build's default workers are amd64, so a
  plain `docker build` produces the right architecture — no emulation).
- **Requirements.** The machine running Terraform needs the `gcloud` CLI (Terraform shells out to it).
  If `enable_apis = false`, an admin must have already enabled `cloudbuild.googleapis.com`.
- **Escape hatch.** Set `build_image = false` if you build and push the image yourself (CI or an
  air-gapped registry) — then push it to the tag the seed/engines consume (`seed_image_tag`, default
  `latest`) before running any compute. The equivalent manual command is documented in the `container`
  module header and `docker/cloudbuild.yaml`.

### The example dataset

`run_seed` defaults to **true**, so a fresh deploy comes with data to forecast against immediately —
the "solution-in-a-box" promise. On the **first** `terraform apply` the `seed` module submits a
Dataproc Serverless Spark batch that:

- generates **100,000 deterministic synthetic time series** (`seed_num_series`, default `100000`;
  reproducible from `seed_master_seed`), and
- writes them to **both** source tables — `source_series_iceberg` (managed Apache Iceberg on GCS) and
  `source_series_native` (native BigQuery) — from a **single generated panel** (Spark `.cache()`s the
  DataFrame and writes it twice), so the series are **byte-identical across formats**. That's the
  point: you can benchmark Iceberg vs native storage on the *same* data.

Two things worth knowing about cost and re-runs:

- **`terraform apply` blocks** until the batch reaches a terminal state (~8.5 min of compute at 100k,
  measured at ~$0.15; the provider wait is set to 60 min to cover provisioning too).
- **The batch is content-addressed** — its id embeds the series count, run label, and an md5 of the
  seed source. Terraform won't re-run an existing batch, so the seed runs **once** and does *not*
  re-spend on later applies unless you change `seed_num_series`, `seed_run_label`, or the seed code.
  Reseeds are deliberate, never per-apply.
- **Re-seeding soon after a `direct` seed?** The Iceberg variant clears with `DELETE WHERE TRUE`,
  which is blocked by the Storage Write API's ~90-min streaming buffer right after a `direct` write —
  so an immediate re-seed should use `seed_write_method = "indirect"` (no buffer) or wait it out. The
  native variant is unaffected (it clears with `TRUNCATE`).

To skip the example data entirely (you'll point runs at your own source table), set
`run_seed = false`. To smoke-test cost/runtime first, set `seed_num_series = 100` (cents, ~2 min),
review, then rerun at `100000`. Select one format with `seed_variant = "iceberg"` or `"native"`
(default `"both"`).

> The **same generator** produces the local playground's sample panel — see the
> [local quickstart](../README.md#quickstart). `playground.sample_data()` calls the identical
> `generate_panel()` with the same master seed, just 3 series in-memory instead of 100k written to
> BigQuery. So what you explore locally is a small slice of the same deterministic dataset the cloud
> seed materializes (G1: same code path local and at scale).

---

## Greenfield vs brownfield

The same deployment serves two very different situations, controlled by a handful of `create_*`
toggles (all default to the greenfield/quickstart behavior).

- **Greenfield** — a fresh project you fully control (the 5-minute quickstart). Leave every toggle at
  its default; Terraform creates everything.
- **Brownfield** — a locked-down org where admins already own the project, the network, the SAs, or
  the APIs. Flip off what they manage and *pass the existing resource in by variable*. The module
  then creates nothing and just threads your value through to the outputs the jobs consume.

| Variable | Default | Turn off / change when… | What you must supply instead |
|----------|---------|--------------------------|------------------------------|
| `enable_apis` | `true` | An admin already enabled the APIs | Nothing — just ensure they're on |
| `create_service_accounts` | `true` | You bring your own SAs | `runner_sa_email` + `compute_sa_email` (and your admin owns their grants) |
| `create_network` | `true` | Your org already manages a VPC | `subnetwork_uri` — a subnet with **Private Google Access** + an **internal-ingress** firewall rule |
| `create_composer` | `false` | You want scheduled DAG runs | Nothing — flip **on** (starts the ~$300–400/mo meter); flip off to stop it |
| `build_image` | `true` | You build/push the runtime image yourself (CI / air-gapped) | Push your image to the `seed_image_tag` before running compute |
| `run_seed` | `true` | You'll bring your own source table (skip the example data) | Flip **off**; then point runs at your own `source_series_*` table |
| `create_project` *(bootstrap)* | `true` | Your org pre-creates projects | An existing `project_id` |

The BYO pattern is identical across `apis`, `iam`, `network`, `composer`, and `seed`: `create = false`
→ the module builds nothing and passes your existing resource through. A locked-down org owns its own
VPC/SAs/APIs and this deployment stays out of the way.

**One brownfield gotcha worth calling out:** if you bring your own network, the subnet *must* have
Private Google Access enabled and an internal-ingress firewall rule — Dataproc Serverless executors
talk to each other on arbitrary ports and reach Google APIs (BigQuery, GCS, Artifact Registry) over
private access with no external IPs. The greenfield `network` module sets this up for you; a BYO
subnet without it will fail at batch-submission time, not at apply time.

---

## GCP services, and how each is used

The `apis` module enables exactly these, grouped by what they're for:

**Data + lineage**
- **BigQuery** (`bigquery.googleapis.com`) — the run registry (three native tables + backtest OOF),
  the BigQuery-native models (`ARIMA_PLUS`, `TimesFM`, SQL-only), and the example
  input tables. This is the system's spine: every run's config, metrics, forecasts, and artifact
  links land here.
- **BigLake / Cloud Resource connection** (`bigqueryconnection.googleapis.com`) — the managed-Iceberg
  source variant (`source_series_iceberg`) reads/writes its GCS files *through* this connection's
  Google-managed service agent, not through the caller. The registry tables are native and need no
  connection; the connection exists only for the Iceberg input format.
- **Cloud Storage** (`storage.googleapis.com`) — three buckets (below).

**Python compute (one runtime per run)**
- **Dataproc Serverless** (`dataproc.googleapis.com`) — the Spark engine and the seed batch. No
  cluster to manage; you submit a batch and it runs.
- **Vertex AI** (`aiplatform.googleapis.com`) — Ray on Vertex. The runner SA creates an autoscaling
  Ray cluster (a Vertex `PersistentResource`), runs the job, and tears it down.

**Networking for that compute**
- **Compute Engine** (`compute.googleapis.com`) — the networking substrate: the VPC, subnet,
  firewall, Cloud NAT, and the PSC-I network attachment.
- **Service Networking** (`servicenetworking.googleapis.com`) — Private Services Access peering, the
  private path Vertex Managed Ray needs to reach the cluster over internal IPs.

**Image supply chain**
- **Artifact Registry** (`artifactregistry.googleapis.com`) — holds the one shared Spark/Ray runtime
  image, so the *same* code + deps run local == Dataproc == Ray.
- **Cloud Build** (`cloudbuild.googleapis.com`) — builds that image from `docker/Dockerfile` on the
  first apply (`build_image`, on by default).

**Orchestration + governance**
- **Composer** (`composer.googleapis.com`) — Composer 3 (Airflow), gated off by default; schedules
  and fans out the pipeline when you turn it on.
- **Cloud Billing + Budgets** (`cloudbilling.googleapis.com`, `billingbudgets.googleapis.com`) — link
  the project to billing and create the budget/alerts.
- **IAM + Resource Manager + Service Usage** (`iam`, `cloudresourcemanager`, `serviceusage`) — service
  accounts, project-level role bindings, and API enablement itself.

### Storage: three buckets on purpose

`warehouse`, `artifacts`, and `code` are separate buckets rather than one bucket with folder
prefixes, because **GCS applies IAM, versioning, lifecycle, and force-destroy at the bucket level** —
"folders" are just name prefixes, not real boundaries:

- **`warehouse`** is the decisive one: the BigLake connection's service agent is granted object access
  scoped to *this bucket only*. Fold everything into one bucket and that least-privilege grant would
  also expose the code and model artifacts.
- **`artifacts`** carry lineage (a forecast row points back to the exact fitted model) — they must
  never be force-destroyed and want no aggressive TTL.
- **`code`** is a derivable deploy artifact (source of truth is GitHub) — it tolerates `force_destroy`
  and a TTL, so it's the **only** bucket with a noncurrent-version lifecycle rule (prune archived
  versions past the 3 most recent after 30 days) — its per-deploy churn doesn't accrete cost, while
  `warehouse`/`artifacts` keep every version. One bucket can't hold both retention stances at once.

All three enforce **uniform bucket-level access** (no legacy ACLs) and **public-access prevention =
`enforced`** — anonymous (`allUsers` / `allAuthenticatedUsers`) grants are hard-blocked regardless of
org policy, so your data and models can never be made public by a stray ACL.

Cost is identical either way (GCS bills per byte + operations, not per bucket), so the split is free
and buys policy isolation.

### Networking: what each piece is for

The `network` module looks large, but each resource earns its place:

- **VPC + custom subnet** with **Private Google Access** — fresh projects here have no `default`
  network, and serverless compute needs a subnet that can reach Google APIs over private access with
  no external IPs.
- **Internal-ingress firewall** (subnet CIDR only) — Dataproc Serverless driver/executors talk to
  each other on arbitrary ports.
- **Private Services Access (PSA) peering** — Vertex Managed Ray provisions the cluster in a Google
  tenant project and peers it into this VPC; without PSA the cluster only gets a public dashboard
  endpoint whose origin is unreachable off-cluster.
- **PSA-range ingress firewall** — the Ray dashboard's proxy→origin hop is sourced from the reserved
  peering range, not the subnet CIDR, so it needs its own rule (this is what fixed the historical
  `524` on the Ray Jobs handshake).
- **Cloud NAT** — outbound-only internet for a VPC-attached client with no external IP (many orgs deny
  external IPs org-wide), so it can `pip install` the Ray SDK. No inbound exposure.
- **PSC-I network attachment** — the newer private path for Vertex Managed Ray; Vertex attaches an
  interface into this VPC to reach the cluster head node. `ACCEPT_AUTOMATIC` means the attachment
  auto-accepts Vertex's tenant project, so the service agent only needs to *consume* it.

Dataproc Serverless needs only the subnet + Private Google Access; the PSA/PSC-I/NAT machinery exists
for the Ray path.

---

## Permissions: why each is granted, and who uses it

This is the section a reviewer usually cares about most. The design principle throughout is
**least privilege**: where a predefined role would over-grant, we define a **custom role** with
exactly the permissions needed. All access is via **ADC + impersonation — no service-account keys,
ever**.

### The two service accounts

| SA | Who runs as it | What it does |
|----|----------------|--------------|
| `scale-forecasting-runner` | The orchestrator (you locally, or Composer) | Reads/writes BigQuery, submits Dataproc/Ray jobs, and creates/tears down its own Ray cluster |
| `scale-forecasting-compute` | Attached to Dataproc/Ray **workers** | BigQuery data + GCS artifacts only — no job-submission or cluster-lifecycle power |

The runner is allowed to **impersonate** the compute SA (`roles/iam.serviceAccountUser`) so it can
attach it to worker jobs — again, no keys.

### Runner SA roles

| Role | Why | Custom? |
|------|-----|---------|
| `roles/bigquery.dataEditor` | Write registry rows + create the registry tables | predefined |
| `roles/bigquery.jobUser` | Run queries / load jobs | predefined |
| **`sfConnectionDelegate`** | Get + use + **delegate** the BigLake connection (delegate is what lets it create managed-Iceberg tables *through* the connection's agent) | **custom** |
| `roles/storage.objectAdmin` | Read/write the warehouse + artifacts + code buckets | predefined |
| `roles/dataproc.editor` | Submit Dataproc Serverless batches | predefined |
| `roles/aiplatform.user` | Submit Ray-on-Vertex jobs (get/list clusters) | predefined |
| **`sfRayClusterManager`** | Create + delete + get + list the Ray cluster it runs on | **custom** |

### Compute SA roles

| Role | Why | Custom? |
|------|-----|---------|
| `roles/bigquery.dataEditor` | Read `source_series`, write results | predefined |
| `roles/bigquery.jobUser` | Run queries | predefined |
| `roles/bigquery.readSessionUser` | Storage Read API — the spark-bigquery connector reads the input | predefined |
| **`sfConnectionDelegate`** | Get + use + delegate the BigLake connection | **custom** |
| `roles/storage.objectAdmin` | Read/write model artifacts | predefined |
| `roles/dataproc.worker` | Batch runtime SA: logs, metrics, staging | predefined |
| `roles/artifactregistry.reader` | Pull the shared runtime image | predefined |

### Why three custom roles instead of predefined ones

Each custom role exists because the one permission we actually need ships *only* inside a much broader
predefined role that would over-grant:

- **`sfConnectionDelegate`** — creating managed-Iceberg tables through the connection needs
  `bigquery.connections.delegate`. Among predefined roles that permission ships **only** in
  `connectionAdmin`, which also carries `setIamPolicy` + delete. We take the three exact permissions
  (`connections.get` / `.use` / `.delegate`) instead.
- **`sfRayClusterManager`** — headless Ray must create and delete its own cluster (a Vertex
  `PersistentResource`). `create`/`delete` on persistent resources ship **only** in
  `aiplatform.admin` (440+ permissions). We take the four exact permissions instead.
- **`sfNetworkAttachmentConsumer`** — consuming a PSC-I attachment needs
  `compute.networkAttachments.get`/`.use`/`.update`/`.list`. `networkAdmin` would cover them but
  grants project-wide network admin to a service agent. We take the four exact permissions instead.

### Grants to Google-managed service agents (not the two SAs)

A few grants go to Google's own service agents, because these managed services reach back into your
project *as themselves*:

- **The BigLake connection's service agent** gets `storage.objectUser` **and**
  `storage.legacyBucketReader` on the warehouse bucket. It needs *both* because the Storage Write API
  streaming path checks object access **and** `storage.buckets.get`, and no single predefined role
  carries both without over-granting (only `storage.admin` does). This was verified empirically —
  with object access alone, `append_rows` failed 403 on `storage.buckets.get`.
- **The Vertex AI service agent** gets `roles/compute.networkUser` **plus** the custom
  `sfNetworkAttachmentConsumer` role, so the managed Vertex tenant can reach back through the PSC-I
  attachment into your VPC. This is the single-project topology, where Google's docs prescribe the
  broader `compute.networkAdmin`; we take the *leaner* `networkUser` (for the `compute.subnetworks.use`
  that interface-IP allocation needs) plus the four exact attachment verbs in the custom role. A
  further trim to a subnet-scoped custom role is possible but deferred until a greenfield Ray run
  confirms the working permission floor (over-tightening here silently 403s cluster creation).
- **The Cloud Build SA** (the project's Compute Engine default SA) gets `cloudbuild.builds.builder` +
  `artifactregistry.writer`, scoped to exactly the build-and-push path, so `gcloud builds submit` can
  build the runtime image and push it to Artifact Registry.

In brownfield mode (`create_service_accounts = false`) the `iam` module grants **nothing** — your
admin owns all of the above, and you simply hand in the two SA emails.

### Human users (running jobs + notebooks)

The Terraform grants roles to the two **service accounts**, not to people — so a **human** who will
submit runs or drive the notebooks needs their own grants on top. The design keeps this small: the
data-plane power lives on the SAs, and a human mostly needs permission to **act as** the right SA and
to **read results**. There are two personas, and a workshop presenter is usually both.

**Notebook user (Colab Enterprise).** The deployed runtime templates **execute as the runner SA**
(`colab` module `service_account = runner_email`), which already holds every BigQuery / Storage /
Dataproc / Ray role. So a notebook user needs almost nothing on the data itself — only the right to
launch a runtime that runs as that SA:

| Role | Granted on | Why |
|------|-----------|-----|
| `roles/aiplatform.user` | project | Create + use Colab Enterprise runtimes from the `sf-main` template. |
| `roles/iam.serviceAccountUser` | the **runner** SA | Launch a runtime that runs **as** the runner SA (which carries the data roles). Without it, runtime creation is denied. |
| `roles/bigquery.dataViewer` + `roles/bigquery.jobUser` | project | *Optional* — only to browse the registry tables/views yourself in **BigQuery Studio**. The notebooks read as the runner SA, so this is for the human's own console poking. |

**CLI submitter (populating the run history — the workshop's Act 1).** Running
`python -m scale_forecasting.submit` / `ray_submit` from **Cloud Shell** (or locally) under your own
identity, you submit jobs that
**run as the compute SA** — so you need job-submission roles **plus** the right to act as that SA, and
write access to the code bucket the submitter stages the package zip into:

| Role | Granted on | Why |
|------|-----------|-----|
| `roles/dataproc.editor` | project | Submit Dataproc Serverless batches (`submit.py`, all three Spark methods). |
| `roles/aiplatform.user` | project | Submit Ray-on-Vertex jobs (`ray_submit.py`). |
| **`sfRayClusterManager`** | project | Create + delete the autoscaling Ray cluster `ray_submit` stands up (the same custom role the runner uses). Ray only. |
| `roles/iam.serviceAccountUser` | the **compute** SA | The batch/cluster runs **as** the compute SA; submitting a job that impersonates it requires this. |
| `roles/storage.objectAdmin` | the **code** bucket | The submitter stages the code zip + launcher to `gs://<project>-code/` before submitting. Bucket-scoped, not project-wide. |
| `roles/bigquery.jobUser` + `roles/bigquery.dataViewer` | project | Resolve the config and review results (`v_run_summary` / `v_model_leaderboard`) after the run lands. |

> **Why direct grants and not impersonation here.** The runner SA already holds the full submission
> role set, so an alternative is to grant the human `roles/iam.serviceAccountTokenCreator` on the
> runner and `login --impersonate-service-account=<runner>` — inheriting everything with no direct
> data grants. We document the **direct** grants above because they're explicit and easy to reason
> about on stage; the impersonation path is a valid, keyless alternative if your org prefers it.

All of the above are grants to a **human principal** (`user:you@example.com` or a `group:`), e.g.:

```bash
PROJECT=<your project_id>
RUNNER=scale-forecasting-runner@$PROJECT.iam.gserviceaccount.com
COMPUTE=scale-forecasting-compute@$PROJECT.iam.gserviceaccount.com
USER=user:presenter@example.com     # or group:workshop-attendees@example.com

# Notebook user (Colab Enterprise):
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=roles/aiplatform.user
gcloud iam service-accounts add-iam-policy-binding $RUNNER --member=$USER --role=roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=roles/bigquery.dataViewer
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=roles/bigquery.jobUser

# CLI submitter (Act 1) — additionally:
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=roles/dataproc.editor
gcloud iam service-accounts add-iam-policy-binding $COMPUTE --member=$USER --role=roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=roles/aiplatform.user
gcloud projects add-iam-policy-binding $PROJECT --member=$USER --role=projects/$PROJECT/roles/sfRayClusterManager
gcloud storage buckets add-iam-policy-binding gs://$PROJECT-code --member=$USER --role=roles/storage.objectAdmin
```

For a workshop, grant a **group** once and add attendees to it — one binding, not one per person.

---

## What you get back (outputs)

After `terraform apply`, `terraform output` gives you the values the application config and the seed
job consume — the dataset id, the BigLake connection ref, the warehouse URI, the three bucket names,
the two SA emails, the runtime-image repo path, and (for the Ray private path) the subnet URI,
network id, and PSC-I attachment id. These feed the run config and the `SF_*` environment the writers
resolve; see the notebooks and `terraform/README.md` for the exact wiring.

---

## Before you run it

- Read [`terraform/README.md`](../terraform/README.md) for the exact command sequence (bootstrap →
  main) and the cost note.
- **`gcloud` CLI required** on the machine running Terraform (unless `build_image = false`) — the main
  stage shells out to `gcloud builds submit` to build the runtime image.
- **Three levers cost money.** `build_image` is **on by default** — the first apply runs a one-time
  Cloud Build (a few minutes, cents) to build + push the runtime image; it rebuilds only when
  `docker/` deps change. `run_seed` is **on by default** — the first apply runs the example-data seed
  batch once (~$0.15 / ~8.5 min at 100k; set `seed_num_series = 100` to smoke first, or
  `run_seed = false` to skip). `create_composer` is **off by default** — the only real at-rest cost.
- Run `terraform plan` and read it. Nothing is created until `apply`, and the plan is the honest
  preview of exactly what this document describes.
