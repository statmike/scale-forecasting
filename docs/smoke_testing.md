# Smoke testing

A **smoke** is a small, real run — ~100 series — that proves one runtime/hardware/ensemble
combination of the platform works end to end against live Google Cloud. The smoke library under
[`configs/smokes/`](https://github.com/statmike/scale-forecasting/tree/main/configs/smokes) covers
every combination once; the driver
([`tests/smokes/smoke_harness.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/smokes/smoke_harness.py))
runs one config through the full lifecycle and checks the result; and this page is both the
**runbook** for executing them and the **living results log** of what has been validated.

The point is confidence: a green smoke means that combination submits, runs, lands its predictions,
scores onto the leaderboard, re-runs idempotently, and reverse-traces to a clickable job — on real
infrastructure, not a mock.

## What each smoke proves

Run them cheap → expensive; each is numbered in that order.

| # | Config | Proves |
|---|--------|--------|
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML families) |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) in BigQuery |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning family on an L4) |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU |
| 05 | `05_cluster_reuse.json` | Spark reusing a standing Dataproc cluster by name |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (deep-learning family on a T4) |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU (deep-learning family on a T4) |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode (blend once after all base jobs) |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode (drain series as they complete) |
| 13 | `13_native_format.json` | Reading the **native** BigQuery source table (dual-format) |
| 14 | `14_full_dag.json` | The flagship: all families + native + ensemble, one run_id |
| 15 | `15_airflow_multi_engine.json` | The whole DAG **orchestrated by Composer/Airflow** — three engines (Spark + Ray GPU + BigQuery) under a microbatch ensemble |
| 16 | `16_cluster_split_hardware.json` | **Two** Dataproc clusters under one run — a CPU cluster and a GPU cluster, created and torn down together |
| 17 | `17_gpu_absent_serverless.json` | Serverless GPU with the card hidden — the run must **refuse**, not finish on CPU |
| 18 | `18_gpu_absent_cluster.json` | The same refusal on a Dataproc cluster GPU worker |
| 19 | `19_gpu_absent_ray.json` | The same refusal on a Ray GPU worker |
| 20 | `20_gpu_intent_cpu_family.json` | The **opposite** mistake: `use_gpu: true` with the deep-learning family overridden to `cpu`. The run must finish on CPU, having bought no accelerator |

Every other smoke reads the managed-Iceberg source table, so 13 gives the native-format read its own
proof; together they validate both source formats.

**Why 16 exists, given 04 and 06 already cover CPU and GPU clusters.** They cover them one run at a
time. A Dataproc cluster has exactly one worker machine type, so a run whose ephemeral cluster
families span both hardware kinds gets *two* clusters — `sf-cluster-<run_id>-cpu` and
`sf-cluster-<run_id>-gpu` — and that is a genuinely different path: a second create, two distinct
names, a region resolved per cluster (a capacity hop can move one and not the other), and two
teardowns. 04 has two cluster families but both are CPU, so it takes the single-cluster path. 16 is
the only config that forces the split, and an offline tripwire
(`test_a_smoke_needs_two_dataproc_clusters_at_once`) fails if it ever stops doing so.

Smokes 01–14 launch the run directly (`main.run`); smoke 15 is the one that proves the **Airflow
layer** actually orchestrates the same building blocks — see
[Orchestrating on Composer](#orchestrating-on-composer-airflow-smoke) below.

**Why 17–19 are expected to fail.** 03, 06 and 08 each provision an accelerator and come back
green. That tells you the GPU check did not object; it does not tell you the check *can* object,
and a check that cannot fail is worth nothing. 17–19 are the same three services with the card
taken away after it was bought:

```bash
SF_HIDE_DEVICES=probe .venv/bin/python tests/smokes/smoke_harness.py \
  configs/smokes/17_gpu_absent_serverless.json --no-rerun
```

`SF_HIDE_DEVICES` arms a fault on a **GPU** job and carries it to the workers on the same
executor-env seam that already carries the provisioned-hardware fact — Spark through
`spark.executorEnv.*`, Ray through `runtime_env.env_vars`. There are two modes, and they hide the
device at different depths:

| Value | What the worker sees | Use it for |
|-------|----------------------|------------|
| `probe` | The device probe reports `cpu` while CUDA itself is untouched | The default. Reaches `_require_device`, so the contract gets to speak. |
| `cuda` (or any other value, including the older `1`) | `CUDA_VISIBLE_DEVICES=""` — the card is gone as far as every CUDA library on the box is concerned | Reproducing what a real provisioning failure does to the whole stack. |

**Forgetting the variable used to be silent, and is not any more.** Run one of these three configs
without `SF_HIDE_DEVICES` and nothing is hidden, so the job succeeds normally and every verifier is
satisfied — the report reads `PASS` for a check that never happened, which is worse than a failure
because it looks like evidence. The harness now refuses to start an unarmed `*_gpu_absent_*` config
and exits `2` without submitting anything, so the mistake costs nothing instead of a GPU cluster.
If you genuinely want one of them as an ordinary positive run, say so with `--allow-unarmed`.

Read their results the other way round from every other smoke: **`RESULT: FAIL`, with every cell
refused and the run closing `FAILED`, is the passing outcome.** A negative arm that reports `PASS`
means the fault never reached the code under test.

`cuda` mode is the more faithful imitation of a lost card and the less useful test, because on two
of the three services something below us dies before our check runs: a Ray worker holding a GPU
slot crashes rather than raising, and the Serverless RAPIDS plugin aborts the executor, which Spark
then replaces over and over. `probe` mode hides the device only from *our* probe, so the job gets as
far as the contract check on every service. Anything other than `probe` reads as `cuda` on purpose —
a typo'd value failing open would turn a negative arm into a positive one and report a pass.

The expected outcome either way is a **failed** job; under `probe` its message names the family,
the service and the config field that turns the device off. A run that reaches `COMPLETED` is the
finding. The harness reports a failed run as a failed smoke, so read the report rather than the exit
code here.

The switch is infrastructure, not config: it never enters `ComputeConfig`, so arming it does not
move a `run_id`, and a config runs under one identity whether the card is hidden or not. Six cells
each, because there is no reason to buy a hundred series' worth of fleet to watch a job refuse to
start.

### Arming the Ray poll recovery (`SF_RAY_POLL_FAULT`)

The same idea as `SF_HIDE_DEVICES`, aimed at a different unproven promise, and unlike 17–19 it needs
no config of its own — it rides a run you were going to make anyway.

A Ray job is watched by a poll loop that talks to Vertex's managed dashboard proxy every fifteen
seconds for as long as the run lasts. Sooner or later one of those requests dies in transit, or the
client's hour-long OAuth token expires underneath it, and neither says anything about the job. The
loop forgives both and reconnects. That behaviour was written after a dropped request ended a
100,000-series run at minute 79 and tore down a twenty-node fleet — and since then **every long Ray
run has polled cleanly, so the recovery has never actually executed against the live proxy.** Green
runs do not prove it works; they prove the fault did not arrive.

```bash
SF_RAY_POLL_FAULT=transport,auth .venv/bin/python tests/smokes/smoke_harness.py \
  configs/smokes/07_ray_cpu.json --force
```

The first poll of **each Ray job** in the run raises a fault instead of calling the dashboard, the
recovery absorbs it, and `_connect_job_client` mints a genuinely fresh token before the poll
resumes. Two values, one per door the recovery has: `transport` imitates a proxy 5xx, `auth`
imitates the expired bearer token. Comma-separate them to spend two of the loop's four attempts on
one poll, which proves both doors and the shared reconnect in a single run. Anything else truthy
reads as `transport`, for the same reason a typo'd `SF_HIDE_DEVICES` still arms something.

**The run still completes normally — that is the point, and it is also why the evidence is a log
line rather than a result.** Look in the driver's output for:

```
SF_RAY_POLL_FAULT is armed: injecting a transport fault into this poll of sf-<run_id>-…
Ray job poll failed on attempt 1/4 (…); reconnecting and retrying
```

Both lines together are the proof: the first says the fault was delivered, the second says the
recovery caught it. A run with the first and not the second is a finding.

**Both are logged at WARNING, and that is deliberate — it used to be the bug.** The first armed run
(2026-09-22, smoke 07) printed four injection lines and not one answering retry, because the retry
logged at INFO while the harness logs at WARNING. The recovery had in fact worked and the run
reached `COMPLETED`; what was broken was that the runbook's own invocation could not show it. If you
ever see the two lines split across levels again, that is the regression, not a recovery failure —
`test_both_halves_of_the_evidence_survive_the_same_log_level` is what holds them together.

Like `SF_HIDE_DEVICES`, this is infrastructure rather than config, so arming it does not move a
`run_id`. Do not arm more than three faults — the loop forgives three consecutive failures and gives
up on the fourth, which ends the run and tears the fleet down, exactly as it would for a channel
that really had gone.

**Why 20 exists, and why it is the mirror image of 17–19.** Those three ask what happens when a job
is told to use a device it cannot see. 20 asks the opposite question: what happens when the two
places that decide about accelerators disagree in the other direction. Its config sets the flat
`compute.use_gpu: true` **and** overrides the deep-learning family to `hardware: "cpu"`, which is a
perfectly reasonable thing for someone to write while moving a workload off GPUs — they flip the
family and forget the legacy flat flag underneath it. There is only one correct outcome: the
per-family override wins, `resolve_family_compute` returns `cpu` with no `gpu_type`, the submitter
provisions a CPU pool because it reads that same resolver, and the run completes having bought no
accelerator. The failure this guards against is the one where routing and provisioning read
different variables — the fleet is GPU, the tasks request CPU, nothing is ever schedulable, and the
run hangs until something times out rather than failing with a message. A hundred cells rather than
six, because a hang only shows up once there is real work to place.

## Prerequisites

- **Identity** — the `SF_*` environment plus Application Default Credentials
  (`gcloud auth application-default login`). The harness **submits** (not just reads), so it needs the
  full set, not only the five `Settings` vars from [running & reviewing](./running_and_reviewing.md):
  `Settings.resolve()` reads `SF_PROJECT_ID` / `SF_CONNECTION` / `SF_WAREHOUSE_URI` (+ optional
  `SF_DATASET_ID` / `SF_REGION`), **and** `BatchInfra.resolve()` reads `SF_CODE_BUCKET` /
  `SF_CONTAINER_IMAGE` / `SF_COMPUTE_SA` / `SF_SUBNETWORK_URI`, **and** the Ray smokes' `RayInfra.resolve()`
  reads `SF_RAY_NETWORK_ATTACHMENT` (or `SF_RAY_NETWORK`) plus optional `SF_RAY_VERSION` /
  `SF_RUNTIME_VERSION`. Wire them straight from Terraform:

  ```bash
  cd terraform/main
  eval "$(terraform output -json | python -c 'import json,sys
  o=json.load(sys.stdin); g=lambda k: o[k]["value"]
  print(f"export SF_PROJECT_ID={g(\"project_id\")}")
  print(f"export SF_CONNECTION={g(\"iceberg_connection\")}")
  print(f"export SF_WAREHOUSE_URI={g(\"warehouse_uri\")}")
  print(f"export SF_DATASET_ID={g(\"dataset_id\")}")
  print(f"export SF_CODE_BUCKET={g(\"code_bucket\")}")
  print(f"export SF_CONTAINER_IMAGE={g(\"runtime_image_repo\")}:latest")
  print(f"export SF_COMPUTE_SA={g(\"compute_sa\")}")
  print(f"export SF_SUBNETWORK_URI={g(\"subnetwork_uri\")}")
  print(f"export SF_RAY_NETWORK_ATTACHMENT={g(\"network_attachment_id\")}")
  print(f"export SF_VENV_ARCHIVE={g(\"venv_archive_uri\")}")
  gpu=o.get("gpu_image_uri", {}).get("value")
  print(f"export SF_GPU_IMAGE={gpu}") if gpu else None')"
  export SF_REGION=us-central1   # or your deploy region
  ```
  `SF_GPU_IMAGE` is exported only when the deploy built the pre-baked GPU cluster image
  (`build_gpu_image = true`); without it, GPU cluster smokes install the driver at cluster-create
  time instead (slower). See [runtime_dependencies.md](./runtime_dependencies.md#gpu-clusters--the-driver-init-action).
  Note the `.get` above rather than a plain lookup: Terraform does not emit the output at all when
  the image was not built, and a hand-set `SF_GPU_IMAGE` pointing at an older baked image fails the
  cluster at create with *"Selected software image version … can no longer be used to create new
  clusters"* once Dataproc retires the sub-version inside it. The fallback path pins a sub-minor of
  its own (`2.2.85-debian12`, because the driver build needs that kernel), so it can hit the same
  message eventually — but recovering is a one-line change rather than an image rebuild, and the
  error says so.
- **Source tables** — both `source_series_iceberg` and `source_series_native` must exist in the
  deployment dataset (they are created by the Terraform + seed step).
- **Deep-learning smokes (03, 06, 08, 09, 10, 14)** — the container must carry the `models` extra so
  `neuralprophet` can fit; the run image built by the deploy already includes it.
- **GPU smokes (03, 06, 08, 09, 10, 14)** — GPU quota in the run's region: L4 for Serverless
  (03, 14), T4 for cluster/Ray (06, 08, 09, 10).
- **GPU cluster smoke (06)** — a Dataproc *cluster* is a bare set of VMs, so the host NVIDIA driver
  is the deploy's to supply. With `build_gpu_image = true` the deploy bakes a custom VM image with
  the driver pre-baked and exports it as `SF_GPU_IMAGE`, so the cluster boots ready. Without it the
  cluster installs the driver at create time — a source compile that can exceed Dataproc's
  cluster-create window. Prefer the pre-baked image for this smoke.
- **Cluster capacity failover** — GPU (and sometimes CPU) capacity is *zonal* and can stock out
  transiently even with quota to spare, so an ephemeral cluster create walks a candidate list: the
  deploy region's auto-zone first (unchanged), then that region's other zones, then — opt-in — other
  regions. The candidate list is the user-editable
  [`configs/compute_fallback.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/compute_fallback.json)
  (or `SF_COMPUTE_FALLBACK`), prepopulated with the US regions/zones; same-region zone failover works
  with no edits, while cross-region failover activates only for regions you give a `subnetwork_uri`
  (a subnet with Cloud NAT + Private Google Access in that region). Non-capacity errors still fail
  fast, and the list never affects `run_id`.
- **Cluster smokes (04, 05, 06)** — a Dataproc **cluster** can't use the custom container, so it
  gets its dependencies from the **self-contained venv archive** instead. `SF_VENV_ARCHIVE` must point
  at it (the `venv_archive_uri` Terraform output, wired above); the deploy's Cloud Build packs + uploads
  it alongside the image. See [runtime_dependencies.md](./runtime_dependencies.md#dataproc-cluster--self-contained-venv-archive).
- **Cluster-reuse smoke (05)** — a standing Dataproc cluster named `sf-smoke-cluster` must already
  exist; the run submits to it rather than creating one. Delete it when the campaign is done.

## Running one smoke

Each smoke is one command; it blocks until the run is terminal, then prints a report.

```bash
.venv/bin/python tests/smokes/smoke_harness.py configs/smokes/01_serverless_cpu.json
```

The driver walks the lifecycle a reviewer would run by hand:

1. **dry** (`plan_run`) — resolve the run_id, the per-runtime model split, the fanout, and the
   exists-vs-new verdict; touches no GCS.
2. **stage** (`stage_run`) — upload the config (and, for Spark, the code zip) and write the
   reproducibility manifest `runs/<run_id>.plan.json`; capture the runnable launch commands.
3. **run** (`main.run`) — submit every family on its runtime under one run_id and block to terminal.
4. **verify** — read the registry views back and check the run reached `COMPLETED`, every expected
   family ran and succeeded with a real platform job id, and every configured model (plus the
   ensembles, when enabled) scored onto the leaderboard.
5. **rerun / collision** — re-run the same config with no `--force`; it must resolve the **same**
   run_id and, via append-only + dedupe-on-read, leave the leaderboard counts unchanged.
6. **reverse-trace** — print each family's stored `system_job_id` and the service it resolves to
   (Dataproc batch / Dataproc cluster job / Vertex Ray submission / BigQuery job).

Flags:

- `--force` — bump the attempt (a fresh job under the same run_id), instead of the default re-run.
- `--no-rerun` — run once and skip the rerun/collision check.

The command exits non-zero if any check fails, so it drops straight into CI or a `for` loop.

## Orchestrating on Composer (Airflow smoke)

Smokes 01–14 launch the run directly. Smoke 15 instead drives the run **through Composer**, proving
the emitted Airflow DAG orchestrates the same building blocks the direct smokes launch by hand. It
takes the identical config, resolves its `run_id`, stages its artifacts, emits `dag_<run_id>.py`
([`airflow_emit.emit_airflow_dag`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/airflow_emit.py)),
imports it into the environment, triggers it, and waits for the run to land in the registry — the
same terminal signal the direct harness polls. Because the `run_id` is a digest of the config, a
Composer-orchestrated run writes the registry under the **identical** id a local run would, so
success is direct proof of *same code local↔Composer*. Verification reuses the direct harness's
checkers (`verify_run_jobs` / `verify_leaderboard` / `verify_predictions` / `verify_cells`), holding
both smokes to one standard.

Two levels of proof, cheap → expensive:

- **Parse-under-Airflow (always-on, free).** The offline emitter tests `compile()` the DAG and walk
  its `ast`; they never import Airflow. `tests/unit/test_airflow_dagbag.py` closes that gap — it
  loads an emitted DAG through a real `airflow.models.DagBag` and asserts no import errors, catching
  operator-kwarg / import-chain mistakes the string checks can't. Airflow is heavy and conflicts
  with our torch/ray/spark pins, so it is **not** in `uv.lock`; the test is marked `@airflow` and
  skips cleanly when Airflow is absent (like `@spark`/`@ray`). A dedicated CI job (`airflow-parse`)
  installs Airflow isolated against its official constraints and runs it on every push, so it can
  never destabilize the main offline gate.
- **Live on Composer (gated).** Provisioning Composer costs money and time, so it is gated behind
  `create_composer=true` and run only on explicit go.

**How the code and config reach Composer.** A Composer worker is just another *launch point* — it
runs the same driver code a local launch does. It never runs the model code (that ships per-job as
the `src/` zip and executes in Dataproc/Ray/BigQuery), but it does need three things to *be* a launch
point, and none of them come from GitHub at runtime:

- **The `SF_*` identity** — set as environment variables on the environment, wired from the Terraform
  outputs (`terraform/main` builds this map and passes it to the composer module). This is infra, so
  `terraform apply -var create_composer=true` sets it.
- **The submit-side dependencies** — the `pypi_packages` the driver imports to talk to the services
  (BigQuery registry, Dataproc/Vertex submit, the Ray `JobSubmissionClient` handshake). **Not** the
  model stack (torch/darts/neuralprophet/pyspark) — that runs in-service. Also set by the apply.
- **The code** (`src/`) — delivered by `make composer-sync`, which `gsutil rsync`s **this working
  tree's** `src/` into the environment's plugins prefix (on the workers' `PYTHONPATH`). The worker
  then imports the driver **and** re-zips that same `src/` to ship code to the jobs — so your
  clone/fork/customizations flow through with no image rebuild. GitHub is only the origin: you pull
  and modify locally, and `composer-sync` is what carries the result to the environment. This is a
  bootstrap step (code changes more often than infra), not baked into Terraform.

**Prerequisite — a running Composer environment.** Composer is off by default
([`terraform/main/modules/composer`](https://github.com/statmike/scale-forecasting/tree/main/terraform/main/modules/composer)).
Provision it, deliver the code, run the smoke, then turn the meter back off:

```bash
# 1. Provision (~40 min build, measured 2026-09-20; ~$300–400/mo while up — the smallest env).
#    This also sets the workers' SF_* env + submit-side pypi packages.
#    Read the plan first: a *full* apply also re-submits the seed batch if the seed code has moved
#    since the last one, which blocks for about an hour at 100k series. `-target=module.composer`
#    provisions the environment and nothing else.
cd terraform/main
terraform plan -var create_composer=true -target=module.composer -out=composer.tfplan
terraform apply composer.tfplan

# 2. Deliver this working tree's src/ to the workers (the code-delivery step; re-run after edits).
cd ..
make composer-sync

# 3. Run the Airflow smoke end-to-end (stage → emit → import → trigger → wait → verify).
.venv/bin/python tests/smokes/airflow_smoke.py configs/smokes/15_airflow_multi_engine.json \
    --composer-env scale-forecasting --location "$SF_REGION"

# 4. Stop the meter — destroys just the environment; data/registry/buckets untouched.
#    Same targeting on the way down, and same reason: `terraform show` the plan and confirm it
#    reads "1 to destroy" before applying it.
cd terraform/main
terraform plan -var create_composer=false -target=module.composer -out=composer-down.tfplan
terraform show -no-color composer-down.tfplan
terraform apply composer-down.tfplan
```

!!! warning "Why the harness talks REST, not `gcloud composer environments run`"
    The obvious way to confirm a DAG parsed and to trigger it is the Airflow CLI, wrapped as
    `gcloud composer environments run <env> ... dags list` / `dags trigger`. **Do not use it here.**
    Those verbs route through the Composer `executeAirflowCommand` API, which spins up a pod per
    invocation; against a healthy, `RUNNING` environment a single `dags list-import-errors` was
    observed to produce no output and hang past a 400-second timeout. In a poll loop that gates a
    live billing run, that is a hang with a meter attached.

    The harness instead reads the environment's `airflowUri` with a plain `describe` (a control-plane
    read — about a second, no pod) and then calls the **Airflow REST API** directly with ADC
    credentials: `GET /api/v1/dags/<dag_id>` to confirm the parse, `PATCH` to unpause,
    `POST .../dagRuns` to trigger. It answers in well under a second and returns
    `has_import_errors` explicitly — which `dags list` does not, so the REST path distinguishes
    *"not parsed yet"* from *"parsed and broken"* where the CLI could only time out. Only the DAG
    **upload** stays on `gcloud` (`storage dags import`), because that is a GCS copy and is
    reliable.

The deep-learning family on Ray GPU is the long pole in this smoke, and it is why `backtest.n_folds`
is **3** rather than the 10 the other backtesting smokes use. At the measured **7.6 fits/min per GPU
node** ([quota and scale](quota_and_scale.md)), 200 series × (3 folds + the final fit) is ~800
NeuralProphet fits — comfortably under an hour including cluster provisioning, which keeps the run
inside the ~60-min bearer-token expiry (a known limit). Ten folds would be ~2,200 fits and would
trip it. What this smoke exists to prove is
*Airflow orchestrating three engines*, not deep-learning throughput; three folds is enough to
exercise backtesting, the error-weighted ensemble strategies, and many microbatch intervals. The
config is the knob if a live run still runs long — drop folds further or move `deep_learning` to CPU.

## Verifying by hand in BigQuery

The harness reads the same views you can query directly for the run_id it prints:

```sql
-- how did the run go, and how efficiently?
SELECT * FROM `PROJECT.DATASET.v_run_summary`     WHERE run_id = 'RUN_ID';
-- which families ran, on what runtime/hardware, and their real job ids (reverse-trace)
SELECT * FROM `PROJECT.DATASET.v_run_jobs`        WHERE run_id = 'RUN_ID';
-- which model won, per run
SELECT * FROM `PROJECT.DATASET.v_model_leaderboard` WHERE run_id = 'RUN_ID' ORDER BY mean_wape;
```

The `system_job_id` in `v_run_jobs` is the real, console-resolvable id for each family's job — paste
it into the Dataproc / Vertex / BigQuery job history to click straight through.

## Offline guardrails

The config library is checked in the offline gate so a broken config never reaches a live submit:

- `tests/smokes/test_smoke_configs.py` — every config loads, validates, and plans a DAG; the library
  still spans every runtime/hardware/ensemble combination; both source formats are exercised.
- `tests/smokes/test_harness.py` — the harness's verify/trace logic (what decides PASS/FAIL) is
  unit-tested with fixture rows.
- `tests/smokes/test_airflow_smoke.py` — the Airflow smoke's pure command-builders (which `gcloud
  composer` argv it shells out) and its `dag_id` derivation, so a typo can't point the live smoke at
  the wrong environment or DAG.

Both run in the standard offline suite (`pytest -m "not gcp and not spark and not ray"`). The
parse-under-Airflow test (`tests/unit/test_airflow_dagbag.py`) is separate: marked `@airflow`, it
skips unless Airflow is installed and runs in its own CI job (see
[Orchestrating on Composer](#orchestrating-on-composer-airflow-smoke)).

## Results log

Live results are recorded in the **[validation ledger](validation.md)** — the single source of
truth for what has been proven on real infrastructure. It is kept there rather than here because a
result only means something relative to the architecture it ran on, and the ledger records that:
each entry declares the architecture axes it depended on, so a later change that invalidates a
passing result is caught mechanically by `tests/unit/test_validation_ledger.py` in the offline gate
instead of going unnoticed.

After running a smoke, add its row to the ledger — including the `run_id`, which is what lets
anyone reverse-trace the result back to the platform job.
