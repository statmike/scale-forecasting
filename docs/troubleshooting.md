# Troubleshooting

The single home for known issues seen building and running this at scale, each as
**symptom → cause → fix / where it's handled**. Most are already handled in code or config; this
doc explains *why* the handling exists so you recognize the symptom fast.

Grouped by where it bites: [Ray](#ray-on-vertex) · [Spark](#spark-on-dataproc) ·
[Registry / writes](#registry--writes) · [Versions](#versions--runtimes) ·
[Deploy / Terraform](#deploy--terraform) · [Notebooks](#notebooks).

---

## Ray on Vertex

### Ray job submission hangs, or HTTP 524 on `/api/version`
**Symptom:** the driver connects but job submission never returns, or the client gets a 524 timeout
hitting the cluster's `/api/version`.
**Cause:** the cluster needs **both** a PSC-I network attachment **and** a sufficiently large head
node (`n1-standard-16`+). Either one alone still 524s — it is a cluster-side capacity/networking
issue, not the client's location.
**Fix:** ensure the Terraform network attachment is applied and `compute.ray_head_machine_type` is
`n1-standard-16` or larger. See [deploying_on_gcp.md](./deploying_on_gcp.md).

### Ray job fails at env setup — `No matching distribution for torch==…+cu126`
**Symptom:** the cluster comes up and the job is submitted, but it goes straight to `FAILED` with
`runtime_env setup failed` and, in the job logs, `ERROR: No matching distribution found for
torch==2.13.0+cu126`. No cells are written.
**Cause:** the x86_64/linux `torch` pin is a CUDA local build (`+cu126`) that exists **only** on the
PyTorch wheel index, not PyPI. The Ray `runtime_env` **`uv`** install needs the same `--extra-index-url`
the container image build uses; without it — and without letting `uv` prefer that index — the pinned
wheel can't resolve and the whole job dies before any work runs.
**Fix:** handled — `build_runtime_env()` passes `--extra-index-url
https://download.pytorch.org/whl/cu126` **and** `--index-strategy unsafe-best-match` in the `uv`
plugin's `uv_pip_install_options` (kept in lockstep with `docker/Dockerfile`, with PyPI still primary;
`unsafe-best-match` is what lets `uv` pick the `+cu126` build from the extra index even though a
same-named wheel exists on PyPI). If you re-pin torch to a different CUDA minor, update the URL in both
places.

### Ray driver OOM — no traceback, killed at "uploading package"
**Symptom:** the run dies with no Python traceback, often right after logging the package upload.
**Cause:** the driver/orchestrator RSS outgrew a small VM. An OOM-kill leaves no traceback.
**Fix:** run the orchestrator on a VM with ≥64 GB (e.g. `e2-highmem-8`). See
[operations.md](./operations.md).

### Long Ray run 401s after ~60 minutes
**Symptom:** a run polling for status starts returning 401 Unauthorized well into the run.
**Cause:** the bearer token in the poll loop hit its TTL (~60 min). Neural Prophet on GPU is the
long pole that pushes total runtime past the expiry.
**Fix:** handled — the status poller does a single-retry re-auth on a 401. If you see a hard 401,
confirm ADC is still valid on the orchestrator.

### Regional GPU capacity stockout
**Symptom:** cluster create fails to obtain T4s in a region.
**Cause:** transient regional accelerator stockout.
**Fix:** `compute.ray_regions` is a fallback list — the launcher tries them in order. Add more
regions if one is chronically short.

### Ray cluster won't shrink when idle / can't grow under load
**Symptom:** the expensive GPU pool stays allocated while idle, or the CPU pool can't grow to work
through the queue.
**Cause:** the run used fixed-size sizing (`ray_autoscale=false`).
**Fix:** autoscaling is the default now (`ray_autoscale=true`), with independent per-pool
`[min, max]`. Set a low `ray_gpu_min_nodes` to shrink idle T4s and a high `ray_cpu_max_nodes` to
grow under load. See the compute block in
[configuration_reference.md](./configuration_reference.md).

---

## Spark on Dataproc

### 100k Spark run OOM — exit 137 → FetchFailedException
**Symptom:** executors killed with exit 137, cascading into `FetchFailedException` at large scale.
**Cause:** bucket sizing. Sizing buckets by `max_parallelism` silently fattened each pandas frame
until executors OOM'd.
**Fix:** buckets are sized by `compute.bucket_target_cells` (default 8) — `buckets = ceil(cells /
target)` bounds per-task memory at every scale. Lower `bucket_target_cells` if a heavy model set
still OOMs. See [configuration_reference.md](./configuration_reference.md).

### 100k batch cancelled at ~4 hours
**Symptom:** a large batch is cancelled around the 4-hour mark.
**Cause:** the default Dataproc batch TTL.
**Fix:** handled — the submitter sets an explicit 24h batch TTL.

---

## Registry / writes

### `RegistryError: Storage Write API … 500/503/429`, or `400 Cannot route on empty project id`
**Symptom:** intermittent write failures over long, high-volume runs.
**Cause:** transient backend/routing blips on the Storage Write API over sustained streams.
**Fix:** handled — `_append_via_write_api` retries with exponential backoff (all four engines route
writes through it). A *genuine* 400 (real bad request) still fails fast.

### An immediate re-run double-counts rows
**Symptom:** re-running the same config right away appears to duplicate rows.
**Cause:** the Storage Write API streaming buffer (~90 min) blocks a same-key DELETE, so the design
is **append-only + dedupe-on-read**, never delete-then-write.
**Fix:** this is by design and the serving views dedupe on `run_id`. If you want a physically
separate run, use a fresh/timestamped `run_name` (which changes the `run_id`). See
[output_schemas.md](./output_schemas.md).

---

## Versions / runtimes

### `PYTHON_VERSION_MISMATCH`, or "why can't I move to 3.12?"
**Symptom:** a version-mismatch error between driver and workers, or a desire to bump to 3.12.
**Cause:** the whole system is pinned to **Python 3.11**: Vertex Ray caps at 3.11, and Spark Connect
uses runtime 2.3 (not 3.0). 3.11-everywhere is the only configuration that satisfies all runtimes.
**Fix:** stay on 3.11. Full rationale and the per-surface matrix:
[version_matrix.md](./version_matrix.md).

### `libgomp1` missing, or neuralprophet + pandas-3 breakage
**Symptom:** an import/link error for `libgomp1`, or neuralprophet failing against pandas 3.
**Cause:** neuralprophet needs the OpenMP runtime, and it is incompatible with pandas 3
(`Series.view` removal).
**Fix:** handled — the image includes `libgomp1` and pandas is pinned `<3`. neuralprophet is an
optional extra; the registry registers-and-skips it when unavailable.

---

## Deploy / Terraform

### Cloud Shell `Bus error` running terraform
**Symptom:** terraform dies with a `Bus error`.
**Cause:** the Cloud Shell disk filled up.
**Fix:** disk-hygiene first (clear caches / large artifacts), then retry. See disk-hygiene in
[operations.md](./operations.md). For long runs, prefer a persistent VM over Cloud Shell.

### `google_dataproc_batch` apply errors, but the batch SUCCEEDED
**Symptom:** terraform/apply reports an error after submitting a batch that actually ran fine.
**Cause:** the provider's client-side wait raced the batch; the batch's state in GCP is the source
of truth.
**Fix:** treat the GCP batch state as authoritative; `terraform import` reconciles the resource.

---

## Notebooks

### `--ack-out-of-org` FAILED_PRECONDITION on notebook fan-out
**Symptom:** the notebook fan-out step fails a precondition for a project not in the corp org.
**Cause:** the out-of-org acknowledgement flag applies only to corp-org projects.
**Fix:** for a non-corp-org project, the flag is not applicable — see the headless/Colab specifics
in [notebook_runtimes.md](./notebook_runtimes.md).
