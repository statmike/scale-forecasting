# Quota and scale

**How much GCP quota do you need to forecast N series?** This page answers that with numbers
measured on real runs, not with a rule of thumb.

The short version: **this product is quota-bound long before it is architecture-bound.** A default
GCP project gives you enough headroom to run tens of thousands of series comfortably. Past that,
every additional unit of parallelism is a quota request, and nothing in a config file can substitute
for one. The point of this page is to let you file that request *before* you discover you needed it,
with a number you can defend.

---

## 1. The arithmetic

Three quantities, and everything follows from them.

**Cells.** The unit of work is one *cell* — one model fitted to one series for one backtest fold.

```
cells = series x models x folds
```

10,000 series with 6 models and no backtesting is 60,000 cells. Turn on 2-fold backtesting and it is
120,000 — backtesting multiplies, it does not add. Watch that number; it is the one that sets your
bill and your wall clock.

**Throughput.** Cells are independent, so they run in parallel and throughput is linear in the size
of the fleet. What matters is throughput *per node*, measured end to end:

```
wall clock = cells / (nodes x cells-per-minute-per-node)
```

**Quota.** Nodes come out of a regional CPU allowance, and GPU nodes additionally out of a
per-accelerator allowance. Those two numbers are the ceiling on `nodes`, and therefore on how fast
any of this can go.

```
vCPUs needed = (worker nodes x vCPUs per worker) + vCPUs for the head/driver node
```

---

## 2. What one node actually delivers

Measured, not estimated. The anchor is `ray-100k-dcc77a9d1e9b` — 100,000 series x 4 models =
400,000 cells on Ray, 20 x `n1-standard-8` workers, five and a half hours. Throughput was flat for
the whole steady state, which is what makes it usable for planning:

| Minutes in | Active nodes | Cells/min |
|---|---|---|
| 0–30 | 26 | 1,061 *(autoscaler still ramping)* |
| 30–240 | 20 | **1,425–1,457** |
| 270+ | 20 | 826 *(tail — work running out)* |

So for a mixed statistical + ML workload on 8-vCPU nodes:

> **~72 cells per minute per 8-vCPU node.**

Per-cell fit times behind that figure, from the same run:

| Model | Family | Avg fit | p95 fit |
|---|---|---|---|
| `holtwinters` | statistical | 0.41 s | 0.49 s |
| `theta` | statistical | 0.42 s | 0.87 s |
| `xgboost` | ml | 0.75 s | 0.88 s |
| `sarimax` | statistical | 1.68 s | 2.41 s |

**Your mileage will vary with the model mix, and `sarimax` is the reason.** It is 4x the cost of
`theta` on the same data. A `theta`-only run is much faster per cell than the table above; a run
dominated by `sarimax` is slower. Series length matters too — these were ~4 years of daily
observations.

!!! warning "This anchor predates the density fix — treat it as a floor"
    72 cells/min/node was measured when a node was fitting roughly **one cell at a time** despite
    having eight vCPUs, because of the memory-request defect described in
    [§5](#5-cores-are-the-unit-and-it-took-a-bug-to-find-out). Throughput scaled with *nodes* rather
    than cores, and the figure above is what that regime delivered. The fix landed on 2026-09-04 and
    a comparable CPU workload came in 1.5–2x faster, so **plan with 72 cells/min/node and expect to
    beat it.** A re-run of this anchor at 100,000 series has not been done; when it is, this section
    gets a bigger number.

**Deep learning is a different regime entirely.** `neuralprophet` measures at **21–65 s/fit**
against sub-second statistical models — 50x or more — and it needs a GPU. Measured at 10,000 series
on 12 T4s, it lands at **7.6 cells/min per T4 node** (the 2026-09-04 `all_families_10k` run, 1 h
50 m for 10,000 cells). Unlike the CPU anchor above, this one is post-fix; the arithmetic in
[§3](#3-the-table) is built on it.

---

## 3. The table

Assumptions: **6 models, 1 fold** (so `cells = series x 6`), the mixed statistical + ML profile
measured above, `n1-standard-8` workers (8 vCPU) and an `n1-highmem-32` head node (32 vCPU). The
head node is a fixed cost you pay once per run.

| Series | Cells | Wall clock on a **default 200-vCPU** project (20 nodes) | Nodes for a **~1-hour** run | vCPUs to request |
|---|---|---|---|---|
| **10** | 60 | seconds | 1 | 40 — **default is fine** |
| **100** | 600 | < 1 min | 1 | 40 — **default is fine** |
| **1,000** | 6,000 | ~4 min | 2 | 48 — **default is fine** |
| **10,000** | 60,000 | ~42 min | 14 | 144 — **default is fine** |
| **100,000** | 600,000 | **~7 hours** | 139 | **~1,150** |
| **1,000,000** | 6,000,000 | **~70 hours** | 1,389 | **~11,150** |

!!! warning "Add the start-up floor to the top two rows"
    Provisioning a cluster takes **10–15 minutes** regardless of how little work you then give it.
    At 10 or 100 series the compute is instant and the wall clock is essentially all start-up. That
    floor does not scale, so it disappears into the noise by 10,000 series — but it is why a tiny
    run does not feel fast, and why iterating on config is better done against a few hundred series
    than a few dozen.

Read it as two separate questions.

**"Can I do this at all today?"** — column 3. A stock project runs 10,000 series in well under an
hour and 100,000 series overnight. Both are useful; neither needs a support ticket. This is why the
demonstration configs in `configs/` top out at 10,000 series: it is the largest scale that a
default project runs comfortably, which makes it the largest scale a reader can *reproduce*.

**"How much quota to make it fast?"** — columns 4 and 5. Wanting 100,000 series in an hour rather
than seven means asking for roughly **1,150 vCPUs**, about 6x a default allowance. A million series
in an hour is ~11,000 vCPUs and is a conversation with your account team, not a form.

The relationship is linear in both directions, so interpolate freely: **halve the wall clock, double
the quota.**

### Deep learning is on a separate, much lower ceiling

`neuralprophet` needs a T4 and fits ~50x slower than a statistical model, so GPU allowance is the
hardest limit in this product. **Which** allowance depends on the runtime, and the two are not the
same number:

| Deep-learning runtime | Quota metric | Typical default |
|---|---|---|
| Spark on a **Dataproc** cluster | `NVIDIA_T4_GPUS` (Compute Engine) | **4** |
| **Ray on Vertex** | `custom_model_training_nvidia_t4_gpus` (Vertex AI) | **12** |

These are separate pools in separate services. Raising one does nothing for the other, and the
`gcloud compute regions describe` recipe below shows only the first — see
[§4](#4-which-quotas-and-where) for how to read the Vertex one. The Vertex allowance is also far more
regionally uneven than the Compute Engine one: on the project used for the
[validation ledger](validation.md) it is 12 in `us-central1` but **2** in both `us-east1` and
`us-west1`, so a failover region may be a third of the capacity you sized for.

At **7.6 cells/min per T4** — measured, see below — one deep-learning model over the same series
counts:

| Series | Wall clock on **4 T4s** (Dataproc default) | On **12 T4s** (Ray default) | T4s for a **~1-hour** run |
|---|---|---|---|
| **100** | ~3 min | ~1 min | 1 |
| **1,000** | ~33 min | ~11 min | 3 |
| **10,000** | **~5.5 hours** | **~1.8 hours** | 22 |
| **100,000** | **~55 hours** | ~18 hours | 219 |
| **1,000,000** | ~23 days | ~7.6 days | 2,193 |

**A default project runs out of deep-learning headroom somewhere around 10,000–20,000 series** for a
run you are willing to sit through, an order of magnitude before it runs out of CPU headroom. This
is why `all_families_10k.json` is the largest config here that includes `neuralprophet`. It pins
`ray_gpu_max_nodes: 12`, the Vertex default for `us-central1`; it previously pinned `4`, which was
the Compute Engine number applied to a Ray run by mistake and left two-thirds of the allowance
unused. If you deploy to a region with a smaller Vertex allowance, lower it — the pool starts at
`ray_gpu_min_nodes` and a `max` above your quota is not an error, just a ceiling the autoscaler
never reaches.

**Where the 7.6 cells/min/T4 anchor comes from.** It replaces an extrapolation from runs of 100
series or fewer, in which cluster start-up was a large fraction of the span. On 2026-09-04
`all_families_10k.json` fit **10,000 `neuralprophet` series across 12 T4s in 1 h 50 m** — 91
cells/min for the fleet. The 10,000-series row above is therefore not a projection; it is that run.

Two things about the number are worth understanding before you plan with it. **It is a throughput
figure, not a latency figure.** The average individual fit in that run took 43.5 s, which is
1.4 cells/min if you watch a single series — roughly seven cells share each T4, and they contend.
Sizing from the per-cell time will over-provision you by about 5x. **And it is `neuralprophet` on a
T4 at this data's shape.** A different model, a different device, or much longer series will move
it. The rate is the right starting point precisely because it is now measured rather than guessed;
it is still your own first run that tells you your number. Plan with these, then measure.

---

## 4. Which quotas, and where

Quotas are **per region**, so a project with 200 vCPUs in `us-central1` has a separate 200 in
`us-east1`. Request them in the console under **IAM & Admin → Quotas & System Limits**, filtered to
the region you deploy into, or with `gcloud`:

```bash
# What you have today
gcloud compute regions describe us-central1 \
  --project="${SF_PROJECT_ID}" \
  --format="table(quotas.metric, quotas.limit, quotas.usage)"
```

The metrics that bind this product:

| Quota metric | What it limits | Typical default |
|---|---|---|
| `CPUS` | Every worker and head/driver vCPU in the region | 200 |
| `NVIDIA_T4_GPUS` | Deep-learning worker nodes **on Dataproc** | 4 |
| `PREEMPTIBLE_NVIDIA_T4_GPUS` | The same, for preemptible pools | 4 |
| `DISKS_TOTAL_GB` | Boot disks across the fleet | 40,960 |
| `IN_USE_ADDRESSES` | Rarely binding — clusters are private | 69 |

**There is no Dataproc capacity quota to look for.** Every `dataproc.googleapis.com` quota is an API
*request rate*, not an amount of hardware — including for Serverless, whose batch vCPUs bill to
Compute Engine `CPUS` in the region. If a Serverless batch is capacity-starved, the meter to raise
is the Compute Engine one above.

**Ray on Vertex is not in that table, and `gcloud compute` cannot show it.** Vertex draws GPUs from
its own service quota, so a Ray run is bound by a metric the recipe above never prints:

```bash
# The quota that actually binds a Ray GPU run
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer="projects/${SF_PROJECT_ID}" \
  --filter="metric:custom_model_training_nvidia_t4_gpus" \
  --format=json
```

Read the `effectiveLimit` for your region out of `quotaBuckets`. Checking only the Compute Engine
number is how you conclude you have 4 T4s on a project that has 12.

**You do not have to run either command.** `--quota` reads the same meters for the run you are about
to launch, in every region it would consider, and prints what they allow:

```bash
python -m scale_forecasting.main --config configs/all_families_10k.json --quota
```

```
quota report for all-families-10k-eb01dcfecfab — candidate regions: us-central1, us-east1, us-west1
  native: on bigquery, no per-job capacity meter to read
  deep_learning on ray/gpu: gpu[1,12]
    us-central1: OK
      quota: gpu pool in us-central1 — aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus = 12, usable nodes = 12
        this run would saturate at 5000 nodes (10000 cells / 2 per node); the ceiling holds it to 12
        now   12 node(s): ~1h29m
        at    24 node(s): ~45m
    us-east1: UNREACHABLE — the PSC-I network attachment is in us-central1; ...
```

It reads only and launches nothing. The wall-clock projections appear when a
[measured profile](configuration_reference.md) exists for the family; without one you still get the
ceiling and the throttle ratio.

A family on an **ephemeral Dataproc cluster** is reported the same way, against Compute Engine's
meters rather than Vertex's, and over the zonal candidate list the Dataproc walk actually uses:

```
  statistical on spark/cluster/gpu: 8 worker(s) + 1 master
    us-central1: CLAMPED
      quota [CLAMPED] compute.googleapis.com/nvidia_t4_gpus allows 4 in us-central1, so the gpu
        pool tops out at 4 node(s) instead of 8
      the cluster would be built with 4 worker(s), not 8
```

Families on Serverless, on BigQuery, and on a cluster you told the run to reuse are named but not
metered: a reused cluster already exists so nothing is being asked for, a Serverless batch has no
fixed allocation to pre-read, and BigQuery has no capacity meter at all.

The launch path runs the same check by itself, before the first cluster create, on both the Ray and
the Dataproc-cluster path. A region that cannot host the fleet is recorded as a hard ceiling and
skipped without spending the ~12 minutes a failed Ray create costs; a region that can host a
*smaller* fleet has this run's pool ceilings — or its physical worker count, on Dataproc — lowered
to fit rather than failing. It never raises a floor: quota is evidence about what you are permitted,
not about what the work needs. And it never changes `run_id`, so the same config is still the same
run in a region that grants less. Set `compute.capacity.preflight: false` to skip it (the only
reason to: a runner service account without `serviceusage.services.get`).

**vCPUs are reported, never clamped.** Compute Engine's regional vCPU pool is shared with every
other VM in the project, so shrinking *your* cluster because something else is using the region is
a decision the product leaves to you — you get the sentence, not a silently smaller fleet. The one
exception is a region that cannot seat even a single worker, which is not a judgement call.

Two things that surprise people:

- **The head node counts.** An `n1-highmem-32` driver consumes 32 vCPUs before a single worker
  starts. On a 200-vCPU project that is 16% of everything you have, which is why the reference
  configs cap CPU workers at 20: `32 + (20 x 8) = 192`, and 200 is the ceiling. **This applies to
  the Dataproc path.** Ray on Vertex bills its vCPUs to `custom_model_training_cpus`, whose
  `us-central1` default is 2,200 — an order of magnitude looser, and the reason a Ray config can
  carry a 12-node GPU pool and a 20-node CPU pool at once (288 vCPU) without CPU ever binding.
- **Vertex's vCPU quota is per machine family.** `custom_model_training_cpus` covers N1 and E2 only;
  `n2`, `n4`, `c2`, `g2`, `a2`, `a3` and `m1` machines each have their own metric
  (`custom_model_training_g2_cpus`, and so on). Reading the N1 meter for a G2 pool reports a number
  about machines your run is not using. The reference configs are all N1, so the default meter is
  the right one unless you change `ray_machine_type`.
- **GPU nodes consume both.** Four T4 workers on `n1-standard-8` cost 4 T4s *and* 32 vCPUs. Raising
  `NVIDIA_T4_GPUS` without also raising `CPUS` will not get you more GPU workers.

When you file the request, ask for the number in the table plus headroom for a concurrent run.
Quota is granted per project and shared across everything in it — two runs at once need two runs'
worth.

---

## 5. Cores are the unit, and it took a bug to find out

This section used to advise the opposite of what it advises now, and the history is worth two
paragraphs because it is the reason to trust the current version.

On `ray-100k-dcc77a9d1e9b`, 37,500 tasks were queued against 20 eight-vCPU nodes and the resource
plan expected 4–8 concurrent cells per node. Measured concurrency was **0.97 cells per node**, flat
across the entire steady state — roughly one core in eight doing arithmetic. With no explanation for
that, the only safe advice was to buy nodes rather than cores.

**The explanation arrived on 2026-09-04.** A live read of a 10,000-series run showed every worker
holding 1.0 of 7.0 CPUs and 17.6 of its 18.1 GiB of memory. Ray schedules on memory as hard as it
schedules on cores, so a task asking for 97 % of a node's memory takes the whole node and the other
six cores are unreachable. The request came from a driver-side sizing pre-pass charging each task
the *driver's* memory footprint rather than a worker's. Fixed the same day, and the identical config
re-run under the fix went from 0.93 to 5.5 concurrent cells per node — **5.9x the density and 3.8x
the wall-clock at identical quota**. Recorded as `ray_slot_memory` in the
[validation ledger](validation.md).

So the advice inverts, and the numbers in this document are the post-fix ones:

- **Cores are the unit.** A node's worth of throughput is its core count divided by the cores a cell
  asks for, and you should expect a fleet to reach it. Twenty `n1-standard-8` workers and ten
  `n1-standard-16` workers are now roughly equivalent for the same vCPU spend; prefer the larger
  node where per-task overhead matters, and the smaller one where a single cell's memory footprint
  is large enough that a big node would be under-filled anyway.
- **Do not size a quota request off node count alone.** Ask for the vCPUs, then check the node shape
  divides into them sensibly.
- **Expect per-cell latency to get worse as density improves, and do not read that as a regression.**
  The re-run above made each `neuralprophet` fit 44 % slower (30.25 s → 43.48 s) while the fleet did
  4.1x the work. Contention between packed cells is what a well-used node looks like.

**If you see one busy core in N, suspect a memory request before you suspect the scheduler.** The
Ray dashboard's `/api/cluster_status` reports `usageByNode`, and a node pinned this way is obvious
in it: cores idle, memory at the ceiling. The product now names the condition itself —
`RuntimeResourcePlan.binding_axis` lands in the run's telemetry and the Ray engine logs a density
note at `WARNING` when memory rather than cores is what limits a pool.

---

## 6. Cheaper than more quota

Before filing for 1,150 vCPUs, three things cost nothing:

**Cut cells, not corners.** `cells = series x models x folds`. Dropping one expensive model from a
100,000-series run removes 100,000 cells. Backtesting with 2 folds doubles the run. Both are
one-line config changes, and both are usually the right answer during development — run the full
model set at 1,000 series, then the shortlist at 100,000.

**Let BigQuery-native models carry their share.** `arima_plus` and `timesfm` run *inside* BigQuery,
on BigQuery's own slots, and consume **none** of your Compute Engine quota. A run that mixes them
with Python models gets that family for free in vCPU terms, and it runs concurrently with the rest
of the DAG rather than after it.

**Use a second region — but deploy into it first.** Quota is regional, so if `us-central1` is capped
and the work splits naturally (different business units, say), two runs in two regions have two
independent allowances. This does not speed up a *single* run — `ray_regions` is failover, not
simultaneous execution — but it does raise total throughput across a workload.

The catch, and it is a real one: **a Ray cluster on the private path can only be created in the
region its network attachment lives in**, and Terraform builds one attachment. Listing three
`ray_regions` against a single-region deployment does not give you three candidates; it gives you
one, and two regions that 404 on the attachment before quota is ever consulted. `--quota` reports
those as `UNREACHABLE` and the launch path skips them. To make a second region real, deploy a
network attachment there too.

---

## See also

- [Deploying on GCP](deploying_on_gcp.md) — check your quotas before the first `terraform apply`
- [Configuration reference](configuration_reference.md) — `ray_cpu_max_nodes`, `max_executors`,
  `series_limit`, `bucket_target_cells`
- [Validation ledger](validation.md) — the runs the numbers on this page were measured from
