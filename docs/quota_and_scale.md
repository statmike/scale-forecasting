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

!!! note "Why per-node and not per-core"
    72 cells/min/node is **not** 9 cells/min/vCPU. On the run above, a node was fitting roughly one
    cell at a time despite having eight vCPUs, so throughput scaled with *nodes*, not with cores.
    That is a known inefficiency, not a design intent — see
    [the caveat below](#5-the-caveat-nodes-are-the-unit-not-cores). Plan with nodes.

**Deep learning is a different regime entirely.** `neuralprophet` measures at **21–65 s/fit**
against sub-second statistical models — 50x or more — and it needs a GPU. Across the GPU smokes it
lands at roughly **4 cells/min per T4 node**. Treat that number as soft: no deep-learning run has
yet executed above **100 series**, so it is an extrapolation from small samples, and cluster
start-up is a large fraction of those spans.

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

`neuralprophet` needs a T4, fits ~50x slower than a statistical model, and the default
`NVIDIA_T4_GPUS` allowance is **4**. Those three facts together are the hardest limit in this
product. At ~4 cells/min per T4, one deep-learning model over the same series counts:

| Series | Wall clock on the **default 4 T4s** | T4s for a **~1-hour** run |
|---|---|---|
| **100** | ~25 s | 1 |
| **1,000** | ~1 hour | 5 |
| **10,000** | **~10 hours** | 42 |
| **100,000** | **~104 hours** | 417 |
| **1,000,000** | ~6 weeks | 4,167 |

**A default project runs out of deep-learning headroom at around 1,000 series**, two orders of
magnitude before it runs out of CPU headroom. This is why `all_families_10k.json` is the largest
config here that includes `neuralprophet`, and why it is expected to take most of a working day: at
10,000 series the deep-learning family alone is ~10 hours, while every other family finishes in
under one.

Treat these figures as **the softest numbers on this page.** No deep-learning run in the
[validation ledger](validation.md) has yet exceeded **100 series**, so the 4 cells/min/T4 anchor is
extrapolated from small samples in which cluster start-up was a large fraction of the span. The true
steady-state figure is probably better. Plan with these, then measure your own.

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
| `NVIDIA_T4_GPUS` | Deep-learning worker nodes | 4 |
| `PREEMPTIBLE_NVIDIA_T4_GPUS` | The same, for preemptible pools | 4 |
| `DISKS_TOTAL_GB` | Boot disks across the fleet | 40,960 |
| `IN_USE_ADDRESSES` | Rarely binding — clusters are private | 69 |

Two things that surprise people:

- **The head node counts.** An `n1-highmem-32` driver consumes 32 vCPUs of your `CPUS` allowance
  before a single worker starts. On a 200-vCPU project that is 16% of everything you have, which is
  why the reference configs cap workers at 20: `32 + (20 x 8) = 192`, and 200 is the ceiling.
- **GPU nodes consume both.** Four T4 workers on `n1-standard-8` cost 4 T4s *and* 32 vCPUs. Raising
  `NVIDIA_T4_GPUS` without also raising `CPUS` will not get you more GPU workers.

When you file the request, ask for the number in the table plus headroom for a concurrent run.
Quota is granted per project and shared across everything in it — two runs at once need two runs'
worth.

---

## 5. The caveat: nodes are the unit, not cores

The 72 cells/min/node figure comes with an asterisk that is worth understanding, because it means
**the table above is conservative and the product should get faster without you asking for
anything.**

On `ray-100k-dcc77a9d1e9b`, 37,500 tasks were queued against 20 eight-vCPU nodes, and the resource
plan expected 4–8 concurrent cells per node. Measured concurrency was **0.97 cells per node**, flat
across the entire steady state. Roughly one core in eight was doing arithmetic.

That gap is under investigation and is recorded as an open item in the
[validation ledger](validation.md). Until it closes, throughput scales with **node count**, so:

- **Prefer more, smaller nodes over fewer, larger ones.** Twenty `n1-standard-8` workers currently
  outperform ten `n1-standard-16` workers, even though the vCPU count is identical and the second
  arrangement costs the same. This is the opposite of the advice you would give for a well-packed
  fleet, and it will stop being true when the packing is fixed.
- **Do not size a quota request off vCPU count alone.** Ask for the vCPUs that buy you the *nodes*
  in the table.

When per-node packing improves, every wall-clock number here drops and every quota number with it.
Nothing in the table becomes wrong — it becomes pessimistic, which is the safe direction for a
document you are using to justify a spend.

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

**Use a second region.** Quota is regional. If `us-central1` is capped and the work splits naturally
(different business units, say), two runs in two regions have two independent allowances. This does
not speed up a *single* run — `ray_regions` is failover, not simultaneous execution — but it does
raise total throughput across a workload.

---

## See also

- [Deploying on GCP](deploying_on_gcp.md) — check your quotas before the first `terraform apply`
- [Configuration reference](configuration_reference.md) — `ray_cpu_max_nodes`, `max_executors`,
  `series_limit`, `bucket_target_cells`
- [Validation ledger](validation.md) — the runs the numbers on this page were measured from
