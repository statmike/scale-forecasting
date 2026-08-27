# Editing code without rebuilding the image

The single most important thing to know as a data scientist working on this platform:

> **The runtime image contains dependencies, not your code.** Edit a model, an engine, the
> worker — anything in `src/` — and the *next run picks it up*. You never rebuild or re-push the
> container to try a code change.

This is a deliberate design guarantee, not an accident, and it's guarded by a test
(`tests/unit/test_code_delivery.py`) so it can't silently regress. Here's why it's true and how to
work with it.

## Why the image doesn't carry your code

The shared runtime image (`docker/Dockerfile`) does exactly one thing: `pip install` the locked
`requirements.txt`. It does **not** `COPY src`, and it does **not** `pip install` this package. So
the image is a frozen dependency environment — Python, Spark/Ray, statsmodels, xgboost, and the rest
— and nothing more. It only changes when the *dependencies* change (a new library, a version bump),
which is rare.

Your code is delivered **at submit time**, freshly, on every run:

| Runtime | How your `src/` reaches the workers |
|---------|-------------------------------------|
| **Dataproc (Spark)** | `submit_batch` zips `src/` and uploads it to the code bucket, then passes it on the batch's `python_file_uris`. A tiny `gs://` shim is the `__main__`; it imports the in-package logic from the uploaded zip. |
| **Ray on Vertex** | `ray_submit` ships `src/` as the job's `runtime_env.working_dir`, so every Ray worker imports the code you just submitted. |
| **The seed job** | Same pattern — Terraform's `seed` module zips `src/` and ships it on `python_file_uris`. |

In every case the image is the *environment* and `src/` is *cargo*. The two are decoupled on
purpose.

## The loop you actually use

```bash
# 1. Edit any file under src/ — a model, the worker, an engine. No build step.
$EDITOR src/scale_forecasting/models/theta.py

# 2. Run it. The submit helper zips your current src/ and ships it.
python -m scale_forecasting.submit --config configs/explode_demo.json

# 3. See the result on the leaderboard. Iterate. Repeat.
```

The same is true from a notebook: the notebooks import `scale_forecasting` from your working tree
(the "Get the code" bootstrap cell is a no-op when the package already imports), so a kernel restart
after an edit is all it takes.

**Prove it to yourself.** Change a model's behavior — e.g. flip a default in
`models/theta.py` — submit a run, and the metrics move, with **no** `docker build`, no Cloud Build,
no image push anywhere in the loop. That's the whole guarantee in one observation.

## When you *do* rebuild the image

Only when you change **dependencies**:

- add or remove a library, or bump a version → edit `pyproject.toml`, re-lock
  `requirements.txt`, and rebuild the image (Cloud Build, content-addressed on `docker/`, so it only
  actually rebuilds when the Dockerfile or the lock changes).

A pure code change never needs this. If you find yourself rebuilding the image to test a code edit,
something is wrong — check that you're not accidentally baking `src/` into the image (the
code-delivery test exists precisely to catch that).

## Why this matters

- **Fast iteration.** A code change costs a submit, not a multi-minute image build + push.
- **One image, many code revisions.** The whole team shares one cached dependency image; each run
  carries whatever `src/` the submitter had. No per-experiment images pile up in Artifact Registry.
- **Reproducibility stays intact.** The run config is staged verbatim to GCS (`--config-uri`) and
  logged to the registry, so *which config* ran is always recorded — independent of the image.
