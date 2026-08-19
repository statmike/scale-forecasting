# Documentation

The one-screen map. Find your task, follow the pointer.

## Deploy
Stand the platform up in a Google Cloud project.
- [deploying_on_gcp.md](./deploying_on_gcp.md) — a reviewer's guide to the Terraform: what gets
  created and why.
- [terraform/README.md](https://github.com/statmike/scale-forecasting/blob/main/terraform/README.md) — the two-stage apply runbook.

## Operate
Run forecasts, review results, and keep a deployment healthy.
- [running_and_reviewing.md](./running_and_reviewing.md) — **the run loop**: submit (Spark / Ray /
  BigQuery), watch it land, review the leaderboard, re-ensemble. Home of the `SF_*` identity setup.
- [operations.md](./operations.md) — rework/reset, disk hygiene, and long-running jobs on a
  persistent VM.

## Demo
Show the system end to end.
- [workshop.md](./workshop.md) — the guided walkthrough and notebook tour.

## Reference
How it works and every knob.
- [architecture.md](./architecture.md) — the module-calling-module call tree; start here to read the
  codebase.
- [configuration_reference.md](./configuration_reference.md) — every config field, type, default,
  constraint.
- [output_schemas.md](./output_schemas.md) — the output tables and the analyst views over them.
- [adding_a_model.md](./adding_a_model.md) + [model_template.py](https://github.com/statmike/scale-forecasting/blob/main/docs/model_template.py) — add a model
  in one file.
- [editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md) — why a code edit ships
  on the next run with no image rebuild.
- [version_matrix.md](./version_matrix.md) — the Python/Spark/Ray version of every surface, and why
  the whole system is pinned to Python 3.11.
- [notebook_runtimes.md](./notebook_runtimes.md) — which Python version each notebook needs and how
  it behaves locally and on Colab.

## SDK
Use it from Python.
- [using_the_sdk.md](./using_the_sdk.md) — the `Forecaster` easy path, and how to drive Spark/Ray
  directly (bypassing the SDK) while reusing the same model machinery.

## Troubleshooting
- [troubleshooting.md](./troubleshooting.md) — known issues, each symptom → cause → fix.
