# Welcome to `scale-forecasting` in Google Cloud Shell

This interactive tutorial walks you through **`scale-forecasting`** — a declarative, multi-runtime time-series forecasting platform for Google Cloud that orchestrates **34 models** across **5 model families**, **21 evaluation metrics**, and **7 compute runtimes** from a single `RunConfig` JSON contract.

You will complete three steps in ~5 minutes:
1. **Step 1 (Zero GCP Required):** Install the pure offline core, probe your environment, and run an offline model benchmark in the playground.
2. **Step 2 (Zero GCP Required):** Validate a multi-family `RunConfig`, inspect its deterministic content-addressed `run_id` and per-family DAG, and query the built-in Model Context Protocol (MCP) server.
3. **Step 3 (Optional Cloud Run):** Connect your Google Cloud project to run preflight checks or provision the BigQuery + Iceberg demo warehouse.

---

## Step 1: Install & Run the Offline Model Playground

Create a virtual environment and install the core package (plus optional statistical/tree model extras if desired):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Inspect what is installed and ready in your Cloud Shell environment using the built-in environment probe:

```bash
python -m scale_forecasting.agent_surfaces --probe-env
```

Now run a real forecasting model (`holtwinters`) on deterministic synthetic panel data with a 3-fold expanding-window backtest — zero cloud credentials required:

```bash
python -m scale_forecasting.playground --model holtwinters --horizon 14 --backtest
```

You can also list all available models in the active environment:

```bash
python -m scale_forecasting.playground --list
```

---

## Step 2: Dry-Run a Multi-Family `RunConfig` & Test the Built-In MCP Server

Every pipeline in `scale-forecasting` is defined by a declarative `RunConfig` JSON file (validated against [`docs/schemas/run_config.schema.json`](./docs/schemas/run_config.schema.json)).

Run an offline dry-run of [`configs/ensemble_demo.json`](./configs/ensemble_demo.json) to compute its deterministic `run_id`, estimate total cell fits, and preview the launch plan:

```bash
python -m scale_forecasting.main --config configs/ensemble_demo.json --dry-run
```

Render a self-contained Cloud Composer 3 / Apache Airflow DAG (`dag_<run_id>.py`) offline with surgical cell-repair (`--with-retry`) wired in:

```bash
python -m scale_forecasting.main --config configs/ensemble_demo.json --emit-airflow --with-retry
```

### Connect Google Antigravity (`agy` CLI & IDE), Claude Code, or Any MCP Agent

Because this repository includes [`.agents/skills.json`](./.agents/skills.json), [`plugin.json`](./plugin.json), [`mcp_config.json`](./mcp_config.json), [`.mcp.json`](./.mcp.json), and [`skills/scale-forecasting/SKILL.md`](./skills/scale-forecasting/SKILL.md), you can test the built-in MCP server (`scale_forecasting.mcp`) right from Cloud Shell:

```bash
python -m scale_forecasting.mcp --probe-env
```

Or launch Google Antigravity (`agy` CLI) or register the MCP server in Claude Code:

```bash
# Google Antigravity CLI (auto-discovers .agents/skills/ & AGENTS.md):
agy

# Claude Code (auto-discovers .mcp.json at repo root, or register globally):
claude mcp add scale-forecasting -- python3 -m scale_forecasting.mcp
```

---

## Step 3: Optional — Connect Google Cloud & Run Live Preflight Checks

If you want to run against BigQuery and Google Cloud compute runtimes (`spark`, `ray`, `vertex`, `gce`, `gke`, `vertex_automl`, `bigquery`), install the `[gcp]` extra:

```bash
pip install -e ".[gcp]"
```

If you have already provisioned the demo infrastructure via [`terraform/`](./terraform/README.md) (or have an existing `.env.infra` file), load your environment variables and run the registry doctor and quota preflight check:

```bash
# Load SF_PROJECT_ID, SF_CONNECTION, SF_WAREHOUSE_URI from .env.infra if present:
if [ -f .env.infra ]; then
  set -a && source .env.infra && set +a
  python -m scale_forecasting.registry.ops doctor
  python -m scale_forecasting.main --config configs/ensemble_demo.json --dry-run --feasibility
else
  echo "No .env.infra found yet. See docs/getting_started.md or terraform/README.md to provision a demo project."
fi
```

---

## Next Steps

<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

- **Full Documentation Site:** <https://statmike.github.io/scale-forecasting/>
- **AI Agents, Skills & MCP Guide:** [`docs/agent_and_mcp_guide.md`](./docs/agent_and_mcp_guide.md)
- **Interactive Notebooks (`00`–`10`):** [`notebooks/README.md`](./notebooks/README.md)
- **Choosing a Runtime:** [`docs/choosing_a_runtime.md`](./docs/choosing_a_runtime.md)
