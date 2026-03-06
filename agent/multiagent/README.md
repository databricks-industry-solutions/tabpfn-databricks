# Sales Multi-Agent Orchestrator

A multi-agent orchestrator that routes natural-language questions to the right backend — a Genie Space for SQL analytics, an external MCP server for TabPFN predictions, or any additional serving endpoint or Databricks App you configure.

## Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 18+ (for the chat UI)
- A Databricks workspace with a serverless SQL warehouse
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) authenticated (`databricks auth login` or a config profile)

## End-to-end setup

Everything below runs from the **repository root** unless noted otherwise.

### 1. Generate the sales data

Open `agent/00_generate_sales_data.ipynb` on your Databricks workspace (or attach it to a cluster) and run all cells. This creates seven Delta tables in `tabpfn_databricks.agent`.

### 2. Create the Genie Space

Run `agent/01_create_genie_space.ipynb` on the same workspace. It annotates the tables with descriptions and creates (or updates) a Genie Space. Copy the `space_id` printed at the end.

### 3. Configure the orchestrator

```bash
cd agent/multiagent
cp .env.example .env
```

Edit `.env`:

| Variable | What to set |
|---|---|
| `DATABRICKS_CONFIG_PROFILE` | Your CLI profile name (or set `DATABRICKS_HOST` + `DATABRICKS_TOKEN` instead) |
| `MLFLOW_EXPERIMENT_ID` | *(optional)* An MLflow experiment ID for tracing |

Then edit `config.yaml` and paste your Genie Space ID into the `space_id` field:

```yaml
subagents:
  - name: genie
    type: genie
    space_id: "<YOUR_GENIE_SPACE_ID>"   # from step 2
```

### 4. Run locally

```bash
cd agent/multiagent
uv run start-app          # backend (port 8000) + chat UI (port 3000)
uv run start-app --no-ui  # backend only
```

The chat UI is auto-cloned from `databricks/app-templates` on first run.

### 5. Deploy to Databricks

From the repo root:

```bash
databricks bundle deploy
databricks bundle run sales_multiagent
```

The bundle (`databricks.yml`) deploys the app and grants it access to the Genie Space and the TabPFN MCP connection.

### 6. Evaluate the agent

```bash
cd agent/multiagent
uv run agent-evaluate
```

Runs a conversation simulator with MLflow scorers (completeness, safety, fluency, etc.) and logs results to your MLflow experiment.

## Project structure

```
agent/
├── 00_generate_sales_data.ipynb   # Step 1 — create Delta tables
├── 01_create_genie_space.ipynb    # Step 2 — create Genie Space
└── multiagent/
    ├── config.yaml                # Subagent definitions (edit this)
    ├── .env.example               # Environment template
    ├── app.yaml                   # Databricks App manifest
    ├── pyproject.toml             # Dependencies & entry points
    ├── agent_server/
    │   ├── agent.py               # Orchestrator logic
    │   ├── utils.py               # MCP URL builder, auth helpers
    │   ├── start_server.py        # MLflow AgentServer entry point
    │   └── evaluate_agent.py      # Conversation simulator + scorers
    └── scripts/
        └── start_app.py           # Local dev launcher (backend + UI)
```

## Adding subagents

Edit `config.yaml` to wire up additional backends. Supported types:

| Type | Connects to | Key config |
|---|---|---|
| `genie` | Databricks Genie Space | `space_id` |
| `mcp` | External MCP server via UC connection | `connection_name` |
| `serving_endpoint` | Model Serving / agent endpoint | `endpoint` |
| `app` | Another Databricks App | `endpoint` |

See the commented examples in `config.yaml`.
