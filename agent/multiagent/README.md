# Sales Multi-Agent Orchestrator

A multi-agent orchestrator for Enterprise Sales Analytics, built with [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) and served via FastAPI on Databricks.

## Architecture

The orchestrator routes user queries to specialized subagents:

- **Genie** — queries a Databricks Genie Space for structured sales data analysis (pipeline, revenue, accounts, reps, products, etc.)
- **TabPFN** — connects to an external MCP server for tabular prediction tasks (classification and regression)

Additional subagent types (App agents, Serving Endpoints, MCP servers) can be added via `config.yaml`.

## Quick Start

1. Copy the environment template and fill in your values:

   ```bash
   cp .env.example .env
   ```

2. Edit `config.yaml` to configure your subagents (Genie space IDs, MCP connections, etc.).

3. Start the application:

   ```bash
   uv run start-app
   ```

## Scripts

| Command | Description |
|---|---|
| `uv run start-app` | Start the FastAPI application |
| `uv run start-server` | Start the agent server directly |
| `uv run agent-evaluate` | Run agent evaluation |
