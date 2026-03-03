# Sales Multi-Agent Orchestrator

Multi-agent orchestrator for Enterprise Sales Analytics, deployed as a Databricks App.

## Architecture

The orchestrator routes user questions to the most appropriate backend:

| Backend | Type | How it's queried |
|---------|------|-----------------|
| **Genie Space** | Structured data | Databricks MCP server |
| **Serving Endpoint** | Model inference | Responses API |
| **Databricks App** | Specialist agent | Responses API |

## Configuration

Edit `config.yaml` to set your Genie Space ID and add additional subagents.

## Local Development

```bash
cp .env.example .env
# Fill in .env values
uv run start-app
```

## Deployment

```bash
# From project root
databricks bundle deploy
databricks bundle run sales_multiagent
```
