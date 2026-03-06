"""
Multi-agent orchestrator with Genie subagent for Enterprise Sales Analytics.

Configuration is loaded from ``config.yaml`` at the project root.  Edit that
file to set your Genie Space ID, add serving-endpoint subagents, or change the
orchestrator model — no code changes required.
"""

from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncGenerator

import mlflow
import yaml
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_user_workspace_client,
    process_agent_stream_events,
    sanitize_output_items,
)

# ---------------------------------------------------------------------------
# Patch: Databricks external MCP proxy returns NDJSON (multiple JSON-RPC
# messages on separate lines) but the MCP client expects a single object.
# Split on newlines and deliver each message individually.
# ---------------------------------------------------------------------------
import logging as _logging

import httpx as _httpx
from mcp.client.streamable_http import StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

_patch_logger = _logging.getLogger(__name__)

_original_handle_json = StreamableHTTPTransport._handle_json_response


async def _patched_handle_json_response(
    self,
    response: _httpx.Response,
    read_stream_writer,
    is_initialization: bool = False,
) -> None:
    content = await response.aread()
    lines = [line for line in content.split(b"\n") if line.strip()]
    if len(lines) <= 1:
        return await _original_handle_json(
            self, response, read_stream_writer, is_initialization
        )
    _patch_logger.debug("NDJSON response with %d messages — splitting", len(lines))
    for line in lines:
        try:
            message = JSONRPCMessage.model_validate_json(line)
            if is_initialization:
                self._maybe_extract_protocol_version_from_message(message)
            await read_stream_writer.send(SessionMessage(message))
        except Exception:
            _patch_logger.exception("Error parsing NDJSON line: %s", line[:200])


StreamableHTTPTransport._handle_json_response = _patched_handle_json_response

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

SUBAGENTS: list[dict] = _config.get("subagents", [])
ORCHESTRATOR_MODEL: str = _config.get("orchestrator_model", "databricks-claude-sonnet-4-5")

assert SUBAGENTS, (
    f"Configure at least one subagent in {_CONFIG_PATH}. "
    "Uncomment an entry and replace placeholder values."
)

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

_tool_client = AsyncDatabricksOpenAI()

# ---------------------------------------------------------------------------
# Subagent tools — one tool per non-genie subagent
# ---------------------------------------------------------------------------


def _make_subagent_tool(subagent: dict):
    """Create a function_tool for a single subagent definition."""
    endpoint = subagent["endpoint"]
    model = f"apps/{endpoint}" if subagent["type"] == "app" else endpoint

    async def _call(question: str) -> str:
        response = await _tool_client.responses.create(
            model=model,
            input=[{"role": "user", "content": question}],
        )
        return response.output_text

    _call.__name__ = f"query_{subagent['name']}"
    _call.__doc__ = subagent["description"]
    return function_tool(_call)


_MCP_TYPES = {"genie", "mcp"}
subagent_tools = [_make_subagent_tool(sa) for sa in SUBAGENTS if sa["type"] not in _MCP_TYPES]


# ---------------------------------------------------------------------------
# MCP server + orchestrator agent
# ---------------------------------------------------------------------------


_DEFAULT_MCP_TIMEOUT = 20
_EXTERNAL_MCP_TIMEOUT = 120


async def init_mcp_servers():
    """Create MCP servers for all genie and external mcp subagents."""
    servers = []
    for sa in SUBAGENTS:
        if sa["type"] == "genie":
            servers.append(McpServer(
                url=build_mcp_url(f"/api/2.0/mcp/genie/{sa['space_id']}"),
                name=sa["description"],
                client_session_timeout_seconds=_DEFAULT_MCP_TIMEOUT,
            ))
        elif sa["type"] == "mcp":
            servers.append(McpServer(
                url=build_mcp_url(f"/api/2.0/mcp/external/{sa['connection_name']}"),
                name=sa.get("server_name", sa["name"]),
                client_session_timeout_seconds=sa.get(
                    "timeout_seconds", _EXTERNAL_MCP_TIMEOUT
                ),
            ))
    return servers


def _build_instructions() -> str:
    """Dynamically build orchestrator instructions from the configured subagents."""
    lines = [
        "You are an orchestrator agent for Enterprise Sales Analytics.",
        "Route the user's request to the most appropriate tool or data source:\n",
    ]
    for sa in SUBAGENTS:
        if sa["type"] == "genie":
            lines.append(
                f"- Use the Genie MCP tools for: {sa['description'].strip()}"
            )
        elif sa["type"] == "mcp":
            lines.append(
                f"- Use the {sa['name']} MCP tools for: {sa['description'].strip()}"
            )
        else:
            lines.append(
                f"- Use query_{sa['name']} for: {sa['description'].strip()}"
            )
    lines.append(
        "\nCRITICAL CONSTRAINTS:"
        "\n- 120-second hard timeout. Each Genie query+poll ≈ 15 s."
        "\n- Workspace token-per-second rate limits are STRICT. Every "
        "extra query risks a 429 REQUEST_LIMIT_EXCEEDED error that "
        "kills the entire conversation. Minimize total tool calls."
        "\n\nTABPFN FOR PREDICTIVE ANALYSIS (MANDATORY):"
        "\nUse TabPFN for ANY request involving: predict, forecast, "
        "recommend, suggest, classify, score, rank, likelihood, "
        "probability, propensity, win probability, churn, next best "
        "action, cross-sell, upsell, what-if, or scenario modeling. "
        "Do NOT predict outcomes using LLM reasoning — always delegate "
        "to TabPFN when structured data is available."
        "\n\n=== ABSOLUTE QUERY BUDGET: 1 GENIE QUERY, 0 RETRIES ==="
        "\nYou may call query_space EXACTLY ONCE and poll_response "
        "EXACTLY ONCE. That is your entire Genie budget. After the "
        "poll returns, you MUST proceed to TabPFN (or final answer) "
        "with whatever data you received — even if columns are missing, "
        "renamed, or NULL. NEVER send a second query_space call. "
        "NEVER send an exploratory or schema-discovery query."
        "\n\nGENIE QUERY RULES:"
        "\n1. Plan BEFORE writing SQL: list tables and columns needed."
        "\n2. Write a single flat SELECT … JOIN … JOIN … WHERE. "
        "Genie has an AI layer that rewrites SQL. Complex CTEs, "
        "UNION ALL, or scalar subqueries cause Genie to drop joins "
        "or replace columns. Keep it simple so Genie preserves it."
        "\n3. NEVER embed test/prediction rows via UNION ALL. Fetch "
        "ONLY training data. Build X_test in-context."
        "\n4. Always LIMIT 50 ORDER BY RAND()."
        "\n5. For promotion analysis, use this exact pattern:"
        "\n   SELECT a.segment, a.annual_revenue_mm, a.industry,"
        "\n     o.acv, o.lead_source, o.days_in_pipeline,"
        "\n     p.promotion_type,"
        "\n     CASE WHEN o.stage='Closed/Won' THEN 1 ELSE 0 END AS won"
        "\n   FROM opportunities o"
        "\n   JOIN accounts a ON o.account_id = a.account_id"
        "\n   JOIN promotions p ON o.opportunity_id = p.opportunity_id"
        "\n   WHERE o.stage IN ('Closed/Won','Closed/Lost')"
        "\n   ORDER BY RAND() LIMIT 50"
        "\n\nHANDLING GENIE REWRITES (IMPORTANT):"
        "\nGenie may silently rewrite your SQL — dropping JOINs, "
        "renaming columns, or replacing a column (e.g. returning "
        "has_promotion instead of promotion_type). When this happens:"
        "\n- Do NOT retry. You have 0 retries."
        "\n- If promotion_type is missing but has_promotion exists, "
        "use has_promotion (true/false) as the feature instead and "
        "build X_test with [has_promotion=true] for each scenario."
        "\n- If key columns are missing entirely, provide a qualitative "
        "recommendation based on the account profile and available "
        "data. Explain that the Genie Space may need to be updated "
        "to include the promotions table."
        "\n\nFULL WORKFLOW (exactly 3 tool calls):"
        "\n  1. query_space  → flat SELECT with JOINs, LIMIT 50"
        "\n  2. poll_response → get results"
        "\n  3. fit_and_predict_inline → X_train, y_train, X_test"
        "\nBuild X_test in-context from the user's question (e.g., "
        "one row per promotion type scenario for the target account)."
        "\n\nTABPFN DATA PREP:"
        "\n- Each X_test row must differ in at least one feature."
        "\n- Build X_train and y_train in LOCKSTEP. Confirm counts match."
        "\n- NEVER retry fit_and_predict_inline."
        "\n\nRESPONSE FORMAT: Keep answers concise — 3-5 key findings "
        "with numbers, then a clear recommendation. No long essays."
        "\n\nIf the user's question doesn't match any tool, ask for "
        "clarification."
    )
    return "\n".join(lines)


def create_orchestrator_agent(mcp_servers: list[McpServer]) -> Agent:
    """Build the orchestrator agent with all tools and MCP servers."""
    return Agent(
        name="Orchestrator",
        instructions=_build_instructions(),
        model=ORCHESTRATOR_MODEL,
        mcp_servers=mcp_servers,
        tools=subagent_tools,
    )


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


def _normalize_input(request: ResponsesAgentRequest) -> list[dict]:
    """Convert request input to dicts the agents SDK can consume.

    The agents SDK ``Converter.items_to_messages`` expects assistant message
    ``content`` to be a list of content-part dicts (e.g.
    ``[{"type": "output_text", "text": "…"}]``).  Callers that send plain
    string content on assistant turns (the Responses API "easy" format) would
    otherwise crash the converter.  This helper normalises such messages.
    """
    messages = [i.model_dump() for i in request.input]
    for msg in messages:
        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
        ):
            msg["content"] = [{"type": "output_text", "text": msg["content"]}]
    return messages


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(s) for s in await init_mcp_servers()]
        agent = create_orchestrator_agent(mcp_servers)
        messages = _normalize_input(request)
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=sanitize_output_items(result.new_items))


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(s) for s in await init_mcp_servers()]
        agent = create_orchestrator_agent(mcp_servers)
        messages = _normalize_input(request)
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
